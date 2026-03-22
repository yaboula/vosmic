"""Integration tests for VOSMIC entry-point wiring and new pipeline features.

These tests validate the glue code without requiring audio hardware or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.config import VosmicConfig
from src.core.pipeline import VosmicPipeline
from src.dsp.noise_gate import NoiseGate
from src.gui.event_bus import EventBus
from src.postprocessing.post_processor import PostProcessor


# ---------- Shared EventBus ----------


class TestSharedEventBus:
    """Verify that the pipeline uses an externally provided EventBus."""

    def test_pipeline_accepts_external_event_bus(self) -> None:
        bus = EventBus()
        p = VosmicPipeline(VosmicConfig(), event_bus=bus)
        assert p.event_bus is bus

    def test_pipeline_creates_own_bus_when_none(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        assert isinstance(p.event_bus, EventBus)

    def test_gui_command_reaches_pipeline_via_shared_bus(self) -> None:
        bus = EventBus()
        p = VosmicPipeline(VosmicConfig(), event_bus=bus)
        pp = MagicMock(spec=PostProcessor)
        p.post_processor = pp

        # Simulate GUI sending a command through the same bus
        bus.send_command("set_bypass", {"enabled": True})
        p.process_commands()
        pp.set_bypass.assert_called_once_with(True)


# ---------- set_noise_gate Command ----------


class TestSetNoiseGateCommand:
    """Verify that set_noise_gate modifies the noise gate threshold at runtime."""

    def test_noise_gate_threshold_updated(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        gate = NoiseGate(threshold_db=-40.0)
        p._noise_gate = gate

        p.event_bus.send_command("set_noise_gate", {"threshold_db": -30.0})
        p.process_commands()

        # NoiseGate converts dB to linear internally; verify change occurred
        expected_linear = 10.0 ** (-30.0 / 20.0)
        assert abs(gate._threshold_linear - expected_linear) < 1e-6

    def test_noise_gate_default_threshold(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        gate = NoiseGate(threshold_db=-20.0)
        p._noise_gate = gate

        # Send without explicit threshold_db -> uses default -40
        p.event_bus.send_command("set_noise_gate", {})
        p.process_commands()

        expected_linear = 10.0 ** (-40.0 / 20.0)
        assert abs(gate._threshold_linear - expected_linear) < 1e-6

    def test_noise_gate_none_does_not_crash(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        p._noise_gate = None
        p.event_bus.send_command("set_noise_gate", {"threshold_db": -20.0})
        p.process_commands()  # Should not raise


# ---------- model_loaded Event Data ----------


class TestModelLoadedEventData:
    """Verify that model_loaded event contains real format and VRAM data."""

    def test_model_loaded_has_format_and_vram(self) -> None:
        bus = EventBus()
        p = VosmicPipeline(VosmicConfig(), event_bus=bus)

        mock_mm = MagicMock()
        mock_mm.model_info = {"name": "test_model", "format": "onnx", "precision": "fp16"}
        mock_mm.get_vram_usage.return_value = {"used_mb": 512.0, "total_mb": 8192.0}
        p.model_manager = mock_mm

        received: list[dict] = []
        bus.subscribe("model_loaded", received.append)

        bus.send_command("load_model", {"model_path": "test.onnx"})
        p.process_commands()

        assert len(received) == 1
        assert received[0]["format"] == "onnx"
        assert received[0]["vram_mb"] == 512.0
        assert received[0]["name"] == "test_model"

    def test_model_load_error_emits_event(self) -> None:
        bus = EventBus()
        p = VosmicPipeline(VosmicConfig(), event_bus=bus)

        mock_mm = MagicMock()
        mock_mm.swap_model.side_effect = RuntimeError("GPU OOM")
        p.model_manager = mock_mm

        errors: list[dict] = []
        bus.subscribe("model_load_error", errors.append)

        bus.send_command("load_model", {"model_path": "bad.onnx"})
        p.process_commands()

        assert len(errors) == 1
        assert "GPU OOM" in errors[0]["error"]


# ---------- emit_threadsafe ----------


class TestEmitThreadsafe:
    """Verify the emit_threadsafe method exists and falls back correctly."""

    def test_emit_threadsafe_method_exists(self) -> None:
        bus = EventBus()
        assert hasattr(bus, "emit_threadsafe")
        assert callable(bus.emit_threadsafe)

    def test_emit_threadsafe_falls_back_without_qt(self) -> None:
        """When Qt is not running, emit_threadsafe should fall back to emit."""
        bus = EventBus()
        received: list[dict] = []
        bus.subscribe("test_event", received.append)

        # Patch the import inside emit_threadsafe to raise ImportError
        with patch.dict("sys.modules", {"PyQt6.QtCore": None}):
            bus.emit_threadsafe("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0]["key"] == "value"
