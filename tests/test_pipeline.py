"""Unit tests for VosmicPipeline — orchestrator logic without audio hardware."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.core.config import VosmicConfig
from src.core.pipeline import VosmicPipeline
from src.postprocessing.post_processor import PostProcessor


class TestPipelineInit:
    def test_creates_buffers(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        assert p.input_buffer.capacity >= 4096
        assert p.output_buffer.capacity >= 4096

    def test_creates_queues(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        assert p.dsp_to_inference_queue is not None
        assert p.inference_to_post_queue is not None

    def test_not_running_initially(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        assert p.is_running is False


class TestProcessCommands:
    def test_bypass_command(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        pp = MagicMock(spec=PostProcessor)
        p.post_processor = pp
        p.event_bus.send_command("set_bypass", {"enabled": True})
        p.process_commands()
        pp.set_bypass.assert_called_once_with(True)

    def test_unknown_command_does_not_crash(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        p.event_bus.send_command("nonexistent", {"x": 1})
        p.process_commands()

    def test_shutdown_stops_pipeline(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        p._is_running = True
        p.event_bus.send_command("shutdown", {})
        p.process_commands()
        assert p.is_running is False


class TestMetrics:
    def test_get_full_metrics_empty(self) -> None:
        p = VosmicPipeline(VosmicConfig())
        m = p.get_full_metrics()
        assert m["is_running"] is False
        assert "estimated_latency_ms" in m

    def test_latency_estimate_includes_block_time(self) -> None:
        config = VosmicConfig()
        config.audio.block_size = 256
        config.audio.sample_rate = 48000
        p = VosmicPipeline(config)
        latency = p._estimate_total_latency()
        block_ms = 256 / 48000 * 1000 * 2
        assert latency >= block_ms - 0.01
