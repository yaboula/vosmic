"""Integration tests for VosmicPipeline — E2E pipeline lifecycle and data flow."""

from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock

import numpy as np

from src.audio_io.ring_buffer import LockFreeRingBuffer
from src.core.config import VosmicConfig
from src.core.pipeline import OutputThread, VosmicPipeline
from src.inference.scheduler import InferenceResult
from src.postprocessing.post_processor import PostProcessor


class TestOutputThread:
    def test_start_and_stop(self) -> None:
        q: queue.Queue = queue.Queue()
        buf = LockFreeRingBuffer(1024)
        pp = MagicMock(spec=PostProcessor)
        pp.process.return_value = np.zeros(256, dtype=np.float32)

        ot = OutputThread(q, buf, pp)
        ot.start()
        assert ot.metrics["is_running"] is True
        ot.stop()
        assert ot.metrics["is_running"] is False

    def test_writes_processed_audio(self) -> None:
        q: queue.Queue = queue.Queue()
        buf = LockFreeRingBuffer(4096)
        pp = MagicMock(spec=PostProcessor)
        audio = np.ones(256, dtype=np.float32) * 0.5
        pp.process.return_value = audio

        ot = OutputThread(q, buf, pp)
        ot.start()

        result = InferenceResult(
            audio=audio,
            original_audio=audio,
            original_rms=0.3,
            was_passthrough=False,
            inference_time_ms=5.0,
            chunk_id=0,
        )
        q.put(result)
        time.sleep(0.1)
        ot.stop()

        assert buf.available_read >= 256
        assert ot.metrics["chunks_written"] >= 1

    def test_postprocess_failure_forwards_original(self) -> None:
        q: queue.Queue = queue.Queue()
        buf = LockFreeRingBuffer(4096)
        pp = MagicMock(spec=PostProcessor)
        pp.process.side_effect = RuntimeError("boom")
        audio = np.ones(256, dtype=np.float32) * 0.3

        ot = OutputThread(q, buf, pp)
        ot.start()

        result = InferenceResult(
            audio=audio,
            original_audio=audio,
            original_rms=0.2,
            was_passthrough=False,
            inference_time_ms=5.0,
            chunk_id=0,
        )
        q.put(result)
        time.sleep(0.1)
        ot.stop()

        assert buf.available_read >= 256


class TestPipelineInit:
    def test_initial_state(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        assert p.is_running is False
        assert p.capture is None
        assert p.output is None

    def test_event_bus_exists(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        assert p.event_bus is not None

    def test_buffers_created(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        assert p.input_buffer is not None
        assert p.output_buffer is not None


class TestCommandHandling:
    def test_bypass_command(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        pp = MagicMock(spec=PostProcessor)
        p.post_processor = pp

        p.event_bus.send_command("set_bypass", {"enabled": True})
        p.process_commands()

        pp.set_bypass.assert_called_with(True)

    def test_shutdown_command(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        p._is_running = True

        p.event_bus.send_command("shutdown", {})
        p.process_commands()

        assert p._is_running is False

    def test_model_load_error_emits_event(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        mm = MagicMock()
        mm.swap_model.side_effect = FileNotFoundError("not found")
        p.model_manager = mm

        errors: list[dict] = []
        p.event_bus.subscribe("model_load_error", lambda d: errors.append(d))
        p.event_bus.send_command("load_model", {"model_path": "/bad.onnx"})
        p.process_commands()

        assert len(errors) == 1
        assert "not found" in errors[0]["error"]

    def test_multiple_commands_drained(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        pp = MagicMock(spec=PostProcessor)
        p.post_processor = pp

        p.event_bus.send_command("set_bypass", {"enabled": True})
        p.event_bus.send_command("set_bypass", {"enabled": False})
        p.process_commands()

        assert pp.set_bypass.call_count == 2


class TestFullMetrics:
    def test_metrics_includes_running(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        m = p.get_full_metrics()
        assert "is_running" in m
        assert m["is_running"] is False

    def test_metrics_includes_latency(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        m = p.get_full_metrics()
        assert "estimated_latency_ms" in m
