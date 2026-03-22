"""Integration tests for error recovery scenarios."""

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


class TestBufferOverflowRecovery:
    def test_overflow_does_not_crash(self) -> None:
        buf = LockFreeRingBuffer(256)
        big_data = np.ones(512, dtype=np.float32)
        ok = buf.write(big_data)
        assert ok is False
        assert buf.available_read == 0

    def test_buffer_recovers_after_overflow(self) -> None:
        buf = LockFreeRingBuffer(256)
        buf.write(np.ones(256, dtype=np.float32))
        overflow = buf.write(np.ones(64, dtype=np.float32))
        assert overflow is False

        data = buf.read(256)
        assert len(data) == 256
        assert buf.available_read == 0

        ok = buf.write(np.ones(128, dtype=np.float32))
        assert ok is True


class TestCorruptModelRejection:
    def test_bad_model_emits_error(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        mm = MagicMock()
        mm.swap_model.side_effect = RuntimeError("corrupt model data")
        p.model_manager = mm

        errors: list[dict] = []
        p.event_bus.subscribe("model_load_error", lambda d: errors.append(d))
        p.event_bus.send_command("load_model", {"model_path": "/corrupt.onnx"})
        p.process_commands()

        assert len(errors) == 1
        assert "corrupt" in errors[0]["error"]


class TestVRAMExhaustionPassthrough:
    def test_inference_failure_uses_passthrough(self) -> None:
        q: queue.Queue = queue.Queue()
        buf = LockFreeRingBuffer(4096)
        pp = MagicMock(spec=PostProcessor)
        pp.process.side_effect = MemoryError("CUDA OOM")

        audio = np.ones(256, dtype=np.float32) * 0.5
        result = InferenceResult(
            audio=audio,
            original_audio=audio,
            original_rms=0.3,
            was_passthrough=True,
            inference_time_ms=0.0,
            chunk_id=0,
        )

        ot = OutputThread(q, buf, pp)
        ot.start()
        q.put(result)
        time.sleep(0.1)
        ot.stop()

        assert buf.available_read >= 256


class TestPipelineGracefulDegradation:
    def test_stop_without_start_is_safe(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        p.stop()
        assert p.is_running is False

    def test_double_stop_is_safe(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        p._is_running = True
        p.stop()
        p.stop()
        assert p.is_running is False

    def test_commands_without_modules_is_safe(self) -> None:
        config = VosmicConfig()
        p = VosmicPipeline(config)
        p.event_bus.send_command("set_input_device", {"device_index": 0})
        p.event_bus.send_command("set_bypass", {"enabled": True})
        p.process_commands()
