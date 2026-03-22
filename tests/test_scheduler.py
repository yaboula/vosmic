"""Tests for InferenceScheduler — FIFO queue, passthrough, timing."""

from __future__ import annotations

import queue
import time

import numpy as np

from src.dsp.feature_bundle import FeatureBundle
from src.inference.inference_core import InferenceEngine
from src.inference.model_manager import ModelManager
from src.inference.scheduler import InferenceResult, InferenceScheduler


def _make_bundle(is_silence: bool = False, chunk_id: int = 0) -> FeatureBundle:
    audio = (
        np.zeros(256, dtype=np.float32)
        if is_silence
        else np.random.randn(256).astype(np.float32) * 0.1
    )
    return FeatureBundle(
        audio=audio,
        f0=np.zeros(10, dtype=np.float32),
        content_embedding=np.zeros((10, 256), dtype=np.float32),
        original_rms=0.0 if is_silence else 0.1,
        is_silence=is_silence,
        timestamp=0.0,
        chunk_id=chunk_id,
    )


def _make_scheduler(
    in_size: int = 16,
    out_size: int = 16,
) -> tuple[InferenceScheduler, queue.Queue, queue.Queue]:
    mm = ModelManager(device="cpu")
    engine = InferenceEngine(mm)
    in_q: queue.Queue[FeatureBundle] = queue.Queue(maxsize=in_size)
    out_q: queue.Queue[InferenceResult] = queue.Queue(maxsize=out_size)
    sched = InferenceScheduler(engine, in_q, out_q, max_inference_ms=25.0)
    return sched, in_q, out_q


class TestSchedulerStartStop:
    def test_start_and_stop(self) -> None:
        sched, _, _ = _make_scheduler()
        sched.start()
        assert sched.is_running is True
        time.sleep(0.05)
        sched.stop()
        assert sched.is_running is False

    def test_stop_within_2s(self) -> None:
        sched, _, _ = _make_scheduler()
        sched.start()
        t0 = time.perf_counter()
        sched.stop()
        assert time.perf_counter() - t0 < 2.0


class TestPassthroughOnNoModel:
    def test_no_model_passthrough(self) -> None:
        sched, in_q, out_q = _make_scheduler()
        in_q.put(_make_bundle(chunk_id=1))

        sched.start()
        time.sleep(0.2)
        sched.stop()

        assert not out_q.empty()
        result = out_q.get_nowait()
        assert isinstance(result, InferenceResult)
        assert result.was_passthrough is True
        assert result.chunk_id == 1
        assert len(result.audio) == 256


class TestSilenceForwarding:
    def test_silence_produces_zeros(self) -> None:
        sched, in_q, out_q = _make_scheduler()
        in_q.put(_make_bundle(is_silence=True, chunk_id=5))

        sched.start()
        time.sleep(0.2)
        sched.stop()

        assert not out_q.empty()
        result = out_q.get_nowait()
        assert result.was_passthrough is True
        assert np.all(result.audio == 0.0)
        assert result.chunk_id == 5


class TestMultipleBundles:
    def test_processes_multiple(self) -> None:
        sched, in_q, out_q = _make_scheduler()
        for i in range(5):
            in_q.put(_make_bundle(chunk_id=i))

        sched.start()
        time.sleep(0.3)
        sched.stop()

        count = 0
        while not out_q.empty():
            out_q.get_nowait()
            count += 1
        assert count >= 5


class TestMetrics:
    def test_metrics_structure(self) -> None:
        sched, _, _ = _make_scheduler()
        m = sched.metrics
        assert "last_inference_ms" in m
        assert "inference_count" in m
        assert "passthrough_count" in m
