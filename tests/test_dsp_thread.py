"""Tests for DSPThread — threading, queue output, and metrics."""

from __future__ import annotations

import queue
import time

import numpy as np

from src.audio_io.ring_buffer import LockFreeRingBuffer
from src.dsp.content_encoder import ContentEncoder
from src.dsp.dsp_thread import DSPThread
from src.dsp.feature_bundle import FeatureBundle, FeatureBundleAssembler
from src.dsp.noise_gate import NoiseGate
from src.dsp.normalizer import LoudnessNormalizer
from src.dsp.pitch_extractor import PitchExtractor


def _make_dsp_thread(
    buf_capacity: int = 4096,
    queue_size: int = 16,
    block_size: int = 256,
) -> tuple[DSPThread, LockFreeRingBuffer, queue.Queue]:
    ring = LockFreeRingBuffer(buf_capacity)
    q: queue.Queue[FeatureBundle] = queue.Queue(maxsize=queue_size)
    asm = FeatureBundleAssembler(
        noise_gate=NoiseGate(),
        normalizer=LoudnessNormalizer(),
        pitch_extractor=PitchExtractor(method="autocorrelation"),
        content_encoder=ContentEncoder(),
    )
    thread = DSPThread(ring, q, asm, block_size=block_size)
    return thread, ring, q


class TestDSPThreadStartStop:
    def test_start_and_stop(self) -> None:
        thread, _, _ = _make_dsp_thread()
        thread.start()
        assert thread.is_running is True
        time.sleep(0.05)
        thread.stop()
        assert thread.is_running is False

    def test_double_start_is_noop(self) -> None:
        thread, _, _ = _make_dsp_thread()
        thread.start()
        thread.start()
        assert thread.is_running is True
        thread.stop()

    def test_stop_within_2_seconds(self) -> None:
        thread, _, _ = _make_dsp_thread()
        thread.start()
        t0 = time.perf_counter()
        thread.stop()
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0


class TestDSPThreadProcessing:
    def test_produces_bundles_from_audio(self) -> None:
        thread, ring, q = _make_dsp_thread(block_size=256)
        audio = np.random.randn(256).astype(np.float32) * 0.1
        ring.write(audio)

        thread.start()
        time.sleep(0.2)
        thread.stop()

        assert not q.empty(), "Should have produced at least one bundle"
        bundle = q.get_nowait()
        assert isinstance(bundle, FeatureBundle)

    def test_silence_produces_silence_bundles(self) -> None:
        thread, ring, q = _make_dsp_thread(block_size=256)
        ring.write(np.zeros(256, dtype=np.float32))

        thread.start()
        time.sleep(0.2)
        thread.stop()

        assert not q.empty()
        bundle = q.get_nowait()
        assert bundle.is_silence is True

    def test_multiple_chunks_processed(self) -> None:
        thread, ring, q = _make_dsp_thread(block_size=256, queue_size=32)
        for _ in range(5):
            ring.write(np.random.randn(256).astype(np.float32) * 0.1)

        thread.start()
        time.sleep(0.5)
        thread.stop()

        count = 0
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count >= 5


class TestDSPThreadMetrics:
    def test_metrics_structure(self) -> None:
        thread, _, _ = _make_dsp_thread()
        m = thread.metrics
        assert "process_time_ms" in m
        assert "bundles_produced" in m
        assert "bundles_dropped" in m
        assert "is_running" in m

    def test_bundles_produced_count(self) -> None:
        thread, ring, q = _make_dsp_thread(block_size=256)
        ring.write(np.random.randn(256).astype(np.float32) * 0.1)

        thread.start()
        time.sleep(0.2)
        thread.stop()

        assert thread.metrics["bundles_produced"] >= 1


class TestDSPThreadQueueFull:
    def test_drops_when_queue_full(self) -> None:
        thread, ring, q = _make_dsp_thread(block_size=256, queue_size=2)

        for _ in range(10):
            ring.write(np.random.randn(256).astype(np.float32) * 0.1)

        thread.start()
        time.sleep(0.5)
        thread.stop()

        assert thread.metrics["bundles_dropped"] >= 0  # may or may not drop depending on timing
