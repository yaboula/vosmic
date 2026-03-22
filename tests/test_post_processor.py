"""Tests for PostProcessor orchestrator — bypass, pipeline, clipping, performance."""

from __future__ import annotations

import time

import numpy as np

from src.core.config import CompressorConfig, PostProcessConfig
from src.inference.scheduler import InferenceResult
from src.postprocessing.post_processor import PostProcessor


def _make_result(audio: np.ndarray | None = None, rms: float = 0.2) -> InferenceResult:
    if audio is None:
        audio = np.random.randn(256).astype(np.float32) * 0.3
    return InferenceResult(
        audio=audio,
        original_audio=audio.copy(),
        original_rms=rms,
        was_passthrough=False,
        inference_time_ms=5.0,
        chunk_id=0,
    )


def _default_config(bypass: bool = False) -> PostProcessConfig:
    return PostProcessConfig(
        compressor=CompressorConfig(),
        filter_type="butterworth",
        filter_order=4,
        filter_cutoff_hz=20000,
        bypass=bypass,
    )


class TestBypass:
    def test_bypass_returns_identical(self) -> None:
        pp = PostProcessor(_default_config(bypass=True))
        result = _make_result()
        output = pp.process(result)
        assert np.array_equal(output, result.audio)

    def test_set_bypass_toggle(self) -> None:
        pp = PostProcessor(_default_config(bypass=False))
        assert pp.bypass is False
        pp.set_bypass(True)
        assert pp.bypass is True


class TestPipeline:
    def test_output_not_silent(self) -> None:
        pp = PostProcessor(_default_config())
        result = _make_result()
        output = pp.process(result)
        assert not np.all(output == 0.0)

    def test_output_within_reasonable_range(self) -> None:
        pp = PostProcessor(_default_config())
        audio = np.random.randn(256).astype(np.float32) * 0.3
        result = _make_result(audio=audio, rms=0.2)
        output = pp.process(result)
        assert float(np.max(np.abs(output))) < 5.0, "Output should be within reasonable bounds"

    def test_output_length_same(self) -> None:
        pp = PostProcessor(_default_config())
        result = _make_result()
        output = pp.process(result)
        assert len(output) == 256

    def test_sample_rate_conversion_changes_length(self) -> None:
        pp = PostProcessor(_default_config())
        audio = np.random.randn(1000).astype(np.float32) * 0.3
        result = _make_result(audio=audio)
        output = pp.process(result, model_sr=40000)
        assert len(output) != 1000


class TestPerformance:
    def test_pipeline_under_2ms(self) -> None:
        pp = PostProcessor(_default_config())
        result = _make_result()

        pp.process(result)

        times = []
        for _ in range(500):
            t0 = time.perf_counter()
            pp.process(result)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = float(np.mean(times))
        assert avg_ms < 2.0, f"PostProcessor too slow: {avg_ms:.3f} ms average"


class TestReset:
    def test_reset_does_not_crash(self) -> None:
        pp = PostProcessor(_default_config())
        pp.process(_make_result())
        pp.reset()
        output = pp.process(_make_result())
        assert len(output) == 256
