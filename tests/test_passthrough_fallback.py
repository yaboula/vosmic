"""Tests for PassthroughFallback — ensures original audio is always forwarded."""

from __future__ import annotations

import numpy as np

from src.dsp.feature_bundle import FeatureBundle
from src.inference.passthrough import PassthroughFallback
from src.inference.scheduler import InferenceResult


def _make_bundle(chunk_id: int = 0) -> FeatureBundle:
    audio = np.random.randn(256).astype(np.float32) * 0.1
    return FeatureBundle(
        audio=audio,
        f0=np.zeros(10, dtype=np.float32),
        content_embedding=np.zeros((10, 256), dtype=np.float32),
        original_rms=0.1,
        is_silence=False,
        timestamp=0.0,
        chunk_id=chunk_id,
    )


class TestPassthroughForward:
    def test_returns_original_audio(self) -> None:
        pt = PassthroughFallback()
        bundle = _make_bundle(chunk_id=7)
        result = pt.forward(bundle)

        assert isinstance(result, InferenceResult)
        assert result.was_passthrough is True
        assert np.array_equal(result.audio, bundle.audio)
        assert result.chunk_id == 7

    def test_audio_is_copy(self) -> None:
        pt = PassthroughFallback()
        bundle = _make_bundle()
        result = pt.forward(bundle)
        result.audio[0] = 999.0
        assert bundle.audio[0] != 999.0, "Should be a copy, not a reference"

    def test_count_increments(self) -> None:
        pt = PassthroughFallback()
        assert pt.count == 0
        pt.forward(_make_bundle())
        pt.forward(_make_bundle())
        assert pt.count == 2

    def test_reset(self) -> None:
        pt = PassthroughFallback()
        pt.forward(_make_bundle())
        pt.reset()
        assert pt.count == 0


class TestNeverSilence:
    def test_voice_bundle_not_zeroed(self) -> None:
        pt = PassthroughFallback()
        bundle = _make_bundle()
        result = pt.forward(bundle)
        assert not np.all(result.audio == 0.0), "Passthrough must NOT output silence for voice"
