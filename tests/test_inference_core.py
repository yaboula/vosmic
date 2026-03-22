"""Tests for InferenceEngine — inference, failure handling, passthrough."""

from __future__ import annotations

import numpy as np

from src.dsp.feature_bundle import FeatureBundle
from src.inference.inference_core import InferenceEngine
from src.inference.model_manager import ModelManager


def _make_bundle(is_silence: bool = False, n: int = 256) -> FeatureBundle:
    audio = (
        np.zeros(n, dtype=np.float32) if is_silence else np.random.randn(n).astype(np.float32) * 0.1
    )
    return FeatureBundle(
        audio=audio,
        f0=np.zeros(10, dtype=np.float32),
        content_embedding=np.zeros((10, 256), dtype=np.float32),
        original_rms=0.0 if is_silence else 0.1,
        is_silence=is_silence,
        timestamp=0.0,
        chunk_id=0,
    )


class TestSilenceHandling:
    def test_silence_returns_none(self) -> None:
        mm = ModelManager(device="cpu")
        engine = InferenceEngine(mm)
        result = engine.infer(_make_bundle(is_silence=True))
        assert result is None


class TestNoModel:
    def test_no_model_returns_none(self) -> None:
        mm = ModelManager(device="cpu")
        engine = InferenceEngine(mm)
        result = engine.infer(_make_bundle())
        assert result is None


class TestInferenceFailure:
    def test_exception_returns_none(self) -> None:
        mm = ModelManager(device="cpu")
        mm._active_model = "broken_model"
        mm._format = "onnx"
        engine = InferenceEngine(mm)

        result = engine.infer(_make_bundle())

        assert result is None
        assert engine.metrics["failure_count"] == 1


class TestMetrics:
    def test_initial_metrics(self) -> None:
        mm = ModelManager(device="cpu")
        engine = InferenceEngine(mm)
        m = engine.metrics
        assert m["inference_count"] == 0
        assert m["failure_count"] == 0
        assert m["model_loaded"] is False
