"""Inference engine — runs RVC model on FeatureBundle, returns converted audio or None."""

from __future__ import annotations

import logging
import time

import numpy as np

from src.dsp.feature_bundle import FeatureBundle
from src.inference.model_manager import ModelManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Executes RVC inference on a FeatureBundle.

    Returns converted audio as float32 ndarray, or None on failure
    (caller is responsible for passthrough fallback).
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager
        self._last_inference_ms: float = 0.0
        self._inference_count: int = 0
        self._failure_count: int = 0

    def infer(self, bundle: FeatureBundle) -> np.ndarray | None:
        """Run inference on a FeatureBundle.

        Returns None (triggering passthrough) when:
        - bundle is silence
        - no model loaded
        - inference raises any exception
        """
        if bundle.is_silence:
            return None

        model = self._model_manager.active_model
        if model is None:
            return None

        try:
            t0 = time.perf_counter()
            audio_out = self._run_model(model, bundle)
            self._last_inference_ms = (time.perf_counter() - t0) * 1000
            self._inference_count += 1
            return audio_out
        except Exception as e:
            self._failure_count += 1
            logger.warning("Inference failed (passthrough activated): %s", e)
            return None

    def _run_model(self, model: object, bundle: FeatureBundle) -> np.ndarray:
        fmt = self._model_manager.format

        if fmt == "pytorch":
            return self._infer_pytorch(model, bundle)
        elif fmt == "onnx":
            return self._infer_onnx(model, bundle)
        elif fmt == "tensorrt":
            return self._infer_tensorrt(model, bundle)
        else:
            raise ValueError(f"Unknown model format: {fmt}")

    def _infer_pytorch(self, model: object, bundle: FeatureBundle) -> np.ndarray:
        import torch

        with torch.no_grad():
            content = torch.from_numpy(bundle.content_embedding).unsqueeze(0)
            f0 = torch.from_numpy(bundle.f0).unsqueeze(0)

            device = (
                next(iter(model.parameters())).device if hasattr(model, "parameters") else "cuda"
            )
            content = content.to(device)
            f0 = f0.to(device)

            output = model(content, f0)
            return output.squeeze().cpu().numpy().astype(np.float32)

    def _infer_onnx(self, model: object, bundle: FeatureBundle) -> np.ndarray:
        inputs = {
            "content": bundle.content_embedding[np.newaxis].astype(np.float32),
            "f0": bundle.f0[np.newaxis].astype(np.float32),
        }
        outputs = model.run(None, inputs)
        return outputs[0].squeeze().astype(np.float32)

    def _infer_tensorrt(self, model: object, bundle: FeatureBundle) -> np.ndarray:
        raise NotImplementedError("TensorRT inference requires engine context setup — see F3-T4")

    @property
    def metrics(self) -> dict:
        return {
            "last_inference_ms": self._last_inference_ms,
            "inference_count": self._inference_count,
            "failure_count": self._failure_count,
            "model_loaded": self._model_manager.is_loaded,
        }
