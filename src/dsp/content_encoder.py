"""Content embedding extraction — ContentVec on GPU with random-projection CPU fallback.

ContentVec produces (T, 768) features that are projected to (T, 256) for RVC.
When ContentVec/torch is unavailable, a deterministic random-projection stub
produces (T, 256) embeddings with correct shape for downstream testing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 256
TARGET_SR = 16000
HOP_SAMPLES_16K = 320  # ~20ms frames at 16kHz


class BaseContentEncoder(ABC):
    @abstractmethod
    def encode(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Return content embeddings (T, 256) float32."""


class StubContentEncoder(BaseContentEncoder):
    """Deterministic stub for testing — produces correctly-shaped embeddings from audio features.

    Uses MFCC-like spectral features projected to the target dimension,
    giving semantically meaningful (though not ContentVec-quality) embeddings.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        self._embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed=42)
        self._projection: np.ndarray | None = None

    def encode(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if len(audio) == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        if sample_rate != TARGET_SR:
            gcd = np.gcd(TARGET_SR, sample_rate)
            audio = resample_poly(audio, TARGET_SR // gcd, sample_rate // gcd).astype(np.float32)

        n_frames = max(1, len(audio) // HOP_SAMPLES_16K)

        frames = np.zeros((n_frames, HOP_SAMPLES_16K), dtype=np.float32)
        for i in range(n_frames):
            start = i * HOP_SAMPLES_16K
            end = min(start + HOP_SAMPLES_16K, len(audio))
            chunk = audio[start:end]
            frames[i, : len(chunk)] = chunk

        spectral = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        spec_dim = spectral.shape[1]

        if self._projection is None or self._projection.shape[0] != spec_dim:
            self._projection = self._rng.standard_normal((spec_dim, self._embedding_dim)).astype(
                np.float32
            ) * (1.0 / np.sqrt(spec_dim))

        embeddings = spectral @ self._projection
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        return embeddings


class ContentVecEncoder(BaseContentEncoder):
    """GPU-accelerated content encoding using ContentVec (HuBERT-based).

    Loads the model once at init, runs inference with torch.no_grad().
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
    ) -> None:
        self._device = device
        self._embedding_dim = EMBEDDING_DIM

        try:
            import torch

            if not torch.cuda.is_available() and device == "cuda":
                raise RuntimeError("CUDA not available")

            self._torch = torch
            self._model, self._projection = self._load_model(model_path)
            logger.info("ContentVec encoder initialized on %s", device)
        except Exception as e:
            raise RuntimeError(f"ContentVec initialization failed: {e}") from e

    def _load_model(self, model_path: str | None) -> tuple[object, object]:
        raise FileNotFoundError(
            "ContentVec model not yet available. Use ContentEncoder factory with fallback."
        )

    def encode(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if sample_rate != TARGET_SR:
            gcd = np.gcd(TARGET_SR, sample_rate)
            audio = resample_poly(audio, TARGET_SR // gcd, sample_rate // gcd).astype(np.float32)

        with self._torch.no_grad():
            tensor = self._torch.from_numpy(audio).to(self._device).unsqueeze(0)
            features = self._model.extract_features(tensor)
            embeddings = self._projection(features)

        return embeddings.squeeze(0).cpu().numpy().astype(np.float32)


class ContentEncoder:
    """Factory that selects the best available backend.

    Tries ContentVec (GPU) first, falls back to stub (CPU).
    """

    def __init__(
        self,
        model: str = "contentvec",
        device: str = "cuda",
        target_sr: int = TARGET_SR,
    ) -> None:
        self._target_sr = target_sr
        self._backend: BaseContentEncoder

        if model == "contentvec":
            try:
                self._backend = ContentVecEncoder(device=device)
                self._model = "contentvec"
                return
            except Exception as e:
                logger.warning("ContentVec unavailable (%s), falling back to stub encoder", e)

        self._backend = StubContentEncoder()
        self._model = "stub"
        logger.info("Using stub content encoder (CPU)")

    def encode(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        return self._backend.encode(audio, sample_rate)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM
