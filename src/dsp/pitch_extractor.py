"""Pitch (F0) extraction — RMVPE on GPU with autocorrelation CPU fallback.

RMVPE requires torch + a pretrained model file. When unavailable the extractor
falls back to a lightweight autocorrelation method that runs on CPU with zero
external dependencies beyond numpy/scipy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MIN_SAMPLES_16K = 1024


class BasePitchExtractor(ABC):
    """Common interface for all pitch extraction backends."""

    @abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Return F0 contour in Hz. 0 = unvoiced."""


class AutocorrelationPitchExtractor(BasePitchExtractor):
    """Lightweight CPU pitch extractor using autocorrelation (YIN-like).

    Used as fallback when RMVPE/CREPE are unavailable. Suitable for
    testing and low-VRAM GPUs.
    """

    def __init__(
        self,
        hop_length: int = 160,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> None:
        self._hop_length = hop_length
        self._f0_min = f0_min
        self._f0_max = f0_max

    def extract(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if len(audio) == 0:
            return np.array([], dtype=np.float32)

        if sample_rate != TARGET_SR:
            gcd = np.gcd(TARGET_SR, sample_rate)
            audio = resample_poly(audio, TARGET_SR // gcd, sample_rate // gcd).astype(np.float32)

        n_frames = max(1, len(audio) // self._hop_length)
        f0 = np.zeros(n_frames, dtype=np.float32)

        min_lag = max(1, int(TARGET_SR / self._f0_max))
        max_lag = int(TARGET_SR / self._f0_min)

        for i in range(n_frames):
            start = i * self._hop_length
            end = min(start + max_lag * 2, len(audio))
            frame = audio[start:end]

            if len(frame) < min_lag * 2:
                continue

            frame = frame - np.mean(frame)
            norm = np.sqrt(np.sum(frame * frame))
            if norm < 1e-8:
                continue

            actual_max_lag = min(max_lag, len(frame) // 2)
            if actual_max_lag <= min_lag:
                continue

            acf = np.correlate(
                frame[: actual_max_lag * 2], frame[: actual_max_lag * 2], mode="full"
            )
            acf = acf[len(acf) // 2 :]
            acf_region = acf[min_lag:actual_max_lag]

            if len(acf_region) == 0:
                continue

            peak_idx = int(np.argmax(acf_region)) + min_lag
            if acf[0] > 0 and acf[peak_idx] / acf[0] > 0.3:
                f0[i] = TARGET_SR / peak_idx

        return f0


class RMVPEPitchExtractor(BasePitchExtractor):
    """GPU-accelerated pitch extraction using RMVPE.

    Accumulates small chunks (256@48kHz -> ~85@16kHz) until enough samples
    are available for the model (1024@16kHz minimum).
    """

    def __init__(
        self,
        device: str = "cuda",
        hop_length: int = 160,
        model_path: str | None = None,
    ) -> None:
        self._hop_length = hop_length
        self._device = device
        self._accumulation_buffer = np.array([], dtype=np.float32)

        try:
            import torch

            if not torch.cuda.is_available() and device == "cuda":
                raise RuntimeError("CUDA not available")

            self._torch = torch
            self._model = self._load_model(model_path)
            logger.info("RMVPE pitch extractor initialized on %s", device)
        except Exception as e:
            raise RuntimeError(f"RMVPE initialization failed: {e}") from e

    def _load_model(self, model_path: str | None) -> object:
        """Load RMVPE model weights. Placeholder until model files are available."""
        raise FileNotFoundError(
            "RMVPE model not yet available. Use PitchExtractor factory with fallback."
        )

    def extract(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if sample_rate != TARGET_SR:
            gcd = np.gcd(TARGET_SR, sample_rate)
            audio_16k = resample_poly(audio, TARGET_SR // gcd, sample_rate // gcd).astype(
                np.float32
            )
        else:
            audio_16k = audio

        self._accumulation_buffer = np.concatenate([self._accumulation_buffer, audio_16k])

        if len(self._accumulation_buffer) < MIN_SAMPLES_16K:
            n_frames = max(1, len(audio_16k) // self._hop_length)
            return np.zeros(n_frames, dtype=np.float32)

        with self._torch.no_grad():
            tensor = self._torch.from_numpy(self._accumulation_buffer).to(self._device).unsqueeze(0)
            f0 = self._model.infer(tensor).cpu().numpy().flatten()

        overlap = 512
        if len(self._accumulation_buffer) > overlap:
            self._accumulation_buffer = self._accumulation_buffer[-overlap:]
        return f0.astype(np.float32)


class PitchExtractor:
    """Factory that selects the best available backend.

    Tries RMVPE (GPU) first, falls back to autocorrelation (CPU).
    """

    def __init__(
        self,
        method: str = "rmvpe",
        device: str = "cuda",
        hop_length: int = 160,
        sample_rate: int = 48000,
    ) -> None:
        self._sample_rate = sample_rate
        self._backend: BasePitchExtractor

        if method == "rmvpe":
            try:
                self._backend = RMVPEPitchExtractor(device=device, hop_length=hop_length)
                self._method = "rmvpe"
                return
            except Exception as e:
                logger.warning("RMVPE unavailable (%s), falling back to autocorrelation", e)

        self._backend = AutocorrelationPitchExtractor(hop_length=hop_length)
        self._method = "autocorrelation"
        logger.info("Using autocorrelation pitch extractor (CPU)")

    def extract(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        sr = sample_rate if sample_rate is not None else self._sample_rate
        return self._backend.extract(audio, sr)

    @property
    def method(self) -> str:
        return self._method
