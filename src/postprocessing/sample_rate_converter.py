"""Sample rate converter — high-quality sinc resampling via scipy."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


class SampleRateConverter:
    """Converts audio between sample rates using polyphase resampling.

    Needed when the RVC model outputs at a different rate than the system (48kHz).
    """

    def __init__(self, target_sr: int = 48000) -> None:
        self._target_sr = target_sr

    def convert(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Resample audio from source_sr to target_sr. No-op if rates match."""
        if source_sr == self._target_sr:
            return audio
        if len(audio) == 0:
            return audio

        gcd_val = math.gcd(self._target_sr, source_sr)
        up = self._target_sr // gcd_val
        down = source_sr // gcd_val

        return resample_poly(audio, up, down).astype(np.float32)

    def is_needed(self, model_sr: int) -> bool:
        return model_sr != self._target_sr

    @property
    def target_sr(self) -> int:
        return self._target_sr
