"""Gain matcher — equalizes converted audio volume to the original level."""

from __future__ import annotations

import numpy as np


class GainMatcher:
    """EMA-smoothed gain matching to prevent volume jumps when voice conversion is toggled.

    Computes the ratio between original and converted RMS, applies it with
    exponential smoothing, and hard-clips to [-1, 1].
    """

    def __init__(
        self, smoothing: float = 0.9, min_gain: float = 0.1, max_gain: float = 10.0
    ) -> None:
        self._smoothing = smoothing
        self._min_gain = min_gain
        self._max_gain = max_gain
        self._current_gain: float = 1.0

    def match(self, converted_audio: np.ndarray, original_rms: float) -> np.ndarray:
        """Match converted audio volume to original RMS level."""
        if original_rms < 1e-8:
            return converted_audio

        converted_rms = float(np.sqrt(np.mean(converted_audio * converted_audio)))
        if converted_rms < 1e-8:
            return converted_audio

        target_gain = np.clip(original_rms / converted_rms, self._min_gain, self._max_gain)
        self._current_gain = (
            self._smoothing * self._current_gain + (1.0 - self._smoothing) * target_gain
        )

        return np.clip(converted_audio * self._current_gain, -1.0, 1.0).astype(np.float32)

    @property
    def current_gain(self) -> float:
        return self._current_gain

    def reset(self) -> None:
        self._current_gain = 1.0
