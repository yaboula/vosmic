"""Loudness normalization with EMA-smoothed gain — avoids clicks from abrupt level changes."""

from __future__ import annotations

import numpy as np


class LoudnessNormalizer:
    """RMS-based loudness normalizer with exponential moving average gain smoothing.

    Smoothing prevents audible clicks when gain changes rapidly between chunks.
    Output is hard-clipped to [-1.0, 1.0].
    """

    def __init__(
        self,
        target_lufs: float = -18.0,
        smoothing: float = 0.95,
        min_gain: float = 0.01,
        max_gain: float = 10.0,
    ) -> None:
        self._target_rms: float = 10.0 ** (target_lufs / 20.0)
        self._smoothing = smoothing
        self._min_gain = min_gain
        self._max_gain = max_gain
        self._current_gain: float = 1.0

    def normalize(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Normalize loudness toward target.

        Returns:
            (normalized_audio, original_rms): clipped to [-1, 1].
        """
        original_rms = float(np.sqrt(np.mean(audio * audio)))

        if original_rms < 1e-8:
            return audio.copy(), 0.0

        desired_gain = np.clip(self._target_rms / original_rms, self._min_gain, self._max_gain)
        self._current_gain = (
            self._smoothing * self._current_gain + (1.0 - self._smoothing) * desired_gain
        )
        normalized = np.clip(audio * self._current_gain, -1.0, 1.0)
        return normalized, original_rms

    @property
    def current_gain(self) -> float:
        return self._current_gain
