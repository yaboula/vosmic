"""Anti-aliasing filter — stateful Butterworth low-pass via scipy sosfilt."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


class AntiAliasingFilter:
    """Butterworth low-pass filter with persistent state for seamless chunk-to-chunk filtering.

    Removes high-frequency artifacts from voice conversion and resampling.
    State is preserved between calls to avoid edge discontinuities.
    """

    def __init__(
        self,
        cutoff_hz: int = 20000,
        order: int = 4,
        sample_rate: int = 48000,
    ) -> None:
        self._cutoff_hz = cutoff_hz
        self._order = order
        self._sample_rate = sample_rate

        nyquist = sample_rate / 2.0
        normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
        self._sos = butter(order, normalized_cutoff, btype="low", output="sos")
        self._zi = sosfilt_zi(self._sos)

    def filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply low-pass filter while maintaining state across calls."""
        if len(audio) == 0:
            return audio
        filtered, self._zi = sosfilt(self._sos, audio, zi=self._zi)
        return filtered.astype(np.float32)

    def reset(self) -> None:
        """Reset filter state (call on device change or model swap)."""
        self._zi = sosfilt_zi(self._sos)

    @property
    def cutoff_hz(self) -> int:
        return self._cutoff_hz
