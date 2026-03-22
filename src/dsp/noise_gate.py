"""Noise gate with hold time — blocks silence while preserving transients."""

from __future__ import annotations

import numpy as np


class NoiseGate:
    """RMS-based noise gate with configurable threshold and hold time.

    Hold time keeps the gate open for short pauses between syllables,
    preventing choppy artifacts in speech.
    """

    def __init__(
        self,
        threshold_db: float = -40.0,
        hold_time_ms: float = 100.0,
        sample_rate: int = 48000,
    ) -> None:
        self._threshold_linear: float = 10.0 ** (threshold_db / 20.0)
        self._hold_samples: int = int(hold_time_ms / 1000.0 * sample_rate)
        self._hold_counter: int = 0
        self._is_open: bool = False

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply noise gate.

        Returns:
            (processed_audio, is_silence): original audio if gate is open,
            zeros if gate is closed.
        """
        rms = float(np.sqrt(np.mean(audio * audio)))

        if rms >= self._threshold_linear:
            self._is_open = True
            self._hold_counter = self._hold_samples
        else:
            if self._hold_counter > 0:
                self._hold_counter -= len(audio)
            else:
                self._is_open = False

        if self._is_open:
            return audio, False
        return np.zeros_like(audio), True

    def set_threshold(self, threshold_db: float) -> None:
        self._threshold_linear = 10.0 ** (threshold_db / 20.0)

    @property
    def is_open(self) -> bool:
        return self._is_open
