"""Dynamic range compressor — vectorized numpy implementation with numba fast-path."""

from __future__ import annotations

import math

import numpy as np

_HAS_NUMBA = False
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    pass


def _compress_loop_python(
    audio: np.ndarray,
    threshold_linear: float,
    threshold_db: float,
    ratio: float,
    attack_coeff: float,
    release_coeff: float,
    envelope: float,
) -> tuple[np.ndarray, float]:
    """Pure-Python sample-by-sample compression (fallback)."""
    out = np.empty_like(audio)
    for i in range(len(audio)):
        abs_s = abs(audio[i])
        if abs_s > envelope:
            envelope = attack_coeff * envelope + (1.0 - attack_coeff) * abs_s
        else:
            envelope = release_coeff * envelope + (1.0 - release_coeff) * abs_s

        if envelope > threshold_linear:
            env_db = 20.0 * math.log10(max(envelope, 1e-10))
            gain_db = threshold_db + (env_db - threshold_db) / ratio
            gain = 10.0 ** (gain_db / 20.0) / max(envelope, 1e-10)
        else:
            gain = 1.0

        out[i] = audio[i] * gain

    return out, envelope


if _HAS_NUMBA:

    @njit(cache=True)
    def _compress_loop_numba(
        audio: np.ndarray,
        threshold_linear: float,
        threshold_db: float,
        ratio: float,
        attack_coeff: float,
        release_coeff: float,
        envelope: float,
    ) -> tuple[np.ndarray, float]:
        out = np.empty_like(audio)
        for i in range(len(audio)):
            abs_s = abs(audio[i])
            if abs_s > envelope:
                envelope = attack_coeff * envelope + (1.0 - attack_coeff) * abs_s
            else:
                envelope = release_coeff * envelope + (1.0 - release_coeff) * abs_s

            if envelope > threshold_linear:
                env_db = 20.0 * math.log10(max(envelope, 1e-10))
                gain_db = threshold_db + (env_db - threshold_db) / ratio
                gain = 10.0 ** (gain_db / 20.0) / max(envelope, 1e-10)
            else:
                gain = 1.0

            out[i] = audio[i] * gain

        return out, envelope


_compress_loop = _compress_loop_numba if _HAS_NUMBA else _compress_loop_python


class DynamicCompressor:
    """Peak-following dynamic range compressor.

    Uses numba JIT when available for ~100x speedup over pure Python.
    Falls back to a numpy-compatible Python loop otherwise.
    """

    def __init__(
        self,
        ratio: float = 3.0,
        threshold_db: float = -12.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        sample_rate: int = 48000,
    ) -> None:
        self._ratio = ratio
        self._threshold_db = threshold_db
        self._threshold_linear: float = 10.0 ** (threshold_db / 20.0)
        self._attack_coeff: float = math.exp(-1.0 / (attack_ms / 1000.0 * sample_rate))
        self._release_coeff: float = math.exp(-1.0 / (release_ms / 1000.0 * sample_rate))
        self._envelope: float = 0.0

    def compress(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic compression. Maintains envelope state across calls."""
        if len(audio) == 0:
            return audio
        result, self._envelope = _compress_loop(
            audio.astype(np.float64),
            self._threshold_linear,
            self._threshold_db,
            self._ratio,
            self._attack_coeff,
            self._release_coeff,
            self._envelope,
        )
        return result.astype(np.float32)

    def reset(self) -> None:
        self._envelope = 0.0

    @property
    def uses_numba(self) -> bool:
        return _HAS_NUMBA
