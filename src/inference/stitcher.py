"""Overlap-add stitcher — eliminates chunk-boundary artifacts with Hanning cross-fade."""

from __future__ import annotations

import numpy as np


class OverlapAddStitcher:
    """Cross-fades consecutive audio chunks using a Hanning window overlap-add.

    Prevents clicks and discontinuities at chunk boundaries in the
    inference output.
    """

    def __init__(self, overlap_ratio: float = 0.5, window_type: str = "hanning") -> None:
        self._overlap_ratio = overlap_ratio
        self._window_type = window_type
        self._prev_tail: np.ndarray | None = None

    def stitch(self, current_chunk: np.ndarray) -> np.ndarray:
        """Stitch the current chunk with the previous tail using overlap-add.

        The first call stores the tail and returns the chunk without the tail region.
        Subsequent calls cross-fade the overlap zone and return the stitched result.
        """
        if len(current_chunk) == 0:
            return current_chunk

        overlap_size = max(1, int(len(current_chunk) * self._overlap_ratio))

        if self._prev_tail is None:
            self._prev_tail = current_chunk[-overlap_size:].copy()
            return current_chunk[:-overlap_size].copy()

        window = np.hanning(overlap_size * 2).astype(np.float32)
        fade_out = window[overlap_size:]  # descending half
        fade_in = window[:overlap_size]  # ascending half

        prev_tail = self._prev_tail
        if len(prev_tail) != overlap_size:
            prev_tail = np.resize(prev_tail, overlap_size)

        current_head = current_chunk[:overlap_size]
        if len(current_head) < overlap_size:
            current_head = np.pad(current_head, (0, overlap_size - len(current_head)))

        crossfaded = prev_tail * fade_out + current_head * fade_in

        self._prev_tail = current_chunk[-overlap_size:].copy()

        middle = (
            current_chunk[overlap_size:-overlap_size]
            if len(current_chunk) > 2 * overlap_size
            else np.array([], dtype=np.float32)
        )
        return np.concatenate([crossfaded, middle])

    def reset(self) -> None:
        """Clear internal state (e.g., on model swap)."""
        self._prev_tail = None

    def flush(self) -> np.ndarray:
        """Return remaining tail samples. Call on stop."""
        if self._prev_tail is not None:
            tail = self._prev_tail.copy()
            self._prev_tail = None
            return tail
        return np.array([], dtype=np.float32)
