"""Lock-free SPSC (Single Producer, Single Consumer) ring buffer for real-time audio.

Zero locks, pre-allocated memory, wrap-around via bitmask.
The audio callback thread writes; the DSP thread reads.
"""

from __future__ import annotations

import numpy as np


def next_power_of_two(n: int) -> int:
    """Round up to the next power of two (or return n if already a power of two)."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


class LockFreeRingBuffer:
    """SPSC lock-free ring buffer with pre-allocated numpy storage.

    Thread-safety model:
      - Exactly ONE thread calls write() (the audio capture callback).
      - Exactly ONE thread calls read() (the DSP/passthrough thread).
      - No locks are used; monotonically increasing pointers provide ordering.
    """

    __slots__ = ("_capacity", "_mask", "_buffer", "_write_pos", "_read_pos")

    def __init__(self, capacity: int = 4096) -> None:
        self._capacity = next_power_of_two(capacity)
        self._mask = self._capacity - 1
        self._buffer = np.zeros(self._capacity, dtype=np.float32)
        self._write_pos: int = 0
        self._read_pos: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available_read(self) -> int:
        return self._write_pos - self._read_pos

    @property
    def available_write(self) -> int:
        return self._capacity - self.available_read

    def write(self, data: np.ndarray) -> bool:
        """Write samples into the buffer. Returns False without writing if there isn't enough space."""
        n = len(data)
        if n == 0:
            return True
        if n > self.available_write:
            return False

        start = self._write_pos & self._mask
        end = start + n

        if end <= self._capacity:
            self._buffer[start:end] = data
        else:
            first = self._capacity - start
            self._buffer[start:] = data[:first]
            self._buffer[: n - first] = data[first:]

        self._write_pos += n
        return True

    def read(self, n_samples: int) -> np.ndarray:
        """Read up to n_samples from the buffer. Returns an empty array if nothing is available."""
        available = self.available_read
        if available == 0:
            return np.empty(0, dtype=np.float32)

        to_read = min(n_samples, available)
        start = self._read_pos & self._mask
        end = start + to_read
        out = np.empty(to_read, dtype=np.float32)

        if end <= self._capacity:
            out[:] = self._buffer[start:end]
        else:
            first = self._capacity - start
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: to_read - first]

        self._read_pos += to_read
        return out

    def peek(self, n_samples: int) -> np.ndarray:
        """Read without advancing the read pointer (for monitoring/debug)."""
        available = self.available_read
        if available == 0:
            return np.empty(0, dtype=np.float32)

        to_read = min(n_samples, available)
        start = self._read_pos & self._mask
        end = start + to_read
        out = np.empty(to_read, dtype=np.float32)

        if end <= self._capacity:
            out[:] = self._buffer[start:end]
        else:
            first = self._capacity - start
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: to_read - first]

        return out

    def clear(self) -> None:
        """Discard all buffered data. Only call when audio is NOT active."""
        self._read_pos = self._write_pos

    def __len__(self) -> int:
        return self.available_read
