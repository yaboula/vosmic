"""Audio capture engine — minimal real-time callback writing to a lock-free ring buffer."""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from src.audio_io.ring_buffer import LockFreeRingBuffer


class AudioDeviceError(Exception):
    """Raised when an audio device cannot be opened or configured."""


class AudioCapture:
    """Captures audio from a microphone into a lock-free ring buffer.

    The sounddevice callback does only memcpy + peak measurement.
    No allocations, no locks, no I/O inside the callback.
    """

    def __init__(
        self,
        ring_buffer: LockFreeRingBuffer,
        device_index: int | None = None,
        sample_rate: int = 48000,
        block_size: int = 256,
        channels: int = 1,
    ) -> None:
        self._ring_buffer = ring_buffer
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._channels = channels
        self._stream: sd.InputStream | None = None
        self._is_running = False

        self._last_callback_time_ms: float = 0.0
        self._callback_count: int = 0
        self._overflow_count: int = 0
        self._peak_level: float = 0.0

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """Real-time audio callback — NO allocations, NO locks, NO I/O."""
        if status:
            self._overflow_count += 1

        audio_mono = indata[:, 0]

        t0 = time.perf_counter_ns()
        if not self._ring_buffer.write(audio_mono):
            self._overflow_count += 1
        t1 = time.perf_counter_ns()

        self._peak_level = float(np.max(np.abs(audio_mono)))
        self._last_callback_time_ms = (t1 - t0) / 1_000_000
        self._callback_count += 1

    def start(self) -> None:
        if self._is_running:
            return
        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                channels=self._channels,
                dtype="float32",
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
            self._is_running = True
        except sd.PortAudioError as e:
            raise AudioDeviceError(f"Cannot open input device: {e}") from e

    def stop(self) -> None:
        if self._stream is not None and self._is_running:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._is_running = False

    def change_device(self, device_index: int) -> None:
        """Hot-swap input device. Brief gap (<200ms) while stream restarts."""
        was_running = self._is_running
        if was_running:
            self.stop()
        self._device_index = device_index
        if was_running:
            self.start()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def metrics(self) -> dict:
        return {
            "callback_time_ms": self._last_callback_time_ms,
            "callback_count": self._callback_count,
            "overflow_count": self._overflow_count,
            "peak_level": self._peak_level,
            "is_running": self._is_running,
            "buffer_available": self._ring_buffer.available_read,
        }

    def __del__(self) -> None:
        self.stop()
