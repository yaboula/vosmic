"""Audio output engine — reads from a lock-free ring buffer into sounddevice OutputStream."""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from src.audio_io.capture import AudioDeviceError
from src.audio_io.ring_buffer import LockFreeRingBuffer


class AudioOutput:
    """Plays audio from a ring buffer to a hardware or virtual output device.

    The output callback only does memcpy from the ring buffer.
    When data is unavailable it outputs clean silence (zeros).
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
        self._stream: sd.OutputStream | None = None
        self._is_running = False

        self._underrun_count: int = 0
        self._last_output_level: float = 0.0

    def _output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """Real-time output callback — NO allocations, NO locks, NO I/O."""
        if status:
            self._underrun_count += 1

        audio = self._ring_buffer.read(frames)
        n = len(audio)

        if n == frames:
            outdata[:, 0] = audio
        elif n > 0:
            outdata[:n, 0] = audio
            outdata[n:, 0] = 0.0
            self._underrun_count += 1
        else:
            outdata[:, 0] = 0.0
            self._underrun_count += 1

        self._last_output_level = float(np.max(np.abs(outdata)))

    def start(self) -> None:
        if self._is_running:
            return
        try:
            self._stream = sd.OutputStream(
                device=self._device_index,
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                channels=self._channels,
                dtype="float32",
                callback=self._output_callback,
                latency="low",
            )
            self._stream.start()
            self._is_running = True
        except sd.PortAudioError as e:
            raise AudioDeviceError(f"Cannot open output device: {e}") from e

    def stop(self) -> None:
        if self._stream is not None and self._is_running:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._is_running = False

    def set_output_device(self, device_index: int) -> None:
        """Hot-swap output device. Brief gap (<200ms) while stream restarts."""
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
            "underrun_count": self._underrun_count,
            "output_level": self._last_output_level,
            "is_running": self._is_running,
        }

    def __del__(self) -> None:
        self.stop()
