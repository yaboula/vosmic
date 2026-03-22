"""Audio passthrough — mic to virtual cable with zero processing, for latency measurement."""

from __future__ import annotations

import sys
import threading
import time

from src.audio_io.capture import AudioCapture
from src.audio_io.devices import find_virtual_cable
from src.audio_io.output import AudioOutput
from src.audio_io.ring_buffer import LockFreeRingBuffer
from src.core.config import AudioConfig, load_config


class AudioPassthrough:
    """Routes microphone audio straight to the output device via ring buffers.

    No DSP or inference — purely for measuring the base audio I/O latency.
    Thread layout: capture callback -> input_buf -> passthrough thread -> output_buf -> output callback.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._input_buffer = LockFreeRingBuffer(config.ring_buffer_capacity)
        self._output_buffer = LockFreeRingBuffer(config.ring_buffer_capacity)

        self._capture = AudioCapture(
            ring_buffer=self._input_buffer,
            device_index=config.input_device,
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            channels=config.channels,
        )

        output_device = config.output_device
        if output_device is None:
            output_device = find_virtual_cable()

        self._output = AudioOutput(
            ring_buffer=self._output_buffer,
            device_index=output_device,
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            channels=config.channels,
        )

        self._is_running = False
        self._passthrough_thread: threading.Thread | None = None

    def _passthrough_loop(self) -> None:
        """Transfer blocks from input buffer to output buffer as fast as possible."""
        block_size = self._config.block_size
        while self._is_running:
            if self._input_buffer.available_read >= block_size:
                data = self._input_buffer.read(block_size)
                self._output_buffer.write(data)
            else:
                time.sleep(0.0001)

    def start(self) -> None:
        if self._is_running:
            return
        self._capture.start()
        self._output.start()
        self._is_running = True
        self._passthrough_thread = threading.Thread(target=self._passthrough_loop, daemon=True)
        self._passthrough_thread.start()

    def stop(self) -> None:
        self._is_running = False
        if self._passthrough_thread is not None:
            self._passthrough_thread.join(timeout=2)
            self._passthrough_thread = None
        self._capture.stop()
        self._output.stop()

    def measure_latency(self) -> dict:
        block_ms = (self._config.block_size / self._config.sample_rate) * 1000
        return {
            "input_callback_ms": self._capture.metrics["callback_time_ms"],
            "theoretical_latency_ms": block_ms * 2,
            "input_overflow": self._capture.metrics["overflow_count"],
            "output_underrun": self._output.metrics["underrun_count"],
            "capture_callbacks": self._capture.metrics["callback_count"],
            "peak_level": self._capture.metrics["peak_level"],
        }


def run_passthrough_benchmark(duration_seconds: int = 10) -> dict:
    """Standalone benchmark: mic -> ring buffers -> output for *duration_seconds*."""
    config = load_config()
    audio_cfg = config.audio

    block_ms = (audio_cfg.block_size / audio_cfg.sample_rate) * 1000

    print("=== VOSMIC Audio I/O Benchmark ===")
    print(f"Sample Rate:          {audio_cfg.sample_rate} Hz")
    print(f"Block Size:           {audio_cfg.block_size} samples")
    print(f"Theoretical Latency:  {block_ms * 2:.1f} ms (2x block)")
    print(f"Ring Buffer Capacity: {audio_cfg.ring_buffer_capacity} samples")
    print(f"Duration:             {duration_seconds} s")
    print()

    passthrough = AudioPassthrough(audio_cfg)
    passthrough.start()

    try:
        for remaining in range(duration_seconds, 0, -1):
            time.sleep(1)
            m = passthrough.measure_latency()
            print(
                f"  [{remaining:3d}s] cb={m['capture_callbacks']:6d}  "
                f"peak={m['peak_level']:.4f}  "
                f"overflow={m['input_overflow']}  "
                f"underrun={m['output_underrun']}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    metrics = passthrough.measure_latency()
    passthrough.stop()

    print()
    print("=== Results ===")
    print(f"Input Callback Time:  {metrics['input_callback_ms']:.3f} ms")
    print(f"Total Callbacks:      {metrics['capture_callbacks']}")
    print(f"Overflows:            {metrics['input_overflow']}")
    print(f"Underruns:            {metrics['output_underrun']}")

    if metrics["input_overflow"] > 0 or metrics["output_underrun"] > 0:
        print("\nWARNING: Timing issues detected")
    else:
        print("\nAudio I/O subsystem OK")

    print("=== Benchmark Complete ===")
    return metrics


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_passthrough_benchmark(duration)
