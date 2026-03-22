"""DSP processing thread — reads from ring buffer, produces FeatureBundles into a queue."""

from __future__ import annotations

import logging
import queue
import threading
import time

from src.audio_io.ring_buffer import LockFreeRingBuffer
from src.dsp.feature_bundle import FeatureBundle, FeatureBundleAssembler

logger = logging.getLogger(__name__)


class DSPThread:
    """Dedicated daemon thread that runs the DSP pipeline.

    Reads audio chunks from the input ring buffer, runs the assembler,
    and pushes FeatureBundles into the output queue (non-blocking).
    """

    def __init__(
        self,
        input_ring_buffer: LockFreeRingBuffer,
        output_queue: queue.Queue[FeatureBundle],
        assembler: FeatureBundleAssembler,
        block_size: int = 256,
    ) -> None:
        self._input_buffer = input_ring_buffer
        self._output_queue = output_queue
        self._assembler = assembler
        self._block_size = block_size

        self._is_running = False
        self._thread: threading.Thread | None = None

        self._bundles_produced: int = 0
        self._bundles_dropped: int = 0
        self._last_process_time_ms: float = 0.0

    def _process_loop(self) -> None:
        while self._is_running:
            if self._input_buffer.available_read >= self._block_size:
                audio = self._input_buffer.read(self._block_size)

                t0 = time.perf_counter()
                bundle = self._assembler.assemble(audio)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._last_process_time_ms = elapsed_ms

                try:
                    self._output_queue.put_nowait(bundle)
                    self._bundles_produced += 1
                except queue.Full:
                    self._bundles_dropped += 1
            else:
                time.sleep(0.0005)

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True, name="vosmic-dsp")
        self._thread.start()
        logger.info("DSP thread started (block_size=%d)", self._block_size)

    def stop(self) -> None:
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("DSP thread stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def metrics(self) -> dict:
        return {
            "process_time_ms": self._last_process_time_ms,
            "bundles_produced": self._bundles_produced,
            "bundles_dropped": self._bundles_dropped,
            "is_running": self._is_running,
        }
