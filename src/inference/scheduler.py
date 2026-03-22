"""Inference scheduler — async FIFO thread consuming FeatureBundles, producing InferenceResults."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from src.dsp.feature_bundle import FeatureBundle
from src.inference.inference_core import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Output of the inference scheduler, consumed by postprocessing."""

    audio: np.ndarray  # Converted audio (or original if passthrough)
    original_audio: np.ndarray  # Original for gain matching
    original_rms: float
    was_passthrough: bool
    inference_time_ms: float
    chunk_id: int


class InferenceScheduler:
    """Dedicated inference thread: reads FeatureBundles, runs inference, writes InferenceResults.

    If inference fails or exceeds max_inference_ms, the original audio
    is forwarded (passthrough). NEVER outputs silence.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        input_queue: queue.Queue[FeatureBundle],
        output_queue: queue.Queue[InferenceResult],
        max_inference_ms: float = 25.0,
    ) -> None:
        self._engine = inference_engine
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._max_inference_ms = max_inference_ms

        self._is_running = False
        self._thread: threading.Thread | None = None

        self._passthrough_count: int = 0
        self._inference_count: int = 0
        self._last_inference_ms: float = 0.0

    def _inference_loop(self) -> None:
        while self._is_running:
            try:
                bundle = self._input_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if bundle.is_silence:
                result = InferenceResult(
                    audio=np.zeros_like(bundle.audio),
                    original_audio=bundle.audio,
                    original_rms=bundle.original_rms,
                    was_passthrough=True,
                    inference_time_ms=0.0,
                    chunk_id=bundle.chunk_id,
                )
                self._put_result(result)
                continue

            t0 = time.perf_counter()
            converted = self._engine.infer(bundle)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._last_inference_ms = elapsed_ms

            if converted is None or elapsed_ms > self._max_inference_ms:
                result = InferenceResult(
                    audio=bundle.audio,
                    original_audio=bundle.audio,
                    original_rms=bundle.original_rms,
                    was_passthrough=True,
                    inference_time_ms=elapsed_ms,
                    chunk_id=bundle.chunk_id,
                )
                self._passthrough_count += 1
            else:
                result = InferenceResult(
                    audio=converted,
                    original_audio=bundle.audio,
                    original_rms=bundle.original_rms,
                    was_passthrough=False,
                    inference_time_ms=elapsed_ms,
                    chunk_id=bundle.chunk_id,
                )
                self._inference_count += 1

            self._put_result(result)

    def _put_result(self, result: InferenceResult) -> None:
        try:
            self._output_queue.put_nowait(result)
        except queue.Full:
            logger.warning("Inference output queue full, dropping chunk %d", result.chunk_id)

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="vosmic-inference"
        )
        self._thread.start()
        logger.info("Inference scheduler started")

    def stop(self) -> None:
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("Inference scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def metrics(self) -> dict:
        return {
            "last_inference_ms": self._last_inference_ms,
            "inference_count": self._inference_count,
            "passthrough_count": self._passthrough_count,
            "is_running": self._is_running,
        }
