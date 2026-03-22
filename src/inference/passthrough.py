"""Passthrough fallback — forwards original audio when GPU cannot keep up.

ABSOLUTE RULE: if inference fails or is too slow, output the original
audio (dry signal). NEVER output silence. NEVER block the audio thread.
"""

from __future__ import annotations

import logging

from src.dsp.feature_bundle import FeatureBundle
from src.inference.scheduler import InferenceResult

logger = logging.getLogger(__name__)


class PassthroughFallback:
    """Converts a FeatureBundle to an InferenceResult containing original audio.

    Used when:
    - No model is loaded
    - GPU inference exceeds time budget
    - Inference raises an exception
    """

    def __init__(self) -> None:
        self._count: int = 0

    def forward(self, bundle: FeatureBundle) -> InferenceResult:
        """Wrap original audio into an InferenceResult flagged as passthrough."""
        self._count += 1
        return InferenceResult(
            audio=bundle.audio.copy(),
            original_audio=bundle.audio,
            original_rms=bundle.original_rms,
            was_passthrough=True,
            inference_time_ms=0.0,
            chunk_id=bundle.chunk_id,
        )

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0
