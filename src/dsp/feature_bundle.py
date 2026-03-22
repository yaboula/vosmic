"""FeatureBundle dataclass and assembler — bundles all DSP outputs for inference."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.dsp.content_encoder import ContentEncoder
from src.dsp.noise_gate import NoiseGate
from src.dsp.normalizer import LoudnessNormalizer
from src.dsp.pitch_extractor import PitchExtractor


@dataclass(frozen=True)
class FeatureBundle:
    """Immutable bundle of all features needed by the inference engine."""

    audio: np.ndarray  # Normalized (N,) float32
    f0: np.ndarray  # Pitch contour (T,) Hz, 0=unvoiced
    content_embedding: np.ndarray  # (T, 256) float32
    original_rms: float  # For gain matching in postprocessing
    is_silence: bool
    timestamp: float
    chunk_id: int


class FeatureBundleAssembler:
    """Orchestrates DSP components to produce FeatureBundles from raw audio."""

    def __init__(
        self,
        noise_gate: NoiseGate,
        normalizer: LoudnessNormalizer,
        pitch_extractor: PitchExtractor,
        content_encoder: ContentEncoder,
    ) -> None:
        self._noise_gate = noise_gate
        self._normalizer = normalizer
        self._pitch_extractor = pitch_extractor
        self._content_encoder = content_encoder
        self._chunk_counter: int = 0

    def assemble(self, audio: np.ndarray, timestamp: float | None = None) -> FeatureBundle:
        """Transform raw audio chunk into a complete FeatureBundle.

        If the noise gate detects silence, pitch and content extraction
        are skipped (returning zero arrays) to save GPU time.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        chunk_id = self._chunk_counter
        self._chunk_counter += 1

        gated, is_silence = self._noise_gate.process(audio)

        if is_silence:
            return FeatureBundle(
                audio=gated,
                f0=np.zeros(1, dtype=np.float32),
                content_embedding=np.zeros((1, 256), dtype=np.float32),
                original_rms=0.0,
                is_silence=True,
                timestamp=timestamp,
                chunk_id=chunk_id,
            )

        normalized, original_rms = self._normalizer.normalize(gated)
        f0 = self._pitch_extractor.extract(normalized)
        embeddings = self._content_encoder.encode(normalized)

        return FeatureBundle(
            audio=normalized,
            f0=f0,
            content_embedding=embeddings,
            original_rms=original_rms,
            is_silence=False,
            timestamp=timestamp,
            chunk_id=chunk_id,
        )
