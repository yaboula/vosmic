"""PostProcessor orchestrator — chains gain, compression, SRC, and filter with bypass."""

from __future__ import annotations

import numpy as np

from src.core.config import PostProcessConfig
from src.inference.scheduler import InferenceResult
from src.postprocessing.anti_alias_filter import AntiAliasingFilter
from src.postprocessing.compressor import DynamicCompressor
from src.postprocessing.gain_matcher import GainMatcher
from src.postprocessing.sample_rate_converter import SampleRateConverter


class PostProcessor:
    """Chains all postprocessing stages: gain -> compress -> SRC -> filter.

    Pipeline budget: <= 2ms total for 256 samples at 48kHz.
    Bypass mode returns audio unmodified (bit-perfect).
    """

    def __init__(self, config: PostProcessConfig, sample_rate: int = 48000) -> None:
        self._gain_matcher = GainMatcher()
        self._compressor = DynamicCompressor(
            ratio=config.compressor.ratio,
            threshold_db=config.compressor.threshold_db,
            attack_ms=config.compressor.attack_ms,
            release_ms=config.compressor.release_ms,
            sample_rate=sample_rate,
        )
        self._src = SampleRateConverter(target_sr=sample_rate)
        self._anti_alias = AntiAliasingFilter(
            cutoff_hz=config.filter_cutoff_hz,
            order=config.filter_order,
            sample_rate=sample_rate,
        )
        self._bypass = config.bypass
        self._sample_rate = sample_rate

    def process(self, result: InferenceResult, model_sr: int = 48000) -> np.ndarray:
        """Run the full postprocessing chain on an InferenceResult."""
        if self._bypass:
            return result.audio

        audio = result.audio

        audio = self._gain_matcher.match(audio, result.original_rms)
        audio = self._compressor.compress(audio)

        if self._src.is_needed(model_sr):
            audio = self._src.convert(audio, model_sr)

        audio = self._anti_alias.filter(audio)

        return audio

    def set_bypass(self, enabled: bool) -> None:
        self._bypass = enabled

    @property
    def bypass(self) -> bool:
        return self._bypass

    def set_compression(
        self,
        ratio: float,
        threshold_db: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        self._compressor = DynamicCompressor(
            ratio=ratio,
            threshold_db=threshold_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
            sample_rate=self._sample_rate,
        )

    def reset(self) -> None:
        """Reset all internal states (call on device/model change)."""
        self._gain_matcher.reset()
        self._compressor.reset()
        self._anti_alias.reset()
