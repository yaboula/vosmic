"""Tests for FeatureBundle dataclass and FeatureBundleAssembler."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.dsp.content_encoder import ContentEncoder
from src.dsp.feature_bundle import FeatureBundle, FeatureBundleAssembler
from src.dsp.noise_gate import NoiseGate
from src.dsp.normalizer import LoudnessNormalizer
from src.dsp.pitch_extractor import PitchExtractor


def _make_assembler() -> FeatureBundleAssembler:
    return FeatureBundleAssembler(
        noise_gate=NoiseGate(threshold_db=-40.0),
        normalizer=LoudnessNormalizer(target_lufs=-18.0),
        pitch_extractor=PitchExtractor(method="autocorrelation", sample_rate=48000),
        content_encoder=ContentEncoder(),
    )


class TestFeatureBundleDataclass:
    def test_frozen(self) -> None:
        bundle = FeatureBundle(
            audio=np.zeros(256, dtype=np.float32),
            f0=np.zeros(1, dtype=np.float32),
            content_embedding=np.zeros((1, 256), dtype=np.float32),
            original_rms=0.0,
            is_silence=True,
            timestamp=0.0,
            chunk_id=0,
        )
        with pytest.raises(AttributeError):
            bundle.is_silence = False  # type: ignore[misc]


class TestAssemblerSilence:
    def test_silence_skips_extraction(self) -> None:
        asm = _make_assembler()
        audio = np.zeros(256, dtype=np.float32)
        bundle = asm.assemble(audio)
        assert bundle.is_silence is True
        assert bundle.original_rms == 0.0
        assert bundle.chunk_id == 0

    def test_silence_has_zero_embeddings(self) -> None:
        asm = _make_assembler()
        audio = np.zeros(256, dtype=np.float32)
        bundle = asm.assemble(audio)
        assert np.all(bundle.f0 == 0)
        assert np.all(bundle.content_embedding == 0)


class TestAssemblerVoice:
    def test_voice_produces_features(self) -> None:
        asm = _make_assembler()
        sr = 48000
        t = np.arange(256, dtype=np.float32) / sr
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

        bundle = asm.assemble(audio)
        assert bundle.is_silence is False
        assert bundle.original_rms > 0
        assert bundle.audio.shape == (256,)
        assert bundle.content_embedding.shape[1] == 256

    def test_chunk_id_increments(self) -> None:
        asm = _make_assembler()
        audio = np.random.randn(256).astype(np.float32) * 0.1
        b1 = asm.assemble(audio)
        b2 = asm.assemble(audio)
        assert b2.chunk_id == b1.chunk_id + 1


class TestAssemblerPerformance:
    def test_assemble_time(self) -> None:
        asm = _make_assembler()
        audio = np.random.randn(256).astype(np.float32) * 0.1

        asm.assemble(audio)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            asm.assemble(audio)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        assert avg_ms < 10.0, f"Assembler too slow: {avg_ms:.2f} ms average"
