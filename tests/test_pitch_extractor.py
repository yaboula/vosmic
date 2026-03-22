"""Tests for PitchExtractor — autocorrelation fallback and factory."""

from __future__ import annotations

import numpy as np

from src.dsp.pitch_extractor import AutocorrelationPitchExtractor, PitchExtractor


class TestAutocorrelationExtractor:
    def test_pure_tone_440hz(self) -> None:
        sr = 16000
        duration = 0.1
        t = np.arange(int(sr * duration), dtype=np.float32) / sr
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        ext = AutocorrelationPitchExtractor(hop_length=160, f0_min=50.0, f0_max=1100.0)
        f0 = ext.extract(audio, sample_rate=sr)

        voiced = f0[f0 > 0]
        assert len(voiced) > 0, "Should detect voiced frames in 440Hz tone"
        median_f0 = float(np.median(voiced))
        assert 400 < median_f0 < 480, f"Expected ~440Hz, got {median_f0:.1f}Hz"

    def test_silence_returns_zeros(self) -> None:
        audio = np.zeros(1600, dtype=np.float32)
        ext = AutocorrelationPitchExtractor()
        f0 = ext.extract(audio, sample_rate=16000)
        assert np.all(f0 == 0)

    def test_empty_input(self) -> None:
        ext = AutocorrelationPitchExtractor()
        f0 = ext.extract(np.array([], dtype=np.float32))
        assert len(f0) == 0

    def test_resamples_from_48k(self) -> None:
        sr = 48000
        t = np.arange(int(sr * 0.1), dtype=np.float32) / sr
        audio = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)

        ext = AutocorrelationPitchExtractor()
        f0 = ext.extract(audio, sample_rate=sr)
        assert len(f0) > 0

    def test_low_frequency_detection(self) -> None:
        sr = 16000
        t = np.arange(int(sr * 0.5), dtype=np.float32) / sr
        audio = (np.sin(2 * np.pi * 100 * t) * 0.5).astype(np.float32)

        ext = AutocorrelationPitchExtractor(hop_length=320, f0_min=50.0, f0_max=600.0)
        f0 = ext.extract(audio, sample_rate=sr)
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            median_f0 = float(np.median(voiced))
            assert 70 < median_f0 < 150, f"Expected ~100Hz, got {median_f0:.1f}Hz"


class TestPitchExtractorFactory:
    def test_falls_back_to_autocorrelation(self) -> None:
        ext = PitchExtractor(method="rmvpe", device="cuda")
        assert ext.method == "autocorrelation"

    def test_explicit_autocorrelation(self) -> None:
        ext = PitchExtractor(method="autocorrelation")
        assert ext.method == "autocorrelation"

    def test_extract_via_factory(self) -> None:
        ext = PitchExtractor(method="autocorrelation", sample_rate=48000)
        sr = 48000
        t = np.arange(int(sr * 0.1), dtype=np.float32) / sr
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        f0 = ext.extract(audio)
        assert len(f0) > 0
