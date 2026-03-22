"""Tests for SampleRateConverter — resampling quality and no-op passthrough."""

from __future__ import annotations

import numpy as np

from src.postprocessing.sample_rate_converter import SampleRateConverter


class TestNoConversion:
    def test_same_rate_returns_same(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        audio = np.random.randn(256).astype(np.float32)
        result = src.convert(audio, source_sr=48000)
        assert np.array_equal(result, audio)

    def test_is_needed_false_when_same(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        assert src.is_needed(48000) is False

    def test_is_needed_true_when_different(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        assert src.is_needed(40000) is True


class TestResampling:
    def test_40k_to_48k_changes_length(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        audio = np.random.randn(1000).astype(np.float32)
        result = src.convert(audio, source_sr=40000)
        expected_len = int(1000 * 48000 / 40000)
        assert abs(len(result) - expected_len) <= 2

    def test_44100_to_48000(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        sr_in = 44100
        t = np.arange(4410, dtype=np.float32) / sr_in
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = src.convert(tone, source_sr=sr_in)
        assert len(result) > len(tone)

    def test_empty_input(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        result = src.convert(np.array([], dtype=np.float32), source_sr=16000)
        assert len(result) == 0

    def test_output_dtype(self) -> None:
        src = SampleRateConverter(target_sr=48000)
        audio = np.random.randn(500).astype(np.float32)
        result = src.convert(audio, source_sr=16000)
        assert result.dtype == np.float32
