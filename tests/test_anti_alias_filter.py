"""Tests for AntiAliasingFilter — Butterworth low-pass with state."""

from __future__ import annotations

import numpy as np

from src.postprocessing.anti_alias_filter import AntiAliasingFilter


class TestFiltering:
    def test_low_frequency_passes(self) -> None:
        filt = AntiAliasingFilter(cutoff_hz=20000, order=4, sample_rate=48000)
        t = np.arange(256, dtype=np.float32) / 48000
        tone_1k = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        result = filt.filter(tone_1k)
        assert float(np.max(np.abs(result))) > 0.5, "1kHz should pass through"

    def test_empty_input(self) -> None:
        filt = AntiAliasingFilter()
        result = filt.filter(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_output_dtype(self) -> None:
        filt = AntiAliasingFilter()
        audio = np.random.randn(256).astype(np.float32)
        result = filt.filter(audio)
        assert result.dtype == np.float32

    def test_output_length_matches_input(self) -> None:
        filt = AntiAliasingFilter()
        audio = np.random.randn(256).astype(np.float32)
        result = filt.filter(audio)
        assert len(result) == 256


class TestStatefulness:
    def test_sequential_calls_maintain_state(self) -> None:
        filt = AntiAliasingFilter(cutoff_hz=5000, order=4, sample_rate=48000)
        chunk1 = np.random.randn(256).astype(np.float32) * 0.1
        chunk2 = np.random.randn(256).astype(np.float32) * 0.1

        r1 = filt.filter(chunk1)
        filt.filter(chunk2)

        filt.reset()
        r1_fresh = filt.filter(chunk1)

        assert np.array_equal(r1, r1_fresh), "First call after reset should match"

    def test_reset_reinitializes(self) -> None:
        filt = AntiAliasingFilter()
        filt.filter(np.random.randn(256).astype(np.float32))
        filt.reset()
        assert filt._zi is not None


class TestCutoff:
    def test_cutoff_property(self) -> None:
        filt = AntiAliasingFilter(cutoff_hz=15000)
        assert filt.cutoff_hz == 15000
