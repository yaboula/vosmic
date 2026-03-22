"""Tests for LoudnessNormalizer — gain smoothing, clipping, and edge cases."""

from __future__ import annotations

import numpy as np

from src.dsp.normalizer import LoudnessNormalizer


class TestBasicNormalization:
    def test_quiet_signal_is_amplified(self) -> None:
        norm = LoudnessNormalizer(target_lufs=-18.0, smoothing=0.0)
        audio = np.full(256, 0.001, dtype=np.float32)
        result, rms = norm.normalize(audio)
        assert float(np.sqrt(np.mean(result * result))) > float(np.sqrt(np.mean(audio * audio)))

    def test_loud_signal_is_attenuated(self) -> None:
        norm = LoudnessNormalizer(target_lufs=-18.0, smoothing=0.0)
        audio = np.full(256, 0.9, dtype=np.float32)
        result, rms = norm.normalize(audio)
        assert float(np.sqrt(np.mean(result * result))) < float(np.sqrt(np.mean(audio * audio)))

    def test_returns_original_rms(self) -> None:
        norm = LoudnessNormalizer()
        audio = np.full(256, 0.5, dtype=np.float32)
        _, rms = norm.normalize(audio)
        expected_rms = float(np.sqrt(np.mean(audio * audio)))
        assert abs(rms - expected_rms) < 1e-6


class TestEdgeCases:
    def test_silence_returns_zeros(self) -> None:
        norm = LoudnessNormalizer()
        audio = np.zeros(256, dtype=np.float32)
        result, rms = norm.normalize(audio)
        assert rms == 0.0
        assert np.all(result == 0.0)

    def test_near_zero_returns_unchanged(self) -> None:
        norm = LoudnessNormalizer()
        audio = np.full(256, 1e-10, dtype=np.float32)
        result, rms = norm.normalize(audio)
        assert rms == 0.0

    def test_output_clipped_to_range(self) -> None:
        norm = LoudnessNormalizer(target_lufs=-6.0, smoothing=0.0, max_gain=100.0)
        audio = np.full(256, 0.5, dtype=np.float32)
        result, _ = norm.normalize(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestEMASmoothing:
    def test_smoothing_reduces_jump(self) -> None:
        norm = LoudnessNormalizer(target_lufs=-18.0, smoothing=0.95)
        quiet = np.full(256, 0.001, dtype=np.float32)
        loud = np.full(256, 0.5, dtype=np.float32)

        for _ in range(50):
            norm.normalize(quiet)

        norm.normalize(loud)
        gain_after_one_loud = norm.current_gain

        # With 0.95 smoothing, gain should not jump instantly to the desired value
        target_rms = 10.0 ** (-18.0 / 20.0)
        desired_gain = target_rms / 0.5
        assert abs(gain_after_one_loud - desired_gain) > 0.01, (
            "Smoothing should prevent instant jump"
        )


class TestPerformance:
    def test_normalize_time(self) -> None:
        import time

        norm = LoudnessNormalizer()
        audio = np.random.randn(256).astype(np.float32) * 0.1

        norm.normalize(audio)

        t0 = time.perf_counter()
        for _ in range(10_000):
            norm.normalize(audio)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_call_ms = elapsed_ms / 10_000
        assert per_call_ms < 0.05, f"Normalizer too slow: {per_call_ms:.4f} ms/call"
