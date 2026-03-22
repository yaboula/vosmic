"""Tests for GainMatcher — volume equalization with EMA smoothing."""

from __future__ import annotations

import numpy as np

from src.postprocessing.gain_matcher import GainMatcher


class TestGainMatching:
    def test_quiet_converted_is_amplified(self) -> None:
        gm = GainMatcher(smoothing=0.0)
        converted = np.full(256, 0.01, dtype=np.float32)
        result = gm.match(converted, original_rms=0.1)
        result_rms = float(np.sqrt(np.mean(result * result)))
        assert result_rms > float(np.sqrt(np.mean(converted * converted)))

    def test_loud_converted_is_attenuated(self) -> None:
        gm = GainMatcher(smoothing=0.0)
        converted = np.full(256, 0.5, dtype=np.float32)
        result = gm.match(converted, original_rms=0.05)
        result_rms = float(np.sqrt(np.mean(result * result)))
        assert result_rms < float(np.sqrt(np.mean(converted * converted)))

    def test_rms_within_1db(self) -> None:
        gm = GainMatcher(smoothing=0.0)
        converted = np.full(256, 0.3, dtype=np.float32)
        original_rms = 0.3
        result = gm.match(converted, original_rms)
        result_rms = float(np.sqrt(np.mean(result * result)))
        db_diff = abs(20 * np.log10(max(result_rms, 1e-10) / max(original_rms, 1e-10)))
        assert db_diff < 1.0, f"RMS diff {db_diff:.2f} dB exceeds 1 dB"


class TestEdgeCases:
    def test_zero_original_rms_unchanged(self) -> None:
        gm = GainMatcher()
        converted = np.full(256, 0.5, dtype=np.float32)
        result = gm.match(converted, original_rms=0.0)
        assert np.array_equal(result, converted)

    def test_zero_converted_rms_unchanged(self) -> None:
        gm = GainMatcher()
        converted = np.zeros(256, dtype=np.float32)
        result = gm.match(converted, original_rms=0.1)
        assert np.all(result == 0.0)

    def test_output_clipped(self) -> None:
        gm = GainMatcher(smoothing=0.0, max_gain=100.0)
        converted = np.full(256, 0.5, dtype=np.float32)
        result = gm.match(converted, original_rms=10.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestSmoothing:
    def test_ema_prevents_instant_jump(self) -> None:
        gm = GainMatcher(smoothing=0.9)
        converted = np.full(256, 0.1, dtype=np.float32)
        gm.match(converted, original_rms=0.1)

        gm.match(converted, original_rms=0.5)
        gain_after = gm.current_gain

        target = 0.5 / 0.1
        assert abs(gain_after - target) > 0.1, "Smoothing should prevent instant jump"


class TestReset:
    def test_reset_to_unity(self) -> None:
        gm = GainMatcher()
        gm._current_gain = 5.0
        gm.reset()
        assert gm.current_gain == 1.0
