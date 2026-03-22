"""Tests for NoiseGate — threshold, hold time, and edge cases."""

from __future__ import annotations

import numpy as np

from src.dsp.noise_gate import NoiseGate


class TestSilenceDetection:
    def test_silence_detected(self) -> None:
        gate = NoiseGate(threshold_db=-40.0)
        audio = np.zeros(256, dtype=np.float32)
        _, is_silence = gate.process(audio)
        assert is_silence is True

    def test_loud_signal_opens_gate(self) -> None:
        gate = NoiseGate(threshold_db=-40.0)
        audio = np.full(256, 0.1, dtype=np.float32)  # -20 dBFS >> -40 threshold
        result, is_silence = gate.process(audio)
        assert is_silence is False
        assert np.array_equal(result, audio)

    def test_signal_at_threshold_opens_gate(self) -> None:
        gate = NoiseGate(threshold_db=-40.0)
        threshold_linear = 10.0 ** (-40.0 / 20.0)
        audio = np.full(256, threshold_linear * 1.01, dtype=np.float32)
        _, is_silence = gate.process(audio)
        assert is_silence is False

    def test_output_is_zeros_when_silent(self) -> None:
        gate = NoiseGate(threshold_db=-40.0)
        audio = np.full(256, 1e-6, dtype=np.float32)
        result, is_silence = gate.process(audio)
        assert is_silence is True
        assert np.all(result == 0.0)


class TestHoldTime:
    def test_hold_keeps_gate_open(self) -> None:
        gate = NoiseGate(threshold_db=-40.0, hold_time_ms=100.0, sample_rate=48000)
        loud = np.full(256, 0.1, dtype=np.float32)
        silent = np.full(256, 1e-6, dtype=np.float32)

        gate.process(loud)
        _, is_silence = gate.process(silent)
        assert is_silence is False, "Hold time should keep gate open"

    def test_hold_expires(self) -> None:
        gate = NoiseGate(threshold_db=-40.0, hold_time_ms=10.0, sample_rate=48000)
        loud = np.full(256, 0.1, dtype=np.float32)
        silent = np.full(256, 1e-6, dtype=np.float32)

        gate.process(loud)
        # 10ms hold = 480 samples @ 48kHz. Two silent chunks (512 samples) should exhaust it.
        gate.process(silent)
        gate.process(silent)
        _, is_silence = gate.process(silent)
        assert is_silence is True


class TestSetThreshold:
    def test_dynamic_threshold_change(self) -> None:
        gate = NoiseGate(threshold_db=-60.0)
        quiet = np.full(256, 0.001, dtype=np.float32)  # ~ -60 dBFS
        _, is_silence = gate.process(quiet)
        assert is_silence is False

        gate.set_threshold(-20.0)
        _, is_silence = gate.process(quiet)
        # After hold expires
        gate._hold_counter = 0
        _, is_silence = gate.process(quiet)
        assert is_silence is True


class TestPerformance:
    def test_process_time(self) -> None:
        import time

        gate = NoiseGate()
        audio = np.random.randn(256).astype(np.float32)

        gate.process(audio)

        t0 = time.perf_counter()
        for _ in range(10_000):
            gate.process(audio)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_call_ms = elapsed_ms / 10_000
        assert per_call_ms < 0.05, f"NoiseGate too slow: {per_call_ms:.4f} ms/call"
