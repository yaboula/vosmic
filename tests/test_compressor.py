"""Tests for DynamicCompressor — peak following, gain reduction, performance."""

from __future__ import annotations

import time

import numpy as np

from src.postprocessing.compressor import DynamicCompressor


class TestCompression:
    def test_loud_signal_is_compressed(self) -> None:
        comp = DynamicCompressor(ratio=3.0, threshold_db=-12.0)
        loud = np.full(256, 0.8, dtype=np.float32)
        result = comp.compress(loud)
        result_rms = float(np.sqrt(np.mean(result * result)))
        input_rms = float(np.sqrt(np.mean(loud * loud)))
        assert result_rms < input_rms, f"Expected compression, RMS {result_rms} >= {input_rms}"

    def test_quiet_signal_unchanged(self) -> None:
        comp = DynamicCompressor(ratio=3.0, threshold_db=-12.0)
        quiet = np.full(256, 0.01, dtype=np.float32)
        result = comp.compress(quiet)
        assert np.allclose(result, quiet, atol=1e-6)

    def test_no_output_exceeds_input_peak(self) -> None:
        comp = DynamicCompressor(ratio=4.0, threshold_db=-6.0)
        audio = np.sin(np.linspace(0, 20 * np.pi, 1024)).astype(np.float32) * 0.9
        result = comp.compress(audio)
        assert float(np.max(np.abs(result))) <= float(np.max(np.abs(audio))) + 0.01

    def test_empty_input(self) -> None:
        comp = DynamicCompressor()
        result = comp.compress(np.array([], dtype=np.float32))
        assert len(result) == 0


class TestEnvelopeState:
    def test_state_persists_across_calls(self) -> None:
        comp = DynamicCompressor()
        loud = np.full(256, 0.9, dtype=np.float32)
        comp.compress(loud)
        env_after_loud = comp._envelope

        quiet = np.full(256, 0.01, dtype=np.float32)
        comp.compress(quiet)
        env_after_quiet = comp._envelope

        assert env_after_quiet < env_after_loud

    def test_reset_clears_envelope(self) -> None:
        comp = DynamicCompressor()
        comp.compress(np.full(256, 0.9, dtype=np.float32))
        comp.reset()
        assert comp._envelope == 0.0


class TestPerformance:
    def test_compress_speed(self) -> None:
        comp = DynamicCompressor()
        audio = np.random.randn(256).astype(np.float32) * 0.3

        comp.compress(audio)

        t0 = time.perf_counter()
        for _ in range(1000):
            comp.compress(audio)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_call_ms = elapsed_ms / 1000
        assert per_call_ms < 2.0, f"Compressor too slow: {per_call_ms:.3f} ms/call"
