"""Tests for LatencyProfiler and BenchmarkResult."""

from __future__ import annotations

import time

from src.core.profiler import BenchmarkResult, LatencyProfiler


class TestLatencyProfiler:
    def test_begin_end_snapshot(self) -> None:
        p = LatencyProfiler()
        p.begin("dsp")
        time.sleep(0.001)
        p.end("dsp")
        snap = p.snapshot()
        assert "dsp" in snap
        assert snap["dsp"] > 0.0
        assert "total" in snap

    def test_multiple_stages(self) -> None:
        p = LatencyProfiler()
        p.begin("dsp")
        p.end("dsp")
        p.begin("inference")
        p.end("inference")
        snap = p.snapshot()
        assert "dsp" in snap
        assert "inference" in snap
        assert snap["total"] >= snap["dsp"] + snap["inference"] - 0.01

    def test_summary_statistics(self) -> None:
        p = LatencyProfiler()
        for _ in range(10):
            p.begin("stage")
            p.end("stage")
            p.snapshot()
        s = p.summary()
        assert "stage" in s
        assert s["stage"]["count"] == 10.0
        assert s["stage"]["min"] <= s["stage"]["avg"] <= s["stage"]["max"]
        assert "total" in s

    def test_empty_summary(self) -> None:
        p = LatencyProfiler()
        assert p.summary() == {}

    def test_reset(self) -> None:
        p = LatencyProfiler()
        p.begin("x")
        p.end("x")
        p.snapshot()
        p.reset()
        assert p.history_count == 0
        assert p.summary() == {}

    def test_history_count(self) -> None:
        p = LatencyProfiler()
        assert p.history_count == 0
        p.begin("a")
        p.end("a")
        p.snapshot()
        p.snapshot()
        assert p.history_count == 2


class TestBenchmarkResult:
    def test_dataclass_fields(self) -> None:
        r = BenchmarkResult(
            duration_seconds=30,
            avg_latency_ms=28.0,
            p99_latency_ms=34.0,
            max_latency_ms=36.0,
            total_overflows=0,
            total_underruns=0,
            vram_peak_mb=1200.0,
        )
        assert r.duration_seconds == 30
        assert r.avg_latency_ms == 28.0
        assert r.samples == []
