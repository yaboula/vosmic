"""Latency profiler — measures per-stage and total pipeline timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageTimestamp:
    """Single stage measurement."""

    name: str
    start_ns: int = 0
    end_ns: int = 0

    @property
    def elapsed_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


class LatencyProfiler:
    """Accumulates per-stage timings and produces summary statistics.

    Usage:
        p = LatencyProfiler()
        p.begin("dsp")
        ... do dsp ...
        p.end("dsp")
        print(p.summary())
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageTimestamp] = {}
        self._history: list[dict[str, float]] = []

    def begin(self, stage: str) -> None:
        self._stages[stage] = StageTimestamp(name=stage, start_ns=time.perf_counter_ns())

    def end(self, stage: str) -> None:
        if stage in self._stages:
            self._stages[stage].end_ns = time.perf_counter_ns()

    def snapshot(self) -> dict[str, float]:
        """Capture current timings and append to history."""
        snap = {s.name: s.elapsed_ms for s in self._stages.values() if s.end_ns > 0}
        snap["total"] = sum(snap.values())
        self._history.append(snap)
        return snap

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute min/avg/max/p99 for each stage across all snapshots."""
        if not self._history:
            return {}
        all_keys = set()
        for h in self._history:
            all_keys.update(h.keys())

        result: dict[str, dict[str, float]] = {}
        for key in sorted(all_keys):
            vals = [h.get(key, 0.0) for h in self._history]
            vals.sort()
            n = len(vals)
            p99_idx = min(int(n * 0.99), n - 1)
            result[key] = {
                "min": vals[0],
                "avg": sum(vals) / n,
                "max": vals[-1],
                "p99": vals[p99_idx],
                "count": float(n),
            }
        return result

    def reset(self) -> None:
        self._stages.clear()
        self._history.clear()

    @property
    def history_count(self) -> int:
        return len(self._history)


@dataclass
class BenchmarkResult:
    """Summary of an E2E benchmark run."""

    duration_seconds: int
    avg_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    total_overflows: int
    total_underruns: int
    vram_peak_mb: float
    samples: list[dict] = field(default_factory=list)
