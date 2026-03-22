"""E2E latency benchmark — measures pipeline performance over a configurable duration.

Usage:
    python scripts/benchmark_e2e.py [--duration 30] [--config configs/default.yaml]

Requires: audio devices available, optional GPU for inference.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.core.config import load_config
from src.core.pipeline import VosmicPipeline
from src.core.profiler import BenchmarkResult


def benchmark_end_to_end(
    config_path: str = "configs/default.yaml",
    duration_seconds: int = 30,
) -> BenchmarkResult:
    config = load_config(config_path)
    pipeline = VosmicPipeline(config)

    print("Starting pipeline for E2E benchmark...")
    try:
        pipeline.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        sys.exit(1)

    print("Warm-up (2s)...")
    time.sleep(2)

    samples: list[dict] = []
    print(f"Collecting metrics for {duration_seconds}s...")
    for i in range(duration_seconds):
        m = pipeline.get_full_metrics()
        sample = {
            "second": i + 1,
            "dsp_ms": m.get("dsp", {}).get("process_time_ms", 0.0),
            "inference_ms": m.get("inference", {}).get("last_inference_ms", 0.0),
            "estimated_total_ms": m.get("estimated_latency_ms", 0.0),
            "input_overflow": m.get("capture", {}).get("overflow_count", 0),
            "output_underrun": m.get("output", {}).get("underrun_count", 0),
        }
        samples.append(sample)
        time.sleep(1)

    pipeline.stop()

    latencies = [s["estimated_total_ms"] for s in samples]
    latencies.sort()
    n = len(latencies)
    p99_idx = min(int(n * 0.99), n - 1)

    result = BenchmarkResult(
        duration_seconds=duration_seconds,
        avg_latency_ms=sum(latencies) / max(n, 1),
        p99_latency_ms=latencies[p99_idx] if latencies else 0.0,
        max_latency_ms=max(latencies) if latencies else 0.0,
        total_overflows=sum(s["input_overflow"] for s in samples),
        total_underruns=sum(s["output_underrun"] for s in samples),
        vram_peak_mb=0.0,
        samples=samples,
    )

    print("\n=== E2E Benchmark Results ===")
    print(f"  Duration:       {result.duration_seconds}s")
    print(f"  Avg latency:    {result.avg_latency_ms:.1f} ms")
    print(f"  P99 latency:    {result.p99_latency_ms:.1f} ms")
    print(f"  Max latency:    {result.max_latency_ms:.1f} ms")
    print(f"  Overflows:      {result.total_overflows}")
    print(f"  Underruns:      {result.total_underruns}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOSMIC E2E Benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    benchmark_end_to_end(args.config, args.duration)
