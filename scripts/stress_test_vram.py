"""VRAM stress test — runs the pipeline for N minutes and checks for memory leaks.

Usage:
    python scripts/stress_test_vram.py [--minutes 30] [--config configs/default.yaml]

Fails if VRAM leak exceeds 50 MB.
"""

from __future__ import annotations

import argparse
import sys
import time


def get_vram_usage_mb() -> float:
    """Best-effort GPU VRAM query."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def stress_test(
    config_path: str = "configs/default.yaml",
    duration_minutes: int = 30,
) -> bool:
    from src.core.config import load_config
    from src.core.pipeline import VosmicPipeline

    config = load_config(config_path)
    pipeline = VosmicPipeline(config)

    print("Starting pipeline for VRAM stress test...")
    try:
        pipeline.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        sys.exit(1)

    time.sleep(2)
    vram_initial = get_vram_usage_mb()
    print(f"Initial VRAM: {vram_initial:.0f} MB")

    vram_samples: list[float] = []
    for minute in range(duration_minutes):
        time.sleep(60)
        vram_now = get_vram_usage_mb()
        vram_samples.append(vram_now)
        print(f"  Minute {minute + 1}: VRAM = {vram_now:.0f} MB")

    pipeline.stop()

    if vram_samples:
        vram_peak = max(vram_samples)
        vram_leak = vram_samples[-1] - vram_initial
    else:
        vram_peak = vram_initial
        vram_leak = 0.0

    print("\n=== VRAM Stress Results ===")
    print(f"  Initial:  {vram_initial:.0f} MB")
    print(f"  Peak:     {vram_peak:.0f} MB")
    print(f"  Leak:     {vram_leak:.0f} MB over {duration_minutes} min")

    ok = vram_leak < 50.0
    print(f"  Status:   {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOSMIC VRAM Stress Test")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    success = stress_test(args.config, args.minutes)
    sys.exit(0 if success else 1)
