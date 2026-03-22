"""Regression suite runner — executes all test suites with recommended flags.

Usage:
    python scripts/run_regression.py [--timeout 120] [--verbose]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_regression(timeout: int = 120, verbose: bool = True) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--tb=short",
        "-x",
        f"--timeout={timeout}",
    ]
    if verbose:
        cmd.append("-v")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".")
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOSMIC Regression Suite")
    parser.add_argument("--timeout", type=int, default=120, help="Per-test timeout in seconds")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    sys.exit(run_regression(args.timeout, args.verbose))
