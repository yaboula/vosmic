"""Tests for CUDA detection — runs on systems with or without GPU."""

from __future__ import annotations

import subprocess
import sys


class TestCheckCudaScript:
    """Verify the check_cuda.py script executes and returns meaningful output."""

    def test_script_executes_without_crash(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_cuda.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "VOSMIC GPU Diagnostic" in result.stdout

    def test_script_exit_code(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_cuda.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Exit 0 if GPU present, exit 1 if not — both are valid for CI
        assert result.returncode in (0, 1)

    def test_script_reports_error_without_torch(self) -> None:
        # When torch is not installed, the script should report it clearly
        # This test passes on both GPU and non-GPU systems
        result = subprocess.run(
            [sys.executable, "scripts/check_cuda.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        has_gpu_info = "CUDA Version" in output
        has_error = "ERROR" in output
        assert has_gpu_info or has_error
