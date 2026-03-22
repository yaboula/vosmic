"""GPU compatibility tests — latency and VRAM budget checks.

These tests validate thresholds. Actual GPU tests require hardware
and are marked with @pytest.mark.gpu (skipped in CI).
"""

from __future__ import annotations

import pytest

RTX_4060_LATENCY_MS = 35.0
GTX_1650_LATENCY_MS = 40.0
RTX_4060_VRAM_MB = 2048.0
GTX_1650_VRAM_MB = 3584.0
VRAM_LEAK_THRESHOLD_MB = 50.0


class TestRTX4060Thresholds:
    def test_latency_target_reasonable(self) -> None:
        assert RTX_4060_LATENCY_MS <= 40.0

    def test_vram_budget(self) -> None:
        assert RTX_4060_VRAM_MB <= 4096.0

    def test_simulated_latency_budget(self) -> None:
        capture_ms = 5.3
        dsp_ms = 2.1
        inference_ms = 18.0
        post_ms = 1.5
        output_ms = 5.3
        total = capture_ms + dsp_ms + inference_ms + post_ms + output_ms
        assert total <= RTX_4060_LATENCY_MS, (
            f"Simulated total {total:.1f} ms > {RTX_4060_LATENCY_MS} ms"
        )


class TestGTX1650Thresholds:
    def test_latency_target_reasonable(self) -> None:
        assert GTX_1650_LATENCY_MS <= 50.0

    def test_vram_budget(self) -> None:
        assert GTX_1650_VRAM_MB <= 4096.0


class TestVRAMLeakPolicy:
    def test_leak_threshold(self) -> None:
        assert VRAM_LEAK_THRESHOLD_MB == 50.0

    def test_simulated_no_leak(self) -> None:
        initial = 1200.0
        after_30_min = 1220.0
        leak = after_30_min - initial
        assert leak < VRAM_LEAK_THRESHOLD_MB


_HAS_TORCH = False
try:
    import torch as _torch

    _HAS_TORCH = _torch.cuda.is_available()
except ImportError:
    pass


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_TORCH, reason="torch + CUDA not available")
class TestGPUHardware:
    """Requires actual GPU hardware. Run with: pytest -m gpu"""

    def test_cuda_available(self) -> None:
        import torch

        assert torch.cuda.is_available(), "CUDA not available"

    def test_vram_sufficient(self) -> None:
        import torch

        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_mem / (1024 * 1024)
        assert total_mb >= 4096, f"Need >= 4GB VRAM, got {total_mb:.0f} MB"
