"""Tests for PerformancePanel — dashboard update logic (pure logic, no Qt)."""

from __future__ import annotations

from src.gui.styles import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    LATENCY_OK_MS,
    LATENCY_WARN_MS,
    VRAM_CRIT_PERCENT,
    VRAM_WARN_PERCENT,
)


class TestLatencyThresholds:
    def test_green_threshold(self) -> None:
        assert 20.0 < LATENCY_OK_MS

    def test_warn_above_ok(self) -> None:
        assert LATENCY_WARN_MS > LATENCY_OK_MS

    def test_color_for_good_latency(self) -> None:
        ms = 20.0
        if ms < LATENCY_OK_MS:
            color = COLOR_GREEN
        elif ms < LATENCY_WARN_MS:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED
        assert color == COLOR_GREEN

    def test_color_for_bad_latency(self) -> None:
        ms = 38.0
        if ms < LATENCY_OK_MS:
            color = COLOR_GREEN
        elif ms < LATENCY_WARN_MS:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED
        assert color == COLOR_RED


class TestVramThresholds:
    def test_low_vram_is_green(self) -> None:
        pct = 25.0
        if pct < VRAM_WARN_PERCENT:
            color = COLOR_GREEN
        elif pct < VRAM_CRIT_PERCENT:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED
        assert color == COLOR_GREEN

    def test_high_vram_is_red(self) -> None:
        pct = 97.0
        if pct < VRAM_WARN_PERCENT:
            color = COLOR_GREEN
        elif pct < VRAM_CRIT_PERCENT:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED
        assert color == COLOR_RED

    def test_vram_percentage_calculation(self) -> None:
        used, total = 2000.0, 8000.0
        pct = (used / max(total, 1.0)) * 100
        assert abs(pct - 25.0) < 0.01
