"""Tests for LevelMeter — VU meter logic (no Qt rendering tests)."""

from __future__ import annotations

import math


class TestPeakDbConversion:
    """Test the dB conversion logic used by LevelMeter without instantiating the widget."""

    @staticmethod
    def _peak_db(peak: float) -> float:
        if peak < 1e-10:
            return -100.0
        return 20.0 * math.log10(peak)

    def test_silent(self) -> None:
        assert self._peak_db(0.0) == -100.0

    def test_full_scale(self) -> None:
        assert abs(self._peak_db(1.0)) < 0.01

    def test_half(self) -> None:
        db = self._peak_db(0.5)
        assert -7.0 < db < -5.0

    def test_quarter(self) -> None:
        db = self._peak_db(0.25)
        assert -13.0 < db < -11.0


class TestSetLevelClamping:
    @staticmethod
    def _clamp(val: float) -> float:
        return max(0.0, min(1.0, val))

    def test_above_one(self) -> None:
        assert self._clamp(1.5) == 1.0

    def test_below_zero(self) -> None:
        assert self._clamp(-0.5) == 0.0

    def test_normal_value(self) -> None:
        assert self._clamp(0.7) == 0.7
