"""LevelMeter — animated horizontal VU meter widget."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter
from PyQt6.QtWidgets import QWidget

from src.gui.styles import (
    COLOR_BG,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_YELLOW,
)


class LevelMeter(QWidget):
    """Horizontal peak meter with green/yellow/red gradient.

    Call set_level(peak_linear) to update; the paintEvent draws the bar.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peak: float = 0.0
        self.setMinimumHeight(18)
        self.setMaximumHeight(22)

    def set_level(self, peak_linear: float) -> None:
        """Set peak level (0.0 .. 1.0 linear). Triggers repaint."""
        self._peak = max(0.0, min(1.0, peak_linear))
        self.update()

    def peak_db(self) -> float:
        if self._peak < 1e-10:
            return -100.0
        return 20.0 * math.log10(self._peak)

    def paintEvent(self, event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor(COLOR_SURFACE))

        if self._peak > 0.001:
            bar_w = int(w * self._peak)
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(COLOR_GREEN))
            grad.setColorAt(0.6, QColor(COLOR_YELLOW))
            grad.setColorAt(1.0, QColor(COLOR_RED))
            p.fillRect(0, 0, bar_w, h, grad)

        db_str = f"{self.peak_db():.0f} dB" if self._peak > 0.001 else "—∞ dB"
        p.setPen(QColor(COLOR_TEXT))
        p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, db_str)

        p.setPen(QColor(COLOR_BG))
        p.drawRect(0, 0, w - 1, h - 1)
        p.end()
