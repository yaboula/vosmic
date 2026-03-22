"""PerformancePanel — real-time dashboard for latency, GPU, VRAM, and buffer."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from src.gui.styles import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    LATENCY_OK_MS,
    LATENCY_WARN_MS,
    VRAM_CRIT_PERCENT,
    VRAM_WARN_PERCENT,
)


def _bar_style(color: str) -> str:
    return f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}"


class PerformancePanel(QWidget):
    """Four progress bars: latency, GPU load, VRAM, buffer fill."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        grp = QGroupBox("PERFORMANCE")
        grid = QGridLayout(grp)

        self._latency_bar, self._latency_lbl = self._add_row(grid, 0, "Latency", 40)
        self._gpu_bar, self._gpu_lbl = self._add_row(grid, 1, "GPU", 100)
        self._vram_bar, self._vram_lbl = self._add_row(grid, 2, "VRAM", 100)
        self._buffer_bar, self._buffer_lbl = self._add_row(grid, 3, "Buffer", 100)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(grp)

    @staticmethod
    def _add_row(
        grid: QGridLayout, row: int, name: str, maximum: int
    ) -> tuple[QProgressBar, QLabel]:
        lbl_name = QLabel(name)
        bar = QProgressBar()
        bar.setMaximum(maximum)
        bar.setValue(0)
        bar.setTextVisible(False)
        lbl_val = QLabel("—")
        grid.addWidget(lbl_name, row, 0)
        grid.addWidget(bar, row, 1)
        grid.addWidget(lbl_val, row, 2)
        return bar, lbl_val

    def update_latency(self, data: dict[str, Any]) -> None:
        ms = data.get("latency_ms", 0.0)
        self._latency_bar.setValue(int(min(ms, 40)))
        self._latency_lbl.setText(f"{ms:.1f} ms")
        if ms < LATENCY_OK_MS:
            self._latency_bar.setStyleSheet(_bar_style(COLOR_GREEN))
        elif ms < LATENCY_WARN_MS:
            self._latency_bar.setStyleSheet(_bar_style(COLOR_YELLOW))
        else:
            self._latency_bar.setStyleSheet(_bar_style(COLOR_RED))

    def update_vram(self, data: dict[str, Any]) -> None:
        used = data.get("used_mb", 0.0)
        total = data.get("total_mb", 1.0)
        pct = (used / max(total, 1.0)) * 100
        self._vram_bar.setValue(int(min(pct, 100)))
        self._vram_lbl.setText(f"{used:.0f} / {total:.0f} MB")
        if pct < VRAM_WARN_PERCENT:
            self._vram_bar.setStyleSheet(_bar_style(COLOR_GREEN))
        elif pct < VRAM_CRIT_PERCENT:
            self._vram_bar.setStyleSheet(_bar_style(COLOR_YELLOW))
        else:
            self._vram_bar.setStyleSheet(_bar_style(COLOR_RED))

    def update_gpu(self, data: dict[str, Any]) -> None:
        pct = data.get("gpu_percent", 0.0)
        self._gpu_bar.setValue(int(min(pct, 100)))
        self._gpu_lbl.setText(f"{pct:.0f}%")

    def update_buffer(self, data: dict[str, Any]) -> None:
        pct = data.get("fill_percent", 0.0)
        self._buffer_bar.setValue(int(min(pct, 100)))
        self._buffer_lbl.setText(f"{pct:.0f}%")
