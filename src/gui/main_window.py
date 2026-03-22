"""MainWindow — primary VOSMIC control surface."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.device_selector import DeviceSelectorPanel
from src.gui.event_bus import EventBus
from src.gui.model_loader import ModelLoaderPanel
from src.gui.performance_panel import PerformancePanel
from src.gui.styles import (
    DASHBOARD_REFRESH_MS,
    MAIN_STYLESHEET,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_TITLE,
)


class MainWindow(QMainWindow):
    """Top-level window assembling all panels.

    Communicates with the audio pipeline ONLY through the EventBus.
    The GUI never calls audio functions directly.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__()
        self._bus = event_bus
        self._bypass = False

        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(MAIN_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.device_panel = DeviceSelectorPanel(event_bus)
        self.model_panel = ModelLoaderPanel(event_bus)
        self.perf_panel = PerformancePanel()

        root.addWidget(self.device_panel)
        root.addWidget(self.model_panel)
        root.addWidget(self.perf_panel)

        controls = self._build_controls()
        root.addLayout(controls)
        root.addStretch()

        self._subscribe_events()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(DASHBOARD_REFRESH_MS)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._active_btn = QPushButton("ACTIVE")
        self._active_btn.setCheckable(True)
        self._active_btn.setChecked(True)
        self._active_btn.clicked.connect(self._toggle_bypass)
        row.addWidget(self._active_btn)

        self._bypass_btn = QPushButton("BYPASS")
        self._bypass_btn.setCheckable(True)
        self._bypass_btn.clicked.connect(self._toggle_bypass)
        row.addWidget(self._bypass_btn)
        return row

    def _toggle_bypass(self) -> None:
        self._bypass = not self._bypass
        self._active_btn.setChecked(not self._bypass)
        self._bypass_btn.setChecked(self._bypass)
        self._bus.send_command("set_bypass", {"enabled": self._bypass})

    def _subscribe_events(self) -> None:
        self._bus.subscribe("latency_update", self.perf_panel.update_latency)
        self._bus.subscribe("vram_update", self.perf_panel.update_vram)
        self._bus.subscribe("buffer_status", self.perf_panel.update_buffer)
        self._bus.subscribe("audio_level", self.device_panel.update_levels)
        self._bus.subscribe("model_loaded", self.model_panel.on_model_loaded)
        self._bus.subscribe("model_load_error", self.model_panel.on_model_error)
        self._bus.subscribe("error", self._show_error)

    def _show_error(self, data: dict[str, Any]) -> None:
        QMessageBox.warning(self, "Error", data.get("message", "Unknown error"))

    def _refresh(self) -> None:
        pass

    def closeEvent(self, event: object) -> None:  # type: ignore[override]  # noqa: N802
        self._bus.send_command("shutdown", {})
        super().closeEvent(event)  # type: ignore[arg-type]

    def show_and_populate(self) -> None:
        """Show window and populate device lists."""
        self.show()
        self.device_panel.populate()
