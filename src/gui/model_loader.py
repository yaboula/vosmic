"""ModelLoaderPanel — model file selection with load/swap and progress feedback."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.event_bus import EventBus


class ModelLoaderPanel(QWidget):
    """Panel showing active model info with Load and Swap buttons."""

    MODEL_FILTER = "Models (*.onnx *.pth *.pt *.trt *.engine)"

    def __init__(self, event_bus: EventBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bus = event_bus

        grp = QGroupBox("VOICE MODEL")
        layout = QVBoxLayout(grp)

        self._name_label = QLabel("No model loaded")
        self._info_label = QLabel("")
        layout.addWidget(self._name_label)
        layout.addWidget(self._info_label)

        btn_row = QHBoxLayout()
        self._load_btn = QPushButton("Load...")
        self._load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(self._load_btn)

        self._swap_btn = QPushButton("Swap...")
        self._swap_btn.setEnabled(False)
        self._swap_btn.clicked.connect(self._on_swap)
        btn_row.addWidget(self._swap_btn)
        layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
        layout.addWidget(self._progress)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(grp)

    def _pick_file(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", self.MODEL_FILTER)
        return path if path else None

    def _on_load(self) -> None:
        path = self._pick_file()
        if path:
            self._progress.show()
            self._load_btn.setEnabled(False)
            self._bus.send_command("load_model", {"model_path": path})

    def _on_swap(self) -> None:
        path = self._pick_file()
        if path:
            self._progress.show()
            self._swap_btn.setEnabled(False)
            self._bus.send_command("load_model", {"model_path": path, "swap": True})

    def on_model_loaded(self, data: dict[str, Any]) -> None:
        """Callback from EventBus 'model_loaded' event."""
        self._progress.hide()
        self._load_btn.setEnabled(True)
        self._swap_btn.setEnabled(True)
        self._name_label.setText(data.get("name", "Unknown"))
        fmt = data.get("format", "")
        vram = data.get("vram_mb", 0.0)
        self._info_label.setText(f"{fmt.upper()}  |  {vram:.0f} MB VRAM")

    def on_model_error(self, data: dict[str, Any]) -> None:
        """Callback from EventBus 'model_load_error' event."""
        self._progress.hide()
        self._load_btn.setEnabled(True)
        self._swap_btn.setEnabled(True)
        self._name_label.setText("Load failed")
        self._info_label.setText(data.get("error", "Unknown error"))
