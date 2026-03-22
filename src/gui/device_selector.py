"""DeviceSelectorPanel — input/output device combo boxes with level meters."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.audio_io.devices import get_input_devices, get_output_devices
from src.gui.event_bus import EventBus
from src.gui.level_meter import LevelMeter


class DeviceSelectorPanel(QWidget):
    """Two grouped combo boxes for input/output audio device selection."""

    def __init__(self, event_bus: EventBus, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bus = event_bus

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_output_group())

    def _build_input_group(self) -> QGroupBox:
        grp = QGroupBox("INPUT")
        vbox = QVBoxLayout(grp)

        self._input_combo = QComboBox()
        self._input_combo.currentIndexChanged.connect(self._on_input_changed)
        vbox.addWidget(self._input_combo)

        self._input_meter = LevelMeter()
        vbox.addWidget(self._input_meter)

        btn = QPushButton("Refresh")
        btn.clicked.connect(self._refresh_inputs)
        vbox.addWidget(btn)

        return grp

    def _build_output_group(self) -> QGroupBox:
        grp = QGroupBox("OUTPUT")
        vbox = QVBoxLayout(grp)

        self._output_combo = QComboBox()
        self._output_combo.currentIndexChanged.connect(self._on_output_changed)
        vbox.addWidget(self._output_combo)

        self._output_meter = LevelMeter()
        vbox.addWidget(self._output_meter)

        btn = QPushButton("Refresh")
        btn.clicked.connect(self._refresh_outputs)
        vbox.addWidget(btn)

        return grp

    def _refresh_inputs(self) -> None:
        self._input_combo.blockSignals(True)
        self._input_combo.clear()
        for dev in get_input_devices():
            self._input_combo.addItem(dev["name"], dev["index"])
        self._input_combo.blockSignals(False)

    def _refresh_outputs(self) -> None:
        self._output_combo.blockSignals(True)
        self._output_combo.clear()
        for dev in get_output_devices():
            self._output_combo.addItem(dev["name"], dev["index"])
        self._output_combo.blockSignals(False)

    def _on_input_changed(self, idx: int) -> None:
        if idx < 0:
            return
        dev_idx = self._input_combo.itemData(idx)
        if dev_idx is not None:
            self._bus.send_command("set_input_device", {"device_index": dev_idx})

    def _on_output_changed(self, idx: int) -> None:
        if idx < 0:
            return
        dev_idx = self._output_combo.itemData(idx)
        if dev_idx is not None:
            self._bus.send_command("set_output_device", {"device_index": dev_idx})

    def update_levels(self, data: dict[str, Any]) -> None:
        """Called by EventBus 'audio_level' events."""
        self._input_meter.set_level(data.get("input_peak", 0.0))
        self._output_meter.set_level(data.get("output_peak", 0.0))

    def populate(self) -> None:
        """Initial load of device lists."""
        self._refresh_inputs()
        self._refresh_outputs()
