"""SystemTray — tray icon with context menu for show/hide, bypass, and quit."""

from __future__ import annotations

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from src.gui.event_bus import EventBus


class SystemTray(QSystemTrayIcon):
    """System tray icon that keeps the app accessible when minimized.

    - Show/Hide toggles main window visibility
    - Active/Bypass toggles voice changer
    - Quit sends shutdown command and closes
    """

    def __init__(
        self,
        event_bus: EventBus,
        main_window: QWidget,
        icon: QIcon | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(icon or QIcon(), parent)
        self._bus = event_bus
        self._main_window = main_window
        self._bypass = False

        menu = QMenu()

        self._show_action = QAction("Hide", menu)
        self._show_action.triggered.connect(self._toggle_window)
        menu.addAction(self._show_action)

        menu.addSeparator()

        self._bypass_action = QAction("Bypass: OFF", menu)
        self._bypass_action.triggered.connect(self._toggle_bypass)
        menu.addAction(self._bypass_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def _toggle_window(self) -> None:
        if self._main_window.isVisible():
            self._main_window.hide()
            self._show_action.setText("Show")
        else:
            self._main_window.show()
            self._main_window.raise_()
            self._show_action.setText("Hide")

    def _toggle_bypass(self) -> None:
        self._bypass = not self._bypass
        self._bypass_action.setText(f"Bypass: {'ON' if self._bypass else 'OFF'}")
        self._bus.send_command("set_bypass", {"enabled": self._bypass})

    def _quit(self) -> None:
        self._bus.send_command("shutdown", {})
        from PyQt6.QtWidgets import QApplication

        QApplication.quit()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._toggle_window()
