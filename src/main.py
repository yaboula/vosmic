"""VOSMIC — Application entry point.

Creates QApplication, wires the shared EventBus between MainWindow and
VosmicPipeline, starts the pipeline in background threads, and polls
pipeline.process_commands() via a QTimer on the Qt event loop.

Usage:
    python -m src.main
    # or, after `pip install -e .`:
    vosmic
"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.core.config import load_config
from src.core.pipeline import VosmicPipeline
from src.gui.event_bus import EventBus
from src.gui.main_window import MainWindow
from src.gui.system_tray import SystemTray

logger = logging.getLogger("vosmic")

COMMAND_POLL_MS = 10  # 100 Hz — fast enough for responsive UI commands


def _setup_logging(level_name: str) -> None:
    """Configure root logger with a clean format."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(config_path: str = "configs/default.yaml") -> None:
    """Application entry point — wires GUI, pipeline, and event bus."""

    # --- Config & logging ---------------------------------------------------
    config = load_config(config_path)
    _setup_logging(config.system.log_level)
    logger.info("VOSMIC starting...")

    # --- Qt application ------------------------------------------------------
    app = QApplication(sys.argv)
    app.setApplicationName("VOSMIC")
    app.setOrganizationName("VOSMIC")

    # --- Shared event bus (one instance for GUI + pipeline) ------------------
    event_bus = EventBus()

    # --- Pipeline (background audio threads) ---------------------------------
    pipeline = VosmicPipeline(config, event_bus=event_bus)

    # --- GUI -----------------------------------------------------------------
    window = MainWindow(event_bus)

    tray = SystemTray(event_bus, window, icon=QIcon())
    window.set_system_tray(tray)
    tray.show()

    # --- Command polling: GUI → pipeline via QTimer --------------------------
    cmd_timer = QTimer()
    cmd_timer.timeout.connect(pipeline.process_commands)
    cmd_timer.start(COMMAND_POLL_MS)

    # --- Start pipeline & show window ----------------------------------------
    pipeline.start()
    window.show_and_populate()
    logger.info("VOSMIC ready")

    # --- Run Qt event loop ---------------------------------------------------
    exit_code = app.exec()

    # --- Clean shutdown ------------------------------------------------------
    pipeline.stop()
    logger.info("VOSMIC shut down (exit code %d)", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
