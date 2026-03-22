"""VOSMIC GUI style constants — centralized theme for all widgets."""

from __future__ import annotations

# Window
WINDOW_TITLE = "VOSMIC — Real-Time Voice Changer"
WINDOW_MIN_WIDTH = 500
WINDOW_MIN_HEIGHT = 600

# Colors
COLOR_BG = "#1e1e2e"
COLOR_SURFACE = "#2a2a3c"
COLOR_BORDER = "#3a3a4c"
COLOR_TEXT = "#e0e0e0"
COLOR_TEXT_DIM = "#888899"
COLOR_ACCENT = "#7c3aed"
COLOR_ACCENT_HOVER = "#6d28d9"

COLOR_GREEN = "#22c55e"
COLOR_YELLOW = "#eab308"
COLOR_RED = "#ef4444"
COLOR_BLUE = "#3b82f6"

# Level meter thresholds (dB)
LEVEL_YELLOW_DB = -6.0
LEVEL_RED_DB = -3.0

# Performance thresholds
LATENCY_OK_MS = 30.0
LATENCY_WARN_MS = 35.0
VRAM_WARN_PERCENT = 80.0
VRAM_CRIT_PERCENT = 95.0

# Timing
DASHBOARD_REFRESH_MS = 100
LEVEL_METER_REFRESH_MS = 50

# Font
FONT_FAMILY = "Segoe UI"
FONT_SIZE_NORMAL = 10
FONT_SIZE_SMALL = 8
FONT_SIZE_HEADER = 14

# Stylesheet fragments
MAIN_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: '{FONT_FAMILY}';
    font-size: {FONT_SIZE_NORMAL}pt;
}}
QGroupBox {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {COLOR_ACCENT};
}}
QComboBox {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 28px;
    color: {COLOR_TEXT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QPushButton {{
    background-color: {COLOR_ACCENT};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: {COLOR_ACCENT_HOVER};
}}
QPushButton:disabled {{
    background-color: {COLOR_BORDER};
    color: {COLOR_TEXT_DIM};
}}
QProgressBar {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    text-align: center;
    min-height: 20px;
    color: {COLOR_TEXT};
}}
QProgressBar::chunk {{
    background-color: {COLOR_GREEN};
    border-radius: 3px;
}}
QLabel {{
    color: {COLOR_TEXT};
}}
"""
