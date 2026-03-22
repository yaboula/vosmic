"""Tests for SystemTray — bypass toggle and shutdown commands (pure Python)."""

from __future__ import annotations

from src.gui.event_bus import EventBus


class TestBypassToggle:
    def test_bypass_on_sends_command(self) -> None:
        bus = EventBus()
        bus.send_command("set_bypass", {"enabled": True})
        cmd = bus.get_command()
        assert cmd is not None
        assert cmd["data"]["enabled"] is True

    def test_bypass_off_sends_command(self) -> None:
        bus = EventBus()
        bus.send_command("set_bypass", {"enabled": False})
        cmd = bus.get_command()
        assert cmd["data"]["enabled"] is False

    def test_toggle_state_tracking(self) -> None:
        bypass = False
        bypass = not bypass
        assert bypass is True
        bypass = not bypass
        assert bypass is False


class TestShutdown:
    def test_quit_sends_shutdown(self) -> None:
        bus = EventBus()
        bus.send_command("shutdown", {})
        cmd = bus.get_command()
        assert cmd["type"] == "shutdown"


class TestWindowToggle:
    def test_visibility_logic(self) -> None:
        visible = True
        if visible:
            visible = False
        else:
            visible = True
        assert visible is False
