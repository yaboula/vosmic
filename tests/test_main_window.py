"""Tests for MainWindow — bypass and shutdown command logic (pure Python)."""

from __future__ import annotations

from src.gui.event_bus import EventBus


class TestBypassCommand:
    def test_bypass_command_via_bus(self) -> None:
        bus = EventBus()
        bus.send_command("set_bypass", {"enabled": True})
        cmd = bus.get_command()
        assert cmd is not None
        assert cmd["type"] == "set_bypass"
        assert cmd["data"]["enabled"] is True

    def test_bypass_toggle_off(self) -> None:
        bus = EventBus()
        bus.send_command("set_bypass", {"enabled": False})
        cmd = bus.get_command()
        assert cmd["data"]["enabled"] is False


class TestShutdownOnClose:
    def test_shutdown_command(self) -> None:
        bus = EventBus()
        bus.send_command("shutdown", {})
        cmd = bus.get_command()
        assert cmd is not None
        assert cmd["type"] == "shutdown"

    def test_shutdown_data_empty(self) -> None:
        bus = EventBus()
        bus.send_command("shutdown")
        cmd = bus.get_command()
        assert cmd["data"] == {}


class TestEventSubscriptions:
    def test_latency_event_routed(self) -> None:
        bus = EventBus()
        received: list[dict] = []
        bus.subscribe("latency_update", lambda d: received.append(d))
        bus.emit("latency_update", {"latency_ms": 25.0})
        assert received[0]["latency_ms"] == 25.0

    def test_error_event_routed(self) -> None:
        bus = EventBus()
        received: list[dict] = []
        bus.subscribe("error", lambda d: received.append(d))
        bus.emit("error", {"message": "something failed"})
        assert received[0]["message"] == "something failed"
