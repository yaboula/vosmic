"""Tests for ModelLoaderPanel — callback logic (pure Python, no Qt instantiation)."""

from __future__ import annotations

from src.gui.event_bus import EventBus


class TestModelLoadCommand:
    def test_load_sends_command(self) -> None:
        bus = EventBus()
        bus.send_command("load_model", {"model_path": "/models/test.onnx"})
        cmd = bus.get_command()
        assert cmd is not None
        assert cmd["type"] == "load_model"
        assert cmd["data"]["model_path"] == "/models/test.onnx"

    def test_swap_sends_command(self) -> None:
        bus = EventBus()
        bus.send_command("load_model", {"model_path": "/models/v2.pth", "swap": True})
        cmd = bus.get_command()
        assert cmd["data"]["swap"] is True


class TestModelLoadedEvent:
    def test_event_dispatched(self) -> None:
        bus = EventBus()
        received: list[dict] = []
        bus.subscribe("model_loaded", lambda d: received.append(d))

        bus.emit("model_loaded", {"name": "Narrator", "format": "onnx", "vram_mb": 1200.0})
        assert len(received) == 1
        assert received[0]["name"] == "Narrator"

    def test_error_event_dispatched(self) -> None:
        bus = EventBus()
        received: list[dict] = []
        bus.subscribe("model_load_error", lambda d: received.append(d))

        bus.emit("model_load_error", {"error": "File not found"})
        assert received[0]["error"] == "File not found"
