"""Tests for EventBus — thread-safe command/event messaging."""

from __future__ import annotations

import threading
import time

from src.gui.event_bus import EventBus


class TestCommandQueue:
    def test_send_and_get(self) -> None:
        bus = EventBus()
        assert bus.send_command("load_model", {"model_path": "a.onnx"})
        cmd = bus.get_command()
        assert cmd is not None
        assert cmd["type"] == "load_model"
        assert cmd["data"]["model_path"] == "a.onnx"

    def test_get_empty_returns_none(self) -> None:
        bus = EventBus()
        assert bus.get_command() is None

    def test_fifo_order(self) -> None:
        bus = EventBus()
        bus.send_command("set_bypass", {"enabled": True})
        bus.send_command("shutdown", {})
        assert bus.get_command()["type"] == "set_bypass"
        assert bus.get_command()["type"] == "shutdown"

    def test_full_queue_returns_false(self) -> None:
        bus = EventBus(max_commands=2)
        assert bus.send_command("a", {})
        assert bus.send_command("b", {})
        assert bus.send_command("c", {}) is False

    def test_pending_commands_count(self) -> None:
        bus = EventBus()
        bus.send_command("a", {})
        bus.send_command("b", {})
        assert bus.pending_commands == 2
        bus.get_command()
        assert bus.pending_commands == 1

    def test_default_data_is_empty_dict(self) -> None:
        bus = EventBus()
        bus.send_command("shutdown")
        cmd = bus.get_command()
        assert cmd["data"] == {}


class TestEventSubscription:
    def test_subscribe_and_emit(self) -> None:
        bus = EventBus()
        received: list[dict] = []
        bus.subscribe("latency_update", lambda d: received.append(d))
        bus.emit("latency_update", {"latency_ms": 25.0})
        assert len(received) == 1
        assert received[0]["latency_ms"] == 25.0

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        a: list[dict] = []
        b: list[dict] = []
        bus.subscribe("error", lambda d: a.append(d))
        bus.subscribe("error", lambda d: b.append(d))
        bus.emit("error", {"message": "oops"})
        assert len(a) == 1
        assert len(b) == 1

    def test_emit_unknown_event_is_noop(self) -> None:
        bus = EventBus()
        bus.emit("nonexistent", {"x": 1})

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[dict] = []
        cb = lambda d: received.append(d)  # noqa: E731
        bus.subscribe("error", cb)
        bus.unsubscribe("error", cb)
        bus.emit("error", {"message": "test"})
        assert len(received) == 0

    def test_subscriber_count(self) -> None:
        bus = EventBus()
        bus.subscribe("a", lambda d: None)
        bus.subscribe("b", lambda d: None)
        bus.subscribe("b", lambda d: None)
        assert bus.subscriber_count == 3

    def test_callback_exception_does_not_break_others(self) -> None:
        bus = EventBus()
        received: list[dict] = []

        def bad_cb(d: dict) -> None:
            raise ValueError("boom")

        bus.subscribe("error", bad_cb)
        bus.subscribe("error", lambda d: received.append(d))
        bus.emit("error", {"message": "test"})
        assert len(received) == 1


class TestClear:
    def test_clear_drains_commands(self) -> None:
        bus = EventBus()
        bus.send_command("a", {})
        bus.send_command("b", {})
        bus.subscribe("x", lambda d: None)
        bus.clear()
        assert bus.get_command() is None
        assert bus.subscriber_count == 0


class TestThreadSafety:
    def test_concurrent_send_get(self) -> None:
        bus = EventBus(max_commands=1024)
        n = 500
        received: list[dict] = []
        errors: list[str] = []

        def producer() -> None:
            for i in range(n):
                bus.send_command("cmd", {"i": i})

        def consumer() -> None:
            count = 0
            deadline = time.monotonic() + 5.0
            while count < n and time.monotonic() < deadline:
                cmd = bus.get_command()
                if cmd:
                    received.append(cmd)
                    count += 1
                else:
                    time.sleep(0.0001)
            if count < n:
                errors.append(f"Only got {count}/{n}")

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors, errors
        assert len(received) == n

    def test_concurrent_emit(self) -> None:
        bus = EventBus()
        counter = {"n": 0}
        lock = threading.Lock()

        def inc(d: dict) -> None:
            with lock:
                counter["n"] += 1

        bus.subscribe("tick", inc)

        threads = [threading.Thread(target=lambda: bus.emit("tick", {})) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert counter["n"] == 100


class TestPerformance:
    def test_send_command_under_01ms(self) -> None:
        bus = EventBus()
        t0 = time.perf_counter()
        for _ in range(10000):
            bus.send_command("set_bypass", {"enabled": True})
            bus.get_command()
        elapsed_us = (time.perf_counter() - t0) * 1e6 / 10000
        assert elapsed_us < 100, f"send+get averaged {elapsed_us:.1f} µs, exceeds 100µs"
