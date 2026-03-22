"""EventBus — thread-safe messaging between GUI and audio pipeline.

GUI -> Pipeline: commands via queue.Queue (get_command / send_command)
Pipeline -> GUI: events via registered callbacks (subscribe / emit)
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

CommandData = dict[str, Any]
EventCallback = Callable[[dict[str, Any]], None]


class EventBus:
    """Bidirectional, thread-safe message bus.

    Command direction: GUI -> Pipeline (queue-based, non-blocking).
    Event direction:   Pipeline -> GUI (callback-based, invoked in caller thread).
    """

    COMMAND_TYPES = frozenset(
        {
            "set_input_device",
            "set_output_device",
            "load_model",
            "set_bypass",
            "set_noise_gate",
            "shutdown",
        }
    )

    EVENT_TYPES = frozenset(
        {
            "latency_update",
            "vram_update",
            "buffer_status",
            "model_loaded",
            "model_load_error",
            "audio_level",
            "error",
        }
    )

    def __init__(self, max_commands: int = 32) -> None:
        self._command_queue: queue.Queue[CommandData] = queue.Queue(maxsize=max_commands)
        self._callbacks: dict[str, list[EventCallback]] = defaultdict(list)
        self._lock = threading.Lock()

    # --- GUI -> Pipeline (commands) ---

    def send_command(self, command_type: str, data: dict[str, Any] | None = None) -> bool:
        """Enqueue a command for the pipeline. Returns False if queue is full."""
        cmd: CommandData = {"type": command_type, "data": data or {}}
        try:
            self._command_queue.put_nowait(cmd)
            return True
        except queue.Full:
            logger.warning("Command queue full, dropping: %s", command_type)
            return False

    def get_command(self) -> CommandData | None:
        """Non-blocking dequeue for the pipeline thread."""
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def pending_commands(self) -> int:
        return self._command_queue.qsize()

    # --- Pipeline -> GUI (events) ---

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            self._callbacks[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            cbs = self._callbacks.get(event_type, [])
            if callback in cbs:
                cbs.remove(callback)

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Fire all callbacks for an event type. Safe to call from any thread."""
        with self._lock:
            cbs = list(self._callbacks.get(event_type, []))
        for cb in cbs:
            try:
                cb(data or {})
            except Exception:
                logger.exception("Error in event callback for %s", event_type)

    def emit_threadsafe(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Emit ensuring callbacks run on the Qt main thread.

        Use this when emitting from a non-GUI thread (e.g. metrics reporter).
        Falls back to regular emit() if Qt event loop is not running.
        """
        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, lambda: self.emit(event_type, data))
        except (ImportError, RuntimeError):
            self.emit(event_type, data)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._callbacks.values())

    def clear(self) -> None:
        """Drain all pending commands and remove all subscriptions."""
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._callbacks.clear()
