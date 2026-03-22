"""Audio device enumeration and discovery — input, output, and virtual cables."""

from __future__ import annotations

import sounddevice as sd

_VIRTUAL_CABLE_KEYWORDS = ("virtual", "vb-", "cable")


def get_input_devices() -> list[dict]:
    """Return all available audio input devices with metadata."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    result = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                    "latency_low": dev["default_low_input_latency"],
                    "latency_high": dev["default_high_input_latency"],
                    "host_api": sd.query_hostapis(dev["hostapi"])["name"],
                    "is_default": idx == default_input,
                }
            )

    return result


def get_output_devices() -> list[dict]:
    """Return all available audio output devices, flagging virtual cables."""
    devices = sd.query_devices()
    default_output = sd.default.device[1]
    result = []

    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            name_lower = dev["name"].lower()
            result.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_output_channels"],
                    "sample_rate": dev["default_samplerate"],
                    "latency_low": dev["default_low_output_latency"],
                    "latency_high": dev["default_high_output_latency"],
                    "host_api": sd.query_hostapis(dev["hostapi"])["name"],
                    "is_virtual_cable": any(kw in name_lower for kw in _VIRTUAL_CABLE_KEYWORDS),
                    "is_default": idx == default_output,
                }
            )

    return result


def find_virtual_cable() -> int | None:
    """Auto-detect a virtual cable output device index, or None if not found."""
    for dev in get_output_devices():
        if dev["is_virtual_cable"]:
            return dev["index"]
    return None


def validate_device(device_index: int, direction: str, sample_rate: int) -> bool:
    """Check whether a device supports the requested sample rate.

    Args:
        device_index: sounddevice device index.
        direction: ``"input"`` or ``"output"``.
        sample_rate: desired sample rate in Hz.
    """
    try:
        if direction == "input":
            sd.check_input_settings(device=device_index, samplerate=sample_rate)
        else:
            sd.check_output_settings(device=device_index, samplerate=sample_rate)
        return True
    except Exception:
        return False
