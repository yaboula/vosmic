"""Tests for AudioCapture — unit tests that do NOT require real audio hardware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio_io.capture import AudioCapture, AudioDeviceError
from src.audio_io.ring_buffer import LockFreeRingBuffer


class TestAudioCaptureInit:
    def test_default_construction(self) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        assert cap.is_running is False
        assert cap.metrics["callback_count"] == 0
        assert cap.metrics["overflow_count"] == 0

    def test_custom_params(self) -> None:
        buf = LockFreeRingBuffer(8192)
        cap = AudioCapture(buf, device_index=3, sample_rate=44100, block_size=512, channels=2)
        assert cap._sample_rate == 44100
        assert cap._block_size == 512
        assert cap._channels == 2


class TestAudioCallback:
    def test_callback_writes_to_buffer(self) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        indata = np.random.randn(256, 1).astype(np.float32)

        cap._audio_callback(indata, 256, None, None)

        assert buf.available_read == 256
        assert cap.metrics["callback_count"] == 1
        assert cap.metrics["overflow_count"] == 0

    def test_callback_tracks_peak_level(self) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        indata = np.full((256, 1), 0.75, dtype=np.float32)

        cap._audio_callback(indata, 256, None, None)

        assert cap.metrics["peak_level"] == pytest.approx(0.75)

    def test_callback_counts_status_overflow(self) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        indata = np.zeros((256, 1), dtype=np.float32)
        status = MagicMock()
        status.__bool__ = lambda self: True

        cap._audio_callback(indata, 256, None, status)

        assert cap.metrics["overflow_count"] == 1

    def test_callback_counts_buffer_overflow(self) -> None:
        buf = LockFreeRingBuffer(128)
        cap = AudioCapture(buf)
        indata = np.zeros((256, 1), dtype=np.float32)

        cap._audio_callback(indata, 256, None, None)

        assert cap.metrics["overflow_count"] >= 1


class TestStartStop:
    @patch("src.audio_io.capture.sd.InputStream")
    def test_start_creates_stream(self, mock_stream_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        cap.start()

        mock_stream_cls.assert_called_once()
        assert cap.is_running is True

    @patch("src.audio_io.capture.sd.InputStream")
    def test_double_start_is_noop(self, mock_stream_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        cap.start()
        cap.start()

        mock_stream_cls.assert_called_once()

    @patch("src.audio_io.capture.sd.InputStream")
    def test_stop_closes_stream(self, mock_stream_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        cap.start()
        cap.stop()

        mock_stream_cls.return_value.stop.assert_called_once()
        mock_stream_cls.return_value.close.assert_called_once()
        assert cap.is_running is False

    @patch(
        "src.audio_io.capture.sd.InputStream",
        side_effect=__import__("sounddevice").PortAudioError(-9999, "No device"),
    )
    def test_start_raises_on_bad_device(self, mock_stream_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf, device_index=999)
        with pytest.raises(AudioDeviceError, match="Cannot open input device"):
            cap.start()


class TestChangeDevice:
    @patch("src.audio_io.capture.sd.InputStream")
    def test_change_device_restarts_stream(self, mock_stream_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        cap = AudioCapture(buf)
        cap.start()
        cap.change_device(5)

        assert cap._device_index == 5
        assert mock_stream_cls.return_value.stop.call_count == 1
        assert mock_stream_cls.call_count == 2
