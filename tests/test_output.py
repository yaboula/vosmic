"""Tests for AudioOutput — unit tests that do NOT require real audio hardware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio_io.capture import AudioDeviceError
from src.audio_io.output import AudioOutput
from src.audio_io.ring_buffer import LockFreeRingBuffer


class TestAudioOutputInit:
    def test_default_construction(self) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf)
        assert out.is_running is False
        assert out.metrics["underrun_count"] == 0


class TestOutputCallback:
    def test_callback_reads_from_buffer(self) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf, block_size=256)

        buf.write(np.ones(256, dtype=np.float32))
        outdata = np.zeros((256, 1), dtype=np.float32)

        out._output_callback(outdata, 256, None, None)

        assert np.all(outdata[:, 0] == 1.0)
        assert out.metrics["underrun_count"] == 0

    def test_callback_partial_data_pads_zeros(self) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf, block_size=256)

        buf.write(np.ones(100, dtype=np.float32))
        outdata = np.full((256, 1), -1.0, dtype=np.float32)

        out._output_callback(outdata, 256, None, None)

        assert np.all(outdata[:100, 0] == 1.0)
        assert np.all(outdata[100:, 0] == 0.0)
        assert out.metrics["underrun_count"] == 1

    def test_callback_empty_buffer_outputs_silence(self) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf, block_size=256)
        outdata = np.full((256, 1), -1.0, dtype=np.float32)

        out._output_callback(outdata, 256, None, None)

        assert np.all(outdata[:, 0] == 0.0)
        assert out.metrics["underrun_count"] == 1

    def test_callback_tracks_output_level(self) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf, block_size=256)
        buf.write(np.full(256, 0.5, dtype=np.float32))
        outdata = np.zeros((256, 1), dtype=np.float32)

        out._output_callback(outdata, 256, None, None)

        assert out.metrics["output_level"] == pytest.approx(0.5)


class TestOutputStartStop:
    @patch("src.audio_io.output.sd.OutputStream")
    def test_start_creates_stream(self, mock_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf)
        out.start()

        mock_cls.assert_called_once()
        assert out.is_running is True

    @patch("src.audio_io.output.sd.OutputStream")
    def test_stop_closes_stream(self, mock_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf)
        out.start()
        out.stop()

        mock_cls.return_value.stop.assert_called_once()
        mock_cls.return_value.close.assert_called_once()
        assert out.is_running is False

    @patch(
        "src.audio_io.output.sd.OutputStream",
        side_effect=__import__("sounddevice").PortAudioError(-9999, "No device"),
    )
    def test_start_raises_on_bad_device(self, mock_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf, device_index=999)
        with pytest.raises(AudioDeviceError, match="Cannot open output device"):
            out.start()


class TestSetOutputDevice:
    @patch("src.audio_io.output.sd.OutputStream")
    def test_hot_swap_restarts_stream(self, mock_cls: MagicMock) -> None:
        buf = LockFreeRingBuffer(4096)
        out = AudioOutput(buf)
        out.start()
        out.set_output_device(7)

        assert out._device_index == 7
        assert mock_cls.return_value.stop.call_count == 1
        assert mock_cls.call_count == 2
