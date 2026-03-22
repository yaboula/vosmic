"""Tests for AudioPassthrough — unit tests using mocked audio streams."""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np

from src.audio_io.passthrough import AudioPassthrough
from src.core.config import AudioConfig


class TestPassthroughConstruction:
    @patch("src.audio_io.passthrough.find_virtual_cable", return_value=None)
    def test_creates_buffers_and_engines(self, _mock_vcable) -> None:
        cfg = AudioConfig(sample_rate=48000, block_size=256, ring_buffer_capacity=4096)
        pt = AudioPassthrough(cfg)
        assert pt._is_running is False
        assert pt._input_buffer.capacity == 4096
        assert pt._output_buffer.capacity == 4096


class TestPassthroughLoop:
    @patch("src.audio_io.passthrough.find_virtual_cable", return_value=None)
    def test_loop_transfers_data(self, _mock_vcable) -> None:
        cfg = AudioConfig(sample_rate=48000, block_size=64, ring_buffer_capacity=1024)
        pt = AudioPassthrough(cfg)

        test_data = np.arange(64, dtype=np.float32)
        pt._input_buffer.write(test_data)

        pt._is_running = True

        def one_iteration() -> None:
            block_size = pt._config.block_size
            if pt._input_buffer.available_read >= block_size:
                data = pt._input_buffer.read(block_size)
                pt._output_buffer.write(data)

        one_iteration()

        assert pt._output_buffer.available_read == 64
        result = pt._output_buffer.read(64)
        assert np.array_equal(result, test_data)


class TestMeasureLatency:
    @patch("src.audio_io.passthrough.find_virtual_cable", return_value=None)
    def test_latency_metrics_structure(self, _mock_vcable) -> None:
        cfg = AudioConfig(sample_rate=48000, block_size=256)
        pt = AudioPassthrough(cfg)
        metrics = pt.measure_latency()

        assert "input_callback_ms" in metrics
        assert "theoretical_latency_ms" in metrics
        assert "input_overflow" in metrics
        assert "output_underrun" in metrics
        assert metrics["theoretical_latency_ms"] > 0

    @patch("src.audio_io.passthrough.find_virtual_cable", return_value=None)
    def test_theoretical_latency_calculation(self, _mock_vcable) -> None:
        cfg = AudioConfig(sample_rate=48000, block_size=256)
        pt = AudioPassthrough(cfg)
        metrics = pt.measure_latency()

        expected = (256 / 48000) * 1000 * 2
        assert abs(metrics["theoretical_latency_ms"] - expected) < 0.01


class TestStartStopWithMocks:
    @patch("src.audio_io.output.sd.OutputStream")
    @patch("src.audio_io.capture.sd.InputStream")
    @patch("src.audio_io.passthrough.find_virtual_cable", return_value=None)
    def test_start_and_stop(self, _mock_vcable, _mock_in, _mock_out) -> None:
        cfg = AudioConfig(sample_rate=48000, block_size=256)
        pt = AudioPassthrough(cfg)

        pt.start()
        assert pt._is_running is True
        assert pt._capture.is_running is True
        assert pt._output.is_running is True

        time.sleep(0.05)

        pt.stop()
        assert pt._is_running is False
