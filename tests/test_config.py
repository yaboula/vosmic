"""Tests for src/core/config.py — YAML loading, defaults, and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    AudioConfig,
    CompressorConfig,
    ConfigValidationError,
    DSPConfig,
    InferenceConfig,
    SystemConfig,
    VosmicConfig,
    load_config,
    save_config,
    validate_config,
)


class TestDefaultValues:
    """Verify all dataclass defaults match the specification."""

    def test_audio_defaults(self) -> None:
        cfg = AudioConfig()
        assert cfg.sample_rate == 48000
        assert cfg.block_size == 256
        assert cfg.channels == 1
        assert cfg.input_device is None
        assert cfg.output_device is None
        assert cfg.ring_buffer_capacity == 4096

    def test_dsp_defaults(self) -> None:
        cfg = DSPConfig()
        assert cfg.noise_gate_threshold_db == -40.0
        assert cfg.loudness_target_lufs == -18.0
        assert cfg.pitch_extractor == "rmvpe"
        assert cfg.pitch_hop_length == 160
        assert cfg.content_encoder == "contentvec"
        assert cfg.content_encoder_sr == 16000

    def test_inference_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.model_path == "models/narrator_v1.onnx"
        assert cfg.format == "onnx"
        assert cfg.precision == "fp16"
        assert cfg.max_inference_time_ms == 25
        assert cfg.queue_max_size == 4

    def test_compressor_defaults(self) -> None:
        cfg = CompressorConfig()
        assert cfg.ratio == 3.0
        assert cfg.threshold_db == -12.0
        assert cfg.attack_ms == 5.0
        assert cfg.release_ms == 50.0

    def test_system_defaults(self) -> None:
        cfg = SystemConfig()
        assert cfg.log_level == "INFO"
        assert cfg.profile_mode is False
        assert cfg.passthrough_on_error is True


class TestLoadConfig:
    """Test YAML loading with real file and missing file scenarios."""

    def test_load_default_yaml(self) -> None:
        cfg = load_config("configs/default.yaml")
        assert cfg.audio.sample_rate == 48000
        assert cfg.audio.block_size == 256
        assert cfg.dsp.pitch_extractor == "rmvpe"
        assert cfg.inference.precision == "fp16"
        assert cfg.system.passthrough_on_error is True

    def test_load_missing_file_returns_defaults(self) -> None:
        cfg = load_config("configs/nonexistent.yaml")
        assert cfg.audio.sample_rate == 48000
        assert cfg.system.passthrough_on_error is True

    def test_load_partial_yaml_fills_defaults(self) -> None:
        partial = {"audio": {"sample_rate": 44100}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(partial, f)
            tmp_path = f.name

        cfg = load_config(tmp_path)
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.block_size == 256  # default filled in
        assert cfg.dsp.pitch_extractor == "rmvpe"  # entire section defaulted
        Path(tmp_path).unlink()

    def test_load_empty_yaml_returns_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name

        cfg = load_config(tmp_path)
        assert cfg.audio.sample_rate == 48000
        Path(tmp_path).unlink()

    def test_postprocessing_filter_loaded(self) -> None:
        cfg = load_config("configs/default.yaml")
        assert cfg.postprocessing.filter_type == "butterworth"
        assert cfg.postprocessing.filter_order == 4
        assert cfg.postprocessing.filter_cutoff_hz == 20000
        assert cfg.postprocessing.bypass is False

    def test_compressor_loaded(self) -> None:
        cfg = load_config("configs/default.yaml")
        assert cfg.postprocessing.compressor.ratio == 3.0
        assert cfg.postprocessing.compressor.threshold_db == -12.0


class TestValidation:
    """Test that validation rejects invalid configurations."""

    def test_invalid_sample_rate(self) -> None:
        cfg = VosmicConfig()
        cfg.audio.sample_rate = 22050
        with pytest.raises(ConfigValidationError, match="sample_rate"):
            validate_config(cfg)

    def test_invalid_block_size(self) -> None:
        cfg = VosmicConfig()
        cfg.audio.block_size = 300
        with pytest.raises(ConfigValidationError, match="block_size"):
            validate_config(cfg)

    def test_invalid_precision(self) -> None:
        cfg = VosmicConfig()
        cfg.inference.precision = "bf16"
        with pytest.raises(ConfigValidationError, match="precision"):
            validate_config(cfg)

    def test_passthrough_on_error_forced_true(self) -> None:
        cfg = VosmicConfig()
        cfg.system.passthrough_on_error = False
        with pytest.raises(ConfigValidationError, match="passthrough_on_error"):
            validate_config(cfg)

    def test_valid_config_passes(self) -> None:
        cfg = VosmicConfig()
        validate_config(cfg)  # should not raise


class TestSaveAndRoundTrip:
    """Test serialization and round-trip consistency."""

    def test_save_creates_file(self) -> None:
        cfg = VosmicConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            save_config(cfg, path)
            assert path.exists()

            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["audio"]["sample_rate"] == 48000

    def test_round_trip(self) -> None:
        original = load_config("configs/default.yaml")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.yaml"
            save_config(original, path)
            reloaded = load_config(path)

        assert original.audio.sample_rate == reloaded.audio.sample_rate
        assert original.audio.block_size == reloaded.audio.block_size
        assert original.dsp.pitch_extractor == reloaded.dsp.pitch_extractor
        assert original.inference.precision == reloaded.inference.precision
        assert original.postprocessing.compressor.ratio == reloaded.postprocessing.compressor.ratio
        assert original.system.passthrough_on_error == reloaded.system.passthrough_on_error
