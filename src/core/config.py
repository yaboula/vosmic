"""VOSMIC — Centralized configuration with YAML loading and validation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_SAMPLE_RATES = {44100, 48000, 96000}
VALID_BLOCK_SIZES = {64, 128, 256, 512, 1024}
VALID_PRECISIONS = {"fp32", "fp16", "int8"}


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    block_size: int = 256
    channels: int = 1
    input_device: int | None = None
    output_device: int | None = None
    ring_buffer_capacity: int = 4096


@dataclass
class DSPConfig:
    noise_gate_threshold_db: float = -40.0
    loudness_target_lufs: float = -18.0
    pitch_extractor: str = "rmvpe"
    pitch_hop_length: int = 160
    content_encoder: str = "contentvec"
    content_encoder_sr: int = 16000


@dataclass
class InferenceConfig:
    model_path: str = "models/narrator_v1.onnx"
    format: str = "onnx"
    precision: str = "fp16"
    max_inference_time_ms: int = 25
    queue_max_size: int = 4


@dataclass
class CompressorConfig:
    ratio: float = 3.0
    threshold_db: float = -12.0
    attack_ms: float = 5.0
    release_ms: float = 50.0


@dataclass
class PostProcessConfig:
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    filter_type: str = "butterworth"
    filter_order: int = 4
    filter_cutoff_hz: int = 20000
    bypass: bool = False


@dataclass
class SystemConfig:
    log_level: str = "INFO"
    profile_mode: bool = False
    passthrough_on_error: bool = True


@dataclass
class VosmicConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    postprocessing: PostProcessConfig = field(default_factory=PostProcessConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


class ConfigValidationError(ValueError):
    """Raised when configuration values violate constraints."""


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _build_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    """Recursively build a dataclass from a dict, applying defaults for missing fields."""
    if data is None:
        return cls()

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            if hasattr(f.type, "__dataclass_fields__") or (
                isinstance(f.type, str) and f.type in _DATACLASS_REGISTRY
            ):
                target_cls = _DATACLASS_REGISTRY.get(f.type, f.type)
                if isinstance(target_cls, str):
                    target_cls = _DATACLASS_REGISTRY[target_cls]
                kwargs[f.name] = _build_dataclass(
                    target_cls, val if isinstance(val, dict) else None
                )
            else:
                kwargs[f.name] = val

    return cls(**kwargs)


_DATACLASS_REGISTRY: dict[str, type] = {
    "AudioConfig": AudioConfig,
    "DSPConfig": DSPConfig,
    "InferenceConfig": InferenceConfig,
    "CompressorConfig": CompressorConfig,
    "PostProcessConfig": PostProcessConfig,
    "SystemConfig": SystemConfig,
}


def validate_config(config: VosmicConfig) -> None:
    """Validate configuration constraints. Raises ConfigValidationError on failure."""
    if config.audio.sample_rate not in VALID_SAMPLE_RATES:
        raise ConfigValidationError(
            f"sample_rate must be one of {VALID_SAMPLE_RATES}, got {config.audio.sample_rate}"
        )

    if config.audio.block_size not in VALID_BLOCK_SIZES:
        raise ConfigValidationError(
            f"block_size must be one of {VALID_BLOCK_SIZES}, got {config.audio.block_size}"
        )

    if not _is_power_of_two(config.audio.block_size):
        raise ConfigValidationError(
            f"block_size must be a power of 2, got {config.audio.block_size}"
        )

    if config.inference.precision not in VALID_PRECISIONS:
        raise ConfigValidationError(
            f"precision must be one of {VALID_PRECISIONS}, got {config.inference.precision}"
        )

    if not config.system.passthrough_on_error:
        raise ConfigValidationError(
            "passthrough_on_error must be True — silence on failure is never acceptable"
        )


def load_config(path: str | Path = "configs/default.yaml") -> VosmicConfig:
    """Load configuration from YAML with defaults for missing fields.

    If the file doesn't exist, returns all-default config with a warning.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("Config file %s not found, using defaults", path)
        return VosmicConfig()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    audio = _build_dataclass(AudioConfig, raw.get("audio"))
    dsp = _build_dataclass(DSPConfig, raw.get("dsp"))
    inference = _build_dataclass(InferenceConfig, raw.get("inference"))

    post_raw = raw.get("postprocessing", {})
    compressor = _build_dataclass(
        CompressorConfig, post_raw.get("compressor") if post_raw else None
    )
    filter_raw = post_raw.get("filter", {}) if post_raw else {}
    postprocessing = PostProcessConfig(
        compressor=compressor,
        filter_type=filter_raw.get("type", "butterworth"),
        filter_order=filter_raw.get("order", 4),
        filter_cutoff_hz=filter_raw.get("cutoff_hz", 20000),
        bypass=post_raw.get("bypass", False) if post_raw else False,
    )

    system = _build_dataclass(SystemConfig, raw.get("system"))
    system.passthrough_on_error = True

    config = VosmicConfig(
        audio=audio,
        dsp=dsp,
        inference=inference,
        postprocessing=postprocessing,
        system=system,
    )

    validate_config(config)
    return config


def save_config(config: VosmicConfig, path: str | Path) -> None:
    """Serialize config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)

    if "postprocessing" in data:
        post = data["postprocessing"]
        post["filter"] = {
            "type": post.pop("filter_type"),
            "order": post.pop("filter_order"),
            "cutoff_hz": post.pop("filter_cutoff_hz"),
        }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
