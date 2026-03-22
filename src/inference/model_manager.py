"""Model manager — load, swap, unload RVC models with VRAM tracking."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"pytorch", "onnx", "tensorrt"}
SWAP_TIMEOUT_S = 10


class ModelError(Exception):
    """Base error for model operations."""


class UnsupportedFormatError(ModelError):
    """Raised when model format is not recognized."""


class ModelSwapError(ModelError):
    """Raised when hot-swap fails (previous model is preserved)."""


class ModelManager:
    """Manages RVC model lifecycle: load, hot-swap, unload, and VRAM tracking.

    The loading lock is ONLY used during model load/swap operations,
    never on the inference hot path.
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._active_model: object | None = None
        self._model_info: dict = {}
        self._format: str = ""
        self._loading_lock = threading.Lock()
        self._torch = _try_import_torch()

    @property
    def active_model(self) -> object | None:
        return self._active_model

    @property
    def model_info(self) -> dict:
        return self._model_info.copy()

    @property
    def format(self) -> str:
        return self._format

    @property
    def is_loaded(self) -> bool:
        return self._active_model is not None

    def load(self, path: str, format: str = "onnx", precision: str = "fp16") -> None:
        """Load a model from disk. Thread-safe via loading lock."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        detected_format = format
        if format == "auto":
            detected_format = _detect_format(path_obj)

        if detected_format not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format '{detected_format}'. Expected one of {SUPPORTED_FORMATS}"
            )

        with self._loading_lock:
            if detected_format == "pytorch":
                model = self._load_pytorch(path, precision)
            elif detected_format == "onnx":
                model = self._load_onnx(path)
            elif detected_format == "tensorrt":
                model = self._load_tensorrt(path)
            else:
                raise UnsupportedFormatError(f"Unknown format: {detected_format}")

            self._active_model = model
            self._format = detected_format
            self._model_info = {
                "path": path,
                "format": detected_format,
                "precision": precision,
                "name": path_obj.stem,
            }

        logger.info("Model loaded: %s (%s, %s)", path_obj.name, detected_format, precision)

    def swap_model(self, new_path: str, format: str = "onnx", precision: str = "fp16") -> None:
        """Hot-swap: load new model before releasing old. Zero downtime."""
        old_model = self._active_model
        old_info = self._model_info.copy()
        old_format = self._format

        t0 = time.monotonic()
        try:
            self.load(new_path, format=format, precision=precision)
        except Exception as e:
            self._active_model = old_model
            self._model_info = old_info
            self._format = old_format
            raise ModelSwapError(f"Swap failed, previous model preserved: {e}") from e

        elapsed = time.monotonic() - t0
        if elapsed > SWAP_TIMEOUT_S:
            logger.warning("Model swap took %.1fs (> %ds timeout)", elapsed, SWAP_TIMEOUT_S)

        _cleanup_old_model(old_model, self._torch)
        logger.info("Model swapped in %.2fs", elapsed)

    def unload(self) -> None:
        """Unload current model and free VRAM."""
        with self._loading_lock:
            _cleanup_old_model(self._active_model, self._torch)
            self._active_model = None
            self._model_info = {}
            self._format = ""
        logger.info("Model unloaded")

    def get_vram_usage(self) -> dict:
        """Return current VRAM usage in MB."""
        if self._torch is None or not self._torch.cuda.is_available():
            return {"used_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}

        used = self._torch.cuda.memory_allocated() / (1024**2)
        total = self._torch.cuda.get_device_properties(0).total_mem / (1024**2)
        return {
            "used_mb": round(used, 1),
            "total_mb": round(total, 1),
            "free_mb": round(total - used, 1),
        }

    def _load_pytorch(self, path: str, precision: str) -> object:
        if self._torch is None:
            raise ModelError("PyTorch not available")
        model = self._torch.load(path, map_location=self._device, weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
        if precision == "fp16" and hasattr(model, "half"):
            model = model.half()
        if hasattr(model, "to"):
            model = model.to(self._device)
        return model

    def _load_onnx(self, path: str) -> object:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModelError("onnxruntime not installed") from e

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        return ort.InferenceSession(path, providers=providers)

    def _load_tensorrt(self, path: str) -> object:
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ModelError("TensorRT not installed") from e

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise ModelError(f"Failed to deserialize TensorRT engine: {path}")
        return engine


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".pth": "pytorch",
        ".pt": "pytorch",
        ".onnx": "onnx",
        ".trt": "tensorrt",
        ".engine": "tensorrt",
    }
    return mapping.get(suffix, "onnx")


def _try_import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _cleanup_old_model(model: object | None, torch_module) -> None:
    if model is None:
        return
    del model
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
