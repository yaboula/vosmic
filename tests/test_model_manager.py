"""Tests for ModelManager — load, swap, unload, error handling."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.inference.model_manager import (
    ModelManager,
    ModelSwapError,
    UnsupportedFormatError,
)


class TestModelManagerInit:
    def test_initial_state(self) -> None:
        mm = ModelManager(device="cpu")
        assert mm.is_loaded is False
        assert mm.active_model is None
        assert mm.model_info == {}

    def test_vram_usage_without_torch(self) -> None:
        mm = ModelManager(device="cpu")
        usage = mm.get_vram_usage()
        assert "used_mb" in usage
        assert "total_mb" in usage


class TestLoadErrors:
    def test_file_not_found(self) -> None:
        mm = ModelManager(device="cpu")
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            mm.load("/nonexistent/model.onnx")

    def test_unsupported_format(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        mm = ModelManager(device="cpu")
        with pytest.raises(UnsupportedFormatError):
            mm.load(tmp, format="unsupported_format")
        Path(tmp).unlink()


class TestFormatDetection:
    def test_detect_pytorch(self) -> None:
        from src.inference.model_manager import _detect_format

        assert _detect_format(Path("model.pth")) == "pytorch"
        assert _detect_format(Path("model.pt")) == "pytorch"

    def test_detect_onnx(self) -> None:
        from src.inference.model_manager import _detect_format

        assert _detect_format(Path("model.onnx")) == "onnx"

    def test_detect_tensorrt(self) -> None:
        from src.inference.model_manager import _detect_format

        assert _detect_format(Path("model.trt")) == "tensorrt"
        assert _detect_format(Path("model.engine")) == "tensorrt"

    def test_unknown_defaults_to_onnx(self) -> None:
        from src.inference.model_manager import _detect_format

        assert _detect_format(Path("model.bin")) == "onnx"


class TestUnload:
    def test_unload_clears_state(self) -> None:
        mm = ModelManager(device="cpu")
        mm._active_model = "fake_model"
        mm._model_info = {"name": "test"}
        mm._format = "onnx"

        mm.unload()
        assert mm.is_loaded is False
        assert mm.active_model is None
        assert mm.model_info == {}
        assert mm.format == ""


class TestSwapModel:
    def test_swap_preserves_old_on_failure(self) -> None:
        mm = ModelManager(device="cpu")
        mm._active_model = "old_model"
        mm._model_info = {"name": "old"}
        mm._format = "onnx"

        with pytest.raises(ModelSwapError, match="previous model preserved"):
            mm.swap_model("/nonexistent/new_model.onnx")

        assert mm.active_model == "old_model"
        assert mm.model_info["name"] == "old"
