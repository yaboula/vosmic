"""VOSMIC - Inference Module: RVC model loading, ONNX/TensorRT execution, and scheduling."""

from src.inference.inference_core import InferenceEngine
from src.inference.model_manager import ModelManager, ModelSwapError, UnsupportedFormatError
from src.inference.passthrough import PassthroughFallback
from src.inference.scheduler import InferenceResult, InferenceScheduler
from src.inference.stitcher import OverlapAddStitcher

__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "InferenceScheduler",
    "ModelManager",
    "ModelSwapError",
    "OverlapAddStitcher",
    "PassthroughFallback",
    "UnsupportedFormatError",
]
