"""TensorRT optimization — converts ONNX model to TensorRT engine for minimum latency."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def optimize_to_tensorrt(
    onnx_path: str,
    trt_output_path: str,
    precision: str = "fp16",
    workspace_gb: int = 1,
) -> dict:
    """Build a TensorRT engine from an ONNX model.

    Returns a dict with build metrics. Requires tensorrt package.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("TensorRT not installed. Install with: pip install tensorrt") from e

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("TensorRT: FP16 mode enabled")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        logger.warning("INT8 requested but no calibrator set — accuracy may degrade")

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(trt_output_path, "wb") as f:
        f.write(engine_bytes)

    file_size_mb = Path(trt_output_path).stat().st_size / (1024**2)
    logger.info("TensorRT engine saved: %s (%.1f MB, %s)", trt_output_path, file_size_mb, precision)

    return {
        "trt_path": trt_output_path,
        "file_size_mb": round(file_size_mb, 2),
        "precision": precision,
    }
