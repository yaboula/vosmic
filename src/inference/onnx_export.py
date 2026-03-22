"""ONNX export — converts PyTorch RVC model to ONNX with equivalence validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MAX_DIFF_THRESHOLD = 1e-3


def export_to_onnx(
    pytorch_model_path: str,
    onnx_output_path: str,
    sample_length: int = 100,
    opset_version: int = 17,
) -> dict:
    """Export a PyTorch RVC model to ONNX with dynamic axes and validate equivalence.

    Returns a dict with export metrics (file size, max diff).
    Requires torch and onnxruntime.
    """
    import torch

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime required for ONNX validation") from e

    model = torch.load(pytorch_model_path, map_location="cuda", weights_only=False)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "cuda"):
        model = model.cuda()

    dummy_content = torch.randn(1, sample_length, 256, device="cuda")
    dummy_f0 = torch.randn(1, sample_length, device="cuda")

    torch.onnx.export(
        model,
        (dummy_content, dummy_f0),
        onnx_output_path,
        input_names=["content", "f0"],
        output_names=["audio"],
        dynamic_axes={
            "content": {1: "time"},
            "f0": {1: "time"},
            "audio": {1: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    onnx_session = ort.InferenceSession(onnx_output_path, providers=providers)

    with torch.no_grad():
        output_pt = model(dummy_content, dummy_f0).cpu().numpy()

    output_onnx = onnx_session.run(
        None,
        {
            "content": dummy_content.cpu().numpy(),
            "f0": dummy_f0.cpu().numpy(),
        },
    )[0]

    max_diff = float(np.max(np.abs(output_pt - output_onnx)))
    file_size_mb = Path(onnx_output_path).stat().st_size / (1024**2)

    result = {
        "onnx_path": onnx_output_path,
        "max_diff": max_diff,
        "file_size_mb": round(file_size_mb, 2),
        "valid": max_diff < MAX_DIFF_THRESHOLD,
    }

    if max_diff >= MAX_DIFF_THRESHOLD:
        logger.error(
            "ONNX export diff too large: %.6f (threshold: %.6f)", max_diff, MAX_DIFF_THRESHOLD
        )
    else:
        logger.info("ONNX export OK: max_diff=%.6f, size=%.1fMB", max_diff, file_size_mb)

    return result
