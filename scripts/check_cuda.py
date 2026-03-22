"""VOSMIC GPU Diagnostic — Verifies CUDA availability and GPU compatibility."""

from __future__ import annotations

import sys
import time


def main() -> int:
    print("=== VOSMIC GPU Diagnostic ===\n")

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.")
        print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        print("  - Verify NVIDIA drivers are installed (nvidia-smi)")
        print("  - Verify PyTorch was installed with CUDA support")
        print("  - Current PyTorch build:", torch.__version__)
        return 1

    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)\n")

    last_gpu_mem = 0
    for gpu in range(device_count):
        props = torch.cuda.get_device_properties(gpu)
        free_mem, total_mem = torch.cuda.mem_get_info(gpu)
        last_gpu_mem = props.total_mem

        print(f"GPU {gpu}: {props.name}")
        print(f"  VRAM Total:          {total_mem / 1024**3:.1f} GB")
        print(f"  VRAM Free:           {free_mem / 1024**3:.1f} GB")
        print(f"  Compute Capability:  {props.major}.{props.minor}")
        print(f"  Multiprocessors:     {props.multi_processor_count}")
        print()

    print(f"CUDA Version:    {torch.version.cuda}")
    cudnn_ver = torch.backends.cudnn.version()
    print(f"cuDNN Version:   {cudnn_ver}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    print("=== Quick VRAM Benchmark ===")
    try:
        tensor_test = torch.randn(1024, 1024, device="cuda")
        t0 = time.perf_counter()
        for _ in range(100):
            result = torch.matmul(tensor_test, tensor_test)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"1024x1024 MatMul x100: {elapsed * 1000:.1f} ms")
        del tensor_test, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"WARNING: Benchmark failed: {e}")

    print()
    if last_gpu_mem < 4 * 1024**3:
        print("WARNING: VRAM < 4 GB. Performance will be limited.")
        print("  Recommended: use INT8 quantization")
    elif last_gpu_mem >= 8 * 1024**3:
        print("GPU compatible for maximum quality (FP16)")
    else:
        print("GPU compatible (FP16 with conservative VRAM management)")

    print("\n=== Diagnostic Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
