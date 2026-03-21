# VOSMIC — Guía de Pruebas para Agentes de Testing

> Cada fase tiene métricas concretas. El Agente de Testing debe verificar TODOS los umbrales.

---

## Fase 0 — Foundation

| Verificación | Comando / Acción | Criterio de Éxito |
|-------------|------------------|-------------------|
| Tests ejecutan | `pytest tests/` | 0 errores |
| CUDA detectada | `python scripts/check_cuda.py` | GPU, CUDA ver, VRAM |
| Lint limpio | `ruff check src/` | 0 warnings |
| Estructura correcta | Comparar con spec | Match exacto |

## Fase 1 — Audio I/O

| Métrica | Umbral |
|---------|--------|
| Ring Buffer correctness | 100% (write == read) |
| Ring Buffer overflow | 0 crashes |
| Passthrough latency | ≤ 12 ms |
| Callback duration | ≤ 1 ms |
| Device hot-swap | 0 crashes, audio < 500 ms |

## Fase 2 — DSP

| Métrica | Umbral |
|---------|--------|
| Noise gate accuracy | ≥ 98% |
| Pitch extraction time | ≤ 3 ms (RTX 4060) |
| Pitch accuracy | ≤ 2 Hz error (sine wave) |
| ContentVec shape | Siempre (T, 256) |
| FeatureBundle fields | 100% no-None con voz |
| DSP total time | ≤ 5 ms (RTX 4060) |

## Fase 3 — Inference

| Métrica | RTX 4060 | GTX 1650 |
|---------|----------|----------|
| PyTorch FP16 latency | ≤ 20 ms | N/A |
| ONNX latency | ≤ 15 ms | ≤ 25 ms |
| TensorRT latency | ≤ 10 ms | ≤ 18 ms |
| ONNX vs PyTorch diff | ≤ 1e-3 | ≤ 1e-3 |
| TRT vs ONNX diff | ≤ 1e-2 | ≤ 1e-2 |
| Model hot-swap | ≤ 2s, 0 dropout | ≤ 3s |
| VRAM FP16 | ≤ 2 GB | N/A |
| VRAM INT8 | N/A | ≤ 1.2 GB |
| Passthrough on overload | 100% | 100% |

## Fase 4 — Postprocessing

| Métrica | Umbral |
|---------|--------|
| Gain match (RMS diff) | ≤ 1 dB |
| Compressor attack accuracy | ± 1 ms |
| SRC quality (THD+N) | ≤ -80 dB |
| Filter cutoff (-3dB) | 20 kHz ± 500 Hz |
| Total post time | ≤ 2 ms |
| Bypass bit-perfect | 100% |

## Fase 5 — GUI

| Métrica | Umbral |
|---------|--------|
| No freeze on model load | GUI responde < 100 ms |
| Level meter accuracy | ≤ 1 dB |
| Latency display accuracy | ± 2 ms |
| Device switch | Audio funcional < 500 ms |
| Model swap via GUI | 0 dropouts |
| GUI crash isolation | Pipeline sigue funcionando |

## Fase 6 — Integration & QA (Criterios Finales)

| Métrica | RTX 4060 | GTX 1650 |
|---------|----------|----------|
| **E2E latency** | **≤ 35 ms** | **≤ 40 ms** |
| VRAM peak | ≤ 2 GB | ≤ 3.5 GB |
| VRAM leak (30 min) | ≤ 50 MB | ≤ 50 MB |
| CPU usage | ≤ 15% | ≤ 25% |
| PESQ score | ≥ 3.5 | ≥ 3.0 |
| Audio dropouts (10 min) | 0 | ≤ 2 |
| Error recovery | ≤ 2s | ≤ 3s |
| Cold start | ≤ 8s | ≤ 12s |
