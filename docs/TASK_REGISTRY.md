# VOSMIC — Registro Maestro de Tareas

> Referencia rápida para Agentes Programadores y de Testing. Cada tarea tiene un ID único, dependencias claras y criterios de aceptación.

## Fase 0 — Foundation

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F0-T1 | Inicializar repositorio | — | ⬜ |
| F0-T2 | Crear `pyproject.toml` | F0-T1 | ⬜ |
| F0-T3 | Configuración CUDA | F0-T2 | ⬜ |
| F0-T4 | Archivo de config base | F0-T1 | ⬜ |
| F0-T5 | CI/CD básico | F0-T2 | ⬜ |

## Fase 1 — Audio I/O

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F1-T1 | Lock-Free Ring Buffer | F0 | ⬜ |
| F1-T2 | Audio Capture Engine | F1-T1 | ⬜ |
| F1-T3 | Audio Output Engine | F1-T1 | ⬜ |
| F1-T4 | Device Enumeration | F1-T2 | ⬜ |
| F1-T5 | Passthrough Test | F1-T2, F1-T3 | ⬜ |

## Fase 2 — DSP Preprocessing

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F2-T1 | Noise Gate | F1 | ⬜ |
| F2-T2 | Loudness Normalization | F2-T1 | ⬜ |
| F2-T3 | Pitch Extractor (RMVPE) | F0-T3 | ⬜ |
| F2-T4 | Content Encoder (ContentVec) | F0-T3 | ⬜ |
| F2-T5 | FeatureBundle Assembler | F2-T1 a F2-T4 | ⬜ |
| F2-T6 | DSP Thread Manager | F1-T1, F2-T5 | ⬜ |

## Fase 3 — RVC Inference

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F3-T1 | Model Manager | F0-T3 | ⬜ |
| F3-T2 | Inference Core (PyTorch) | F3-T1, F2-T5 | ⬜ |
| F3-T3 | ONNX Export | F3-T2 | ⬜ |
| F3-T4 | TensorRT Optimization | F3-T3 | ⬜ |
| F3-T5 | Inference Scheduler | F3-T2 | ⬜ |
| F3-T6 | Overlap-Add Stitcher | F3-T2 | ⬜ |
| F3-T7 | Passthrough Fallback | F3-T5 | ⬜ |

## Fase 4 — Postprocessing

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F4-T1 | Gain Matcher | F3-T2 | ⬜ |
| F4-T2 | Dynamic Range Compressor | F4-T1 | ⬜ |
| F4-T3 | Sample Rate Converter | F4-T1 | ⬜ |
| F4-T4 | Anti-Aliasing Filter | F4-T3 | ⬜ |
| F4-T5 | PostProcessor Orchestrator | F4-T1 a F4-T4 | ⬜ |

## Fase 5 — GUI (en paralelo con F2-F4)

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F5-T1 | Event Bus | F0 | ⬜ |
| F5-T2 | Main Window (PyQt6) | F5-T1 | ⬜ |
| F5-T3 | Device Selector Widget | F5-T2, F1-T4 | ⬜ |
| F5-T4 | Level Meters | F5-T2 | ⬜ |
| F5-T5 | Performance Dashboard | F5-T2 | ⬜ |
| F5-T6 | Model Loader UI | F5-T2, F3-T1 | ⬜ |
| F5-T7 | System Tray | F5-T2 | ⬜ |

## Fase 6 — Integration & QA

| ID | Tarea | Dependencia | Estado |
|----|-------|-------------|--------|
| F6-T1 | Pipeline Orchestrator | F1–F5 | ⬜ |
| F6-T2 | E2E Latency Profiler | F6-T1 | ⬜ |
| F6-T3 | VRAM Stress Test | F6-T1 | ⬜ |
| F6-T4 | Compat RTX 4060 | F6-T1 | ⬜ |
| F6-T5 | Compat GTX 1650 | F6-T1 | ⬜ |
| F6-T6 | Error Recovery Test | F6-T1 | ⬜ |
| F6-T7 | Regression Suite | F6-T1 a F6-T6 | ⬜ |
| F6-T8 | User Acceptance Test | F6-T1 | ⬜ |

---

**Leyenda:** ⬜ Pendiente · 🔄 En Progreso · ✅ Completada · ❌ Bloqueada
