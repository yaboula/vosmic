# PRIMER PROMPT — Ejecución Completa del Proyecto VOSMIC
## Copiar y pegar esto como primer mensaje al Agente Programador

---

```
Eres el Agente Programador asignado al proyecto VOSMIC (Real-Time Voice Changer).
Tu contexto está definido en .agents/rules/contexto_dev.md — léelo primero.

## TU MISIÓN

Implementar el proyecto VOSMIC completo, fase por fase, siguiendo la documentación
arquitectónica existente en docs/phases/. El proyecto ya tiene repositorio Git
configurado y toda la documentación de diseño lista.

## ORDEN DE EJECUCIÓN

Ejecuta las fases en este orden estricto. NO saltes fases.
Cada fase tiene su documento detallado en docs/phases/.

### FASE 0 — Foundation (Prioridad: INMEDIATA)
Lee: docs/phases/FASE_00_FOUNDATION.md

Tareas:
1. F0-T1: Crear .gitignore y estructura de carpetas con __init__.py
2. F0-T2: Crear pyproject.toml con dependencias (verificar CUDA version primero)
3. F0-T3: Implementar scripts/check_cuda.py
4. F0-T4: Implementar src/core/config.py (dataclasses + loader YAML)
5. F0-T5: Crear .github/workflows/ci.yml + .ruff.toml

Verificación: pytest pasa, ruff limpio, check_cuda imprime GPU info
Commit: "feat(core): initialize project structure and config"
Push: git push origin main

### FASE 1 — Audio I/O (Prioridad: CRÍTICA)
Lee: docs/phases/FASE_01_AUDIO_IO.md

Tareas:
1. F1-T1: Implementar LockFreeRingBuffer en src/audio_io/ring_buffer.py
   → Escribir TODOS los tests de test_ring_buffer.py primero (TDD)
2. F1-T2: Implementar AudioCapture en src/audio_io/capture.py
3. F1-T3: Implementar AudioOutput en src/audio_io/output.py
4. F1-T4: Implementar device enumeration en src/audio_io/devices.py
5. F1-T5: Implementar AudioPassthrough + benchmark

Verificación: Passthrough micro→VB-Cable ≤ 12ms, 0 overflows en 60s
Commit por cada tarea completada
Push: git push origin main

### FASE 2 — DSP Preprocessing (Prioridad: ALTA)
Lee: docs/phases/FASE_02_DSP.md

Tareas:
1. F2-T1: NoiseGate en src/dsp/noise_gate.py
2. F2-T2: LoudnessNormalizer en src/dsp/normalizer.py
3. F2-T3: PitchExtractor (RMVPE) en src/dsp/pitch_extractor.py
   → Si RMVPE no disponible, implementar CREPE-tiny como fallback
4. F2-T4: ContentEncoder en src/dsp/content_encoder.py
5. F2-T5: FeatureBundle dataclass + assembler
6. F2-T6: DSPThread en src/dsp/dsp_thread.py

Verificación: FeatureBundle producido en ≤ 5ms, F0 ≤ 2Hz error
Commit por tarea, push al final de la fase

### FASE 3 — Inference Engine (Prioridad: MÁXIMA)
Lee: docs/phases/FASE_03_INFERENCE.md

Tareas:
1. F3-T1: ModelManager con hot-swap double-buffer
2. F3-T2: InferenceEngine (PyTorch FP16 primero)
3. F3-T3: Script ONNX export + validación (diff ≤ 1e-3)
4. F3-T4: TensorRT optimization (si TensorRT disponible)
5. F3-T5: InferenceScheduler (cola FIFO async)
6. F3-T6: OverlapAddStitcher
7. F3-T7: Passthrough fallback

Verificación: Inferencia ≤ 20ms PyTorch / ≤ 10ms TRT en RTX 4060
⚠️ Si inferencia falla → PASSTHROUGH (audio original), NUNCA silencio

### FASE 4 — Postprocessing (Prioridad: MEDIA)
Lee: docs/phases/FASE_04_POSTPROCESSING.md

Tareas:
1. F4-T1: GainMatcher
2. F4-T2: DynamicCompressor (usar numba @njit para performance)
3. F4-T3: SampleRateConverter
4. F4-T4: AntiAliasingFilter (Butterworth sosfilt)
5. F4-T5: PostProcessor orchestrator con bypass

Verificación: Pipeline completo ≤ 2ms, gain match ≤ 1dB diff

### FASE 5 — GUI (Prioridad: MEDIA)
Lee: docs/phases/FASE_05_GUI.md

Tareas:
1. F5-T1: EventBus (queue.Queue thread-safe)
2. F5-T2: MainWindow layout (PyQt6)
3. F5-T3: DeviceSelector dropdown
4. F5-T4: LevelMeter animado
5. F5-T5: PerformancePanel (latencia, GPU, VRAM, buffer)
6. F5-T6: ModelLoader file dialog
7. F5-T7: SystemTray

⚠️ GUI NUNCA llama funciones de audio directamente → solo vía EventBus

### FASE 6 — Integration & QA (Prioridad: FINAL)
Lee: docs/phases/FASE_06_INTEGRATION_QA.md

Tareas:
1. F6-T1: VosmicPipeline orchestrator (conecta todo)
2. F6-T2: E2E latency benchmark
3. F6-T3: VRAM stress test (30 min)
4. F6-T4 + T5: GPU compatibility (RTX 4060 + GTX 1650)
5. F6-T6: Error recovery (5 escenarios)
6. F6-T7: Automated regression suite
7. F6-T8: User acceptance test

Criterio final: E2E ≤ 35ms RTX 4060, ≤ 40ms GTX 1650, 0 dropouts

## INSTRUCCIONES GENERALES

1. LEE el documento de fase ANTES de implementar cada tarea
2. SIGUE el pseudocódigo del documento — no inventes tu propia arquitectura
3. RESPETA los contratos de interfaz de docs/CONTRACTS.md
4. ESCRIBE tests ANTES o JUNTO con la implementación
5. MIDE latencia después de cada módulo de audio/DSP/inferencia
6. HAZ commit + push después de cada tarea verificada
7. SI algo falla o bloquea → reporta ANTES de intentar workarounds
8. NUNCA modifiques docs/phases/ — esos son tus especificaciones

## HARDWARE OBJETIVO
- Principal: NVIDIA RTX 4060 (8GB VRAM) → FP16, calidad máxima
- Compatible: NVIDIA GTX 1650 (4GB VRAM) → INT8, rendimiento

## OBJETIVO FINAL
Que un usuario pueda: abrir VOSMIC → seleccionar su micrófono →
cargar un modelo de voz → hablar en Discord con la voz clonada →
con latencia imperceptible (< 40ms).

Empieza por Fase 0. Lee docs/phases/FASE_00_FOUNDATION.md ahora.
```
