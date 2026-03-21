# VOSMIC — Fases del Proyecto

## Mapa de Dependencias

```
                    ┌──────────┐
                    │  Fase 0  │  Foundation
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  Fase 1  │  Audio I/O
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │                     │
         ┌────▼─────┐         ┌────▼─────┐
         │  Fase 2  │         │  Fase 5  │  GUI (paralelo)
         │  DSP     │         │          │
         └────┬─────┘         └────┬─────┘
              │                     │
         ┌────▼─────┐              │
         │  Fase 3  │              │
         │ Inference│              │
         └────┬─────┘              │
              │                     │
         ┌────▼─────┐              │
         │  Fase 4  │              │
         │  Post    │              │
         └────┬─────┘              │
              │                     │
              └──────────┬──────────┘
                    ┌────▼─────┐
                    │  Fase 6  │  Integration & QA
                    └──────────┘
```

> **Fase 5 (GUI)** se desarrolla en paralelo con Fases 2-4.

## Resumen por Fase

| Fase | Nombre | Módulo | Tareas |
|------|--------|--------|--------|
| 0 | Foundation | `core/` | Repo, deps, CUDA, config, CI |
| 1 | Audio I/O | `audio_io/` | Ring buffer, capture, output, devices |
| 2 | DSP | `dsp/` | Noise gate, normalization, pitch, embeddings |
| 3 | Inference | `inference/` | Model manager, RVC core, ONNX/TRT, scheduler |
| 4 | Postprocessing | `postprocessing/` | Gain, compressor, SRC, filter |
| 5 | GUI | `gui/` | PyQt6, dashboard, meters, model loader |
| 6 | Integration & QA | — | E2E profiling, stress tests, regression |

Para la documentación detallada de cada fase (flujos de datos, contratos de interfaz, tareas y criterios de aceptación), consultar `docs/phases/` o el `implementation_plan.md` del proyecto.
