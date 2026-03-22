# VOSMIC — Roadmap & Timeline
## Plan de Sprints para el Equipo de Desarrollo

> **Duración total estimada:** 8–10 semanas (2 Agentes Programadores en paralelo)  
> **Fecha de inicio de referencia:** Semana 1 post-aprobación

---

## Vista General de Sprints

```
Semana    1    2    3    4    5    6    7    8    9    10
          ├────┤
          │ F0 │ Foundation
          └──┬─┘
             ├────────┤
             │  F1    │ Audio I/O
             └──┬─────┘
                ├──────────────┤        ┌────────────┐
                │     F2       │        │    F5      │ ← PARALELO
                │     DSP      │        │    GUI     │
                └──┬───────────┘        └──────┬─────┘
                   ├──────────────────┤        │
                   │       F3         │        │
                   │    Inference     │        │
                   └──┬───────────────┘        │
                      ├────────┤               │
                      │  F4    │               │
                      │  Post  │               │
                      └──┬─────┘               │
                         └──────────┬──────────┘
                                    ├────────────┤
                                    │    F6      │
                                    │ QA & Final │
                                    └────────────┘
```

---

## Sprint 1 (Semana 1) — Foundation

| ID | Tarea | Agente | Días |
|----|-------|--------|------|
| F0-T1 | Inicializar repositorio & estructura | Prog A | 0.5 |
| F0-T2 | `pyproject.toml` + dependencias | Prog A | 0.5 |
| F0-T3 | Script CUDA diagnostic | Prog A | 0.5 |
| F0-T4 | Config loader (`core/config.py`) | Prog A | 1 |
| F0-T5 | CI/CD (GitHub Actions) | Prog A | 0.5 |
| — | **Testing F0** | Testing | 1 |

**Entregable:** Proyecto que compila, pasa lint, tests placeholder pasan, GPU detectada.  
**Gate de salida:** CI verde en GitHub Actions.

---

## Sprint 2 (Semanas 2–3) — Audio I/O

| ID | Tarea | Agente | Días |
|----|-------|--------|------|
| F1-T1 | Lock-Free Ring Buffer SPSC | Prog A | 2 |
| F1-T2 | Audio Capture Engine | Prog A | 1.5 |
| F1-T3 | Audio Output Engine | Prog A | 1.5 |
| F1-T4 | Device Enumeration | Prog A | 1 |
| F1-T5 | Passthrough Test & Benchmark | Prog A | 1 |
| — | **Testing F1** | Testing | 2 |

**Entregable:** Audio pasa del micro al cable virtual en passthrough con ≤ 12ms.  
**Gate de salida:** Benchmark passthrough documentado con métricas.

---

## Sprint 3 (Semanas 3–5) — DSP + GUI en Paralelo

### Track A: DSP (Agente Programador A)

| ID | Tarea | Días |
|----|-------|------|
| F2-T1 | Noise Gate | 1 |
| F2-T2 | Loudness Normalization | 0.5 |
| F2-T3 | Pitch Extractor (RMVPE) | 3 |
| F2-T4 | Content Encoder (ContentVec) | 2 |
| F2-T5 | FeatureBundle Assembler | 1 |
| F2-T6 | DSP Thread Manager | 1.5 |

### Track B: GUI (Agente Programador B) — EN PARALELO

| ID | Tarea | Días |
|----|-------|------|
| F5-T1 | Event Bus | 1 |
| F5-T2 | Main Window (PyQt6) | 2 |
| F5-T3 | Device Selector Widget | 1 |
| F5-T4 | Level Meters | 1.5 |
| F5-T5 | Performance Dashboard | 1.5 |
| F5-T6 | Model Loader UI | 1.5 |
| F5-T7 | System Tray | 1 |

**Entregable Track A:** Audio → FeatureBundle en ≤ 5ms con F0 y embeddings.  
**Entregable Track B:** GUI funcional con widgets de monitoreo (sin pipeline conectado aún).  
**Gate de salida:** DSP benchmark + GUI se abre sin crash.

---

## Sprint 4 (Semanas 5–7) — Inference Engine

| ID | Tarea | Agente | Días |
|----|-------|--------|------|
| F3-T1 | Model Manager (carga/swap) | Prog A | 2 |
| F3-T2 | Inference Core (PyTorch FP16) | Prog A | 3 |
| F3-T3 | ONNX Export + validación | Prog A | 1.5 |
| F3-T4 | TensorRT Optimization | Prog A | 2 |
| F3-T5 | Inference Scheduler (async) | Prog A | 1.5 |
| F3-T6 | Overlap-Add Stitcher | Prog A | 1 |
| F3-T7 | Passthrough Fallback | Prog A | 0.5 |
| — | **Testing F3** | Testing | 2 |

> **Prog B** continúa refinando GUI + conectando EventBus con módulos ya listos.

**Entregable:** Modelo RVC infiere en ≤ 20ms (PyTorch) / ≤ 10ms (TensorRT).  
**Gate de salida:** Benchmark de inferencia + hot-swap sin dropout.

---

## Sprint 5 (Semanas 7–8) — Postprocessing + Integración Parcial

| ID | Tarea | Agente | Días |
|----|-------|--------|------|
| F4-T1 | Gain Matcher | Prog A | 0.5 |
| F4-T2 | Dynamic Range Compressor | Prog A | 1.5 |
| F4-T3 | Sample Rate Converter | Prog A | 0.5 |
| F4-T4 | Anti-Aliasing Filter | Prog A | 0.5 |
| F4-T5 | PostProcessor Orchestrator | Prog A | 1 |
| — | **Integración parcial (sin GUI)** | Prog A | 2 |

**Entregable:** Pipeline completo micro → cable virtual funcional sin GUI.  
**Gate de salida:** Voz clonada audible a través de VB-Cable.

---

## Sprint 6 (Semanas 8–10) — Integration, QA & Polish

| ID | Tarea | Agente | Días |
|----|-------|--------|------|
| F6-T1 | Pipeline Orchestrator (con GUI) | Prog A + B | 2 |
| F6-T2 | E2E Latency Profiler | Prog A | 1 |
| F6-T3 | VRAM Stress Test (30 min) | Testing | 1 |
| F6-T4 | Compatibility RTX 4060 | Testing | 1 |
| F6-T5 | Compatibility GTX 1650 | Testing | 1 |
| F6-T6 | Error Recovery Tests (5 escenarios) | Testing | 2 |
| F6-T7 | Automated Regression Suite | Prog A | 2 |
| F6-T8 | User Acceptance Test (Discord 10 min) | Todos | 1 |

**Entregable:** Producto final con benchmarks documentados.  
**Gate de salida:** Todos los criterios de aceptación cumplidos.

---

## Milestones Clave

| Milestone | Semana | Criterio |
|-----------|--------|----------|
| 🟢 **M1: Build funcional** | 1 | CI verde, GPU detectada |
| 🟢 **M2: Audio passthrough** | 3 | Micro → VB-Cable ≤ 12ms |
| 🟡 **M3: First voice conversion** | 7 | Voz clonada audible (calidad TBD) |
| 🟡 **M4: Full pipeline** | 8 | E2E ≤ 35ms, GUI conectada |
| 🔴 **M5: Production ready** | 10 | Todos los tests pasan, UAT aprobado |

---

## Distribución de Agentes

```
        Semana 1   2   3   4   5   6   7   8   9   10
Prog A: [F0][──── F1 ────][──── F2/DSP ────][──── F3 ────][F4][── F6 ──]
Prog B:                   [────── F5/GUI ──────────────────────][── F6 ──]
Testing:    [T0]    [T1]       [T2]          [T3]      [T4] [── T6 ──]
```

**Nota:** Si solo hay **1 Agente Programador**, la Fase 5 (GUI) se pospone a después de F4, y el timeline se extiende a **12–14 semanas**.

---

## Decisiones Pendientes (Resolver en Sprint 1)

| Decisión | Opciones | Impacto | Cuándo |
|----------|----------|---------|--------|
| ¿CUDA 12.1 o 12.4? | Depende de drivers instalados | Fija versiones torch/onnxrt | Sprint 1 |
| ¿Modelo RVC base? | Necesario para testing F3+ | Sin modelo no hay benchmark | Sprint 1 |
| ¿WASAPI Shared o Exclusive? | Exclusive = menor latencia, más riesgo | Latencia base del sistema | Sprint 2 |
| ¿PyQt6 definitivo? | vs Electron | Impacta F5 completa | Sprint 3 |

---

## Riesgos del Timeline

| Riesgo | Impacto en Timeline | Mitigación |
|--------|-------------------|-----------|
| RMVPE tarda en integrar | +3 días en Sprint 3 | Empezar con CREPE-tiny, migrar después |
| TensorRT export falla | +2 días en Sprint 4 | Quedarse en ONNX (funcional, algo más lento) |
| GTX 1650 no cumple 40ms | +1 semana en Sprint 6 | Ajustar block_size, INT8 más agresivo |
| Solo 1 programador | +4–6 semanas total | GUI después de F4, no en paralelo |
| Modelo RVC no disponible | Bloquea Sprint 4+ | Descargar modelo open-source pre-entrenado ASAP |
