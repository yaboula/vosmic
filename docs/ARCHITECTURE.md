# VOSMIC — Arquitectura del Sistema

## Diagrama de Bloques

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  MICRÓFONO   │───▸│  AUDIO I/O    │───▸│  PREPROCESAMIENTO│───▸│  INFERENCIA RVC   │───▸│  POSTPROCESAMIENTO│
│  (Hardware)  │    │  (Ring Buffer) │    │  DSP             │    │  (GPU - Async)    │    │  & SALIDA         │
└──────────────┘    └───────────────┘    └──────────────────┘    └───────────────────┘    └──────────────────┘
                           ▲                                                                       │
                           │                    ┌──────────────────┐                                │
                           └────────────────────│  GUI / DASHBOARD │◀───────────────────────────────┘
                                                └──────────────────┘
                                                        │
                                                ┌───────▼──────────┐
                                                │  CABLE VIRTUAL   │
                                                │  (VB-Cable/WASAPI)│
                                                └──────────────────┘
```

## Principios de Diseño

1. **Audio Thread es Sagrado** — Cero allocaciones, cero locks, cero I/O de disco en el callback de audio.
2. **Inferencia Asíncrona** — GPU en hilo separado. El audio thread nunca espera.
3. **Fail-Safe Passthrough** — Si GPU tarda demasiado → audio original (nunca silencio).
4. **Modular y Desacoplado** — Cada módulo con interfaz clara. Fallo aislado, degradación graceful.
5. **VRAM-Conscious** — Modelo cargado una vez. Buffers reutilizados. FP16/INT8.

## Presupuesto de Latencia (< 40 ms)

| Etapa | Presupuesto |
|-------|-------------|
| Captura (buffer input) | ≤ 5 ms |
| Preprocesamiento DSP | ≤ 3 ms |
| Inferencia RVC (GPU) | ≤ 20 ms |
| Postprocesamiento | ≤ 2 ms |
| Salida (cable virtual) | ≤ 5 ms |
| **TOTAL** | **≤ 35 ms** (5 ms margen) |

## Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| Lenguaje | Python 3.11+ |
| Audio | sounddevice (PortAudio) |
| IA | PyTorch → ONNX → TensorRT |
| GUI | PyQt6 |
| Cable Virtual | VB-Audio Virtual Cable |
