# 🎙️ VOSMIC — Real-Time Voice Changer

> Software de clonación de voz en tiempo real con latencia < 40ms, optimizado para NVIDIA GPUs.

## ⚡ Requisitos

- **Python** 3.11+
- **NVIDIA GPU**: RTX 4060 (calidad máxima) / GTX 1650 (rendimiento)
- **CUDA** 12.x + cuDNN
- **VB-Audio Virtual Cable** instalado

## 📁 Estructura del Proyecto

```
vosmic/
├── src/
│   ├── audio_io/          # Captura y salida de audio
│   ├── dsp/               # Preprocesamiento DSP
│   ├── inference/          # Motor de inferencia RVC
│   ├── postprocessing/     # Postprocesamiento y salida
│   ├── gui/               # Interfaz gráfica (PyQt6)
│   └── core/              # Orquestador del pipeline
├── models/                # Modelos .pth / .onnx / .trt
├── tests/                 # Tests unitarios + integración
├── configs/               # Configuración YAML
├── docs/                  # Documentación del proyecto
└── scripts/               # Build, export, profiling
```

## 📄 Documentación

| Documento | Descripción |
|-----------|------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Visión arquitectónica y diagrama de bloques |
| [PHASES.md](docs/PHASES.md) | Mapa de fases y dependencias |
| [TASK_REGISTRY.md](docs/TASK_REGISTRY.md) | Registro maestro de tareas (40+) |
| [CONTRACTS.md](docs/CONTRACTS.md) | Contratos de interfaz por módulo |
| [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) | Guía de pruebas y criterios de aceptación |

## 🎯 Objetivo de Latencia

| Etapa | Presupuesto |
|-------|-------------|
| Captura de audio | ≤ 5 ms |
| DSP (pitch + features) | ≤ 3 ms |
| Inferencia RVC (GPU) | ≤ 20 ms |
| Postprocesamiento | ≤ 2 ms |
| Salida a cable virtual | ≤ 5 ms |
| **Total** | **≤ 35 ms** |
