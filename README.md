# 🎙️ VOSMIC — Real-Time Voice Changer

> Software de clonación de voz en tiempo real con latencia < 40ms, optimizado para NVIDIA GPUs.

## ⚡ Quickstart

```powershell
# 1. Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -e ".[dev]"          # solo tests/GUI
pip install -e ".[gpu,dev]"      # + GPU (torch + onnxruntime-gpu)

# 3. Lanzar
python -m src.main
```

> Ver [INSTALL.md](docs/INSTALL.md) para la guía completa de instalación.

---

## 🖥️ Requisitos

- **Python** 3.11+
- **Windows** 10/11 (WASAPI para baja latencia)
- **NVIDIA GPU**: RTX 4060 (calidad máxima) / GTX 1650 (rendimiento) — *opcional*
- **CUDA** 12.x + cuDNN — *solo si usas GPU*
- **VB-Audio Virtual Cable** — para enrutar el audio a otras apps

---

## 📁 Estructura del Proyecto

```
vosmic/
├── src/
│   ├── main.py            # ← Punto de entrada de la aplicación
│   ├── audio_io/          # Captura y salida de audio (sounddevice)
│   ├── dsp/               # Noise gate, pitch, content encoder
│   ├── inference/         # Motor RVC (ONNX / PyTorch / TensorRT)
│   ├── postprocessing/    # Compresor, convertidor de SR, anti-alias
│   ├── gui/               # Interfaz gráfica PyQt6 + EventBus
│   └── core/              # Orquestador (VosmicPipeline) + config
├── models/                # Modelos .pth / .onnx / .trt
├── tests/                 # Tests unitarios + integración
├── configs/               # Configuración YAML
├── docs/                  # Documentación del proyecto
└── scripts/               # Benchmark, export, profiling
```

---

## 🎯 Objetivo de Latencia

| Etapa | Presupuesto |
|-------|-------------|
| Captura de audio | ≤ 5 ms |
| DSP (pitch + features) | ≤ 3 ms |
| Inferencia RVC (GPU) | ≤ 20 ms |
| Postprocesamiento | ≤ 2 ms |
| Salida a cable virtual | ≤ 5 ms |
| **Total** | **≤ 35 ms** |

---

## 📄 Documentación

| Documento | Descripción |
|-----------|-------------|
| [INSTALL.md](docs/INSTALL.md) | Guía completa de instalación y configuración |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Visión arquitectónica y diagrama de bloques |
| [PHASES.md](docs/PHASES.md) | Mapa de fases y dependencias |
| [CONTRACTS.md](docs/CONTRACTS.md) | Contratos de interfaz por módulo |
| [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) | Guía de pruebas y criterios de aceptación |
| [TASK_REGISTRY.md](docs/TASK_REGISTRY.md) | Registro maestro de tareas (40+) |

---

## 🧪 Tests

```powershell
# Tests sin GPU (CI-safe)
pytest tests/ -v -m "not gpu"

# Todos los tests
pytest tests/ -v

# Regresión completa
python scripts/run_regression.py
```

---

## 📝 Notas de arquitectura

- **EventBus compartido**: la GUI y el pipeline usan la misma instancia. La GUI envía comandos (`send_command`) y el pipeline emite eventos (`emit_threadsafe`) → nunca se llaman funciones de audio desde el hilo de Qt.
- **Modo passthrough**: sin modelo cargado, el audio pasa sin conversión. Sin interrupciones.
- **Cierre a bandeja**: cerrar la ventana minimiza a la bandeja del sistema. El pipeline sigue activo. Salir desde **clic derecho → Quit**.
