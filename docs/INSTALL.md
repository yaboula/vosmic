# Guía de instalación — VOSMIC

## Requisitos previos

| Requisito | Versión mínima | Notas |
|-----------|---------------|-------|
| **Windows** | 10 / 11 | WASAPI para audio de baja latencia |
| **Python** | 3.11+ | Recomendado: descargar de [python.org](https://python.org) |
| **NVIDIA GPU** | GTX 1650 / RTX 4060 | Opcional — sin GPU corre en passthrough |
| **CUDA Toolkit** | 12.x | Solo si usas GPU — [Descargar](https://developer.nvidia.com/cuda-downloads) |
| **cuDNN** | Compatible con tu CUDA | [Descargar](https://developer.nvidia.com/cudnn) |
| **VB-Audio Virtual Cable** | Cualquiera | [Descargar](https://vb-audio.com/Cable/) — para enrutamiento de audio |

---

## 1. Clonar el repositorio

```powershell
git clone <url-del-repo>
cd vosmic
```

---

## 2. Crear y activar el entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **IMPORTANTE:** Todos los comandos siguientes asumen que el entorno virtual está activo.  
> Si el prompt no empieza con `(.venv)`, primero ejecuta `.\.venv\Scripts\Activate.ps1`.

---

## 3. Instalar dependencias

### Opción A — Solo tests, sin GPU ni GUI

```powershell
pip install -e ".[dev]"
```

### Opción B — GUI + audio (sin GPU)

```powershell
pip install -e ".[dev]"
```

La GUI (PyQt6) y el audio (sounddevice) están incluidos en las dependencias base.

### Opción C — Completo (GPU + GUI + tests)

```powershell
pip install -e ".[gpu,dev]"
```

Instala `torch`, `torchaudio` y `onnxruntime-gpu`. Asegúrate de tener CUDA 12.x instalado.

---

## 4. Verificar la instalación

```powershell
# Ejecutar todos los tests (sin hardware de audio ni GPU)
pytest tests/ -v -m "not gpu"

# Verificar PyQt6, numpy, sounddevice, etc.
python -c "import PyQt6, numpy, sounddevice, yaml; print('OK')"

# Verificar CUDA (solo si instalaste la opción GPU)
python scripts/check_cuda.py
```

Resultado esperado en los tests:

```
============ XX passed in X.XXs ============
```

---

## 5. Configurar VB-Audio Virtual Cable

1. Descargar e instalar [VB-Cable](https://vb-audio.com/Cable/)
2. Ir a: **Configuración → Sistema → Sonido → Avanzadas → Dispositivos de grabación**
3. VB-Cable Input aparece como micrófono virtual
4. En VOSMIC: seleccionar **VB-Cable Input** como salida de audio
5. En Discord / Zoom / OBS: seleccionar **VB-Cable Output** como micrófono

---

## 6. Colocar un modelo de voz

Coloca un modelo RVC exportado en la carpeta `models/`:

```
models/
└── narrator_v1.onnx      # .onnx (recomendado), .pth o .trt
```

Edita `configs/default.yaml` para apuntar a tu modelo:

```yaml
inference:
  model_path: "models/tu_modelo.onnx"
  format: "onnx"         # "pytorch", "onnx", "tensorrt"
  precision: "fp16"
```

> Si no hay modelo, el pipeline corre en modo **passthrough** (audio sin conversión).

---

## 7. Lanzar la aplicación

```powershell
# Con el venv activo:
python -m src.main

# O si instalaste el paquete con console_scripts:
vosmic
```

Al arrancar verás:
- **Ventana principal** con selector de dispositivos, panel de rendimiento y controles bypass
- **Icono en la bandeja del sistema** — clic derecho para mostrar/ocultar, bypass, y salir
- **Pipeline de audio** iniciado en hilos en segundo plano (logs en consola)

> Cerrar la ventana **minimiza a la bandeja** (el pipeline sigue activo).  
> Para salir completamente: **clic derecho en el icono → Quit**.

---

## 8. Tabla de comandos rápida

| Objetivo | Comando |
|----------|---------|
| Activar venv | `.\.venv\Scripts\Activate.ps1` |
| Instalar base | `pip install -e .` |
| + Tests/dev | `pip install -e ".[dev]"` |
| + GPU | `pip install -e ".[gpu,dev]"` |
| Lanzar app | `python -m src.main` |
| Tests unitarios | `pytest tests/ -v -m "not gpu"` |
| Tests con GPU | `pytest tests/ -v` |
| Regresión completa | `python scripts/run_regression.py` |
| Benchmark E2E | `python scripts/benchmark_e2e.py --duration 30` |
| Stress VRAM | `python scripts/stress_test_vram.py --minutes 5` |
| Verificar CUDA | `python scripts/check_cuda.py` |
