# FASE 0 — FUNDACIÓN Y ENTORNO
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 1–2 días  
> **Dependencias:** Ninguna (es la primera fase)  
> **Output esperado:** Proyecto funcional con CI/CD, CUDA verificada, y config base

---

## 1. OBJETIVO DE ESTA FASE

Crear la infraestructura base del proyecto: repositorio limpio, dependencias resueltas, verificación de hardware GPU, configuración centralizada, y pipeline de CI/CD. Al finalizar esta fase, el proyecto debe compilar, pasar lint, y ejecutar un test placeholder.

**REGLA DE LATENCIA:** Esta fase no involucra audio real, pero sienta las bases: el `default.yaml` define los parámetros de buffer y sample rate que impactarán directamente la latencia en fases posteriores. Cualquier cambio aquí tiene efecto cascada.

---

## 2. ESTRUCTURA DE ARCHIVOS A CREAR

```
vosmic/
├── src/
│   ├── __init__.py
│   ├── audio_io/
│   │   └── __init__.py
│   ├── dsp/
│   │   └── __init__.py
│   ├── inference/
│   │   └── __init__.py
│   ├── postprocessing/
│   │   └── __init__.py
│   ├── gui/
│   │   └── __init__.py
│   └── core/
│       ├── __init__.py
│       └── config.py              ← Loader de configuración YAML
├── models/
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_config.py             ← Test del loader de config
│   └── test_cuda.py               ← Test de detección CUDA
├── configs/
│   └── default.yaml               ← Ya creado (verificar completitud)
├── scripts/
│   └── check_cuda.py              ← Script de verificación GPU
├── .github/
│   └── workflows/
│       └── ci.yml                 ← GitHub Actions
├── pyproject.toml
├── .gitignore
├── .ruff.toml                     ← Configuración de linter
└── README.md                      ← Ya creado
```

---

## 3. TAREA F0-T1: INICIALIZAR REPOSITORIO

### Instrucciones para el Agente Programador

**Archivo: `.gitignore`**
```
CONTENIDO REQUERIDO:
  - Patrones Python estándar: __pycache__/, *.pyc, *.pyo, .eggs/
  - Entornos virtuales: .venv/, venv/, env/
  - PyTorch/CUDA: *.pth, *.onnx, *.trt, *.engine (archivos de modelo pesados)
  - IDEs: .vscode/, .idea/, *.swp
  - OS: Thumbs.db, .DS_Store
  - Distribución: dist/, build/, *.egg-info/
  - Logs y temporales: *.log, /tmp/
  - NOTA: NO ignorar configs/*.yaml (deben versionarse)
  - NOTA: NO ignorar models/.gitkeep (para mantener la carpeta)
```

**Todos los `__init__.py`**
```
CONTENIDO:
  - Vacíos por ahora
  - Comentario docstring: """VOSMIC - [Nombre del módulo]"""
  - Ejemplo para audio_io: """VOSMIC - Audio I/O Module: Capture and output with lock-free ring buffers."""
```

### Criterio de Éxito (Testing)
- La estructura de carpetas existe exactamente como se especifica
- `git status` muestra todos los archivos trackeados correctamente
- Ningún archivo `.pth`, `.onnx`, `.trt` es trackeado por git

---

## 4. TAREA F0-T2: CREAR `pyproject.toml`

### Instrucciones para el Agente Programador

**Archivo: `pyproject.toml`**

```
SECCIONES REQUERIDAS:

[project]
  name = "vosmic"
  version = "0.1.0"
  description = "Real-time voice cloning with < 40ms latency"
  requires-python = ">=3.11"

[project.dependencies]  (CORE - requeridas siempre)
  - numpy >= 1.24
  - sounddevice >= 0.4.6          ← Audio I/O (PortAudio binding)
  - pyyaml >= 6.0                 ← Config loader
  - scipy >= 1.11                 ← Signal processing (SRC, filtros)

[project.optional-dependencies]
  gpu = [
    "torch >= 2.1",               ← PyTorch con CUDA
    "torchaudio >= 2.1",          ← Para operaciones de audio en GPU
    "onnxruntime-gpu >= 1.16",    ← Inferencia ONNX optimizada
  ]
  dev = [
    "pytest >= 7.4",
    "ruff >= 0.1",
    "mypy >= 1.7",
    "pytest-cov >= 4.1",
  ]

[build-system]
  requires = ["setuptools >= 68.0"]
  build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
  where = ["."]
  include = ["src*"]
```

### ADVERTENCIA: CONFLICTO DE DEPENDENCIAS
```
⚠️ CUELLO DE BOTELLA CONOCIDO:
  - PyTorch y onnxruntime-gpu DEBEN usar la MISMA versión de CUDA
  - Si CUDA 12.1: torch==2.1.x+cu121 y onnxruntime-gpu==1.16.x
  - Si CUDA 12.4: torch==2.4.x+cu124 y onnxruntime-gpu==1.19.x
  - El Agente Programador DEBE verificar la versión de CUDA instalada
    ANTES de fijar versiones en pyproject.toml
  - NUNCA mezclar CUDA 11.x y 12.x en el mismo entorno

⚠️ SOUNDDEVICE vs PYAUDIO:
  - Decisión: sounddevice (NO PyAudio)
  - Razón: PyAudio no tiene wheels actualizados para Python 3.11+
  - sounddevice usa PortAudio directamente y soporta WASAPI Exclusive
  - Si el Agente detecta PyAudio en algún punto → ALERTAR INMEDIATAMENTE
```

### Criterio de Éxito (Testing)
- `pip install -e ".[dev,gpu]"` completa sin errores
- `python -c "import sounddevice; print(sounddevice.__version__)"` funciona
- `python -c "import torch; print(torch.cuda.is_available())"` retorna `True`
- No hay conflictos de dependencias reportados por pip

---

## 5. TAREA F0-T3: CONFIGURACIÓN CUDA

### Instrucciones para el Agente Programador

**Archivo: `scripts/check_cuda.py`**

```
PSEUDOCÓDIGO:

FUNCIÓN main():
    IMPRIMIR "=== VOSMIC GPU Diagnostic ==="
    
    # 1. Verificar que torch está instalado
    INTENTAR importar torch
    SI falla:
        IMPRIMIR "ERROR: PyTorch no instalado"
        SALIR código 1
    
    # 2. Verificar CUDA disponible
    SI torch.cuda.is_available() es False:
        IMPRIMIR "ERROR: CUDA no disponible"
        IMPRIMIR "  - Verificar drivers NVIDIA instalados"
        IMPRIMIR "  - Verificar que PyTorch tiene soporte CUDA"
        SALIR código 1
    
    # 3. Información del dispositivo
    PARA CADA gpu EN range(torch.cuda.device_count()):
        propiedades = torch.cuda.get_device_properties(gpu)
        IMPRIMIR "GPU {gpu}: {propiedades.name}"
        IMPRIMIR "  - VRAM Total: {propiedades.total_mem / 1024**3:.1f} GB"
        IMPRIMIR "  - VRAM Libre: {torch.cuda.mem_get_info(gpu)[0] / 1024**3:.1f} GB"
        IMPRIMIR "  - Compute Capability: {propiedades.major}.{propiedades.minor}"
        IMPRIMIR "  - Multiprocessors: {propiedades.multi_processor_count}"
    
    # 4. Verificar versión CUDA
    IMPRIMIR "CUDA Version: {torch.version.cuda}"
    IMPRIMIR "cuDNN Version: {torch.backends.cudnn.version()}"
    IMPRIMIR "PyTorch Version: {torch.__version__}"
    
    # 5. Benchmark rápido de VRAM
    IMPRIMIR "=== Quick VRAM Benchmark ==="
    INTENTAR:
        tensor_test = torch.randn(1024, 1024, device='cuda')
        tiempo_inicio = time.time()
        PARA i EN range(100):
            resultado = torch.matmul(tensor_test, tensor_test)
        torch.cuda.synchronize()
        tiempo = time.time() - tiempo_inicio
        IMPRIMIR "1024x1024 MatMul x100: {tiempo*1000:.1f} ms"
        LIBERAR tensor_test, resultado
    CAPTURAR error:
        IMPRIMIR "WARNING: Benchmark falló: {error}"
    
    # 6. Verificar compatibilidad con el proyecto
    SI propiedades.total_mem < 4 * 1024**3:
        IMPRIMIR "⚠️ WARNING: VRAM < 4GB. Rendimiento limitado."
        IMPRIMIR "   Recomendado: usar INT8 quantization"
    SI propiedades.total_mem >= 8 * 1024**3:
        IMPRIMIR "✅ GPU compatible para calidad máxima (FP16)"
    
    IMPRIMIR "=== Diagnostic Complete ==="
    SALIR código 0
```

### Criterio de Éxito (Testing)
- El script ejecuta sin error en máquina con GPU NVIDIA
- Imprime nombre de GPU, VRAM total/libre, versión CUDA
- Retorna exit code 0 en sistema compatible
- Retorna exit code 1 en sistema sin GPU con mensaje claro

---

## 6. TAREA F0-T4: LOADER DE CONFIGURACIÓN

### Instrucciones para el Agente Programador

**Archivo: `src/core/config.py`**

```
PSEUDOCÓDIGO:

IMPORTAR yaml, pathlib, dataclasses

@dataclass
CLASE AudioConfig:
    sample_rate: int = 48000
    block_size: int = 256
    channels: int = 1
    input_device: int | None = None      ← None = auto-detect
    output_device: int | None = None
    ring_buffer_capacity: int = 4096

@dataclass
CLASE DSPConfig:
    noise_gate_threshold_db: float = -40.0
    loudness_target_lufs: float = -18.0
    pitch_extractor: str = "rmvpe"
    pitch_hop_length: int = 160
    content_encoder: str = "contentvec"
    content_encoder_sr: int = 16000

@dataclass
CLASE InferenceConfig:
    model_path: str = "models/narrator_v1.onnx"
    format: str = "onnx"
    precision: str = "fp16"
    max_inference_time_ms: int = 25
    queue_max_size: int = 4

@dataclass
CLASE CompressorConfig:
    ratio: float = 3.0
    threshold_db: float = -12.0
    attack_ms: float = 5.0
    release_ms: float = 50.0

@dataclass
CLASE PostProcessConfig:
    compressor: CompressorConfig
    filter_type: str = "butterworth"
    filter_order: int = 4
    filter_cutoff_hz: int = 20000
    bypass: bool = False

@dataclass
CLASE SystemConfig:
    log_level: str = "INFO"
    profile_mode: bool = False
    passthrough_on_error: bool = True     ← CRÍTICO: nunca silencio

@dataclass
CLASE VosmicConfig:
    audio: AudioConfig
    dsp: DSPConfig
    inference: InferenceConfig
    postprocessing: PostProcessConfig
    system: SystemConfig

FUNCIÓN load_config(path: str = "configs/default.yaml") → VosmicConfig:
    """
    Carga configuración desde YAML y la valida.
    Si el archivo no existe → usar valores por defecto.
    Si un campo falta → usar valor por defecto del dataclass.
    """
    SI el archivo existe:
        LEER yaml
        PARSEAR a dataclasses con defaults
    SINO:
        IMPRIMIR WARNING "Config no encontrada, usando defaults"
        RETORNAR VosmicConfig con todos los defaults
    
    # VALIDACIÓN CRÍTICA:
    ASSERT config.audio.sample_rate EN [44100, 48000, 96000]
    ASSERT config.audio.block_size EN [64, 128, 256, 512, 1024]
    ASSERT config.audio.block_size es potencia de 2
    ASSERT config.inference.precision EN ["fp32", "fp16", "int8"]
    ASSERT config.system.passthrough_on_error es True
        ← FORZADO: esta configuración NO es opcional
    
    RETORNAR config

FUNCIÓN save_config(config: VosmicConfig, path: str) → None:
    """Serializa config a YAML para persistencia."""
    ESCRIBIR config como YAML al path
```

### Consideraciones de Latencia
```
⚠️ IMPACTO EN LATENCIA:
  - block_size = 256 @ 48kHz = 5.33ms por callback
  - block_size = 128 @ 48kHz = 2.67ms por callback (más CPU, menos latencia)
  - block_size = 512 @ 48kHz = 10.67ms por callback (menos CPU, más latencia)
  
  El default de 256 es el balance óptimo para RTX 4060.
  Para GTX 1650, considerar block_size = 512 para dar más tiempo a la GPU.
  
  ring_buffer_capacity = 4096 = ~85ms de audio a 48kHz
  Esto da margen para 4 chunks de inferencia en cola.
```

### Criterio de Éxito (Testing)
- `load_config()` carga correctamente `configs/default.yaml`
- Valores por defecto se aplican cuando faltan campos
- Validación rechaza valores inválidos (ej. block_size=300)
- `passthrough_on_error` siempre es `True` (no se puede desactivar)
- Round-trip: `save_config(load_config(path), path2)` → el contenido es equivalente

---

## 7. TAREA F0-T5: CI/CD CON GITHUB ACTIONS

### Instrucciones para el Agente Programador

**Archivo: `.github/workflows/ci.yml`**

```
PSEUDOCÓDIGO DEL WORKFLOW:

NOMBRE: "VOSMIC CI"
TRIGGER: push a main, pull requests

JOBS:
  lint:
    RUNNER: ubuntu-latest
    PASOS:
      1. Checkout del código
      2. Setup Python 3.11
      3. pip install ruff
      4. ruff check src/
      5. ruff format --check src/

  typecheck:
    RUNNER: ubuntu-latest
    PASOS:
      1. Checkout del código
      2. Setup Python 3.11
      3. pip install -e ".[dev]"
      4. mypy src/ --ignore-missing-imports

  test:
    RUNNER: ubuntu-latest
    PASOS:
      1. Checkout del código
      2. Setup Python 3.11
      3. pip install -e ".[dev]"
      4. pytest tests/ -v --tb=short
      
  # NOTA: No se incluye test de GPU en CI
  # Los tests de GPU se ejecutan localmente por el Agente de Testing
```

**Archivo: `.ruff.toml`**

```
CONTENIDO:
  line-length = 100
  target-version = "py311"
  
  [lint]
  select = ["E", "F", "W", "I", "N", "UP"]
  ignore = ["E501"]    ← No forzar líneas cortas en docstrings
  
  [format]
  quote-style = "double"
  indent-style = "space"
```

### Criterio de Éxito (Testing)
- Push a `main` dispara el workflow en GitHub Actions
- El workflow pasa: lint, typecheck, tests
- Ruff reporta 0 errores sobre el código actual
- El pipeline tarda < 2 minutos

---

## 8. RESUMEN DE ENTREGABLES DE ESTA FASE

| Entregable | Archivo | Validación |
|-----------|---------|-----------|
| Estructura de carpetas | Todo `src/`, `tests/`, `scripts/`, `configs/` | Existe y está completa |
| Dependencias | `pyproject.toml` | `pip install` sin errores |
| GPU Diagnostic | `scripts/check_cuda.py` | Imprime info GPU correcta |
| Config Loader | `src/core/config.py` | Tests pasan, validación funciona |
| CI/CD | `.github/workflows/ci.yml` | Actions verde en GitHub |
| Lint | `.ruff.toml` | `ruff check` sin errores |
| Git setup | `.gitignore` | Modelos no trackeados |

---

## 9. RIESGOS Y MITIGACIONES

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| CUDA version mismatch (torch vs onnxruntime) | No compila | Verificar `nvidia-smi` ANTES de fijar versiones |
| Python < 3.11 | Type hints fallan | Forzar `requires-python >= 3.11` |
| sounddevice sin PortAudio | No audio | En Windows, PortAudio viene bundled. En Linux: `apt install libportaudio2` |
| `.gitignore` mal configurado | Modelos de 500MB en el repo | Verificar con `git ls-files` que no hay .pth/.onnx |
