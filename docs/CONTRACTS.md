# VOSMIC — Contratos de Interfaz

> Cada módulo expone interfaces estrictas. Los Agentes Programadores deben implementar exactamente estos contratos.

---

## Módulo 1: Audio I/O

### `LockFreeRingBuffer`
```
Constructor(capacity: int = 4096)

write(data: np.ndarray) → bool           # True si hay espacio
read(n_samples: int) → np.ndarray        # Lee n samples o array vacío
available_read() → int                   # Samples disponibles
available_write() → int                  # Espacio libre

RESTRICCIONES: Cero locks. Solo atómicos. Memoria pre-allocated.
```

### `AudioCapture`
```
Constructor(device_index, sample_rate=48000, block_size=256, channels=1)

start() → None
stop() → None
get_devices() → List[Dict]

OUTPUT vía ring buffer: audio_chunk (block_size,) float32 + timestamp
```

### `AudioOutput`
```
Constructor(device_index, sample_rate=48000, block_size=256)

start() → None
stop() → None
set_output_device(device_index: int) → None

INPUT desde ring buffer de salida: audio_chunk (block_size,) float32
```

---

## Módulo 2: DSP Preprocessing

### `DSPPreprocessor`
```
process(audio_chunk: np.ndarray) → FeatureBundle
set_noise_gate_threshold(db: float) → None
```

### `FeatureBundle` (dataclass)
```
audio: np.ndarray              # Audio normalizado
f0: np.ndarray                 # Contorno de pitch (Hz)
content_embedding: np.ndarray  # (T, 256)
is_silence: bool
timestamp: float
```

### `PitchExtractor`
```
extract(audio: np.ndarray, sample_rate=48000, hop_length=160) → np.ndarray  # (T,) Hz

RESTRICCIÓN: < 3ms para 256 samples
```

### `ContentEncoder`
```
encode(audio: np.ndarray) → np.ndarray  # (T, 256)

RESTRICCIÓN: Modelo cargado UNA VEZ al inicio
```

---

## Módulo 3: Inference

### `InferenceEngine`
```
infer(feature_bundle: FeatureBundle) → np.ndarray   # audio convertido
load_model(model_path: str, precision="fp16") → None
unload_model() → None
get_vram_usage() → Dict[str, float]

RESTRICCIÓN: Nunca bloquea audio thread. Si falla → retorna None (passthrough)
```

### `ModelManager`
```
load(path: str, format="onnx") → None
swap_model(new_path: str) → None          # Hot-swap double-buffer
get_info() → Dict
optimize(target="tensorrt") → None

RESTRICCIÓN: Swap usa double-buffer. Timeout 10s.
```

### `OverlapAddStitcher`
```
stitch(current_chunk, previous_tail, window="hanning", overlap=0.5) → np.ndarray
```

---

## Módulo 4: Postprocessing

### `PostProcessor`
```
process(converted_audio: np.ndarray, original_rms: float) → np.ndarray
set_compression(ratio, threshold, attack, release) → None
bypass(enabled: bool) → None
```

---

## Módulo 5: GUI

### `GUIController`
```
COMANDOS (→ pipeline):
  set_input_device(device_index: int) → None
  set_output_device(device_index: int) → None
  load_model(model_path: str) → None
  set_bypass(enabled: bool) → None
  set_noise_gate_threshold(db: float) → None

CALLBACKS (← pipeline):
  on_latency_update(callback: Callable[[float], None])
  on_vram_update(callback: Callable[[float, float], None])
  on_error(callback: Callable[[str], None])

COMUNICACIÓN: Vía EventBus (queue.Queue). GUI nunca llama audio directamente.
```
