# FASE 1 — MÓDULO DE AUDIO I/O
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 3–5 días  
> **Dependencias:** Fase 0 completada (repo, deps, config)  
> **Output esperado:** Captura de audio y salida a cable virtual con latencia ≤ 12ms en passthrough

---

## 1. OBJETIVO DE ESTA FASE

Construir el subsistema de tiempo real más crítico del proyecto. Este módulo captura audio desde el micrófono, lo deposita en un ring buffer lock-free, y reproduce audio procesado a través del cable virtual. **Este es el módulo donde la latencia se gana o se pierde.**

### Regla de Oro del Audio Thread
```
╔══════════════════════════════════════════════════════════════╗
║  DENTRO DEL CALLBACK DE AUDIO ESTÁ PROHIBIDO:                ║
║  ❌ Allocar memoria (malloc, new, append, list comprehension) ║
║  ❌ Usar locks (mutex, semaphore, threading.Lock)             ║
║  ❌ Hacer I/O de disco (open, read, write, log)               ║
║  ❌ Llamar a funciones de red (socket, HTTP)                  ║
║  ❌ Usar print() o logging                                    ║
║  ❌ Acceder a variables compartidas sin atomicidad            ║
║                                                               ║
║  ✅ PERMITIDO: memcpy, operaciones atómicas, aritmética       ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 2. ESTRUCTURA DE ARCHIVOS A CREAR

```
src/audio_io/
├── __init__.py
├── ring_buffer.py          ← F1-T1: Lock-free ring buffer SPSC
├── capture.py              ← F1-T2: Motor de captura de audio
├── output.py               ← F1-T3: Motor de salida de audio
├── devices.py              ← F1-T4: Enumeración de dispositivos
└── passthrough.py          ← F1-T5: Modo passthrough para testing

tests/
├── test_ring_buffer.py     ← Tests del ring buffer
├── test_capture.py         ← Tests de captura
├── test_output.py          ← Tests de salida
└── test_passthrough.py     ← Test de latencia end-to-end
```

---

## 3. TAREA F1-T1: LOCK-FREE RING BUFFER

### Visión Técnica
El ring buffer es la **pieza central de comunicación** entre el hilo de audio (que escribe a velocidad de tiempo real) y el hilo de procesamiento (que lee a la velocidad que pueda). Debe ser **absolutamente libre de locks** — un solo lock aquí destruye la garantía de latencia.

### Archivo: `src/audio_io/ring_buffer.py`

```
PSEUDOCÓDIGO DETALLADO:

IMPORTAR numpy as np
IMPORTAR threading (solo para atómicos, NUNCA para locks)

CLASE LockFreeRingBuffer:
    """
    Buffer circular SPSC (Single Producer, Single Consumer) lock-free.
    - Un solo hilo escribe (audio capture callback)
    - Un solo hilo lee (DSP thread)
    - Cero locks: usa punteros atómicos
    - Memoria pre-allocated: NUNCA alloca durante operación
    """
    
    CONSTRUCTOR(capacity: int = 4096):
        # Redondear al siguiente potencia de 2 (para modular con AND)
        self._capacity = siguiente_potencia_de_2(capacity)
        self._mask = self._capacity - 1    ← Para wrap-around con AND bitwise
        
        # Buffer pre-allocated UNA VEZ
        self._buffer = np.zeros(self._capacity, dtype=np.float32)
        
        # Punteros atómicos (int, no necesitan lock en CPython por GIL)
        # NOTA: En producción real, usar ctypes.c_long para atomicidad
        self._write_pos = 0    ← Posición del próximo write
        self._read_pos = 0     ← Posición del próximo read
    
    PROPIEDAD available_read → int:
        """Cuántos samples disponibles para leer."""
        RETORNAR (self._write_pos - self._read_pos)
        # Esto funciona incluso con wrap-around porque usamos enteros
        # que crecen monotónicamente
    
    PROPIEDAD available_write → int:
        """Cuánto espacio libre para escribir."""
        RETORNAR self._capacity - self.available_read
    
    MÉTODO write(data: np.ndarray) → bool:
        """
        Escribe datos al buffer.
        RETORNA True si escribió todo, False si no hay espacio.
        NUNCA bloquea.
        """
        n = len(data)
        
        SI n > self.available_write:
            RETORNAR False    ← Overflow: el caller decide qué hacer
        
        # Calcular posición real en el buffer circular
        start = self._write_pos & self._mask
        end = start + n
        
        SI end <= self._capacity:
            # Caso simple: no cruza el borde
            self._buffer[start:end] = data
        SINO:
            # Caso wrap-around: dividir en dos copias
            first_part = self._capacity - start
            self._buffer[start:] = data[:first_part]
            self._buffer[:n - first_part] = data[first_part:]
        
        # Actualizar write pointer DESPUÉS de escribir datos
        # (el orden es crítico para consistencia SPSC)
        self._write_pos += n
        RETORNAR True
    
    MÉTODO read(n_samples: int) → np.ndarray:
        """
        Lee n_samples del buffer.
        RETORNA array con los datos, o array vacío si no hay suficientes.
        NUNCA bloquea.
        """
        available = self.available_read
        
        SI available == 0:
            RETORNAR np.array([], dtype=np.float32)
        
        # Leer lo que hay (hasta n_samples)
        to_read = min(n_samples, available)
        
        start = self._read_pos & self._mask
        end = start + to_read
        
        # Pre-allocated output array (para evitar allocation en hot path)
        out = np.empty(to_read, dtype=np.float32)
        
        SI end <= self._capacity:
            out[:] = self._buffer[start:end]
        SINO:
            first_part = self._capacity - start
            out[:first_part] = self._buffer[start:]
            out[first_part:] = self._buffer[:to_read - first_part]
        
        # Actualizar read pointer DESPUÉS de leer datos
        self._read_pos += to_read
        RETORNAR out
    
    MÉTODO peek(n_samples: int) → np.ndarray:
        """Lee sin avanzar el puntero (para debugging/monitoring)."""
        # Igual que read() pero SIN modificar self._read_pos
    
    MÉTODO clear():
        """Vacía el buffer (solo llamar cuando NO hay audio activo)."""
        self._read_pos = self._write_pos
    
    MÉTODO __len__() → int:
        RETORNAR self.available_read

FUNCIÓN siguiente_potencia_de_2(n: int) → int:
    """Redondea N al siguiente potencia de 2."""
    # Ejemplo: 4096 → 4096, 3000 → 4096, 5000 → 8192
    p = 1
    MIENTRAS p < n:
        p <<= 1
    RETORNAR p
```

### Edge Cases que el Agente DEBE Manejar

| Caso | Comportamiento Esperado |
|------|------------------------|
| Escribir más datos que capacidad | `write()` retorna `False`, no escribe nada |
| Leer cuando buffer vacío | `read()` retorna array vacío `[]` |
| Escribir 0 samples | `write()` retorna `True`, no hace nada |
| Múltiples writes antes de un read | Datos se acumulan correctamente |
| Buffer exactamente lleno | `available_write == 0`, `available_read == capacity` |
| Wrap-around en el borde | Los datos se leen en orden correcto |

### Tests Requeridos — `tests/test_ring_buffer.py`

```
TESTS OBLIGATORIOS:

TEST test_write_and_read_basic:
    buffer = LockFreeRingBuffer(1024)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ASSERT buffer.write(data) == True
    ASSERT buffer.available_read == 4
    result = buffer.read(4)
    ASSERT np.array_equal(result, data)
    ASSERT buffer.available_read == 0

TEST test_overflow_returns_false:
    buffer = LockFreeRingBuffer(4)    ← Se redondea a 4
    data = np.ones(5, dtype=np.float32)
    ASSERT buffer.write(data) == False
    ASSERT buffer.available_read == 0    ← Nada fue escrito

TEST test_empty_read_returns_empty:
    buffer = LockFreeRingBuffer(1024)
    result = buffer.read(100)
    ASSERT len(result) == 0

TEST test_wrap_around_correctness:
    buffer = LockFreeRingBuffer(8)
    # Escribir 6, leer 6, escribir 6 más (fuerza wrap-around)
    buffer.write(np.arange(6, dtype=np.float32))
    buffer.read(6)
    new_data = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    buffer.write(new_data)
    result = buffer.read(6)
    ASSERT np.array_equal(result, new_data)

TEST test_concurrent_write_read:
    """Simular SPSC con dos hilos."""
    buffer = LockFreeRingBuffer(4096)
    errors = []
    
    HILO producer:
        PARA i EN range(1000):
            chunk = np.full(256, i, dtype=np.float32)
            MIENTRAS NOT buffer.write(chunk):
                time.sleep(0.0001)    ← Spin-wait (solo en test)
    
    HILO consumer:
        total_read = 0
        MIENTRAS total_read < 256000:
            data = buffer.read(256)
            SI len(data) > 0:
                total_read += len(data)
    
    EJECUTAR ambos hilos
    ASSERT len(errors) == 0

TEST test_performance_no_allocation:
    """Verificar que read/write no allocan memoria nueva."""
    buffer = LockFreeRingBuffer(4096)
    data = np.ones(256, dtype=np.float32)
    
    # Warm up
    buffer.write(data)
    buffer.read(256)
    
    # Medir allocations
    import tracemalloc
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    PARA _ EN range(10000):
        buffer.write(data)
        buffer.read(256)
    
    snapshot2 = tracemalloc.take_snapshot()
    diff = snapshot2.compare_to(snapshot1, 'lineno')
    total_leaked = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
    
    ASSERT total_leaked < 1024    ← Menos de 1KB de allocation nueva
```

### Criterio de Éxito (Testing)
- Todos los tests pasan
- Benchmark de 10K write/read ciclos: < 1ms total (0.1μs por operación)
- Zero memory allocations en hot path (verificar con tracemalloc)
- Thread safety: test concurrente con 1M total samples sin error

---

## 4. TAREA F1-T2: AUDIO CAPTURE ENGINE

### Archivo: `src/audio_io/capture.py`

```
PSEUDOCÓDIGO DETALLADO:

IMPORTAR sounddevice as sd
IMPORTAR numpy as np
IMPORTAR time

CLASE AudioCapture:
    """
    Captura audio del micrófono y lo deposita en un ring buffer.
    El callback de audio es minimal: SOLO memcpy al buffer.
    """
    
    CONSTRUCTOR(
        ring_buffer: LockFreeRingBuffer,
        device_index: int | None = None,
        sample_rate: int = 48000,
        block_size: int = 256,
        channels: int = 1
    ):
        self._ring_buffer = ring_buffer
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._channels = channels
        self._stream = None
        self._is_running = False
        
        # Métricas (actualizadas desde el callback, leídas desde GUI thread)
        self._last_callback_time_ms = 0.0
        self._callback_count = 0
        self._overflow_count = 0
        self._peak_level = 0.0
    
    MÉTODO _audio_callback(indata, frames, time_info, status):
        """
        ⚠️ CALLBACK DE AUDIO — HILO DE TIEMPO REAL
        REGLAS:
          - NO allocar memoria
          - NO usar locks
          - NO hacer I/O
          - NO llamar print/logging
          - SOLO: copiar datos al ring buffer
        """
        SI status:
            # Solo marcar flag, NO imprimir
            self._overflow_count += 1
        
        # Copiar audio mono al ring buffer
        audio_mono = indata[:, 0]    ← Tomar primer canal (ya es float32)
        
        # Medir tiempo del callback (para dashboard)
        t_start = time.perf_counter_ns()
        
        success = self._ring_buffer.write(audio_mono)
        
        SI NOT success:
            self._overflow_count += 1
        
        # Calcular peak level (para meters de la GUI)
        self._peak_level = float(np.max(np.abs(audio_mono)))
        
        t_end = time.perf_counter_ns()
        self._last_callback_time_ms = (t_end - t_start) / 1_000_000
        self._callback_count += 1
    
    MÉTODO start():
        """Iniciar captura de audio."""
        SI self._is_running:
            RETORNAR
        
        INTENTAR:
            self._stream = sd.InputStream(
                device=self._device_index,
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                channels=self._channels,
                dtype='float32',
                callback=self._audio_callback,
                latency='low'              ← Pedir latencia mínima al driver
            )
            self._stream.start()
            self._is_running = True
        CAPTURAR sd.PortAudioError como error:
            LANZAR AudioDeviceError(f"No se pudo abrir micrófono: {error}")
    
    MÉTODO stop():
        """Detener captura."""
        SI self._stream Y self._is_running:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._is_running = False
    
    PROPIEDAD metrics → dict:
        """Métricas para el dashboard de la GUI."""
        RETORNAR {
            "callback_time_ms": self._last_callback_time_ms,
            "callback_count": self._callback_count,
            "overflow_count": self._overflow_count,
            "peak_level": self._peak_level,
            "is_running": self._is_running,
            "buffer_available": self._ring_buffer.available_read,
        }
    
    MÉTODO change_device(device_index: int):
        """
        Cambiar dispositivo de entrada en caliente.
        Estrategia: stop → reconfigurar → start (gap < 200ms)
        """
        was_running = self._is_running
        SI was_running:
            self.stop()
        self._device_index = device_index
        SI was_running:
            self.start()
    
    MÉTODO __del__():
        self.stop()

CLASE AudioDeviceError(Exception):
    """Error de dispositivo de audio."""
    PASAR
```

### Consideraciones de Latencia

```
⚠️ CONFIGURACIÓN DE LATENCIA DE SOUNDDEVICE:

El parámetro `latency='low'` le pide a PortAudio/WASAPI la menor latencia posible.
En Windows con WASAPI Shared:
  - Latencia típica: 10-20ms
  - blocksize=256 @ 48kHz → 5.3ms teóricos

Para menor latencia (si se necesita):
  - Usar WASAPI Exclusive: sd.InputStream(..., extra_settings=sd.WasapiSettings(exclusive=True))
  - Usar ASIO (requiere ASIO4ALL o driver del hardware)
  
El Agente Programador DEBE exponer la opción de backend WASAPI en la configuración.

⚠️ TIMING DEL CALLBACK:
  - El callback DEBE completar en < 1ms
  - Si tarda más → audio glitches (underruns/overruns)
  - Medir con time.perf_counter_ns() (resolución de nanosegundos)
  - numpy.max + memcpy de 256 floats ≈ 0.01ms → muy dentro del presupuesto
```

### Criterio de Éxito (Testing)
- Captura audio de micrófono real durante 5 segundos sin error
- `callback_time_ms` promedio ≤ 0.5 ms, max ≤ 1.0 ms
- `overflow_count` == 0 durante operación normal de 60 segundos
- `peak_level` refleja nivel de audio real (> 0 cuando se habla)
- `change_device()` completa en < 500ms y audio sigue funcional

---

## 5. TAREA F1-T3: AUDIO OUTPUT ENGINE

### Archivo: `src/audio_io/output.py`

```
PSEUDOCÓDIGO DETALLADO:

CLASE AudioOutput:
    """
    Reproduce audio procesado a través del cable virtual (VB-Cable).
    Lee del ring buffer de salida en un callback de tiempo real.
    """
    
    CONSTRUCTOR(
        ring_buffer: LockFreeRingBuffer,
        device_index: int | None = None,
        sample_rate: int = 48000,
        block_size: int = 256,
        channels: int = 1
    ):
        self._ring_buffer = ring_buffer
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._channels = channels
        self._stream = None
        self._is_running = False
        
        # Buffer de silencio pre-allocated (para cuando no hay datos)
        self._silence = np.zeros(block_size, dtype=np.float32)
        
        # Métricas
        self._underrun_count = 0
        self._last_output_level = 0.0
    
    MÉTODO _output_callback(outdata, frames, time_info, status):
        """
        ⚠️ CALLBACK DE AUDIO — HILO DE TIEMPO REAL
        Mismas reglas que el input callback.
        """
        SI status:
            self._underrun_count += 1
        
        # Intentar leer audio procesado del ring buffer
        audio = self._ring_buffer.read(frames)
        
        SI len(audio) == frames:
            # Caso normal: hay suficiente audio procesado
            outdata[:, 0] = audio
        ELIF len(audio) > 0:
            # Caso parcial: hay algo pero no suficiente
            outdata[:len(audio), 0] = audio
            outdata[len(audio):, 0] = 0.0    ← Zero-pad el resto
            self._underrun_count += 1
        SINO:
            # Caso vacío: no hay audio procesado → silencio
            outdata[:, 0] = 0.0
            self._underrun_count += 1
        
        # Medir peak output level
        self._last_output_level = float(np.max(np.abs(outdata)))
    
    MÉTODO start():
        """Iniciar reproducción en cable virtual."""
        INTENTAR:
            self._stream = sd.OutputStream(
                device=self._device_index,
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                channels=self._channels,
                dtype='float32',
                callback=self._output_callback,
                latency='low'
            )
            self._stream.start()
            self._is_running = True
        CAPTURAR sd.PortAudioError como error:
            LANZAR AudioDeviceError(f"No se pudo abrir salida: {error}")
    
    MÉTODO stop():
        SI self._stream Y self._is_running:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._is_running = False
    
    MÉTODO set_output_device(device_index: int):
        """Hot-swap del dispositivo de salida."""
        was_running = self._is_running
        SI was_running:
            self.stop()
        self._device_index = device_index
        SI was_running:
            self.start()
    
    PROPIEDAD metrics → dict:
        RETORNAR {
            "underrun_count": self._underrun_count,
            "output_level": self._last_output_level,
            "is_running": self._is_running,
        }
```

### Criterio de Éxito (Testing)
- Reproduce audio en VB-Cable sin artefactos durante 60 segundos
- `underrun_count` == 0 durante operación normal con datos disponibles
- Output level es 0.0 cuando no hay datos (silencio limpio, no ruido)
- `set_output_device()` cambia sin crash en < 500ms

---

## 6. TAREA F1-T4: ENUMERACIÓN DE DISPOSITIVOS

### Archivo: `src/audio_io/devices.py`

```
PSEUDOCÓDIGO DETALLADO:

FUNCIÓN get_input_devices() → List[Dict]:
    """
    Retorna lista de dispositivos de entrada (micrófonos) disponibles.
    """
    devices = sd.query_devices()
    inputs = []
    
    PARA idx, dev EN enumerate(devices):
        SI dev['max_input_channels'] > 0:
            inputs.append({
                "index": idx,
                "name": dev['name'],
                "channels": dev['max_input_channels'],
                "sample_rate": dev['default_samplerate'],
                "latency_low": dev['default_low_input_latency'],
                "latency_high": dev['default_high_input_latency'],
                "host_api": sd.query_hostapis(dev['hostapi'])['name'],
                "is_default": idx == sd.default.device[0],
            })
    
    RETORNAR inputs

FUNCIÓN get_output_devices() → List[Dict]:
    """
    Retorna lista de dispositivos de salida disponibles.
    Incluye cables virtuales (VB-Cable, etc.)
    """
    devices = sd.query_devices()
    outputs = []
    
    PARA idx, dev EN enumerate(devices):
        SI dev['max_output_channels'] > 0:
            outputs.append({
                "index": idx,
                "name": dev['name'],
                "channels": dev['max_output_channels'],
                "sample_rate": dev['default_samplerate'],
                "latency_low": dev['default_low_output_latency'],
                "latency_high": dev['default_high_output_latency'],
                "host_api": sd.query_hostapis(dev['hostapi'])['name'],
                "is_virtual_cable": "virtual" EN dev['name'].lower() 
                                   O "vb-" EN dev['name'].lower()
                                   O "cable" EN dev['name'].lower(),
                "is_default": idx == sd.default.device[1],
            })
    
    RETORNAR outputs

FUNCIÓN find_virtual_cable() → int | None:
    """
    Busca automáticamente el cable virtual (VB-Cable Input).
    Retorna el device index o None si no se encuentra.
    """
    outputs = get_output_devices()
    PARA dev EN outputs:
        SI dev['is_virtual_cable']:
            RETORNAR dev['index']
    RETORNAR None

FUNCIÓN validate_device(device_index: int, direction: str, sample_rate: int) → bool:
    """
    Verifica que un dispositivo soporta el sample rate deseado.
    direction: "input" o "output"
    """
    INTENTAR:
        sd.check_input_settings(device=device_index, samplerate=sample_rate)
        RETORNAR True
    CAPTURAR:
        RETORNAR False
```

### Criterio de Éxito (Testing)
- `get_input_devices()` lista al menos 1 dispositivo
- `get_output_devices()` lista dispositivos incluyendo VB-Cable (si instalado)
- `find_virtual_cable()` retorna index correcto de VB-Cable
- `validate_device()` retorna False para sample rates no soportados
- Información consistente con la que muestra el Panel de Control de Windows

---

## 7. TAREA F1-T5: MODO PASSTHROUGH (TEST DE LATENCIA)

### Archivo: `src/audio_io/passthrough.py`

```
PSEUDOCÓDIGO DETALLADO:

CLASE AudioPassthrough:
    """
    Modo de prueba: micro → ring buffer → ring buffer de salida → cable virtual.
    SIN procesamiento DSP ni inferencia.
    Propósito: medir la latencia BASE del subsistema de audio.
    """
    
    CONSTRUCTOR(config: AudioConfig):
        self._config = config
        self._input_buffer = LockFreeRingBuffer(config.ring_buffer_capacity)
        self._output_buffer = LockFreeRingBuffer(config.ring_buffer_capacity)
        self._capture = AudioCapture(self._input_buffer, ...)
        self._output = AudioOutput(self._output_buffer, ...)
        self._is_running = False
        self._passthrough_thread = None
        
        # Métricas de latencia
        self._latency_samples = []
    
    MÉTODO _passthrough_loop():
        """
        Hilo que copia datos del input buffer al output buffer.
        Simula el pipeline completo pero sin procesamiento.
        """
        MIENTRAS self._is_running:
            # Leer lo que haya disponible
            available = self._input_buffer.available_read
            SI available >= self._config.block_size:
                data = self._input_buffer.read(self._config.block_size)
                self._output_buffer.write(data)
            SINO:
                # Espera activa muy corta (no bloquea el hilo de audio)
                time.sleep(0.0001)    ← 0.1ms sleep
    
    MÉTODO start():
        self._capture.start()
        self._output.start()
        self._is_running = True
        self._passthrough_thread = Thread(target=self._passthrough_loop, daemon=True)
        self._passthrough_thread.start()
    
    MÉTODO stop():
        self._is_running = False
        SI self._passthrough_thread:
            self._passthrough_thread.join(timeout=2)
        self._capture.stop()
        self._output.stop()
    
    MÉTODO measure_latency() → Dict:
        """
        Retorna métricas de latencia del passthrough.
        """
        RETORNAR {
            "input_callback_ms": self._capture.metrics["callback_time_ms"],
            "theoretical_latency_ms": (self._config.block_size / self._config.sample_rate) * 1000 * 2,
            "input_overflow": self._capture.metrics["overflow_count"],
            "output_underrun": self._output.metrics["underrun_count"],
        }

FUNCIÓN run_passthrough_benchmark(duration_seconds: int = 10) → Dict:
    """
    Script standalone para medir latencia del subsistema de audio.
    """
    config = load_config()
    passthrough = AudioPassthrough(config.audio)
    
    IMPRIMIR "=== VOSMIC Audio I/O Benchmark ==="
    IMPRIMIR f"Sample Rate: {config.audio.sample_rate} Hz"
    IMPRIMIR f"Block Size: {config.audio.block_size} samples"
    IMPRIMIR f"Theoretical Latency: {(config.audio.block_size/config.audio.sample_rate)*1000*2:.1f} ms"
    IMPRIMIR f"Running for {duration_seconds} seconds..."
    
    passthrough.start()
    time.sleep(duration_seconds)
    metrics = passthrough.measure_latency()
    passthrough.stop()
    
    IMPRIMIR f"Input Callback Time: {metrics['input_callback_ms']:.3f} ms"
    IMPRIMIR f"Overflows: {metrics['input_overflow']}"
    IMPRIMIR f"Underruns: {metrics['output_underrun']}"
    
    SI metrics['input_overflow'] > 0 O metrics['output_underrun'] > 0:
        IMPRIMIR "⚠️ WARNING: Detectados problemas de timing"
    SINO:
        IMPRIMIR "✅ Audio I/O subsystem OK"
    
    RETORNAR metrics
```

### Criterio de Éxito (Testing)
- Passthrough funciona durante 60 segundos sin error
- Audio escuchado a través del cable virtual es reconocible (no corrupto)
- Latencia teórica ≤ 10.7ms (2 × 256/48000 × 1000)
- 0 overflows y 0 underruns durante la prueba
- El benchmark imprime métricas correctas

---

## 8. RESUMEN DE ENTREGABLES

| Entregable | Archivo | Validación |
|-----------|---------|-----------|
| Ring Buffer SPSC | `src/audio_io/ring_buffer.py` | 6 tests obligatorios pasan |
| Audio Capture | `src/audio_io/capture.py` | Captura audio real sin overflow |
| Audio Output | `src/audio_io/output.py` | Reproduce en VB-Cable sin underrun |
| Device Enum | `src/audio_io/devices.py` | Lista dispositivos correctamente |
| Passthrough | `src/audio_io/passthrough.py` | Benchmark: ≤ 12ms, 0 errores |

---

## 9. RIESGOS Y MITIGACIONES

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| VB-Cable no instalado | No hay salida | Detectar con `find_virtual_cable()`, mostrar error claro |
| Driver WASAPI Exclusive falla | Mayor latencia | Fallback a WASAPI Shared, log warning |
| ring_buffer overflow por DSP lento | Pérdida de audio | Aumentar capacity, alertar en dashboard |
| Micrófono USB con sample rate fijado | Mismatch de SR | `validate_device()` detecta, resampling en la captura |
| callback_time > 1ms | Audio glitches | Alert en dashboard, reducir operaciones en callback |

---

## 10. DIAGRAMA DE HILOS Y COMUNICACIÓN

```
┌────────────────────────────────────────────────────────────────┐
│  HILO 1: Audio Input Callback (C-level, ~5ms periodo)          │
│  ┌──────────────────────────┐                                  │
│  │ indata → memcpy → ring_buffer_in.write()                    │
│  │ Tiempo máximo: 1ms                                          │
│  └──────────────┬───────────┘                                  │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────┐                                  │
│  │     RING BUFFER INPUT     │                                  │
│  │    (lock-free, SPSC)      │                                  │
│  └──────────────┬───────────┘                                  │
│                 │                                               │
│                 ▼                                               │
│  HILO 2: Passthrough/DSP Thread (Python-level)                 │
│  ┌──────────────────────────┐                                  │
│  │ ring_buffer_in.read() → [PROCESS] → ring_buffer_out.write() │
│  └──────────────┬───────────┘                                  │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────┐                                  │
│  │    RING BUFFER OUTPUT     │                                  │
│  │    (lock-free, SPSC)      │                                  │
│  └──────────────┬───────────┘                                  │
│                 │                                               │
│                 ▼                                               │
│  HILO 3: Audio Output Callback (C-level, ~5ms periodo)         │
│  ┌──────────────────────────┐                                  │
│  │ ring_buffer_out.read() → memcpy → outdata                   │
│  │ Tiempo máximo: 1ms                                          │
│  └──────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────┘
```
