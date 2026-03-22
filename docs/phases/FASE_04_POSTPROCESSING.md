# FASE 4 — POSTPROCESAMIENTO Y SALIDA
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 3–4 días | **Dependencias:** Fase 3 (InferenceResult)  
> **Output esperado:** Audio procesado listo para cable virtual con gain matching en ≤ 2ms

---

## 1. OBJETIVO

Recibir audio convertido del motor de inferencia, aplicar ajustes finales (ganancia, compresión, filtro anti-aliasing) y depositarlo en el ring buffer de salida. Presupuesto: **≤ 2ms por chunk**.

---

## 2. ARCHIVOS

```
src/postprocessing/
├── __init__.py
├── gain_matcher.py        (F4-T1)
├── compressor.py          (F4-T2)
├── sample_rate_converter.py (F4-T3)
├── anti_alias_filter.py   (F4-T4)
└── post_processor.py      (F4-T5) Orquestador
```

---

## 3. F4-T1: GAIN MATCHER (`gain_matcher.py`)

### Pseudocódigo
```
CLASE GainMatcher:
    """
    Iguala el volumen del audio convertido al nivel original.
    Evita saltos de volumen molestos al activar/desactivar voice changer.
    """
    CONSTRUCTOR():
        current_gain = 1.0
        smoothing = 0.9   ← EMA para transiciones suaves

    match(converted_audio, original_rms) → np.ndarray:
        SI original_rms < 1e-8: RETORNAR converted_audio
        
        converted_rms = sqrt(mean(converted_audio²))
        SI converted_rms < 1e-8: RETORNAR converted_audio
        
        target_gain = original_rms / converted_rms
        target_gain = clip(target_gain, 0.1, 10.0)
        
        # Transición suave
        current_gain = smoothing * current_gain + (1-smoothing) * target_gain
        
        RETORNAR clip(converted_audio * current_gain, -1.0, 1.0)
```

### Criterio: diferencia RMS input vs output ≤ 1dB, sin clicks

---

## 4. F4-T2: DYNAMIC RANGE COMPRESSOR (`compressor.py`)

### Pseudocódigo
```
CLASE DynamicCompressor:
    """
    Controla picos y normaliza rango dinámico.
    Evita distorsión y protege speakers/headphones.
    """
    CONSTRUCTOR(ratio=3.0, threshold_db=-12.0, attack_ms=5.0, release_ms=50.0,
                sample_rate=48000):
        threshold_linear = 10^(threshold_db/20)
        attack_coeff = exp(-1.0 / (attack_ms/1000 * sample_rate))
        release_coeff = exp(-1.0 / (release_ms/1000 * sample_rate))
        envelope = 0.0
    
    compress(audio) → np.ndarray:
        output = np.copy(audio)
        
        PARA CADA sample EN range(len(audio)):
            # Detectar envelope (seguidor de pico)
            abs_sample = abs(audio[sample])
            SI abs_sample > envelope:
                envelope = attack_coeff * envelope + (1-attack_coeff) * abs_sample
            SINO:
                envelope = release_coeff * envelope + (1-release_coeff) * abs_sample
            
            # Calcular ganancia de compresión
            SI envelope > threshold_linear:
                # Región de compresión
                gain_db = threshold_db + (20*log10(envelope) - threshold_db) / ratio
                gain = 10^(gain_db/20) / envelope
            SINO:
                gain = 1.0
            
            output[sample] = audio[sample] * gain
        
        RETORNAR output
    
    # ⚠️ OPTIMIZACIÓN: El loop sample-by-sample es lento en Python.
    # Vectorizar con numpy o usar numba @jit para < 1ms
```

### Nota de Optimización
```
⚠️ RENDIMIENTO CRÍTICO:
El compresor sample-by-sample es ~10x más lento que vectorizado.
OPCIONES DE OPTIMIZACIÓN (en orden de prioridad):
  1. numba @njit: JIT-compile el loop → ~100x speedup
  2. numpy vectorizado: procesar en bloques de 64 samples
  3. C extension: si numba no disponible
  
El Agente DEBE verificar que compress() completa en < 0.5ms para 256 samples.
```

### Criterio: attack accuracy ±1ms, sin distorsión audible, ≤ 0.5ms

---

## 5. F4-T3: SAMPLE RATE CONVERTER (`sample_rate_converter.py`)

### Pseudocódigo
```
CLASE SampleRateConverter:
    """
    Resamplea audio si el modelo RVC output ≠ 48kHz del sistema.
    Usa sinc interpolation (alta calidad, no linear).
    """
    CONSTRUCTOR(target_sr=48000):
        self.target_sr = target_sr
        self._resample_cache = {}  ← Cache de filtros para evitar recálculo
    
    convert(audio, source_sr) → np.ndarray:
        SI source_sr == self.target_sr:
            RETORNAR audio    ← No conversion needed
        
        gcd_val = gcd(source_sr, self.target_sr)
        up = self.target_sr // gcd_val
        down = source_sr // gcd_val
        
        RETORNAR scipy.signal.resample_poly(audio, up, down)
    
    is_needed(model_sr) → bool:
        RETORNAR model_sr != self.target_sr
```

### Criterio: THD+N ≤ -80dB, sin aliasing audible

---

## 6. F4-T4: ANTI-ALIASING FILTER (`anti_alias_filter.py`)

### Pseudocódigo
```
CLASE AntiAliasingFilter:
    """
    Low-pass Butterworth para eliminar artefactos de alta frecuencia
    producidos por la conversión de voz y el resampleo.
    """
    CONSTRUCTOR(cutoff_hz=20000, order=4, sample_rate=48000):
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist
        self._sos = scipy.signal.butter(order, normalized_cutoff, 
                                         btype='low', output='sos')
        # Estado del filtro (para procesamiento continuo sin artefactos de borde)
        self._zi = scipy.signal.sosfilt_zi(self._sos)
        self._zi = np.repeat(self._zi[:, :, np.newaxis], 1, axis=2)
    
    filter(audio) → np.ndarray:
        # sosfilt mantiene estado entre llamadas → sin artefactos de borde
        filtered, self._zi = scipy.signal.sosfilt(self._sos, audio, zi=self._zi)
        RETORNAR filtered.astype(np.float32)
    
    reset():
        self._zi = scipy.signal.sosfilt_zi(self._sos)
```

### Criterio: -3dB point a 20kHz ± 500Hz, THD+N acceptable, fase mínima

---

## 7. F4-T5: POST PROCESSOR ORCHESTRATOR (`post_processor.py`)

### Pseudocódigo
```
CLASE PostProcessor:
    """
    Encadena todos los postprocesadores en orden:
    gain → compressor → SRC → filter
    Con opción de bypass global.
    """
    CONSTRUCTOR(config: PostProcessConfig):
        self.gain_matcher = GainMatcher()
        self.compressor = DynamicCompressor(**config.compressor)
        self.src = SampleRateConverter(config.target_sr)
        self.anti_alias = AntiAliasingFilter(config.filter_cutoff_hz, ...)
        self.bypass_enabled = config.bypass
    
    process(result: InferenceResult, model_sr: int = 48000) → np.ndarray:
        SI self.bypass_enabled:
            RETORNAR result.audio    ← Bit-perfect passthrough
        
        audio = result.audio
        
        # 1. Gain matching (igualar volumen)
        audio = self.gain_matcher.match(audio, result.original_rms)
        
        # 2. Compresión dinámica
        audio = self.compressor.compress(audio)
        
        # 3. Sample rate conversion (si necesario)
        SI self.src.is_needed(model_sr):
            audio = self.src.convert(audio, model_sr)
        
        # 4. Anti-aliasing filter
        audio = self.anti_alias.filter(audio)
        
        RETORNAR audio
    
    set_bypass(enabled: bool):
        self.bypass_enabled = enabled
```

### Tests Clave
- Bypass: `process()` con bypass retorna audio identical (bit-perfect)
- Pipeline completo: ≤ 2ms para chunk de 256 samples
- Sin clipping: output nunca excede [-1.0, 1.0]
- Gain match: RMS diff ≤ 1dB

---

## 8. RIESGOS

| Riesgo | Mitigación |
|--------|-----------|
| Compresor lento (sample-by-sample) | Usar numba @njit |
| sosfilt state corruption | Reset state en change_device |
| Resampleo artifacts | Usar resample_poly (high quality sinc) |
| Bypass state desincronizado con GUI | Event bus actualiza estado atómicamente |
