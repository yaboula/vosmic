# FASE 2 — MÓDULO DE PREPROCESAMIENTO DSP
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 5–7 días | **Dependencias:** Fase 0 (CUDA), Fase 1 (Ring Buffer)  
> **Output esperado:** Módulo que transforma audio crudo en `FeatureBundle` en ≤ 5ms

---

## 1. OBJETIVO

Transformar audio crudo en features para RVC: **pitch (F0)** con RMVPE, **embeddings** con ContentVec, y **noise gate + normalización**. Todo en **hilo dedicado** separado del audio thread.

### Presupuesto de Latencia: ≤ 5ms total
- Noise gate: ~0.01ms | Normalization: ~0.01ms | Pitch (RMVPE GPU): ~2ms | ContentVec (GPU): ~2ms

---

## 2. ARCHIVOS A CREAR

```
src/dsp/
├── __init__.py          ├── noise_gate.py        (F2-T1)
├── normalizer.py        (F2-T2)
├── pitch_extractor.py   (F2-T3)
├── content_encoder.py   (F2-T4)
├── feature_bundle.py    (F2-T5)
└── dsp_thread.py        (F2-T6)
```

---

## 3. F2-T1: NOISE GATE (`noise_gate.py`)

### Pseudocódigo
```
CLASE NoiseGate:
    CONSTRUCTOR(threshold_db=-40.0, hold_time_ms=100.0, sample_rate=48000):
        threshold_linear = 10^(threshold_db/20)
        hold_samples = hold_time_ms/1000 * sample_rate
        hold_counter = 0, is_open = False
    
    process(audio) → (audio_procesado, is_silence):
        rms = sqrt(mean(audio²))
        SI rms >= threshold: is_open=True, hold_counter=hold_samples
        SINO: SI hold_counter > 0: hold_counter -= len(audio)
              SINO: is_open = False
        RETORNAR (audio, False) SI is_open SINO (audio*0, True)
```

### Edge Cases
| Caso | Comportamiento |
|------|---------------|
| Audio todo ceros | `is_silence = True` |
| Justo en umbral | Gate se abre (≥) |
| Pausa entre sílabas | Hold time mantiene gate abierto |

### Tests
- Silencio → `is_silence == True`
- Voz (-20dB) → `is_silence == False`, audio sin modificar
- Hold time evita cortes: voz → silencio breve → gate sigue abierto
- Hold expira: suficiente silencio → gate se cierra

### Criterio: ≥98% accuracy, ≤0.05ms/chunk

---

## 4. F2-T2: LOUDNESS NORMALIZATION (`normalizer.py`)

### Pseudocódigo
```
CLASE LoudnessNormalizer:
    CONSTRUCTOR(target_lufs=-18.0):
        target_rms = 10^(target_lufs/20)
        current_gain = 1.0, smoothing = 0.95
        min_gain = 0.01, max_gain = 10.0
    
    normalize(audio) → (audio_normalizado, original_rms):
        original_rms = sqrt(mean(audio²))
        SI original_rms < 1e-8: RETORNAR (audio, 0.0)
        desired_gain = clip(target_rms / original_rms, min_gain, max_gain)
        current_gain = smoothing*current_gain + (1-smoothing)*desired_gain  ← EMA
        normalized = clip(audio * current_gain, -1.0, 1.0)
        RETORNAR (normalized, original_rms)
```

### Criterio: ±2dB del target, no clicks por cambios bruscos, ≤0.05ms/chunk

---

## 5. F2-T3: PITCH EXTRACTOR (`pitch_extractor.py`)

### Pseudocódigo
```
CLASE PitchExtractor:
    CONSTRUCTOR(method="rmvpe", device="cuda", hop_length=160, sample_rate=48000):
        Cargar modelo RMVPE a GPU (una vez)
        accumulation_buffer = [], min_samples_16k = 1024
        SI RMVPE falla: fallback a CREPE-tiny (CPU)
    
    extract(audio) → f0:
        audio_16k = resample(audio, 48000, 16000)  ← RMVPE espera 16kHz
        Acumular en buffer hasta tener ≥ 1024 samples
        SI buffer < min: RETORNAR zeros (unvoiced)
        
        CON torch.no_grad():  ← SIEMPRE en inferencia
            tensor = from_numpy(audio_16k).to(device).unsqueeze(0)
            f0 = model.infer(tensor).cpu().numpy()
        
        Limpiar buffer (mantener 512 overlap)
        RETORNAR f0  ← (T,) Hz, 0=unvoiced
```

### Notas GPU Críticas
- RMVPE usa ~200MB VRAM
- `torch.no_grad()` reduce VRAM ~50% y mejora velocidad ~2x
- Chunks de 256@48kHz = solo ~85 samples@16kHz → acumulación necesaria
- Resampleo con `scipy.signal.resample_poly` (alta calidad)

### Tests
- Tono puro 440Hz → resultado 438-442Hz (≤2Hz error)
- Voz masculina → 80-300Hz
- Silencio → F0 == 0

### Criterio: ≤2Hz error, ≤3ms RTX 4060, ≤250MB VRAM

---

## 6. F2-T4: CONTENT ENCODER (`content_encoder.py`)

### Pseudocódigo
```
CLASE ContentEncoder:
    CONSTRUCTOR(model="contentvec", device="cuda", target_sr=16000):
        Cargar ContentVec pretrained → model.eval()
        embedding_dim = 256
    
    encode(audio, sample_rate=48000) → embeddings:
        SI sample_rate != 16000: audio = resample(audio, sample_rate, 16000)
        CON torch.no_grad():
            tensor = from_numpy(audio).to(device).unsqueeze(0)
            features = model.extract_features(tensor)  ← (1, T, 768)
            embeddings = projection(features)  ← (1, T, 256)
        RETORNAR embeddings.squeeze(0).cpu().numpy()  ← (T, 256)
```

### Tests
- Output shape siempre `(T, 256)`
- Misma frase → cosine similarity > 0.9
- Diferentes frases → cosine similarity < 0.7

### Criterio: ≤3ms RTX 4060, ≤350MB VRAM

---

## 7. F2-T5: FEATURE BUNDLE (`feature_bundle.py`)

### Pseudocódigo
```
@dataclass(frozen=True)
CLASE FeatureBundle:
    audio: ndarray           # Normalizado (N,) float32
    f0: ndarray              # Pitch (T,) Hz
    content_embedding: ndarray # (T, 256)
    original_rms: float      # Para gain matching (Fase 4)
    is_silence: bool
    timestamp: float
    chunk_id: int

CLASE FeatureBundleAssembler:
    CONSTRUCTOR(noise_gate, normalizer, pitch_extractor, content_encoder)
    
    assemble(audio, timestamp) → FeatureBundle:
        gated, is_silence = noise_gate.process(audio)
        SI is_silence: RETORNAR bundle con zeros y is_silence=True
        normalized, rms = normalizer.normalize(gated)
        f0 = pitch_extractor.extract(normalized)
        embeddings = content_encoder.encode(normalized)
        RETORNAR FeatureBundle(normalized, f0, embeddings, rms, False, timestamp, id++)
```

### Paralelismo GPU Opcional
```
Pitch y ContentVec son INDEPENDIENTES → pueden correr en CUDA streams paralelas:
  Secuencial: 2ms + 2ms = 4ms
  Paralelo:   max(2ms, 2ms) = 2ms
RECOMENDACIÓN: Empezar secuencial. Optimizar si Fase 6 lo requiere.
```

---

## 8. F2-T6: DSP THREAD MANAGER (`dsp_thread.py`)

### Pseudocódigo
```
CLASE DSPThread:
    CONSTRUCTOR(input_ring_buffer, output_queue, assembler, block_size=256):
        Hilo daemon dedicado
    
    _process_loop():
        MIENTRAS running:
            SI input_buffer.available >= block_size:
                audio = input_buffer.read(block_size)
                bundle = assembler.assemble(audio, timestamp)
                INTENTAR: output_queue.put_nowait(bundle)
                CAPTURAR Full: bundles_dropped++  ← NO bloquear
            SINO:
                sleep(0.0005)  ← 0.5ms, no spin-wait
```

### Criterio: process_time ≤5ms, bundles_dropped==0 en operación normal, stop() en <2s

---

## 9. RIESGOS

| Riesgo | Mitigación |
|--------|-----------|
| RMVPE no disponible | Fallback CREPE-tiny (CPU, más lento) |
| ContentVec grande en VRAM | Versión tiny o INT8 para GTX 1650 |
| Chunks pequeños para RMVPE | Buffer de acumulación |
| CUDA stream contention | Medir antes/después, revertir si peor |
