# FASE 3 — MOTOR DE INFERENCIA RVC
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 7–10 días | **Dependencias:** Fase 0 (CUDA), Fase 2 (FeatureBundle)  
> **Output esperado:** Motor de inferencia que convierte voz en ≤ 20ms con hot-swap de modelos

---

## 1. OBJETIVO

Módulo más crítico. Recibe `FeatureBundle` (pitch + embeddings), ejecuta el modelo RVC en GPU para generar voz clonada, y entrega audio procesado al postprocessor. Toda inferencia en **hilo async separado**. Si GPU tarda demasiado → **passthrough** (audio original, NUNCA silencio).

### Presupuesto: ≤ 20ms por chunk (RTX 4060 FP16)

---

## 2. ARCHIVOS

```
src/inference/
├── __init__.py
├── model_manager.py       (F3-T1) Carga/swap de modelos
├── inference_core.py      (F3-T2) Inferencia PyTorch
├── onnx_export.py         (F3-T3) Export a ONNX
├── tensorrt_optimize.py   (F3-T4) Export a TensorRT
├── scheduler.py           (F3-T5) Cola FIFO async
├── stitcher.py            (F3-T6) Overlap-add cross-fade
└── passthrough.py         (F3-T7) Fallback when GPU overloaded
```

---

## 3. F3-T1: MODEL MANAGER (`model_manager.py`)

### Pseudocódigo
```
CLASE ModelManager:
    CONSTRUCTOR(device="cuda"):
        active_model = None
        active_model_info = {}
        loading_lock = threading.Lock()  ← Solo para carga, NUNCA en hot path
    
    load(path, format="onnx", precision="fp16"):
        # Detectar formato por extensión
        SI format == "pytorch" o path.endswith(".pth"):
            model = cargar_pytorch(path)
            SI precision == "fp16": model = model.half()
            model = model.to(device).eval()
        ELIF format == "onnx":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            model = onnxruntime.InferenceSession(path, providers=providers)
        ELIF format == "tensorrt":
            model = cargar_tensorrt_engine(path)
        
        active_model = model
        active_model_info = {name, format, precision, vram_used}
    
    swap_model(new_path) → None:
        """
        Hot-swap: carga nuevo modelo ANTES de descargar el viejo.
        Garantiza cero downtime.
        """
        old_model = self.active_model
        INTENTAR:
            self.load(new_path)    ← Carga nuevo a VRAM
            # Si éxito → liberar viejo
            del old_model
            torch.cuda.empty_cache()
        CAPTURAR error:
            # Si falla → mantener viejo
            self.active_model = old_model
            LANZAR ModelSwapError(error)
        TIMEOUT: 10 segundos máximo para carga
    
    get_vram_usage() → Dict:
        used = torch.cuda.memory_allocated() / 1024²
        total = torch.cuda.get_device_properties(0).total_memory / 1024²
        RETORNAR {"used_mb": used, "total_mb": total}
    
    unload():
        del active_model
        torch.cuda.empty_cache()
```

### Edge Cases
| Caso | Comportamiento |
|------|---------------|
| Archivo .pth no existe | Lanzar `FileNotFoundError` con path |
| VRAM insuficiente para nuevo modelo | Mantener modelo actual, lanzar error |
| swap_model tarda > 10s | Timeout, mantener modelo actual |
| Formato no reconocido | Lanzar `UnsupportedFormatError` |

### Criterio: swap ≤ 2s sin dropout, VRAM FP16 ≤ 2GB, FP32 fallback funcional

---

## 4. F3-T2: INFERENCE CORE (`inference_core.py`)

### Pseudocódigo
```
CLASE InferenceEngine:
    CONSTRUCTOR(model_manager: ModelManager):
        self.model_manager = model_manager
    
    infer(bundle: FeatureBundle) → np.ndarray | None:
        """
        ENTRADA: FeatureBundle (content_embedding, f0, audio)
        SALIDA: audio convertido (N,) float32
        RETORNA None si falla (caller usa passthrough)
        
        ⚠️ NUNCA bloquea el audio thread — se llama desde inference thread
        """
        SI bundle.is_silence:
            RETORNAR None    ← No procesar silencio
        
        model = self.model_manager.active_model
        SI model es None:
            RETORNAR None
        
        INTENTAR:
            CON torch.no_grad():
                # Preparar inputs según formato del modelo
                SI formato == "pytorch":
                    content = torch.from_numpy(bundle.content_embedding).to(device)
                    f0 = torch.from_numpy(bundle.f0).to(device)
                    # speaker_embedding se carga del modelo
                    
                    output = model(content.unsqueeze(0), f0.unsqueeze(0))
                    audio_out = output.squeeze().cpu().numpy()
                
                ELIF formato == "onnx":
                    inputs = {
                        "content": bundle.content_embedding[np.newaxis],
                        "f0": bundle.f0[np.newaxis],
                    }
                    outputs = model.run(None, inputs)
                    audio_out = outputs[0].squeeze()
            
            RETORNAR audio_out.astype(np.float32)
        
        CAPTURAR Exception como error:
            # Loggear error pero NO crashear
            RETORNAR None    ← Passthrough fallback
    
    get_inference_time_ms(bundle) → float:
        """Benchmark: medir tiempo de inferencia."""
        start = time.perf_counter()
        self.infer(bundle)
        torch.cuda.synchronize()
        RETORNAR (time.perf_counter() - start) * 1000
```

### Notas Críticas para el Agente Programador
```
⚠️ RVC MODEL INPUT/OUTPUT:

El modelo RVC típico tiene estos inputs:
  1. content_embedding: shape (1, T, 256) ← del ContentVec
  2. f0: shape (1, T) ← del pitch extractor
  3. sid: int ← speaker ID (si multi-speaker)

Output:
  - audio: shape (1, N) ← audio de voz convertida a sample_rate del modelo

El sample_rate del output puede ser DIFERENTE al input:
  - Modelo entrenado a 40000 Hz → output a 40kHz
  - Modelo entrenado a 48000 Hz → output a 48kHz
  - Si hay mismatch → Fase 4 (PostProcessor) hace el resampleo

⚠️ torch.cuda.synchronize():
  - Llamar DESPUÉS de la inferencia para medir tiempo real
  - CUDA es asíncrono: sin synchronize(), el timing es falso
  - En producción NO llamar synchronize() → deja que CUDA sea async
  - Solo llamar synchronize() en benchmarks/profiling
```

### Criterio
| Métrica | RTX 4060 | GTX 1650 |
|---------|----------|----------|
| PyTorch FP16 | ≤ 20ms | N/A |
| ONNX | ≤ 15ms | ≤ 25ms |
| TensorRT | ≤ 10ms | ≤ 18ms |
| Fallo → retorna None | 100% | 100% |

---

## 5. F3-T3: ONNX EXPORT (`onnx_export.py`)

### Pseudocódigo
```
FUNCIÓN export_to_onnx(pytorch_model_path, onnx_output_path, sample_length=256):
    model = cargar_pytorch(pytorch_model_path).eval().cuda()
    
    # Crear inputs dummy para tracing
    dummy_content = torch.randn(1, 100, 256).cuda()
    dummy_f0 = torch.randn(1, 100).cuda()
    
    torch.onnx.export(
        model,
        (dummy_content, dummy_f0),
        onnx_output_path,
        input_names=["content", "f0"],
        output_names=["audio"],
        dynamic_axes={
            "content": {1: "time"},    ← Longitud variable
            "f0": {1: "time"},
            "audio": {1: "samples"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    # Verificar equivalencia
    modelo_onnx = onnxruntime.InferenceSession(onnx_output_path)
    output_pytorch = model(dummy_content, dummy_f0).cpu().numpy()
    output_onnx = modelo_onnx.run(None, {
        "content": dummy_content.cpu().numpy(),
        "f0": dummy_f0.cpu().numpy()
    })[0]
    
    max_diff = np.max(np.abs(output_pytorch - output_onnx))
    ASSERT max_diff < 1e-3, f"Diff too large: {max_diff}"
    IMPRIMIR f"✅ ONNX export OK. Max diff: {max_diff:.6f}"
```

### Criterio: max diff ≤ 1e-3, archivo ONNX generado y validado

---

## 6. F3-T4: TENSORRT OPTIMIZATION (`tensorrt_optimize.py`)

### Pseudocódigo
```
FUNCIÓN optimize_to_tensorrt(onnx_path, trt_output_path, precision="fp16"):
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    
    parser.parse_from_file(onnx_path)
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  ← 1GB
    
    SI precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    ELIF precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 requiere calibración con datos reales
        config.int8_calibrator = crear_calibrador(datos_audio_reales)
    
    engine = builder.build_serialized_network(network, config)
    
    CON open(trt_output_path, "wb") como f:
        f.write(engine)
    
    # Benchmark: comparar vs ONNX
    benchmark_trt = medir_latencia_trt(trt_output_path)
    benchmark_onnx = medir_latencia_onnx(onnx_path)
    IMPRIMIR f"ONNX: {benchmark_onnx:.1f}ms → TRT: {benchmark_trt:.1f}ms"
```

### Criterio: TRT ≤ 10ms RTX 4060, diff vs ONNX ≤ 1e-2

---

## 7. F3-T5: INFERENCE SCHEDULER (`scheduler.py`)

### Pseudocódigo
```
CLASE InferenceScheduler:
    """
    Hilo dedicado de inferencia.
    Consume FeatureBundles de la cola DSP,
    ejecuta inferencia, deposita resultado en cola de salida.
    """
    
    CONSTRUCTOR(inference_engine, input_queue, output_queue,
                max_inference_ms=25, passthrough_enabled=True):
        self.engine = inference_engine
        self.input_queue = input_queue    ← De DSP Thread
        self.output_queue = output_queue  ← Hacia PostProcessor
        self.max_ms = max_inference_ms
        self.passthrough = passthrough_enabled
    
    _inference_loop():
        MIENTRAS running:
            INTENTAR:
                bundle = input_queue.get(timeout=0.01)
            CAPTURAR Empty:
                CONTINUAR
            
            SI bundle.is_silence:
                # Silencio → pasar silencio directamente
                output_queue.put(InferenceResult(
                    audio=np.zeros_like(bundle.audio),
                    original_audio=bundle.audio, ...))
                CONTINUAR
            
            t_start = perf_counter()
            converted = engine.infer(bundle)
            t_ms = (perf_counter() - t_start) * 1000
            
            SI converted es None O t_ms > self.max_ms:
                # PASSTHROUGH: enviar audio original
                output_queue.put(InferenceResult(
                    audio=bundle.audio,    ← ORIGINAL, no silencio
                    original_audio=bundle.audio,
                    was_passthrough=True, ...))
            SINO:
                output_queue.put(InferenceResult(
                    audio=converted,
                    original_audio=bundle.audio,
                    was_passthrough=False, ...))

@dataclass
CLASE InferenceResult:
    audio: np.ndarray           # Audio convertido (o original si passthrough)
    original_audio: np.ndarray  # Audio original (para gain matching)
    original_rms: float
    was_passthrough: bool
    inference_time_ms: float
    chunk_id: int
```

### Criterio: passthrough activa al 100% cuando GPU overloaded, 0 silencio

---

## 8. F3-T6: OVERLAP-ADD STITCHER (`stitcher.py`)

### Pseudocódigo
```
CLASE OverlapAddStitcher:
    """
    Elimina artefactos en los bordes de chunks
    aplicando ventana Hanning y overlap-add.
    """
    CONSTRUCTOR(overlap_ratio=0.5, window_type="hanning"):
        self.overlap = overlap_ratio
        self.prev_tail = None
    
    stitch(current_chunk) → np.ndarray:
        overlap_size = int(len(current_chunk) * self.overlap)
        window = np.hanning(overlap_size * 2)
        
        SI self.prev_tail is None:
            self.prev_tail = current_chunk[-overlap_size:]
            RETORNAR current_chunk[:-overlap_size]
        
        # Cross-fade entre cola anterior y inicio actual
        fade_out = self.prev_tail * window[overlap_size:]  ← Parte descendente
        fade_in = current_chunk[:overlap_size] * window[:overlap_size]  ← Parte ascendente
        crossfaded = fade_out + fade_in
        
        # Guardar nueva cola
        self.prev_tail = current_chunk[-overlap_size:]
        
        RETORNAR np.concatenate([crossfaded, current_chunk[overlap_size:-overlap_size]])
```

### Criterio: no clicks audibles en bordes, aprobación subjetiva humana

---

## 9. F3-T7: PASSTHROUGH FALLBACK (`passthrough.py`)

### Regla Absoluta
```
╔═══════════════════════════════════════════════════╗
║  SI LA GPU NO PUEDE COMPLETAR A TIEMPO:           ║
║  → ENVIAR AUDIO ORIGINAL (DRY SIGNAL)             ║
║  → NUNCA ENVIAR SILENCIO                           ║
║  → NUNCA BLOQUEAR EL AUDIO THREAD                 ║
║                                                    ║
║  El usuario escucha su propia voz sin cambiar      ║
║  en lugar de un corte de audio.                    ║
╚═══════════════════════════════════════════════════╝
```

---

## 10. ESTRATEGIA DOUBLE-BUFFERING

```
Tiempo ──────────────────────────────────────────▶

Audio Thread:  [reproduce N-2] [reproduce N-1] [reproduce N]
GPU Thread:         [infiere N-1]───┘  [infiere N]───┘
DSP Thread:   [procesa N]──────────────┘

Latencia = 1 chunk audio (5ms) + 1 chunk inferencia (~20ms) = ~25ms
SI GPU tarda > 1 chunk → passthrough automático
```

---

## 11. RIESGOS

| Riesgo | Mitigación |
|--------|-----------|
| OOM al cargar modelo grande | Verificar VRAM antes, FP16/INT8 |
| TensorRT no disponible | Quedarse en ONNX (algo más lento) |
| ONNX export falla en ops custom | Identificar ops incompatibles, reescribir |
| Hot-swap causa dropout | Double-buffer: cargar nuevo antes de soltar viejo |
| INT8 degrada calidad | Benchmark PESQ, calibrar con datos reales |
