# FASE 6 — INTEGRACIÓN, PROFILING Y QA
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 5–7 días | **Dependencias:** TODAS las fases (F0-F5)  
> **Output esperado:** Pipeline E2E funcional con benchmarks documentados y suite de tests

---

## 1. OBJETIVO

Conectar todos los módulos en un pipeline end-to-end, ejecutar benchmarks exhaustivos, y validar que el sistema cumple los criterios de calidad finales. Esta fase es donde todo se integra y se prueba bajo condiciones reales.

---

## 2. ARCHIVOS

```
src/core/
├── __init__.py
├── config.py              (ya creado en F0)
├── pipeline.py            (F6-T1) Pipeline Orchestrator
└── profiler.py            (F6-T2) Latency profiler

scripts/
├── check_cuda.py          (ya creado en F0)
├── benchmark_e2e.py       (F6-T2) E2E benchmark
├── stress_test_vram.py    (F6-T3) VRAM stress
└── run_regression.py      (F6-T7) Regression runner

tests/
├── integration/
│   ├── test_e2e_pipeline.py
│   ├── test_error_recovery.py    (F6-T6)
│   └── test_compat_gpu.py        (F6-T4/T5)
└── ...
```

---

## 3. F6-T1: PIPELINE ORCHESTRATOR (`src/core/pipeline.py`)

### Pseudocódigo
```
CLASE VosmicPipeline:
    """
    Orquestador principal que inicializa, conecta y gestiona
    todos los módulos del sistema.
    """
    CONSTRUCTOR(config: VosmicConfig):
        self.config = config
        self.event_bus = EventBus()
        
        # Ring Buffers
        self.input_buffer = LockFreeRingBuffer(config.audio.ring_buffer_capacity)
        self.output_buffer = LockFreeRingBuffer(config.audio.ring_buffer_capacity)
        
        # Colas inter-thread
        self.dsp_to_inference_queue = queue.Queue(maxsize=config.inference.queue_max_size)
        self.inference_to_post_queue = queue.Queue(maxsize=config.inference.queue_max_size)
        
        # Módulos (inicializados en start())
        self.capture = None
        self.output = None
        self.dsp_thread = None
        self.inference_scheduler = None
        self.post_processor = None
        self.output_thread = None
        
        self._is_running = False
    
    start():
        """
        Inicializa y arranca todos los módulos en el orden correcto.
        Orden de arranque:
        1. Cargar modelo de voz a GPU
        2. Inicializar DSP (pitch + content models)
        3. Inicializar Audio Output (para que el callback esté listo)
        4. Inicializar Audio Capture (último, para que todo esté listo cuando llegue audio)
        5. Arrancar hilos de procesamiento
        """
        # 1. Modelo RVC
        model_mgr = ModelManager(device="cuda")
        model_mgr.load(self.config.inference.model_path, 
                       precision=self.config.inference.precision)
        
        # 2. DSP components
        noise_gate = NoiseGate(self.config.dsp.noise_gate_threshold_db)
        normalizer = LoudnessNormalizer(self.config.dsp.loudness_target_lufs)
        pitch = PitchExtractor(method=self.config.dsp.pitch_extractor)
        content = ContentEncoder()
        assembler = FeatureBundleAssembler(noise_gate, normalizer, pitch, content)
        
        # 3. Inference Engine
        engine = InferenceEngine(model_mgr)
        
        # 4. Post Processor
        self.post_processor = PostProcessor(self.config.postprocessing)
        
        # 5. Audio I/O
        self.output = AudioOutput(self.output_buffer, 
                                   device_index=self.config.audio.output_device)
        self.capture = AudioCapture(self.input_buffer,
                                     device_index=self.config.audio.input_device)
        
        # 6. Processing Threads
        self.dsp_thread = DSPThread(self.input_buffer, self.dsp_to_inference_queue,
                                     assembler, self.config.audio.block_size)
        self.inference_scheduler = InferenceScheduler(
            engine, self.dsp_to_inference_queue, self.inference_to_post_queue,
            max_inference_ms=self.config.inference.max_inference_time_ms)
        self.output_thread = OutputThread(
            self.inference_to_post_queue, self.output_buffer,
            self.post_processor)
        
        # 7. Start all (order matters!)
        self.output.start()           ← Output primero
        self.capture.start()          ← Capture último
        self.dsp_thread.start()
        self.inference_scheduler.start()
        self.output_thread.start()
        
        self._is_running = True
        self._start_metrics_reporter()
    
    stop():
        """Shutdown limpio en orden inverso."""
        self._is_running = False
        self.capture.stop()           ← Capture primero (no más audio entrante)
        self.dsp_thread.stop()
        self.inference_scheduler.stop()
        self.output_thread.stop()
        self.output.stop()            ← Output último
    
    _start_metrics_reporter():
        """Hilo que recopila métricas de todos los módulos y emite al EventBus."""
        FUNCIÓN report_loop():
            MIENTRAS self._is_running:
                self.event_bus.emit("latency_update", {
                    "latency_ms": calcular_latencia_total()})
                self.event_bus.emit("vram_update", 
                    model_mgr.get_vram_usage())
                self.event_bus.emit("audio_level", {
                    "input_peak": self.capture.metrics["peak_level"],
                    "output_peak": self.output.metrics["output_level"]})
                self.event_bus.emit("buffer_status", {
                    "fill_percent": self.input_buffer.available_read / 
                                    self.config.audio.ring_buffer_capacity * 100})
                time.sleep(0.1)    ← 10 Hz reporting
        
        Thread(target=report_loop, daemon=True).start()
    
    # ---- Command handler (from GUI) ----
    process_commands():
        """Procesar comandos del EventBus de la GUI."""
        cmd = self.event_bus.get_command()
        MIENTRAS cmd:
            SI cmd["type"] == "set_input_device":
                self.capture.change_device(cmd["data"]["device_index"])
            ELIF cmd["type"] == "load_model":
                INTENTAR:
                    model_mgr.swap_model(cmd["data"]["model_path"])
                    self.event_bus.emit("model_loaded", model_mgr.active_model_info)
                CAPTURAR error:
                    self.event_bus.emit("model_load_error", {"error": str(error)})
            ELIF cmd["type"] == "set_bypass":
                self.post_processor.set_bypass(cmd["data"]["enabled"])
            ELIF cmd["type"] == "shutdown":
                self.stop()
            cmd = self.event_bus.get_command()

CLASE OutputThread:
    """Hilo que toma InferenceResult, aplica postprocessing, escribe a output buffer."""
    _loop():
        MIENTRAS running:
            INTENTAR:
                result = inference_to_post_queue.get(timeout=0.01)
                audio = post_processor.process(result)
                output_buffer.write(audio)
            CAPTURAR Empty:
                CONTINUAR
```

---

## 4. F6-T2: E2E LATENCY PROFILER (`scripts/benchmark_e2e.py`)

### Pseudocódigo
```
FUNCIÓN benchmark_end_to_end(duration_seconds=30):
    """
    Mide latencia real micro → cable virtual usando impulse response.
    
    Método:
    1. Enviar un click (impulse) al micrófono
    2. Medir cuándo aparece en la salida del cable virtual
    3. La diferencia = latencia real E2E
    """
    pipeline = VosmicPipeline(load_config())
    pipeline.start()
    
    # Dar tiempo de warm-up
    sleep(2)
    
    # Recopilar métricas cada segundo
    latencies = []
    PARA _ EN range(duration_seconds):
        metrics = {
            "dsp_time_ms": pipeline.dsp_thread.metrics["process_time_ms"],
            "inference_time_ms": pipeline.inference_scheduler.metrics["inference_time_ms"],
            "total_estimated_ms": calcular_total(),
            "input_overflow": pipeline.capture.metrics["overflow_count"],
            "output_underrun": pipeline.output.metrics["underrun_count"],
            "vram_mb": pipeline.model_manager.get_vram_usage()["used_mb"],
        }
        latencies.append(metrics)
        sleep(1)
    
    pipeline.stop()
    
    # Report
    IMPRIMIR "=== E2E Benchmark Results ==="
    IMPRIMIR f"Average latency: {mean(l['total_estimated_ms'] for l in latencies):.1f} ms"
    IMPRIMIR f"P99 latency: {percentile(latencies, 99):.1f} ms"
    IMPRIMIR f"Max latency: {max(l['total_estimated_ms'] for l in latencies):.1f} ms"
    IMPRIMIR f"Total overflows: {sum(l['input_overflow'] for l in latencies)}"
    IMPRIMIR f"Total underruns: {sum(l['output_underrun'] for l in latencies)}"
    IMPRIMIR f"VRAM peak: {max(l['vram_mb'] for l in latencies):.0f} MB"
```

---

## 5. F6-T3: VRAM STRESS TEST (`scripts/stress_test_vram.py`)

### Pseudocódigo
```
FUNCIÓN stress_test(duration_minutes=30):
    pipeline = VosmicPipeline(load_config())
    pipeline.start()
    
    vram_initial = get_vram_usage()
    vram_samples = []
    
    PARA minuto EN range(duration_minutes):
        sleep(60)
        vram_current = get_vram_usage()
        vram_samples.append(vram_current)
        IMPRIMIR f"Minute {minuto+1}: VRAM = {vram_current:.0f} MB"
    
    pipeline.stop()
    
    vram_leak = vram_samples[-1] - vram_initial
    IMPRIMIR f"VRAM leak: {vram_leak:.0f} MB over {duration_minutes} min"
    ASSERT vram_leak < 50, f"VRAM leak too large: {vram_leak} MB"
```

---

## 6. F6-T4/T5: COMPATIBILITY TESTS

### RTX 4060 (target principal)

| Métrica | Umbral | Método |
|---------|--------|--------|
| E2E latency | ≤ 35ms | benchmark_e2e.py |
| VRAM peak | ≤ 2 GB | stress_test_vram.py |
| VRAM leak (30min) | ≤ 50 MB | stress_test_vram.py |
| CPU usage | ≤ 15% | Task Manager / psutil |
| Audio quality (PESQ) | ≥ 3.5 | pesq library vs reference |
| Dropouts (10min) | 0 | overflow + underrun counters |

### GTX 1650 (compatible)

| Métrica | Umbral | Notas |
|---------|--------|-------|
| E2E latency | ≤ 40ms | Con INT8 + block_size=512 |
| VRAM peak | ≤ 3.5 GB | INT8 model requerido |
| VRAM leak (30min) | ≤ 50 MB | Misma tolerancia |
| CPU usage | ≤ 25% | Mayor carga CPU por cuantización |
| PESQ | ≥ 3.0 | Ligeramente menor por INT8 |
| Dropouts (10min) | ≤ 2 | Tolerancia ligeramente mayor |

---

## 7. F6-T6: ERROR RECOVERY TEST

### Escenarios a Probar

```
ESCENARIO 1: Micrófono desconectado mid-session
    ACCIÓN: Desconectar USB mic mientras pipeline corre
    ESPERADO: Pipeline detecta error, audio output → silencio
    RECUPERACIÓN: Al reconectar mic → audio reanuda en < 2 seconds
    VERIFICACIÓN: No crash, no memory leak

ESCENARIO 2: VB-Cable desinstalado
    ACCIÓN: Desinstalar VB-Cable mientras pipeline corre
    ESPERADO: Output engine lanza error, pipeline entra en degraded mode
    RECUPERACIÓN: Al reinstalar → pipeline puede reconfigurar output
    VERIFICACIÓN: No crash, error reportado a GUI

ESCENARIO 3: VRAM agotada (simular)
    ACCIÓN: Allocar tensores grandes para llenar VRAM
    ESPERADO: Inferencia falla → passthrough activa
    RECUPERACIÓN: Al liberar VRAM → inferencia reanuda
    VERIFICACIÓN: Audio NUNCA es silencio (passthrough)

ESCENARIO 4: Overflow de ring buffer
    ACCIÓN: Bloquear DSP thread artificialmente (sleep largo)
    ESPERADO: Ring buffer overflow → datos más viejos se pierden
    RECUPERACIÓN: Al desbloquear → pipeline reanuda
    VERIFICACIÓN: No crash, métricas reportan overflow

ESCENARIO 5: Modelo corrupto
    ACCIÓN: Cargar archivo .onnx corrupto
    ESPERADO: ModelManager rechaza, mantiene modelo actual
    VERIFICACIÓN: Error reportado a GUI, audio sigue fluyendo
```

---

## 8. F6-T7: AUTOMATED REGRESSION SUITE

### Pseudocódigo
```
ESTRUCTURA de tests/integration/:

test_e2e_pipeline.py:
    - test_pipeline_starts_and_stops
    - test_pipeline_processes_audio
    - test_pipeline_passthrough_mode
    - test_pipeline_model_swap
    - test_pipeline_device_change

test_error_recovery.py:
    - test_mic_disconnect_recovery
    - test_vram_exhaustion_passthrough
    - test_buffer_overflow_recovery
    - test_corrupt_model_rejection

test_compat_gpu.py:
    - test_rtx4060_latency_target
    - test_gtx1650_latency_target
    - test_vram_within_budget

COMANDO: pytest tests/ -v --tb=short -x --timeout=120
    -v: verbose
    --tb=short: tracebacks cortos
    -x: parar al primer fallo
    --timeout=120: máximo 2 min por test
```

---

## 9. F6-T8: USER ACCEPTANCE TEST

### Protocolo
```
PASO 1: Abrir VOSMIC con modelo "El Narrador"
PASO 2: Unirse a una llamada de Discord / Zoom / Google Meet
PASO 3: Configurar VB-Cable como micrófono en la app
PASO 4: Hablar durante 10 minutos con conversación natural
PASO 5: Evaluar subjetivamente:
    - ¿La voz suena natural?
    - ¿Hay clicks, artefactos, o cortes?
    - ¿El delay es perceptible al hablar?
    - ¿Hay eco o feedback?
PASO 6: Pedir al interlocutor que evalúe calidad
PASO 7: Documentar hallazgos en un reporte
```

---

## 10. CRITERIOS FINALES DE ACEPTACIÓN

| Métrica | RTX 4060 | GTX 1650 |
|---------|----------|----------|
| **E2E latency** | **≤ 35ms** | **≤ 40ms** |
| VRAM peak | ≤ 2 GB | ≤ 3.5 GB |
| VRAM leak (30 min) | ≤ 50 MB | ≤ 50 MB |
| CPU usage | ≤ 15% | ≤ 25% |
| PESQ score | ≥ 3.5 | ≥ 3.0 |
| Audio dropouts (10 min) | 0 | ≤ 2 |
| Error recovery time | ≤ 2s | ≤ 3s |
| Cold start | ≤ 8s | ≤ 12s |
| **All regression tests** | **PASS** | **PASS** |

---

## 11. RIESGOS

| Riesgo | Mitigación |
|--------|-----------|
| Latencia > 40ms en GTX 1650 | Aumentar block_size a 512, usar INT8 |
| VRAM leak por torch cache | torch.cuda.empty_cache() periódico |
| CI no puede testear GPU | Tests GPU marcados como @pytest.mark.gpu, skip en CI |
| Discord detecta VB-Cable como "mala calidad" | Verificar sample rate match 48kHz |
| PESQ score bajo | Reentrenar modelo con más datos, ajustar postprocessing |
