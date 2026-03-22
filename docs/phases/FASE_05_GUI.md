# FASE 5 — GUI / SUPERFICIE DE CONTROL
## Documento Detallado para Agentes Programadores y de Testing

> **Tiempo estimado:** 5–7 días | **Dependencias:** Fase 1 (devices), contratos de interfaces  
> **Desarrollo PARALELO con Fases 2-4**  
> **Output esperado:** Interfaz PyQt6 con dashboard de rendimiento en tiempo real

---

## 1. OBJETIVO

Interfaz gráfica que permite: seleccionar dispositivos, cargar modelos, monitorear latencia/VRAM y ajustar parámetros. **La GUI NUNCA ejecuta lógica de audio** — se comunica con el pipeline vía EventBus (queue thread-safe).

---

## 2. ARCHIVOS

```
src/gui/
├── __init__.py
├── event_bus.py           (F5-T1) Messaging thread-safe
├── main_window.py         (F5-T2) Layout principal
├── device_selector.py     (F5-T3) Dropdown de dispositivos
├── level_meter.py         (F5-T4) VU meters animados
├── performance_panel.py   (F5-T5) Dashboard latencia/VRAM
├── model_loader.py        (F5-T6) File dialog + progress
├── system_tray.py         (F5-T7) Tray icon + context menu
└── styles.py              (constantes de estilo)
```

---

## 3. F5-T1: EVENT BUS (`event_bus.py`)

### Pseudocódigo
```
CLASE EventBus:
    """
    Sistema de mensajería thread-safe entre GUI y Pipeline.
    GUI envía comandos → Pipeline recibe y ejecuta.
    Pipeline envía métricas → GUI recibe y actualiza.
    
    ⚠️ NUNCA llamar funciones de audio directamente desde la GUI.
    """
    CONSTRUCTOR():
        self._command_queue = queue.Queue(maxsize=32)
        self._event_callbacks = {}  ← Dict[str, List[Callable]]
    
    # --- GUI → Pipeline (Commands) ---
    send_command(command_type: str, data: Dict):
        """GUI llama esto para enviar comandos al pipeline."""
        self._command_queue.put_nowait({"type": command_type, "data": data})
    
    get_command() → Dict | None:
        """Pipeline llama esto para obtener comandos pendientes."""
        INTENTAR: RETORNAR self._command_queue.get_nowait()
        CAPTURAR Empty: RETORNAR None
    
    # --- Pipeline → GUI (Events) ---
    subscribe(event_type: str, callback: Callable):
        """GUI registra callbacks para recibir eventos."""
        SI event_type NO en self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)
    
    emit(event_type: str, data: Dict):
        """Pipeline emite eventos hacia la GUI."""
        SI event_type EN self._event_callbacks:
            PARA callback EN self._event_callbacks[event_type]:
                # Usar QTimer.singleShot(0, ...) para ejecutar en GUI thread
                callback(data)

TIPOS DE COMANDOS (GUI → Pipeline):
    "set_input_device"    → {"device_index": int}
    "set_output_device"   → {"device_index": int}
    "load_model"          → {"model_path": str}
    "set_bypass"          → {"enabled": bool}
    "set_noise_gate"      → {"threshold_db": float}
    "shutdown"            → {}

TIPOS DE EVENTOS (Pipeline → GUI):
    "latency_update"      → {"latency_ms": float}
    "vram_update"         → {"used_mb": float, "total_mb": float}
    "buffer_status"       → {"fill_percent": float}
    "model_loaded"        → {"name": str, "format": str, "vram_mb": float}
    "model_load_error"    → {"error": str}
    "audio_level"         → {"input_peak": float, "output_peak": float}
    "error"               → {"message": str}
```

### Criterio: thread-safe, ≤ 0.1ms por operación, 0 deadlocks

---

## 4. F5-T2: MAIN WINDOW (`main_window.py`)

### Wireframe
```
┌─────────────────────────────────────────────────────────────┐
│  🎙️ VOSMIC — Real-Time Voice Changer              [─][□][×]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─ INPUT ────────────┐    ┌─ OUTPUT ───────────────────┐  │
│  │ 🎤 [Realtek HD ▾]  │    │ 🔈 [VB-Cable       ▾]     │  │
│  │ ████████░░ -12dB   │    │ ██████░░░░ -18dB           │  │
│  └────────────────────┘    └────────────────────────────┘  │
│                                                             │
│  ┌─ VOICE MODEL ────────────────────────────────────────┐  │
│  │ 🗣️ "El Narrador v2.1"  [Load...] [Swap...]           │  │
│  │ TensorRT FP16 │ 1.2GB VRAM │ 48kHz                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ PERFORMANCE ────────────────────────────────────────┐  │
│  │ Latency:  ████░░░░░░  28ms / 40ms target             │  │
│  │ GPU:      ██████░░░░  62%                             │  │
│  │ VRAM:     ███░░░░░░░  1.2 / 8.0 GB                   │  │
│  │ Buffer:   █░░░░░░░░░  12%                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  [🟢 ACTIVE]  [⏸ BYPASS]  [⚙ SETTINGS]                    │
└─────────────────────────────────────────────────────────────┘
```

### Pseudocódigo
```
CLASE MainWindow(QMainWindow):
    CONSTRUCTOR(event_bus: EventBus):
        self.event_bus = event_bus
        
        # Layout principal: QVBoxLayout
        central = QWidget()
        layout = QVBoxLayout()
        
        # Paneles
        self.device_panel = DeviceSelectorPanel(event_bus)     ← F5-T3
        self.model_panel = ModelLoaderPanel(event_bus)           ← F5-T6
        self.perf_panel = PerformancePanel()                    ← F5-T5
        self.controls = ControlsPanel(event_bus)
        
        layout.addWidget(self.device_panel)
        layout.addWidget(self.model_panel)
        layout.addWidget(self.perf_panel)
        layout.addWidget(self.controls)
        
        # Suscribirse a eventos del pipeline
        event_bus.subscribe("latency_update", self.perf_panel.update_latency)
        event_bus.subscribe("vram_update", self.perf_panel.update_vram)
        event_bus.subscribe("audio_level", self.device_panel.update_levels)
        event_bus.subscribe("model_loaded", self.model_panel.on_model_loaded)
        event_bus.subscribe("error", self._show_error)
        
        # Timer para refrescar dashboard (no bloquea GUI)
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_dashboard)
        self._timer.start(100)    ← 100ms = 10 FPS para dashboard
    
    closeEvent(event):
        event_bus.send_command("shutdown", {})
        event.accept()
    
    _show_error(data):
        QMessageBox.warning(self, "Error", data["message"])
    
    _refresh_dashboard():
        # El timer solo pide datos, no bloquea
        PASAR  ← Los eventos del pipeline actualizan los widgets directamente
```

### Criterio: GUI no freeze > 100ms, cierre limpio con shutdown

---

## 5. F5-T3 / F5-T4 / F5-T5: WIDGETS

### Device Selector (`device_selector.py`)
```
CLASE DeviceSelectorPanel(QWidget):
    Dos QComboBox: input y output
    Botón refresh que llama a devices.get_input_devices() / get_output_devices()
    Al cambiar selección → event_bus.send_command("set_input_device", {index})
    Incluye LevelMeter widget para input y output
```

### Level Meters (`level_meter.py`)
```
CLASE LevelMeter(QWidget):
    Barra horizontal animada que muestra peak level
    Actualiza cada 50ms vía evento "audio_level"
    Colores: verde (normal), amarillo (> -6dB), rojo (> -3dB)
    paintEvent: dibujar barra con QPainter, gradiente de color
```

### Performance Panel (`performance_panel.py`)
```
CLASE PerformancePanel(QWidget):
    4 QProgressBars: Latency, GPU Load, VRAM, Buffer
    Cada una con label de valor numérico
    update_latency(data): actualizar barra + label + color
        Verde: < 30ms, Amarillo: 30-35ms, Rojo: > 35ms
    update_vram(data): similar con umbrales de VRAM
```

---

## 6. F5-T6: MODEL LOADER (`model_loader.py`)

### Pseudocódigo
```
CLASE ModelLoaderPanel(QWidget):
    Muestra: nombre modelo activo, formato, VRAM
    Botón "Load": QFileDialog → seleccionar .onnx/.pth/.trt
    Botón "Swap": hot-swap modelo sin detener pipeline
    
    _on_load_clicked():
        path = QFileDialog.getOpenFileName(filter="Models (*.onnx *.pth *.trt)")
        SI path:
            self._progress.show()
            event_bus.send_command("load_model", {"model_path": path})
    
    on_model_loaded(data):
        """Callback cuando pipeline confirma modelo cargado."""
        self._progress.hide()
        self.name_label.setText(data["name"])
        self.format_label.setText(data["format"])
        self.vram_label.setText(f"{data['vram_mb']:.0f} MB")
```

---

## 7. F5-T7: SYSTEM TRAY (`system_tray.py`)

### Pseudocódigo
```
CLASE SystemTray(QSystemTrayIcon):
    Icono en system tray con context menu:
        - "Show/Hide" → toggle ventana principal
        - "Active / Bypass" → toggle voice changer
        - "Quit" → shutdown completo
    
    Al cerrar ventana → minimizar a tray (no cerrar app)
    Double-click en tray → restaurar ventana
```

---

## 8. REGLA CRÍTICA DE AISLAMIENTO

```
╔═══════════════════════════════════════════════════════╗
║  GUI → Pipeline: SOLO vía EventBus commands           ║
║  Pipeline → GUI: SOLO vía EventBus events             ║
║                                                        ║
║  SI LA GUI CRASHEA:                                    ║
║  → El pipeline SIGUE funcionando                       ║
║  → El audio SIGUE pasando por el cable virtual         ║
║  → Se puede reiniciar la GUI sin reiniciar el pipeline ║
╚═══════════════════════════════════════════════════════╝
```

### Tests de Aislamiento
- Crashear GUI con signal → pipeline sigue procesando audio
- Cerrar y re-abrir ventana → pipeline mantiene estado
- 1000 commands rápidos al EventBus → sin deadlock ni crash

---

## 9. RIESGOS

| Riesgo | Mitigación |
|--------|-----------|
| GUI freeze al cargar modelo | Carga en thread separado vía EventBus |
| QTimer too fast → flickering | 100ms mínimo para dashboard, 50ms para meters |
| QComboBox sin respuesta | Refresh devices en background thread |
| Memory leak en widgets | Usar deleteLater() en widgets dinámicos |
