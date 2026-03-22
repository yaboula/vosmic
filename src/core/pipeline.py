"""VosmicPipeline — end-to-end orchestrator connecting all modules."""

from __future__ import annotations

import logging
import queue
import threading
import time

from src.audio_io.capture import AudioCapture
from src.audio_io.output import AudioOutput
from src.audio_io.ring_buffer import LockFreeRingBuffer
from src.core.config import VosmicConfig
from src.dsp.content_encoder import ContentEncoder
from src.dsp.dsp_thread import DSPThread
from src.dsp.feature_bundle import FeatureBundleAssembler
from src.dsp.noise_gate import NoiseGate
from src.dsp.normalizer import LoudnessNormalizer
from src.dsp.pitch_extractor import PitchExtractor
from src.gui.event_bus import EventBus
from src.inference.inference_core import InferenceEngine
from src.inference.model_manager import ModelManager
from src.inference.scheduler import InferenceResult, InferenceScheduler
from src.postprocessing.post_processor import PostProcessor

logger = logging.getLogger(__name__)


class OutputThread:
    """Takes InferenceResults from the queue, applies postprocessing, writes to output buffer."""

    def __init__(
        self,
        input_queue: queue.Queue[InferenceResult],
        output_buffer: LockFreeRingBuffer,
        post_processor: PostProcessor,
    ) -> None:
        self._input_queue = input_queue
        self._output_buffer = output_buffer
        self._post_processor = post_processor
        self._is_running = False
        self._thread: threading.Thread | None = None
        self._chunks_written: int = 0

    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vosmic-output")
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self) -> None:
        while self._is_running:
            try:
                result = self._input_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            try:
                audio = self._post_processor.process(result)
                self._output_buffer.write(audio)
                self._chunks_written += 1
            except Exception:
                logger.exception("OutputThread: postprocessing failed, forwarding original")
                self._output_buffer.write(result.audio)

    @property
    def metrics(self) -> dict:
        return {
            "chunks_written": self._chunks_written,
            "is_running": self._is_running,
        }


class VosmicPipeline:
    """Master orchestrator — initializes, connects, and manages all modules.

    Startup order: model -> DSP -> output -> capture -> threads.
    Shutdown order: capture -> threads -> output (reverse).
    """

    def __init__(self, config: VosmicConfig) -> None:
        self.config = config
        self.event_bus = EventBus()

        self.input_buffer = LockFreeRingBuffer(config.audio.ring_buffer_capacity)
        self.output_buffer = LockFreeRingBuffer(config.audio.ring_buffer_capacity)

        self.dsp_to_inference_queue: queue.Queue = queue.Queue(
            maxsize=config.inference.queue_max_size
        )
        self.inference_to_post_queue: queue.Queue = queue.Queue(
            maxsize=config.inference.queue_max_size
        )

        self.capture: AudioCapture | None = None
        self.output: AudioOutput | None = None
        self.dsp_thread: DSPThread | None = None
        self.inference_scheduler: InferenceScheduler | None = None
        self.output_thread: OutputThread | None = None
        self.post_processor: PostProcessor | None = None
        self.model_manager: ModelManager | None = None

        self._is_running = False
        self._metrics_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Initialize and start all modules in the correct order."""
        if self._is_running:
            return
        logger.info("Starting VOSMIC pipeline...")

        self.model_manager = ModelManager(device="cuda")
        try:
            self.model_manager.load(
                self.config.inference.model_path,
                precision=self.config.inference.precision,
            )
        except Exception:
            logger.warning("No model loaded at startup, running in passthrough mode")

        noise_gate = NoiseGate(self.config.dsp.noise_gate_threshold_db)
        normalizer = LoudnessNormalizer(self.config.dsp.loudness_target_lufs)
        pitch = PitchExtractor(method=self.config.dsp.pitch_extractor)
        content = ContentEncoder()
        assembler = FeatureBundleAssembler(noise_gate, normalizer, pitch, content)

        engine = InferenceEngine(self.model_manager)

        self.post_processor = PostProcessor(
            self.config.postprocessing, sample_rate=self.config.audio.sample_rate
        )

        self.output = AudioOutput(
            self.output_buffer,
            device_index=self.config.audio.output_device,
            sample_rate=self.config.audio.sample_rate,
            block_size=self.config.audio.block_size,
        )
        self.capture = AudioCapture(
            self.input_buffer,
            device_index=self.config.audio.input_device,
            sample_rate=self.config.audio.sample_rate,
            block_size=self.config.audio.block_size,
        )

        self.dsp_thread = DSPThread(
            self.input_buffer,
            self.dsp_to_inference_queue,
            assembler,
            block_size=self.config.audio.block_size,
        )
        self.inference_scheduler = InferenceScheduler(
            engine,
            self.dsp_to_inference_queue,
            self.inference_to_post_queue,
            max_inference_ms=self.config.inference.max_inference_time_ms,
        )
        self.output_thread = OutputThread(
            self.inference_to_post_queue, self.output_buffer, self.post_processor
        )

        self.output.start()
        self.capture.start()
        self.dsp_thread.start()
        self.inference_scheduler.start()
        self.output_thread.start()

        self._is_running = True
        self._start_metrics_reporter()
        logger.info("VOSMIC pipeline started")

    def stop(self) -> None:
        """Clean shutdown in reverse startup order."""
        if not self._is_running:
            return
        self._is_running = False
        logger.info("Stopping VOSMIC pipeline...")

        if self.capture:
            self.capture.stop()
        if self.dsp_thread:
            self.dsp_thread.stop()
        if self.inference_scheduler:
            self.inference_scheduler.stop()
        if self.output_thread:
            self.output_thread.stop()
        if self.output:
            self.output.stop()

        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=2)
            self._metrics_thread = None

        logger.info("VOSMIC pipeline stopped")

    def _start_metrics_reporter(self) -> None:
        """10 Hz metrics reporter thread emitting to EventBus."""

        def report_loop() -> None:
            while self._is_running:
                try:
                    self.event_bus.emit(
                        "latency_update",
                        {"latency_ms": self._estimate_total_latency()},
                    )
                    if self.model_manager:
                        self.event_bus.emit("vram_update", self.model_manager.get_vram_usage())
                    if self.capture and self.output:
                        self.event_bus.emit(
                            "audio_level",
                            {
                                "input_peak": self.capture.metrics.get("peak_level", 0.0),
                                "output_peak": self.output.metrics.get("output_level", 0.0),
                            },
                        )
                    cap = self.config.audio.ring_buffer_capacity
                    fill = self.input_buffer.available_read / max(cap, 1) * 100
                    self.event_bus.emit("buffer_status", {"fill_percent": fill})
                except Exception:
                    logger.exception("Metrics reporter error")
                time.sleep(0.1)

        self._metrics_thread = threading.Thread(
            target=report_loop, daemon=True, name="vosmic-metrics"
        )
        self._metrics_thread.start()

    def _estimate_total_latency(self) -> float:
        """Sum of measured component latencies."""
        total = 0.0
        if self.dsp_thread:
            total += self.dsp_thread.metrics.get("process_time_ms", 0.0)
        if self.inference_scheduler:
            total += self.inference_scheduler.metrics.get("last_inference_ms", 0.0)
        block_ms = self.config.audio.block_size / self.config.audio.sample_rate * 1000
        total += block_ms * 2
        return total

    def process_commands(self) -> None:
        """Drain and handle all pending GUI commands."""
        cmd = self.event_bus.get_command()
        while cmd is not None:
            try:
                self._handle_command(cmd)
            except Exception:
                logger.exception("Error handling command %s", cmd.get("type"))
            cmd = self.event_bus.get_command()

    def _handle_command(self, cmd: dict) -> None:
        cmd_type = cmd.get("type", "")
        data = cmd.get("data", {})

        if cmd_type == "set_input_device":
            if self.capture:
                self.capture.change_device(data["device_index"])
        elif cmd_type == "set_output_device":
            if self.output:
                self.output.set_output_device(data["device_index"])
        elif cmd_type == "load_model":
            if self.model_manager:
                try:
                    self.model_manager.swap_model(data["model_path"])
                    self.event_bus.emit(
                        "model_loaded", {"name": data["model_path"], "format": "", "vram_mb": 0.0}
                    )
                except Exception as e:
                    self.event_bus.emit("model_load_error", {"error": str(e)})
        elif cmd_type == "set_bypass":
            if self.post_processor:
                self.post_processor.set_bypass(data["enabled"])
        elif cmd_type == "set_noise_gate":
            pass
        elif cmd_type == "shutdown":
            self.stop()

    def get_full_metrics(self) -> dict:
        """Aggregate metrics from all modules."""
        m: dict = {"is_running": self._is_running}
        if self.capture:
            m["capture"] = self.capture.metrics
        if self.output:
            m["output"] = self.output.metrics
        if self.dsp_thread:
            m["dsp"] = self.dsp_thread.metrics
        if self.inference_scheduler:
            m["inference"] = self.inference_scheduler.metrics
        if self.output_thread:
            m["output_thread"] = self.output_thread.metrics
        m["estimated_latency_ms"] = self._estimate_total_latency()
        return m
