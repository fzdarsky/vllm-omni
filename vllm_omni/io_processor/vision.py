# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VisionIOProcessor for V-JEPA video ingest.

This processor runs GStreamer in a separate process to avoid Python GIL
contention, delivering frames to Stage 0 via OmniConnector.

Architecture:
    VisionIOProcessor (separate process)
        └── GStreamer Pipeline (decode, composite, scale)
                └── OmniConnector.put() → Stage 0 FrameBuffer

The processor handles:
- Multi-source video input (file, RTSP, V4L2, CSI)
- Hardware-accelerated decode (NVDEC, VA-API, software fallback)
- Multi-camera compositing
- Frame delivery with backpressure
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.io_processor.gstreamer import (
    GST_AVAILABLE,
    GStreamerPipeline,
    HardwareDetector,
    PipelineConfig,
    SourceConfig,
)

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase

logger = logging.getLogger(__name__)


@dataclass
class VisionProcessorConfig:
    """Configuration for VisionIOProcessor."""
    sources: list[dict] = field(default_factory=list)
    target_width: int = 256
    target_height: int = 256
    target_fps: int = 30
    max_buffers: int = 2
    ntp_server: str | None = None
    # Connector routing
    from_stage: str = "io_processor"
    to_stage: str = "stage_0"


class VisionIOProcessor:
    """Vision IOProcessor using GStreamer for video ingest.

    Runs GStreamer in the current process, delivering frames via
    OmniConnector to Stage 0. For GIL isolation, spawn this in
    a separate process using VisionIOProcessor.spawn().

    Attributes:
        config: Processor configuration
        hw_target: Detected hardware acceleration target
    """

    def __init__(self, config: VisionProcessorConfig):
        self.config = config
        self._pipeline: GStreamerPipeline | None = None
        self._connector: OmniConnectorBase | None = None
        self._frame_count: int = 0
        self._session_id: str = ""

        if GST_AVAILABLE and HardwareDetector:
            self.hw_target = HardwareDetector.detect()
            logger.info(f"VisionIOProcessor hardware target: {self.hw_target.value}")
        else:
            self.hw_target = None
            logger.warning("GStreamer not available")

    def configure(
        self,
        connector: OmniConnectorBase,
        session_id: str,
    ) -> None:
        """Configure the processor with connector and session.

        Args:
            connector: OmniConnector for frame delivery
            session_id: Session identifier for request keys
        """
        self._connector = connector
        self._session_id = session_id

        # Build source configs
        source_configs = []
        for src in self.config.sources:
            source_configs.append(SourceConfig(
                type=src.get("type", "file"),
                device=src.get("device"),
                uri=src.get("uri"),
                sensor_id=src.get("sensor_id"),
                path=src.get("path"),
            ))

        # Build pipeline config
        pipeline_config = PipelineConfig(
            sources=source_configs,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            target_fps=self.config.target_fps,
            max_buffers=self.config.max_buffers,
            ntp_server=self.config.ntp_server,
        )

        # Create pipeline with frame callback
        self._pipeline = GStreamerPipeline(
            config=pipeline_config,
            on_frame_callback=self._on_frame,
        )
        self._pipeline.build()

    def _on_frame(
        self,
        tensor: torch.Tensor,
        pts_ns: int,
        timing_meta: dict | None,
    ) -> None:
        """Callback when GStreamer delivers a frame.

        Sends frame to Stage 0 via OmniConnector.
        """
        if self._connector is None:
            logger.warning("No connector configured, dropping frame")
            return

        self._frame_count += 1
        request_key = f"{self._session_id}_frame_{self._frame_count}"

        # Package frame data for Stage 0
        frame_data = {
            "tensor": tensor,
            "pts_ns": pts_ns,
            "frame_idx": self._frame_count,
            "timing": timing_meta,
        }

        # Send via connector (blocks if Stage 0 is slow → backpressure)
        success, size, meta = self._connector.put(
            from_stage=self.config.from_stage,
            to_stage=self.config.to_stage,
            put_key=request_key,
            data=frame_data,
        )

        if not success:
            logger.warning(f"Failed to deliver frame {self._frame_count}")

    def start(self, blocking: bool = False) -> None:
        """Start the GStreamer pipeline.

        Args:
            blocking: If True, block until pipeline stops.
        """
        if self._pipeline is None:
            raise RuntimeError("Processor not configured. Call configure() first.")

        self._pipeline.start(blocking=blocking)
        logger.info(f"VisionIOProcessor started (session={self._session_id})")

    def stop(self) -> None:
        """Stop the GStreamer pipeline."""
        if self._pipeline:
            self._pipeline.stop()
        logger.info(f"VisionIOProcessor stopped (frames={self._frame_count})")

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._pipeline.is_running if self._pipeline else False

    @property
    def frames_processed(self) -> int:
        """Number of frames processed."""
        return self._frame_count

    # =========================================================================
    # Separate Process Support
    # =========================================================================

    @staticmethod
    def _run_in_process(
        config_dict: dict,
        connector_config: dict,
        session_id: str,
        ready_event: mp.Event,
        stop_event: mp.Event,
    ) -> None:
        """Entry point for subprocess.

        Creates connector and processor, runs until stop_event is set.
        """
        import sys
        import traceback

        try:
            from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
            from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

            # Recreate config and connector in subprocess
            config = VisionProcessorConfig(**config_dict)

            # Create connector from config
            spec = ConnectorSpec(
                name=connector_config["name"],
                extra=connector_config.get("extra", {}),
            )
            connector = OmniConnectorFactory.create_connector(spec)

            # Create and configure processor
            processor = VisionIOProcessor(config)
            processor.configure(connector, session_id)

            # Signal ready
            ready_event.set()

            # Start pipeline (non-blocking)
            processor.start(blocking=False)

            # Wait for stop signal
            while not stop_event.is_set() and processor.is_running:
                time.sleep(0.1)

        except Exception as e:
            # Log to stderr so it's visible in container logs
            print(f"VisionIOProcessor subprocess error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            # Don't set ready_event - parent will timeout and raise

        processor.stop()
        connector.close()

    @classmethod
    def spawn(
        cls,
        config: VisionProcessorConfig,
        connector_config: dict,
        session_id: str,
    ) -> "VisionProcessorHandle":
        """Spawn processor in a separate process.

        Uses 'spawn' start method to avoid CUDA re-initialization issues
        when the parent process has already initialized CUDA.

        Args:
            config: Processor configuration
            connector_config: OmniConnector configuration dict
            session_id: Session identifier

        Returns:
            Handle for controlling the subprocess
        """
        # Use spawn context to avoid "Cannot re-initialize CUDA in forked subprocess"
        ctx = mp.get_context("spawn")
        ready_event = ctx.Event()
        stop_event = ctx.Event()

        config_dict = {
            "sources": config.sources,
            "target_width": config.target_width,
            "target_height": config.target_height,
            "target_fps": config.target_fps,
            "max_buffers": config.max_buffers,
            "ntp_server": config.ntp_server,
            "from_stage": config.from_stage,
            "to_stage": config.to_stage,
        }

        process = ctx.Process(
            target=cls._run_in_process,
            args=(config_dict, connector_config, session_id, ready_event, stop_event),
            name=f"VisionIOProcessor-{session_id}",
            daemon=True,
        )
        process.start()

        # Wait for processor to be ready
        # Spawn context reimports all modules (~11s for vllm/torch/gstreamer)
        if not ready_event.wait(timeout=30.0):
            process.terminate()
            raise RuntimeError("VisionIOProcessor failed to start")

        return VisionProcessorHandle(process, stop_event, session_id)


@dataclass
class VisionProcessorHandle:
    """Handle for controlling a VisionIOProcessor subprocess."""
    process: mp.Process
    stop_event: mp.Event
    session_id: str
    worker_id: int | None = None  # Set when using VisionWorkerPool

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the session gracefully.

        For pooled workers (worker_id is set), this stops the current session
        but keeps the worker alive for reuse. For non-pooled workers, this
        terminates the subprocess entirely.
        """
        self.stop_event.set()

        # For pooled workers, don't join/terminate - worker stays alive for next session
        if self.worker_id is not None:
            # Just give the worker a moment to acknowledge the stop
            time.sleep(0.1)
            return

        # For non-pooled workers, wait for process to exit
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            logger.warning(f"Force-terminating VisionIOProcessor {self.session_id}")
            self.process.terminate()

    @property
    def is_alive(self) -> bool:
        """Check if subprocess is running."""
        return self.process.is_alive()


class VisionWorkerPool:
    """Pre-warmed subprocess pool for VisionIOProcessor.

    Spawns subprocesses at startup that do all heavy imports (torch, vllm,
    gstreamer). When a session starts, configuration is sent via queue
    and the subprocess configures its GStreamer pipeline.

    This avoids the ~11s import latency on first user request.
    """

    def __init__(self, pool_size: int = 1):
        self.pool_size = pool_size
        self._ctx = mp.get_context("spawn")
        self._workers: list[_PrewarmedWorker] = []
        self._available: mp.Queue = self._ctx.Queue()
        self._started = False

    def start(self) -> None:
        """Spawn pre-warmed workers. Call at server startup."""
        if self._started:
            return

        logger.info(f"Starting VisionWorkerPool with {self.pool_size} workers")

        for i in range(self.pool_size):
            worker = _PrewarmedWorker(self._ctx, i)
            worker.start()
            self._workers.append(worker)
            self._available.put(i)

        # Wait for all workers to be warm
        for worker in self._workers:
            if not worker.wait_warm(timeout=30.0):
                raise RuntimeError(f"Worker {worker.worker_id} failed to warm up")

        self._started = True
        logger.info(f"VisionWorkerPool ready: {self.pool_size} warm workers")

    def assign_session(
        self,
        config: VisionProcessorConfig,
        connector_config: dict,
        session_id: str,
        timeout: float = 5.0,
    ) -> VisionProcessorHandle:
        """Assign a warm worker to handle a session.

        Returns immediately since worker is already warm.
        """
        if not self._started:
            raise RuntimeError("Worker pool not started")

        try:
            worker_id = self._available.get(timeout=timeout)
        except Exception:
            raise RuntimeError("No available workers in pool")

        worker = self._workers[worker_id]

        # Respawn if worker process died (e.g., native crash in GStreamer)
        if not worker.process.is_alive():
            logger.warning("Worker %d is dead, respawning", worker_id)
            worker.stop()
            worker = _PrewarmedWorker(self._ctx, worker_id)
            worker.start()
            if not worker.wait_warm(timeout=30.0):
                self._available.put(worker_id)
                raise RuntimeError(f"Worker {worker_id} failed to respawn")
            self._workers[worker_id] = worker

        try:
            handle = worker.configure_session(config, connector_config, session_id)
        except Exception:
            self._available.put(worker_id)
            raise
        return handle

    def return_worker(self, worker_id: int) -> None:
        """Return a worker to the available pool."""
        self._available.put(worker_id)

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self._workers:
            worker.stop()
        self._workers.clear()
        self._started = False


@dataclass
class _PrewarmedWorker:
    """A pre-warmed subprocess that waits for session configuration."""

    ctx: Any
    worker_id: int
    process: mp.Process = None
    config_queue: mp.Queue = None
    ready_event: Any = None
    warm_event: Any = None
    stop_event: Any = None  # Kills worker entirely (pool shutdown)
    session_stop_event: Any = None  # Stops current session, keeps worker alive
    _current_session: str = None

    def __post_init__(self):
        self.config_queue = self.ctx.Queue()
        self.ready_event = self.ctx.Event()
        self.warm_event = self.ctx.Event()
        self.stop_event = self.ctx.Event()
        self.session_stop_event = self.ctx.Event()

    def start(self) -> None:
        """Start the subprocess."""
        self.process = self.ctx.Process(
            target=self._worker_main,
            args=(
                self.worker_id,
                self.config_queue,
                self.ready_event,
                self.warm_event,
                self.stop_event,
                self.session_stop_event,
            ),
            name=f"VisionWorker-{self.worker_id}",
            daemon=True,
        )
        self.process.start()

    def wait_warm(self, timeout: float = 30.0) -> bool:
        """Wait for worker to complete imports."""
        return self.warm_event.wait(timeout=timeout)

    def configure_session(
        self,
        config: VisionProcessorConfig,
        connector_config: dict,
        session_id: str,
    ) -> VisionProcessorHandle:
        """Send configuration to worker and wait for pipeline to start."""
        self._current_session = session_id
        self.ready_event.clear()
        self.session_stop_event.clear()  # Reset for new session

        config_dict = {
            "sources": config.sources,
            "target_width": config.target_width,
            "target_height": config.target_height,
            "target_fps": config.target_fps,
            "max_buffers": config.max_buffers,
            "ntp_server": config.ntp_server,
            "from_stage": config.from_stage,
            "to_stage": config.to_stage,
        }

        self.config_queue.put({
            "config": config_dict,
            "connector_config": connector_config,
            "session_id": session_id,
        })

        # Pipeline configuration is fast (<1s) since imports are done
        if not self.ready_event.wait(timeout=5.0):
            raise RuntimeError(f"Worker {self.worker_id} failed to configure session")

        # For pooled workers, use session_stop_event (stops session, keeps worker)
        return VisionProcessorHandle(
            process=self.process,
            stop_event=self.session_stop_event,  # Use session event, not worker event
            session_id=session_id,
            worker_id=self.worker_id,
        )

    def stop(self) -> None:
        """Stop the worker."""
        self.stop_event.set()
        if self.process:
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()

    @staticmethod
    def _worker_main(
        worker_id: int,
        config_queue: mp.Queue,
        ready_event: mp.Event,
        warm_event: mp.Event,
        stop_event: mp.Event,
        session_stop_event: mp.Event,
    ) -> None:
        """Worker subprocess entry point."""
        import sys
        import traceback

        try:
            # Heavy imports - done once at startup
            from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
            from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

            # Signal that imports are complete
            warm_event.set()
            logger.info(f"VisionWorker-{worker_id} warm and ready")

            processor = None
            connector = None

            while not stop_event.is_set():
                try:
                    # Wait for session configuration
                    try:
                        msg = config_queue.get(timeout=0.5)
                    except Exception:
                        continue

                    # Clean up previous session if any
                    if processor is not None:
                        processor.stop()
                    if connector is not None:
                        connector.close()

                    # Configure new session
                    config = VisionProcessorConfig(**msg["config"])
                    spec = ConnectorSpec(
                        name=msg["connector_config"]["name"],
                        extra=msg["connector_config"].get("extra", {}),
                    )
                    connector = OmniConnectorFactory.create_connector(spec)

                    processor = VisionIOProcessor(config)
                    processor.configure(connector, msg["session_id"])

                    # Signal ready
                    ready_event.set()

                    # Start pipeline
                    processor.start(blocking=False)

                    # Run until session_stop requested or pipeline ends
                    # Note: session_stop_event stops current session but keeps worker alive
                    while not stop_event.is_set() and not session_stop_event.is_set() and processor.is_running:
                        time.sleep(0.1)

                    # Pipeline finished (EOS) but consumer may still be reading
                    # SHM segments. Write EOF sentinel so consumer knows no
                    # more frames are coming, then wait before cleanup.
                    if not session_stop_event.is_set() and not stop_event.is_set():
                        total_frames = processor.frames_processed
                        eof_key = f"{msg['session_id']}_frame_eof"
                        eof_data = {"eof": True, "total_frames": total_frames}
                        if connector is not None:
                            try:
                                connector.put(
                                    from_stage="io_processor",
                                    to_stage="stage_0",
                                    put_key=eof_key,
                                    data=eof_data,
                                )
                                logger.info(
                                    "VisionWorker-%d: wrote EOF sentinel after %d frames",
                                    worker_id, total_frames,
                                )
                            except Exception as e:
                                logger.warning(
                                    "VisionWorker-%d: failed to write EOF: %s",
                                    worker_id, e,
                                )

                        processor.stop()
                        processor = None
                        # Keep connector alive — consumer still reading SHM
                        session_stop_event.wait(timeout=300)

                    # Clean up current session
                    if processor is not None:
                        processor.stop()
                        processor = None
                    if connector is not None:
                        connector.close()
                        connector = None

                except Exception as e:
                    print(f"VisionWorker-{worker_id} error: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()

            # Final cleanup (when worker is shutting down)
            if processor is not None:
                processor.stop()
            if connector is not None:
                connector.close()

        except Exception as e:
            print(f"VisionWorker-{worker_id} fatal error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
