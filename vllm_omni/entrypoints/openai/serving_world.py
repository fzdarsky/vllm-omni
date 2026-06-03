# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""World Model serving handler for V-JEPA.

Provides session-based inference for V-JEPA world models with support for:
- Long-running video inference sessions
- Reconnection to active sessions
- SSE streaming of predictions
- Multiple video source types (file, RTSP, V4L2, CSI)

Architecture:
    Session creation spawns VisionIOProcessor in a separate process.
    VisionIOProcessor uses GStreamer for HW-accelerated video decode
    and delivers frames via OmniConnector to Stage 0 (VJepa2Model).
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import numpy as np
import torch

from vllm_omni.entrypoints.openai.protocol.world import (
    PredictionLabel,
    SessionStatus,
    SourceConfigRequest,
    SourceType,
    WorldHeadInfo,
    WorldHeadsResponse,
    WorldPrediction,
    WorldPredictionEvent,
    WorldSessionConfig,
    WorldSessionCreateRequest,
    WorldSessionCreateResponse,
    WorldSessionDeleteResponse,
    WorldSessionListResponse,
    WorldSessionStatusResponse,
    WorldUploadResponse,
)

if TYPE_CHECKING:
    from vllm_omni.io_processor import MoQSession, VisionProcessorHandle, VisionWorkerPool

from vllm_omni.entrypoints.openai.telemetry import (
    get_tracer,
    init_world_telemetry,
    record_span,
    trace_span,
)

logger = logging.getLogger(__name__)


class TracingHooks:
    """Manages OTel spans for PyTorch module forward passes.

    Uses paired pre/post hooks to create spans that accurately measure
    submodule execution time, including GPU synchronization.
    Ported from vjepa2-demo/app/model.py.
    """

    def __init__(self, device: str):
        self.device = device
        self._active_spans: dict[int, Any] = {}
        self._tracer = get_tracer()

    def _sync_device(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _make_pre_hook(self, span_name: str):
        def hook(module: Any, args: tuple) -> None:
            self._sync_device()
            span = self._tracer.start_span(span_name)
            self._active_spans[id(module)] = span
        return hook

    def _make_post_hook(self, span_name: str):
        def hook(module: Any, args: tuple, output: Any) -> None:
            self._sync_device()
            span = self._active_spans.pop(id(module), None)
            if span is not None:
                span.end()
        return hook

    def register(self, module: Any, span_name: str) -> None:
        module.register_forward_pre_hook(self._make_pre_hook(span_name))
        module.register_forward_hook(self._make_post_hook(span_name))


# Check GStreamer availability
try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False
    logger.warning("GStreamer not available - video streaming disabled")


@dataclass
class SourceConfig:
    """Internal source configuration for GStreamer pipeline."""

    type: str
    path: str | None = None
    uri: str | None = None
    device: str | None = None
    sensor_id: int | None = None


@dataclass
class Session:
    """Represents an active inference session.

    Manages the lifecycle of video processing and model inference,
    providing predictions via an async queue for SSE delivery.
    """

    session_id: str
    model_name: str
    source: SourceConfigRequest
    config: WorldSessionConfig
    head: str | None = None
    status: SessionStatus = SessionStatus.CREATED
    frames_processed: int = 0
    predictions_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    error_message: str | None = None

    # Internal state (not serialized)
    _prediction_queue: asyncio.Queue | None = field(default=None, repr=False)
    _vision_processor: "VisionProcessorHandle | None" = field(default=None, repr=False)
    _moq_session: "MoQSession | None" = field(default=None, repr=False)
    _gst_pipeline: Any = field(default=None, repr=False)
    # Frame consumer state
    _frame_connector: Any = field(default=None, repr=False)
    _frame_consumer_task: asyncio.Task | None = field(default=None, repr=False)
    _stop_consumer: asyncio.Event | None = field(default=None, repr=False)
    # Pool worker tracking (for returning to pool on session end)
    _pool_worker_id: int | None = field(default=None, repr=False)

    def to_status_response(self) -> WorldSessionStatusResponse:
        """Convert to API response."""
        return WorldSessionStatusResponse(
            session_id=self.session_id,
            status=self.status,
            model=self.model_name,
            config=self.config,
            source=self.source,
            frames_processed=self.frames_processed,
            predictions_count=self.predictions_count,
            created_at=self.created_at,
            started_at=self.started_at,
            error_message=self.error_message,
        )


class OmniOpenAIServingWorld:
    """World model serving handler for V-JEPA.

    Manages inference sessions with support for:
    - Session creation and lifecycle management
    - Video source configuration (file, RTSP, V4L2, CSI)
    - SSE streaming of predictions
    - Reconnection to running sessions

    This handler integrates with vLLM-Omni's serving infrastructure
    and can be used alongside other modality handlers (chat, speech, etc.).
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str,
        stage_configs: list[Any] | None = None,
        storage_path: str = "/tmp/storage",
        heads_config: list[dict] | None = None,
    ):
        """Initialize world model serving handler.

        Args:
            engine_client: vLLM-Omni engine client (AsyncOmni)
            model_name: Model identifier for this server
            stage_configs: Pipeline stage configurations
            storage_path: Directory for storing uploaded video files
            heads_config: Optional list of head configs from deploy YAML
        """
        self.engine_client = engine_client
        self.model_name = model_name
        self.stage_configs = stage_configs
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._gc_task: asyncio.Task | None = None

        # Load V-JEPA model directly for inference
        # TODO: Integrate with vLLM-Omni multi-stage pipeline for production
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_vjepa_model(model_name)

        # Multi-head registry (name → ClassificationHead)
        self._heads: dict[str, Any] = {}
        self._default_head: str | None = None
        self._load_heads(heads_config or [])

        # Initialize OpenTelemetry for World Model traces
        init_world_telemetry()

        # Pre-warm vision worker pool (avoids ~11s import latency on first request)
        self._vision_pool: VisionWorkerPool | None = None
        if GST_AVAILABLE:
            from vllm_omni.io_processor import VisionWorkerPool
            self._vision_pool = VisionWorkerPool(pool_size=1)
            self._vision_pool.start()

        # Start TTL-based session GC (RFC #3745-compatible)
        self._gc_task = asyncio.ensure_future(self._session_gc_loop())

        logger.info(
            "World model serving initialized: model=%s, gstreamer=%s, storage=%s, device=%s",
            model_name,
            GST_AVAILABLE,
            self.storage_path,
            self._device,
        )

    def _load_vjepa_model(self, model_name: str) -> None:
        """Load V-JEPA model for direct inference.

        For the PoC, this loads the HuggingFace model directly. In production,
        this should integrate with vLLM-Omni's multi-stage pipeline.
        """
        logger.info("_load_vjepa_model called: model=%s, device=%s", model_name, self._device)

        try:
            from transformers import AutoModelForVideoClassification, AutoVideoProcessor

            logger.info("Loading V-JEPA model: %s on %s", model_name, self._device)

            # V-JEPA2 requires video-specific HuggingFace classes
            self._model = AutoModelForVideoClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(self._device)
            # Set model to evaluation mode
            self._model.train(False)

            # Load video processor for preprocessing
            self._processor = AutoVideoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            logger.info(
                "V-JEPA model loaded successfully: %s (%d labels)",
                type(self._model).__name__,
                getattr(self._model.config, "num_labels", "unknown"),
            )

            # Register tracing hooks on model submodules
            self._tracing_hooks = TracingHooks(self._device)
            hook_config = [
                ("vjepa2.encoder", "jepa_encode"),
                ("vjepa2.predictor", "jepa_predict"),
                ("pooler", "jepa_pool"),
            ]
            for module_path, span_name in hook_config:
                try:
                    module = self._model
                    for part in module_path.split("."):
                        module = getattr(module, part)
                    self._tracing_hooks.register(module, span_name)
                    logger.info("Registered tracing hook: %s → %s", module_path, span_name)
                except AttributeError:
                    logger.debug("Submodule %s not found, skipping hook", module_path)

            # Warmup: run a dummy forward pass to trigger CUDA kernel JIT
            # This moves the ~2s first-inference penalty to server startup
            self._warmup_model()

        except ImportError as e:
            logger.warning("Cannot import transformers for V-JEPA model: %s", e)
            self._model = None
            self._processor = None
        except Exception as e:
            logger.exception("Failed to load V-JEPA model: %s", e)
            self._model = None
            self._processor = None

    def _warmup_model(self) -> None:
        """Run a dummy forward pass to trigger CUDA kernel compilation.

        Moves the ~2s first-inference penalty from the first user request
        to server startup time where it's not user-visible.
        """
        if self._model is None or self._processor is None:
            return

        try:
            t0 = time.time()
            dummy_frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * 16
            inputs = self._processor(dummy_frames, return_tensors="pt")
            pixel_key = (
                "pixel_values_videos"
                if "pixel_values_videos" in inputs
                else "pixel_values"
            )
            pixel_values = inputs[pixel_key].to(self._device)
            with torch.no_grad():
                self._model(pixel_values)
            if self._device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            logger.info("Model warmup complete in %.1fs", elapsed)
        except Exception as e:
            logger.warning("Model warmup failed (non-fatal): %s", e)

    def _load_heads(self, heads_config: list[dict]) -> None:
        """Load classification heads from config.

        Extracts the default head from the loaded model and optionally
        loads additional heads from checkpoint files.
        """
        if self._model is None:
            return

        from vllm_omni.model_executor.models.vjepa.encoder import ClassificationHead

        # Find the name for the model's built-in head: use the YAML-declared
        # default (if it has no separate checkpoint), otherwise "default".
        builtin_name = "default"
        for hcfg in heads_config:
            if hcfg.get("default", False) and "checkpoint" not in hcfg:
                builtin_name = hcfg.get("name", "default")
                break

        default_head = ClassificationHead.from_hf_model(self._model, name=builtin_name)
        self._heads[builtin_name] = default_head
        self._default_head = builtin_name

        backbone_hidden = self._model.config.hidden_size

        for hcfg in heads_config:
            name = hcfg.get("name", "")
            if not name:
                continue

            if hcfg.get("default", False) and "checkpoint" not in hcfg:
                # Already registered as the builtin head above
                continue

            checkpoint = hcfg.get("checkpoint")
            if not checkpoint:
                continue

            try:
                import json
                from copy import deepcopy
                from pathlib import Path

                import torch.nn as nn

                state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

                num_labels = hcfg.get("num_labels")
                if num_labels is None:
                    classifier_weight = state_dict.get("classifier.weight")
                    num_labels = classifier_weight.shape[0] if classifier_weight is not None else 0

                # Load label map: explicit path > head_meta.json > empty
                label_map: dict[int, str] = {}
                labels_path = hcfg.get("labels")
                if labels_path and Path(labels_path).exists():
                    raw = json.loads(Path(labels_path).read_text())
                    label_map = {int(k): v for k, v in raw.items()}
                elif not labels_path:
                    meta_path = Path(checkpoint).parent / "head_meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        id2label = meta.get("id2label", {})
                        label_map = {int(k): v for k, v in id2label.items()}

                pooler = deepcopy(self._model.pooler)
                classifier = nn.Linear(backbone_hidden, num_labels)

                pooler_state = {
                    k.removeprefix("pooler."): v
                    for k, v in state_dict.items()
                    if k.startswith("pooler.")
                }
                classifier_state = {
                    k.removeprefix("classifier."): v
                    for k, v in state_dict.items()
                    if k.startswith("classifier.")
                }

                pooler.load_state_dict(pooler_state)
                classifier.load_state_dict(classifier_state)

                device = torch.device(self._device)
                dtype = torch.float16 if self._device == "cuda" else torch.float32
                pooler.to(device=device, dtype=dtype)
                classifier.to(device=device, dtype=dtype)

                head = ClassificationHead(
                    pooler=pooler,
                    classifier=classifier,
                    hidden_size=backbone_hidden,
                    num_labels=num_labels,
                    label_map=label_map,
                    name=name,
                )
                self._heads[name] = head
                if hcfg.get("default", False):
                    self._default_head = name
                head_params = sum(p.numel() for p in pooler.parameters()) + sum(p.numel() for p in classifier.parameters())
                logger.info(
                    "Loaded head '%s': num_labels=%d, params=%s",
                    name, num_labels, f"{head_params:,}",
                )
            except Exception:
                logger.exception("Failed to load head '%s' from %s", name, checkpoint)

        total_head_params = sum(
            sum(p.numel() for p in h.parameters())
            for h in self._heads.values()
        )
        logger.info(
            "Heads loaded: %d heads, %s total params, default='%s'",
            len(self._heads), f"{total_head_params:,}", self._default_head,
        )

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: Any,
        model_name: str,
        stage_configs: list[Any] | None = None,
    ) -> "OmniOpenAIServingWorld":
        """Create handler for diffusion mode (compatibility layer)."""
        return cls(
            engine_client=diffusion_engine,
            model_name=model_name,
            stage_configs=stage_configs,
        )

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"sess_{secrets.token_urlsafe(16)}"

    async def _session_gc_loop(self) -> None:
        """Periodically close expired sessions based on TTL."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired: list[str] = []
            for sid, session in self._sessions.items():
                ttl = session.config.ttl_seconds
                if ttl > 0 and (now - session.created_at) > ttl:
                    expired.append(sid)
            for sid in expired:
                try:
                    await self.close_session(sid)
                    logger.info("GC: closed expired session %s", sid)
                except Exception:
                    logger.exception("GC: failed to close session %s", sid)

    async def open_session(
        self,
        request: WorldSessionCreateRequest,
    ) -> WorldSessionCreateResponse:
        """Create a new inference session.

        Args:
            request: Session configuration including source and model settings.

        Returns:
            Response with session ID and initial status.
            For MOQ sources, includes moq_endpoint for WebTransport connection.
        """
        session_id = self._generate_session_id()

        session = Session(
            session_id=session_id,
            model_name=self.model_name,
            source=request.source,
            config=request.config,
            head=request.head,
            status=SessionStatus.CREATED,
            _prediction_queue=asyncio.Queue(maxsize=100),
        )

        moq_relay_url: str | None = None
        moq_broadcast_path: str | None = None

        if request.source.type == SourceType.MOQ:
            moq_relay_url = os.environ.get("MOQ_RELAY_URL", "https://localhost:4443")
            moq_broadcast_path = f"{session_id}.hang"
            logger.info(
                "MOQ session configured: relay=%s, broadcast=%s",
                moq_relay_url, moq_broadcast_path,
            )

        async with self._lock:
            self._sessions[session_id] = session

        logger.info("Created world session: %s", session_id)

        return WorldSessionCreateResponse(
            session_id=session_id,
            status=session.status,
            moq_relay_url=moq_relay_url,
            moq_broadcast_path=moq_broadcast_path,
        )

    async def get_session(
        self,
        session_id: str,
    ) -> WorldSessionStatusResponse | None:
        """Get session status by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session status response, or None if not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.to_status_response()

    async def start_session(self, session_id: str) -> Session:
        """Start processing for a session.

        For MOQ sources: MoQ server decodes H.264 frames directly and pushes
        to connector (no GStreamer subprocess needed).

        For other sources: Spawns VisionIOProcessor in a separate process to
        handle GStreamer video decode and frame delivery via OmniConnector.

        Args:
            session_id: Session to start.

        Returns:
            Updated session with RUNNING status.

        Raises:
            KeyError: If session not found.
            RuntimeError: If GStreamer not available or start fails.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session.status not in (SessionStatus.CREATED, SessionStatus.STOPPED):
            raise RuntimeError(f"Cannot start session in {session.status} state")

        session.status = SessionStatus.STARTING
        session.started_at = time.time()

        # Connector config for SharedMemoryConnector
        connector_config = {
            "name": "SharedMemoryConnector",
            "extra": {
                "shm_threshold_bytes": 65536,
                "connector_get_sleep_s": 0.01,
                "connector_get_max_wait": 500,
            },
        }

        try:
            # MOQ sources use direct frame delivery (no GStreamer subprocess)
            if session.source.type == SourceType.MOQ:
                await self._start_moq_session(session, connector_config)
            else:
                # Other sources use GStreamer subprocess
                if not GST_AVAILABLE:
                    raise RuntimeError("GStreamer not available")
                await self._start_gstreamer_session(session, connector_config)

            session.status = SessionStatus.RUNNING
            logger.info("Started world session: %s", session_id)

        except Exception as e:
            session.status = SessionStatus.ERROR
            session.error_message = str(e)
            logger.exception("Failed to start session %s: %s", session_id, e)
            raise

        return session

    async def _start_moq_session(
        self, session: Session, connector_config: dict
    ) -> None:
        """Start a MoQ-based session using GStreamer moqsrc.

        Subscribes to the MoQ relay broadcast matching this session's ID.
        The browser publishes camera frames via @moq/publish to the same
        broadcast path. GStreamer decodes the H.264 stream and delivers
        raw frames via appsink → connector → FrameBuffer → inference.
        """
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

        session_id = session.session_id
        relay_url = os.environ.get("MOQ_RELAY_URL", "https://localhost:4443")
        broadcast_path = f"{session_id}.hang"

        connector = OmniConnectorFactory.create_connector(
            ConnectorSpec(
                name="SharedMemoryConnector",
                extra=connector_config["extra"],
            )
        )
        session._frame_connector = connector

        frame_count = [0]

        def on_new_sample(appsink):
            """GStreamer appsink callback: extract decoded frame and push to connector."""
            receive_start = time.perf_counter_ns()
            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            width = struct.get_value("width")
            height = struct.get_value("height")

            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            try:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1)

                frame_count[0] += 1
                frame_idx = frame_count[0]
                pts_ns = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else 0

                frame_data = {
                    "tensor": tensor,
                    "pts_ns": pts_ns,
                    "frame_idx": frame_idx,
                    "timing": {
                        "input_receive_start_ns": receive_start,
                        "input_receive_end_ns": time.perf_counter_ns(),
                        "hw_target": "moq-gst",
                        "tensor_shape": str(tensor.shape),
                    },
                }

                request_key = f"{session_id}_frame_{frame_idx}"
                success_put, size, meta = connector.put(
                    from_stage="io_processor",
                    to_stage="stage_0",
                    put_key=request_key,
                    data=frame_data,
                )

                if frame_idx % 30 == 0:
                    logger.debug("MOQ: Delivered frame %d", frame_idx)

            finally:
                buf.unmap(mapinfo)

            return Gst.FlowReturn.OK

        if not GST_AVAILABLE:
            raise RuntimeError(
                "GStreamer not available — required for MoQ camera streaming"
            )

        from gi.repository import Gst

        pipeline_str = (
            f'moqsrc url="{relay_url}" broadcast="{broadcast_path}" tls-disable-verify=true '
            f"! h264parse ! nvh264dec ! cudadownload "
            f"! videoconvert "
            f"! videoscale "
            f"! video/x-raw,format=RGB,width=256,height=256 "
            f"! appsink name=sink emit-signals=true sync=false"
        )

        logger.info("MOQ GStreamer pipeline: %s", pipeline_str)
        pipeline = Gst.parse_launch(pipeline_str)

        appsink = pipeline.get_by_name("sink")
        appsink.connect("new-sample", on_new_sample)

        session._gst_pipeline = pipeline

        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def _on_bus_message(_bus, msg):
            t = msg.type
            if t == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error("MOQ GStreamer error: %s  debug=%s", err.message, debug)
            elif t == Gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                logger.warning("MOQ GStreamer warning: %s  debug=%s", err.message, debug)
            elif t == Gst.MessageType.STATE_CHANGED:
                if msg.src == pipeline:
                    old, new, _pending = msg.parse_state_changed()
                    logger.info("MOQ pipeline state: %s → %s", old.value_nick, new.value_nick)
            elif t == Gst.MessageType.EOS:
                logger.info("MOQ pipeline reached EOS")

        bus.connect("message", _on_bus_message)

        pipeline.set_state(Gst.State.PLAYING)

        logger.info(
            "MOQ session started: relay=%s, broadcast=%s",
            relay_url, broadcast_path,
        )

        session._stop_consumer = asyncio.Event()

        async def _pump_glib_context():
            from gi.repository import GLib
            ctx = GLib.MainContext.default()
            while not session._stop_consumer.is_set():
                while ctx.iteration(False):
                    pass
                await asyncio.sleep(0.01)

        session._glib_pump_task = asyncio.create_task(
            _pump_glib_context(),
            name=f"glib-pump-{session_id}",
        )
        session._frame_consumer_task = asyncio.create_task(
            self._consume_frames(session),
            name=f"frame-consumer-{session_id}",
        )

    async def _start_gstreamer_session(
        self, session: Session, connector_config: dict
    ) -> None:
        """Start a GStreamer-based session for non-MOQ sources."""
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
        from vllm_omni.io_processor import VisionIOProcessor, VisionProcessorConfig

        session_id = session.session_id

        # Build source config from session source
        source_dict = self._source_to_dict(session.source)

        # Create processor config
        processor_config = VisionProcessorConfig(
            sources=[source_dict],
            target_width=256,  # V-JEPA default
            target_height=256,
            target_fps=30,
            from_stage="io_processor",
            to_stage="stage_0",
        )

        # Use pre-warmed pool if available (fast: ~1s), otherwise spawn (slow: ~11s)
        if self._vision_pool is not None:
            handle = self._vision_pool.assign_session(
                config=processor_config,
                connector_config=connector_config,
                session_id=session_id,
            )
            # Track worker_id for returning to pool when session ends
            session._pool_worker_id = handle.worker_id
        else:
            # Fallback to direct spawn (only if pool not available)
            handle = VisionIOProcessor.spawn(
                config=processor_config,
                connector_config=connector_config,
                session_id=session_id,
            )

        session._vision_processor = handle

        # Create connector client to receive frames from VisionIOProcessor
        connector = OmniConnectorFactory.create_connector(
            ConnectorSpec(
                name="SharedMemoryConnector",
                extra=connector_config["extra"],
            )
        )
        session._frame_connector = connector

        # Start frame consumer task
        session._stop_consumer = asyncio.Event()
        session._frame_consumer_task = asyncio.create_task(
            self._consume_frames(session),
            name=f"frame-consumer-{session_id}",
        )

    def _source_to_dict(self, source: SourceConfigRequest) -> dict:
        """Convert SourceConfigRequest to dict for VisionIOProcessor."""
        result = {"type": source.type.value if hasattr(source.type, 'value') else source.type}

        if source.type == SourceType.FILE:
            result["path"] = source.uri.replace("file://", "") if source.uri else None
        elif source.type == SourceType.RTSP:
            result["uri"] = source.uri
        elif source.type == SourceType.V4L2:
            result["device"] = source.device or "/dev/video0"
        elif source.type == SourceType.CSI:
            result["sensor_id"] = source.sensor_id or 0
        elif source.type == SourceType.MOQ:
            # MOQ uses appsrc - frames come from WebTransport server
            result["appsrc"] = True

        return result

    def _run_clip_inference_sync(
        self,
        clip: torch.Tensor,
        is_final: bool = False,
        clip_index: int = 0,
        head: str | None = None,
    ) -> torch.Tensor:
        """Run synchronous preprocessing + model inference on a clip.

        Designed to be called via asyncio.to_thread() to avoid blocking
        the event loop during the ~13s inference on slower GPUs.

        Args:
            clip: Frame tensor of shape (T, C, H, W).
            is_final: Whether this is a padded final clip.
            clip_index: 1-based clip number within session.

        Returns:
            Logits tensor from model output.
        """
        if self._tracing_hooks is not None:
            self._tracing_hooks.clip_index = clip_index

        with trace_span("clip_inference", clip_index=clip_index, num_frames=clip.shape[0], is_final=is_final, head=head or ""):
            with trace_span("input_preprocess", clip_index=clip_index, num_frames=clip.shape[0], is_final=is_final):
                clip_np = clip.cpu().numpy()

                if clip_np.ndim == 4 and clip_np.shape[1] in (1, 3, 4):
                    clip_np = np.transpose(clip_np, (0, 2, 3, 1))

                if clip_np.dtype != np.uint8:
                    if clip_np.max() <= 1.0:
                        clip_np = (clip_np * 255).astype(np.uint8)
                    else:
                        clip_np = clip_np.astype(np.uint8)

                if is_final:
                    inputs = self._processor(videos=list(clip_np), return_tensors="pt")
                else:
                    inputs = self._processor(list(clip_np), return_tensors="pt")

            pixel_key = (
                "pixel_values_videos"
                if "pixel_values_videos" in inputs
                else "pixel_values"
            )

            # Resolve which head to use
            selected_head = self._heads.get(head) if head else None
            if selected_head is None and head is not None:
                selected_head = self._heads.get(self._default_head)

            if is_final:
                device = next(self._model.parameters()).device
                inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                pixel_values = inputs_on_device.get(pixel_key, inputs_on_device.get("pixel_values"))
            else:
                pixel_values = inputs[pixel_key].to(self._device)

            if selected_head is not None:
                # Multi-head path: run encoder only, then selected head
                with torch.no_grad():
                    encoder_out = self._model.vjepa2(pixel_values)
                with trace_span("output_postprocess"):
                    logits = selected_head(encoder_out.last_hidden_state)
            else:
                # Default path: run full model
                with torch.no_grad():
                    outputs = self._model(pixel_values)
                with trace_span("output_postprocess"):
                    logits = outputs.logits

            return logits

    async def _consume_frames(self, session: Session) -> None:
        """Consume frames from VisionIOProcessor and run model inference.

        Uses two concurrent tasks: a reader that pulls frames from SHM
        into a FrameBuffer and enqueues ready clips, and an inferencer
        that processes clips as they become available. This allows frame
        reading to continue during GPU inference — critical for live
        streams where frames arrive continuously at 30fps.

        Args:
            session: The active session to consume frames for.
        """
        from vllm_omni.model_executor.models.vjepa.encoder import FrameBuffer

        connector = session._frame_connector
        if connector is None:
            logger.error("Frame consumer: no connector configured")
            return

        frame_buffer = FrameBuffer(
            num_frames=session.config.num_frames,
            stride=session.config.stride,
        )

        clip_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        frames_read = 0
        clip_index = 0
        last_pts_ns = 0

        logger.info(
            "Frame consumer started for session %s (num_frames=%d, stride=%d)",
            session.session_id,
            session.config.num_frames,
            session.config.stride,
        )

        tracer = get_tracer()

        async def read_frames() -> None:
            """Read frames from SHM into FrameBuffer, enqueue ready clips."""
            nonlocal frames_read, last_pts_ns

            frame_idx = 1
            poll_interval = 0.01
            max_consecutive_misses = 100
            consecutive_misses = 0

            while not session._stop_consumer.is_set():
                try:
                    get_key = f"{session.session_id}_frame_{frame_idx}"

                    result = await asyncio.to_thread(
                        connector.get,
                        from_stage="io_processor",
                        to_stage="stage_0",
                        get_key=get_key,
                    )

                    if result is None:
                        consecutive_misses += 1
                        if consecutive_misses >= max_consecutive_misses:
                            eof_key = f"{session.session_id}_frame_eof"
                            eof_result = await asyncio.to_thread(
                                connector.get,
                                from_stage="io_processor",
                                to_stage="stage_0",
                                get_key=eof_key,
                            )
                            if eof_result is not None:
                                eof_data, _ = eof_result
                                logger.info(
                                    "Frame consumer: received EOF after %d frames (total=%d)",
                                    frame_idx - 1,
                                    eof_data.get("total_frames", -1),
                                )
                                break

                            source_active = False
                            if session._vision_processor is not None:
                                source_active = session._vision_processor.is_alive
                            elif session._gst_pipeline is not None:
                                from gi.repository import Gst
                                _, state, pending = session._gst_pipeline.get_state(0)
                                source_active = state in (Gst.State.PLAYING, Gst.State.PAUSED) or pending == Gst.State.PLAYING
                            elif session._moq_session is not None:
                                source_active = session._moq_session._running

                            if not source_active:
                                logger.info(
                                    "Frame consumer: frame source stopped, ending consumption"
                                )
                                break
                            consecutive_misses = 0
                        await asyncio.sleep(poll_interval)
                        continue

                    consecutive_misses = 0
                    frame_data, size = result
                    frame_idx += 1

                    tensor = frame_data.get("tensor")
                    pts_ns = frame_data.get("pts_ns", 0)
                    timing = frame_data.get("timing")

                    if tensor is None:
                        logger.warning("Frame consumer: frame %d missing tensor", frame_idx - 1)
                        continue

                    if timing:
                        if "input_receive_start_ns" in timing:
                            record_span(
                                "input_receive",
                                start_time_ns=timing["input_receive_start_ns"],
                                end_time_ns=timing["input_receive_end_ns"],
                                frame_index=frame_idx - 1,
                                session_id=session.session_id,
                                hw_target=timing.get("hw_target", "unknown"),
                                pts_ns=pts_ns,
                            )

                        if "input_decode_start_ns" in timing:
                            record_span(
                                "input_decode",
                                start_time_ns=timing["input_decode_start_ns"],
                                end_time_ns=timing["input_decode_end_ns"],
                                frame_index=frame_idx - 1,
                                session_id=session.session_id,
                                hw_target=timing.get("hw_target", "unknown"),
                                tensor_shape=timing.get("tensor_shape", "unknown"),
                            )

                    frame_buffer.add_frame(tensor, pts_ns)
                    frames_read = frame_idx - 1
                    last_pts_ns = pts_ns
                    session.frames_processed = frames_read

                    if frame_buffer.ready():
                        clip, actions = frame_buffer.consume()
                        await clip_queue.put((clip, pts_ns, False))

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(
                        "Frame reader error for session %s: %s",
                        session.session_id, e,
                    )
                    await asyncio.sleep(poll_interval)

            # Flush remaining frames as final padded clip
            remaining = frame_buffer.remaining_frames()
            if remaining > 0:
                logger.info(
                    "Frame consumer: flushing %d remaining frames as padded clip",
                    remaining,
                )
                flush_result = frame_buffer.flush()
                if flush_result is not None:
                    clip, actions = flush_result
                    await clip_queue.put((clip, last_pts_ns, True))

            await clip_queue.put(None)  # Sentinel to stop inferencer

        async def run_inference() -> None:
            """Process clips from the queue, run model inference."""
            nonlocal clip_index

            while True:
                item = await clip_queue.get()
                if item is None:
                    break

                clip, pts_ns, is_final = item

                if self._model is None or self._processor is None:
                    logger.warning(
                        "Frame consumer: model=%s, processor=%s - skipping inference",
                        type(self._model).__name__ if self._model else None,
                        type(self._processor).__name__ if self._processor else None,
                    )
                    continue

                try:
                    clip_index += 1
                    logits = await asyncio.to_thread(
                        self._run_clip_inference_sync,
                        clip, is_final, clip_index, session.head,
                    )

                    # Use head-specific labels if available
                    id2label = getattr(self._model.config, "id2label", None)
                    head_obj = self._heads.get(session.head) if session.head else None
                    if head_obj is not None and head_obj.label_map:
                        id2label = head_obj.label_map

                    self.queue_prediction(
                        session=session,
                        logits=logits,
                        pts_ns=pts_ns,
                        id2label=id2label,
                    )

                    logger.info(
                        "Frame consumer: clip %d done for session %s, "
                        "logits=%s, next frame_idx=%d, stop=%s",
                        clip_index,
                        session.session_id,
                        logits.shape,
                        frames_read + 1,
                        session._stop_consumer.is_set(),
                    )

                except Exception as e:
                    logger.exception(
                        "Frame consumer: inference failed for session %s: %s",
                        session.session_id, e,
                    )

        with tracer.start_as_current_span("stream_inference") as stream_span:
            stream_span.set_attribute("session.id", session.session_id)
            stream_span.set_attribute("config.num_frames", session.config.num_frames)
            stream_span.set_attribute("config.stride", session.config.stride)

            reader_task = asyncio.create_task(read_frames())
            inference_task = asyncio.create_task(run_inference())
            await asyncio.gather(reader_task, inference_task)

        logger.info(
            "Frame consumer stopped for session %s (processed %d frames, %d clips)",
            session.session_id,
            frames_read,
            clip_index,
        )

        if session._vision_processor is not None:
            session._vision_processor.stop()
            session._vision_processor = None

        if session._pool_worker_id is not None and self._vision_pool is not None:
            self._vision_pool.return_worker(session._pool_worker_id)
            logger.info("Returned worker %d to pool (frame consumer completed)", session._pool_worker_id)
            session._pool_worker_id = None

        if session.status == SessionStatus.RUNNING:
            session.status = SessionStatus.STOPPED

    async def stop_session(self, session_id: str) -> Session:
        """Stop a running session.

        Args:
            session_id: Session to stop.

        Returns:
            Updated session with STOPPED status.

        Raises:
            KeyError: If session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session.status in (SessionStatus.STOPPED, SessionStatus.STOPPING):
            return session

        session.status = SessionStatus.STOPPING

        try:
            # Stop frame consumer task first
            if session._stop_consumer is not None:
                session._stop_consumer.set()
            if session._frame_consumer_task is not None:
                session._frame_consumer_task.cancel()
                try:
                    await asyncio.wait_for(session._frame_consumer_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                session._frame_consumer_task = None

            # Close frame connector
            if session._frame_connector is not None:
                try:
                    session._frame_connector.close()
                except Exception:
                    pass  # Best-effort cleanup
                session._frame_connector = None

            # Stop VisionIOProcessor subprocess
            if session._vision_processor is not None:
                session._vision_processor.stop()
                session._vision_processor = None

            # Return worker to pool if it was from the pool
            if session._pool_worker_id is not None and self._vision_pool is not None:
                self._vision_pool.return_worker(session._pool_worker_id)
                logger.info("Returned worker %d to pool", session._pool_worker_id)
                session._pool_worker_id = None

            # Stop MoQ GStreamer pipeline
            if session._gst_pipeline is not None:
                from gi.repository import Gst
                session._gst_pipeline.set_state(Gst.State.NULL)
                session._gst_pipeline = None

            # Stop legacy MoQ WebTransport server (if any)
            if session._moq_session is not None:
                await session._moq_session.stop()
                session._moq_session = None

            session.status = SessionStatus.STOPPED
            logger.info("Stopped world session: %s", session_id)

        except Exception as e:
            session.status = SessionStatus.ERROR
            session.error_message = str(e)
            logger.exception("Error stopping session %s: %s", session_id, e)
            raise

        return session

    async def close_session(
        self,
        session_id: str,
    ) -> WorldSessionDeleteResponse:
        """Delete a session and free resources.

        Args:
            session_id: Session to delete.

        Returns:
            Deletion confirmation response.

        Raises:
            KeyError: If session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        # Stop if running
        if session.status not in (SessionStatus.STOPPED, SessionStatus.ERROR):
            await self.stop_session(session_id)

        # Remove from registry
        async with self._lock:
            del self._sessions[session_id]

        logger.info("Deleted world session: %s", session_id)

        return WorldSessionDeleteResponse(session_id=session_id)

    async def list_sessions(self) -> WorldSessionListResponse:
        """List all sessions.

        Returns:
            List of all session statuses.
        """
        sessions = [s.to_status_response() for s in self._sessions.values()]
        return WorldSessionListResponse(
            sessions=sessions,
            total=len(sessions),
        )

    async def list_heads(self) -> WorldHeadsResponse:
        """List available classification heads."""
        heads = [
            WorldHeadInfo(
                name=name,
                num_labels=head.num_labels,
                is_default=(name == self._default_head),
            )
            for name, head in self._heads.items()
        ]
        return WorldHeadsResponse(heads=heads)

    async def stream_predictions(
        self,
        session_id: str,
    ) -> AsyncIterator[WorldPredictionEvent]:
        """Stream predictions from a session.

        Supports reconnection - can be called multiple times for the
        same session. Each call continues from the current queue state.

        Args:
            session_id: Session to stream from.

        Yields:
            Prediction events as they become available.

        Raises:
            KeyError: If session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session._prediction_queue is None:
            raise RuntimeError("Session not initialized")

        # Auto-start session if needed
        if session.status in (SessionStatus.CREATED, SessionStatus.STOPPED):
            await self.start_session(session_id)

        logger.info("Streaming predictions for session: %s", session_id)

        while session.status in (SessionStatus.RUNNING, SessionStatus.STARTING):
            try:
                # Wait for prediction with timeout (allows status checks)
                prediction = await asyncio.wait_for(
                    session._prediction_queue.get(),
                    timeout=1.0,
                )
                yield WorldPredictionEvent(
                    prediction=prediction,
                    frames_processed=session.frames_processed,
                )
            except asyncio.TimeoutError:
                # Continue loop to check status
                continue
            except asyncio.CancelledError:
                logger.info("Prediction stream cancelled for session: %s", session_id)
                break

        # Drain remaining predictions
        while not session._prediction_queue.empty():
            try:
                prediction = session._prediction_queue.get_nowait()
                yield WorldPredictionEvent(
                    prediction=prediction,
                    frames_processed=session.frames_processed,
                )
            except asyncio.QueueEmpty:
                break

        # Send end event
        yield WorldPredictionEvent(status="ended")
        logger.info("Prediction stream ended for session: %s", session_id)

    def queue_prediction(
        self,
        session: Session,
        logits: torch.Tensor,
        pts_ns: int,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Queue a prediction for SSE delivery.

        Called by the inference pipeline when a prediction is ready.

        Args:
            session: Session to queue prediction for.
            logits: Model output logits.
            pts_ns: Frame timestamp in nanoseconds.
            id2label: Optional label mapping.
        """
        # Get top-5 predictions
        if isinstance(logits, torch.Tensor):
            probs = torch.softmax(logits[0], dim=-1)
            top_k = torch.topk(probs, k=min(5, probs.shape[-1]))
            labels = []
            for idx, prob in zip(top_k.indices.tolist(), top_k.values.tolist()):
                label = (id2label or {}).get(idx, f"class_{idx}")
                labels.append(PredictionLabel(label=label, probability=prob))
        else:
            labels = []

        # Calculate frame range
        stride = session.config.stride
        num_frames = session.config.num_frames
        end_frame = session.frames_processed
        start_frame = max(0, end_frame - num_frames)

        prediction = WorldPrediction(
            labels=labels,
            frame_range=(start_frame, end_frame),
            timestamp_ns=pts_ns,
            request_id=session.session_id,
        )

        session.predictions_count += 1

        # Queue for SSE delivery (non-blocking, drops if full)
        if session._prediction_queue is not None:
            try:
                session._prediction_queue.put_nowait(prediction)
            except asyncio.QueueFull:
                logger.warning("Prediction queue full for session %s", session.session_id)

    async def upload_video(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> WorldUploadResponse:
        """Store uploaded video and return reference.

        Args:
            content: Video file content as bytes.
            filename: Original filename (used to preserve extension).
            content_type: MIME type of the video file.

        Returns:
            Response with video_id and file URI for session creation.
        """
        # Generate unique video ID
        video_id = f"vid_{secrets.token_urlsafe(12)}"

        # Preserve extension from original filename
        ext = Path(filename).suffix or ".mp4"
        file_path = self.storage_path / f"{video_id}{ext}"

        # Write file to storage
        file_path.write_bytes(content)

        logger.info(
            "Video uploaded: video_id=%s, size=%d bytes, path=%s",
            video_id,
            len(content),
            file_path,
        )

        return WorldUploadResponse(
            video_id=video_id,
            uri=f"file://{file_path}",
            size_bytes=len(content),
            content_type=content_type,
        )

    async def infer_video(
        self,
        content: bytes,
        filename: str,
        num_frames: int = 16,
        stride: int | None = None,
        top_k: int = 5,
        obs_timestamp_ms: int | None = None,
        head: str | None = None,
    ) -> dict[str, Any]:
        """Single-shot video inference (KServe-compatible).

        Decodes the video, iterates clips, runs inference on each, and
        returns all predictions in one response. Mirrors the vjepa2-demo
        ``/v2/models/vjepa2/infer`` endpoint for benchmark compatibility.
        """
        import tempfile
        import uuid

        import av

        tracer = get_tracer()
        effective_stride = stride if stride is not None else num_frames

        # Resolve label map
        id2label = getattr(self._model.config, "id2label", None)
        head_obj = self._heads.get(head) if head else None
        if head_obj is not None and head_obj.label_map:
            id2label = head_obj.label_map

        with tracer.start_as_current_span("video_inference") as root_span:
            if obs_timestamp_ms is not None:
                root_span.set_attribute("input.obs_timestamp_ms", obs_timestamp_ms)

            with tracer.start_as_current_span("input_receive") as recv_span:
                recv_span.set_attribute("input.filename", filename)
                recv_span.set_attribute("input.size_bytes", len(content))
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

            try:
                with tracer.start_as_current_span("input_open") as open_span:
                    container = av.open(tmp_path)
                    stream = container.streams.video[0]
                    open_span.set_attribute("input.codec", stream.codec_context.name)
                    open_span.set_attribute("input.width", stream.width)
                    open_span.set_attribute("input.height", stream.height)
                    if stream.frames:
                        open_span.set_attribute("input.total_frames", stream.frames)
                    open_span.set_attribute("input.source_type", "file")

                buffer: list[np.ndarray] = []
                frame_index = 0
                next_clip_start = 0
                clips_results: list[dict[str, Any]] = []
                clip_idx = 0

                for frame in container.decode(video=0):
                    buffer.append(frame.to_ndarray(format="rgb24"))
                    frame_index += 1

                    while len(buffer) >= next_clip_start + num_frames:
                        clip_frames = np.stack(
                            buffer[next_clip_start : next_clip_start + num_frames]
                        )
                        clip_tensor = torch.from_numpy(clip_frames)
                        clip_idx += 1
                        logits = await asyncio.to_thread(
                            self._run_clip_inference_sync,
                            clip_tensor, False, clip_idx, head,
                        )
                        probs = torch.softmax(logits[0], dim=-1)
                        topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
                        preds = []
                        for score, idx in zip(topk.values, topk.indices):
                            label = (
                                id2label.get(idx.item(), f"class_{idx.item()}")
                                if id2label else f"class_{idx.item()}"
                            )
                            preds.append({"label": label, "score": round(score.item(), 6)})
                        clips_results.append({
                            "clip_index": clip_idx - 1,
                            "start_frame": next_clip_start,
                            "end_frame": next_clip_start + num_frames,
                            "partial": False,
                            "predictions": preds,
                        })
                        next_clip_start += effective_stride

                        if stride is None:
                            break

                    if stride is None and clips_results:
                        break

                container.close()

                # Trailing frames
                if next_clip_start < frame_index and buffer and stride is not None:
                    remaining = buffer[next_clip_start:]
                    last_frame = remaining[-1]
                    pad_count = num_frames - len(remaining)
                    padded = np.stack(remaining + [last_frame] * pad_count)
                    clip_tensor = torch.from_numpy(padded)
                    clip_idx += 1
                    logits = await asyncio.to_thread(
                        self._run_clip_inference_sync,
                        clip_tensor, True, clip_idx, head,
                    )
                    probs = torch.softmax(logits[0], dim=-1)
                    topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
                    preds = []
                    for score, idx in zip(topk.values, topk.indices):
                        label = (
                            id2label.get(idx.item(), f"class_{idx.item()}")
                            if id2label else f"class_{idx.item()}"
                        )
                        preds.append({"label": label, "score": round(score.item(), 6)})
                    clips_results.append({
                        "clip_index": clip_idx - 1,
                        "start_frame": next_clip_start,
                        "end_frame": frame_index,
                        "partial": True,
                        "predictions": preds,
                    })

                root_span.set_attribute("video.clips_count", len(clips_results))
                root_span.set_attribute("video.stride", effective_stride)

                model_name = getattr(self._model.config, "_name_or_path", "vjepa2")
                return {
                    "model_name": "vjepa2",
                    "model_version": model_name,
                    "id": f"req-{uuid.uuid4().hex[:8]}",
                    "clips": clips_results,
                }
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    @property
    def gstreamer_available(self) -> bool:
        """Check if GStreamer is available."""
        return GST_AVAILABLE
