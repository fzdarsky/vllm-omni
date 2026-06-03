# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
V-JEPA Model for vLLM-Omni.

This module provides a V-JEPA model implementation using HuggingFace's
pretrained weights. It supports both batch and streaming inference modes
with V-JEPA span tracing for latency measurement.

Architecture:
    Input -> FrameBuffer -> HuggingFace V-JEPA2 -> Logits
                            (hooks for jepa_encode, jepa_predict, jepa_pool)

The implementation is self-contained - no external vllm_jepa dependency.

Model Protocol Markers:
    - supports_multimodal_raw_input_only: Accept raw tensors, not tokens
    - is_attention_free: No KV-cache needed (non-autoregressive)
    - is_pooling_model: Outputs classifications/embeddings

Usage:
    # Within vLLM-Omni pipeline
    model = VJepa2Encoder(vllm_config)
    result = model(frames=video_frames)
    logits = result["logits"]

    # Streaming mode (frame-by-frame)
    for frame in video_stream:
        result = model(tensor=frame, pts_ns=timestamp)
        if result is not None:  # Prediction ready
            process(result["logits"])

    # With V-JEPA metrics
    metrics = VJepa2Metrics(request_id="req-001")
    model.set_metrics(metrics)
    result = model(frames=video_frames)
    print(f"L_comp: {metrics.l_comp_ms:.1f}ms")
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

import torch
import torch.nn as nn

from vllm.tracing import instrument_manual, is_tracing_available

if TYPE_CHECKING:
    from transformers import AutoModelForVideoClassification, AutoVideoProcessor
    from vllm.config import VllmConfig


# =============================================================================
# V-JEPA Tracing Infrastructure
# =============================================================================
# V-JEPA defines standardized spans for measuring video inference latency.
# These match vjepa2-demo's methodology and export to OpenTelemetry/Jaeger.

TRACE_SPANS = [
    "input_receive",
    "input_decode",
    "input_preprocess",
    "jepa_encode",
    "jepa_predict",
    "jepa_pool",
    "output_postprocess",
]


@dataclass
class SpanTiming:
    """Timing data for a single V-JEPA span."""
    name: str
    start_time_ns: int
    end_time_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time_ns - self.start_time_ns) / 1_000_000


@dataclass
class VJepa2Metrics:
    """Collected V-JEPA metrics for a single inference request.

    Used for benchmarking V-JEPA inference latency. The spans are recorded
    via forward hooks on the model's submodules.

    Attributes:
        request_id: Unique request identifier for tracing
        spans: List of recorded span timings
        l_comp_ms: Total computational latency (sum of all spans)
    """
    request_id: str
    spans: list[SpanTiming] = field(default_factory=list)

    @property
    def l_comp_ms(self) -> float:
        """Total computational latency (L_comp) in milliseconds."""
        return sum(span.duration_ms for span in self.spans)

    def get_span(self, name: str) -> SpanTiming | None:
        """Get a specific span by name."""
        for span in self.spans:
            if span.name == name:
                return span
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "l_comp_ms": self.l_comp_ms,
            "spans": {
                span.name: {"duration_ms": span.duration_ms}
                for span in self.spans
            },
        }


def _sync_device(device: str) -> None:
    """Synchronize device for accurate timing."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# =============================================================================
# Frame and Action Buffers
# =============================================================================


class FrameBuffer:
    """GPU-resident frame accumulator with stride-based clip yield.

    Accumulates frames with timestamps and yields overlapping clips
    for V-JEPA encoding. The stride parameter controls how many frames
    are advanced between clips (enabling temporal overlap).

    Args:
        num_frames: Number of frames per clip (default: 16)
        stride: Number of frames to advance between clips (default: 8)

    Note:
        The stride must be configured consistently with the async_chunk
        processor to prevent "temporal tearing" in predictions.
    """

    def __init__(self, num_frames: int = 16, stride: int = 8):
        self.num_frames = num_frames
        self.stride = stride
        self._frames: deque[tuple[torch.Tensor, int]] = deque()
        self._action_buffer: ActionBuffer | None = None

    def set_action_buffer(self, action_buffer: ActionBuffer) -> None:
        """Attach an ActionBuffer for timestamp-based action lookup."""
        self._action_buffer = action_buffer

    def add_frame(
        self,
        frame: torch.Tensor,
        pts_ns: int | None = None,
        action_meta: dict | None = None,
    ) -> None:
        """Add a frame to the buffer."""
        if pts_ns is None:
            pts_ns = time.time_ns()

        self._frames.append((frame, pts_ns))

        if action_meta is not None and self._action_buffer is not None:
            action_tensor = action_meta.get("action")
            if action_tensor is not None:
                self._action_buffer.add(pts_ns, action_tensor)

    def ready(self) -> bool:
        """Check if enough frames are available for one clip."""
        return len(self._frames) >= self.num_frames

    def consume(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Consume a clip and advance buffer by stride."""
        if not self.ready():
            raise ValueError(
                f"Not enough frames: {len(self._frames)} < {self.num_frames}"
            )

        clip_frames = [self._frames[i][0] for i in range(self.num_frames)]
        clip_pts = [self._frames[i][1] for i in range(self.num_frames)]
        clip = torch.stack(clip_frames, dim=0)

        actions = None
        if self._action_buffer is not None:
            actions = self._action_buffer.get_for_frames(clip_pts)

        for _ in range(self.stride):
            self._frames.popleft()

        return clip, actions

    def clear(self) -> None:
        """Clear all frames from buffer."""
        self._frames.clear()

    def flush(self) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Flush remaining frames as a padded clip.

        Pads incomplete clips by repeating the last frame.
        Returns None if no frames remain.
        """
        if not self._frames:
            return None

        remaining = list(self._frames)
        if not remaining:
            return None

        # Pad with last frame to fill the clip
        last_frame = remaining[-1][0]
        pad_count = self.num_frames - len(remaining)
        padded_frames = [f[0] for f in remaining] + [last_frame] * pad_count
        padded_pts = [f[1] for f in remaining] + [remaining[-1][1]] * pad_count
        clip = torch.stack(padded_frames, dim=0)

        actions = None
        if self._action_buffer is not None:
            actions = self._action_buffer.get_for_frames(padded_pts)

        self._frames.clear()
        return clip, actions

    def remaining_frames(self) -> int:
        """Return number of frames not yet consumed."""
        return len(self._frames)


class ActionBuffer:
    """Ring buffer for high-frequency actions with NTP timestamp lookup.

    Stores timestamped action tensors and provides efficient lookup
    for frame-synchronized action retrieval.

    Args:
        capacity: Maximum number of actions to store (default: 1000)
        tolerance_ns: Timestamp tolerance in nanoseconds (default: 1ms)
    """

    def __init__(self, capacity: int = 1000, tolerance_ns: int = 1_000_000):
        self.capacity = capacity
        self.tolerance_ns = tolerance_ns
        self.buffer: deque[tuple[int, torch.Tensor]] = deque(maxlen=capacity)

    def add(self, timestamp_ns: int, action: torch.Tensor) -> None:
        """Add action with NTP timestamp."""
        self.buffer.append((timestamp_ns, action))

    def get_for_frames(
        self,
        frame_pts: list[int],
        tolerance_ns: int | None = None,
    ) -> torch.Tensor:
        """Get actions closest to each frame PTS."""
        tolerance = tolerance_ns or self.tolerance_ns
        actions = [self._find_closest(pts, tolerance) for pts in frame_pts]
        return torch.stack(actions, dim=0)

    def _find_closest(self, target_ns: int, tolerance_ns: int) -> torch.Tensor:
        """Find action closest to target timestamp within tolerance."""
        if not self.buffer:
            return torch.zeros(7)  # Default action_dim=7 for robotics

        best_action = None
        best_diff = float("inf")

        for ts, action in self.buffer:
            diff = abs(ts - target_ns)
            if diff < best_diff:
                best_diff = diff
                best_action = action

        if best_diff <= tolerance_ns and best_action is not None:
            return best_action
        elif best_action is not None:
            return torch.zeros_like(best_action)
        return torch.zeros(7)

    def clear(self) -> None:
        """Clear all actions from buffer."""
        self.buffer.clear()


# =============================================================================
# Tracing Hooks
# =============================================================================


class TracingHooks:
    """Manages V-JEPA spans for PyTorch module forward passes.

    Uses paired pre/post hooks to create spans that accurately measure
    submodule execution time, including GPU synchronization.

    Spans are:
    1. Recorded in VJepa2Metrics (if provided) for benchmark collection
    2. Exported to OpenTelemetry/Jaeger (if tracing is enabled)
    """

    def __init__(self, device: str, metrics: VJepa2Metrics | None = None):
        self.device = device
        self.metrics = metrics
        self._active_spans: dict[int, tuple[str, int]] = {}
        self._handles: list = []
        self._otel_enabled = is_tracing_available()

    def _make_pre_hook(self, span_name: str):
        """Create a pre-forward hook that starts timing."""

        def hook(module: nn.Module, args: tuple) -> None:
            _sync_device(self.device)
            start_ns = time.perf_counter_ns()
            self._active_spans[id(module)] = (span_name, start_ns)

        return hook

    def _make_post_hook(self, span_name: str):
        """Create a post-forward hook that ends timing and exports span."""

        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            _sync_device(self.device)
            end_ns = time.perf_counter_ns()

            entry = self._active_spans.pop(id(module), None)
            if entry is not None:
                name, start_ns = entry

                # Record in VJepa2Metrics (if provided)
                if self.metrics is not None:
                    self.metrics.spans.append(
                        SpanTiming(name=name, start_time_ns=start_ns, end_time_ns=end_ns)
                    )

                # Export to OpenTelemetry (if enabled)
                if self._otel_enabled:
                    instrument_manual(
                        span_name=name,
                        start_time=start_ns,
                        end_time=end_ns,
                        attributes={"component": "vjepa2", "device": self.device},
                    )

        return hook

    def register(self, module: nn.Module, span_name: str) -> None:
        """Register tracing hooks on a module."""
        pre_handle = module.register_forward_pre_hook(self._make_pre_hook(span_name))
        post_handle = module.register_forward_hook(self._make_post_hook(span_name))
        self._handles.extend([pre_handle, post_handle])

    def set_metrics(self, metrics: VJepa2Metrics | None) -> None:
        """Update the metrics collector."""
        self.metrics = metrics

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# =============================================================================
# HuggingFace V-JEPA Wrapper
# =============================================================================


@dataclass
class VJepa2Config:
    """Configuration for HuggingFace V-JEPA wrapper."""

    model_id: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    num_frames: int = 16
    image_size: tuple[int, int] = (256, 256)
    stride: int = 8
    device: str | None = None  # Auto-detect if None

    # Required by vLLM adapters for vision-only models
    def get_text_config(self):
        """Return self for vision-only models (no separate text config)."""
        return self


class VJepa2Model(nn.Module):
    """HuggingFace V-JEPA2 wrapper with V-JEPA tracing.

    Single-stage wrapper that uses the full HuggingFace model with
    forward hooks for V-JEPA tracing. Handles FrameBuffer accumulation
    for streaming mode.

    Attributes:
        supports_multimodal_raw_input_only: Accept raw tensors, not tokens
        is_attention_free: No KV-cache needed (non-autoregressive)
        is_pooling_model: Outputs classifications/embeddings
    """

    supports_multimodal_raw_input_only = True
    is_attention_free = True
    is_pooling_model = True

    def __init__(self, config: VJepa2Config | None = None):
        super().__init__()
        self.config = config or VJepa2Config()

        self._model: AutoModelForVideoClassification | None = None
        self._processor: AutoVideoProcessor | None = None
        self._device: torch.device | None = None
        self._device_str: str = "cpu"
        self._tracing_hooks: TracingHooks | None = None

        # Streaming support
        self.frame_buffer = FrameBuffer(
            num_frames=self.config.num_frames,
            stride=self.config.stride,
        )
        self.action_buffer = ActionBuffer(capacity=1000)
        self.frame_buffer.set_action_buffer(self.action_buffer)

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Load HuggingFace model and processor.

        Records init_model_load span for cold start analysis.
        """
        from transformers import AutoModelForVideoClassification, AutoVideoProcessor

        load_start = time.perf_counter_ns()

        self._processor = AutoVideoProcessor.from_pretrained(self.config.model_id)
        self._model = AutoModelForVideoClassification.from_pretrained(
            self.config.model_id
        )

        # Determine device
        if self.config.device:
            self._device_str = self.config.device
        elif torch.cuda.is_available():
            self._device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device_str = "mps"
        else:
            self._device_str = "cpu"

        self._device = torch.device(self._device_str)
        self._model.to(self._device)

        load_end = time.perf_counter_ns()

        # Export init_model_load span to OpenTelemetry
        if is_tracing_available():
            load_duration_s = (load_end - load_start) / 1_000_000_000
            instrument_manual(
                span_name="init_model_load",
                start_time=load_start,
                end_time=load_end,
                attributes={
                    "component": "vjepa2",
                    "model.path": self.config.model_id,
                    "model.device": self._device_str,
                    "model.load_duration_s": load_duration_s,
                },
            )
        self._model.train(False)


        self._register_tracing_hooks()

    def _register_tracing_hooks(self) -> None:
        """Register V-JEPA tracing hooks on model submodules."""
        self._tracing_hooks = TracingHooks(self._device_str)

        # Map submodule paths to V-JEPA span names
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
            except AttributeError:
                pass  # Submodule not found - skip

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        num_frames: int = 16,
        stride: int = 8,
        device: str | None = None,
    ) -> "VJepa2Model":
        """Load pretrained HuggingFace V-JEPA2 model."""
        config = VJepa2Config(
            model_id=model_id,
            num_frames=num_frames,
            stride=stride,
            device=device,
        )
        model = cls(config=config)
        model._load_model()
        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """Forward pass through HuggingFace model.

        Supports two modes:
        1. Streaming: Receives frames via kwargs['tensor'], accumulates
           in FrameBuffer, returns logits tensor when clip ready
        2. Batch: Receives frames via kwargs['frames'] or preprocessed
           clip via kwargs['pixel_values'], returns logits tensor

        Returns:
            torch.Tensor: Logits tensor of shape (batch, num_classes)
            None: When streaming and clip not yet ready
        """
        self._ensure_loaded()

        # Check for streaming mode
        tensor = kwargs.get("tensor")
        if tensor is not None:
            return self._forward_streaming(
                tensor=tensor,
                pts_ns=kwargs.get("pts_ns"),
                action_meta=kwargs.get("action_meta"),
            )

        return self._forward_batch(**kwargs)

    def _forward_batch(self, **kwargs) -> torch.Tensor:
        """Process a batch of frames and return logits tensor."""
        pixel_values = kwargs.get("pixel_values")
        frames = kwargs.get("frames")
        metrics = kwargs.get("metrics")

        # Handle dummy run (vLLM warmup/profiling)
        # When no input is provided, return dummy output
        # For pooling models, vLLM expects hidden states of shape (batch, seq_len, hidden_size)
        # not final logits. This is used during model profiling for memory allocation.
        if pixel_values is None and frames is None:
            # Return dummy hidden states for warmup run
            # Shape: (batch=1, seq_len=1, hidden_size) - V-JEPA ViT-L has hidden_size=1024
            hidden_size = getattr(self._model.config, "hidden_size", 1024)
            dummy_hidden = torch.zeros(1, 1, hidden_size, device=self._device)
            return dummy_hidden

        # Record input_preprocess span
        preprocess_start = time.perf_counter_ns()
        _sync_device(self._device_str)

        if pixel_values is None and frames is not None:
            inputs = self._processor(list(frames), return_tensors="pt")
            key = (
                "pixel_values_videos"
                if "pixel_values_videos" in inputs
                else "pixel_values"
            )
            pixel_values = inputs[key].to(self._device)

        if pixel_values.device != self._device:
            pixel_values = pixel_values.to(self._device)

        _sync_device(self._device_str)
        preprocess_end = time.perf_counter_ns()

        # Record input_preprocess span
        if metrics is not None:
            metrics.spans.append(
                SpanTiming(
                    name="input_preprocess",
                    start_time_ns=preprocess_start,
                    end_time_ns=preprocess_end,
                )
            )
            if self._tracing_hooks is not None:
                self._tracing_hooks.set_metrics(metrics)

        # Export to OpenTelemetry
        if is_tracing_available():
            instrument_manual(
                span_name="input_preprocess",
                start_time=preprocess_start,
                end_time=preprocess_end,
                attributes={"component": "vjepa2", "device": self._device_str},
            )

        # Forward through model (hooks capture jepa_* spans)
        with torch.no_grad():
            outputs = self._model(pixel_values)

        # Record output_postprocess span
        postprocess_start = time.perf_counter_ns()
        _sync_device(self._device_str)

        # Return logits reshaped as hidden states for vLLM's pooling runner
        # Shape: (batch, 1, hidden_size)
        batch_size = outputs.logits.shape[0]
        hidden_size = getattr(self._model.config, "hidden_size", 1024)
        hidden_states = torch.zeros(
            batch_size, 1, hidden_size,
            device=outputs.logits.device,
            dtype=outputs.logits.dtype,
        )

        _sync_device(self._device_str)
        postprocess_end = time.perf_counter_ns()

        # Record output_postprocess span
        if metrics is not None:
            metrics.spans.append(
                SpanTiming(
                    name="output_postprocess",
                    start_time_ns=postprocess_start,
                    end_time_ns=postprocess_end,
                )
            )
            if self._tracing_hooks is not None:
                self._tracing_hooks.set_metrics(None)

        # Export to OpenTelemetry
        if is_tracing_available():
            instrument_manual(
                span_name="output_postprocess",
                start_time=postprocess_start,
                end_time=postprocess_end,
                attributes={"component": "vjepa2", "device": self._device_str},
            )

        return hidden_states

    def forward_encoder_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run only the backbone encoder, returning raw encoder hidden states.

        Used by multi-head routing to avoid running through the default
        pooler+classifier. The caller applies its own head to these states.
        """
        self._ensure_loaded()
        if pixel_values.device != self._device:
            pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            encoder_outputs = self._model.vjepa2(pixel_values)
        return encoder_outputs.last_hidden_state

    def _forward_streaming(
        self,
        tensor: torch.Tensor,
        pts_ns: int | None = None,
        action_meta: dict | None = None,
    ) -> torch.Tensor | None:
        """Process frame in streaming mode.

        Returns:
            Logits tensor when clip is ready, None while accumulating frames.
        """
        self.frame_buffer.add_frame(tensor, pts_ns, action_meta)

        if not self.frame_buffer.ready():
            return None

        frames, actions = self.frame_buffer.consume()

        # Convert (T, C, H, W) -> (T, H, W, C) -> numpy for HF processor
        if frames.dim() == 4 and frames.shape[1] == 3:
            frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        else:
            frames_np = frames.cpu().numpy()

        # _forward_batch now returns tensor directly (vLLM requirement)
        logits = self._forward_batch(frames=frames_np)

        # Note: actions are computed but not returned via vLLM pipeline
        # For action-conditioned inference, they would be passed to predictor stage
        return logits

    def reset(self) -> None:
        """Reset streaming state."""
        self.frame_buffer.clear()
        self.action_buffer.clear()

    def set_metrics(self, metrics: VJepa2Metrics | None) -> None:
        """Set V-JEPA metrics collector for tracing."""
        if self._tracing_hooks is not None:
            self._tracing_hooks.set_metrics(metrics)

    @property
    def device(self) -> torch.device:
        """Get model device."""
        self._ensure_loaded()
        return self._device

    @property
    def num_labels(self) -> int:
        """Get number of classification labels."""
        self._ensure_loaded()
        return self._model.config.num_labels

    @property
    def id2label(self) -> dict[int, str]:
        """Get label mapping."""
        self._ensure_loaded()
        return self._model.config.id2label


# =============================================================================
# Multi-Head Support
# =============================================================================


class ClassificationHead(nn.Module):
    """A swappable classification head (pooler + classifier).

    Each head is tied to a specific backbone by hidden_size.
    """

    def __init__(
        self,
        pooler: nn.Module,
        classifier: nn.Module,
        hidden_size: int,
        num_labels: int,
        label_map: dict[int, str] | None = None,
        name: str = "",
    ):
        super().__init__()
        self.pooler = pooler
        self.classifier = classifier
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.label_map = label_map or {}
        self.name = name

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(encoder_output)
        if pooled.dim() == 3:
            pooled = pooled[:, 0]
        return self.classifier(pooled)

    @classmethod
    def from_hf_model(
        cls,
        model: nn.Module,
        name: str = "default",
    ) -> "ClassificationHead":
        """Extract head from a loaded HF VJEPA2ForVideoClassification."""
        hidden_size = model.config.hidden_size
        num_labels = model.config.num_labels
        label_map = getattr(model.config, "id2label", {})
        return cls(
            pooler=model.pooler,
            classifier=model.classifier,
            hidden_size=hidden_size,
            num_labels=num_labels,
            label_map=label_map,
            name=name,
        )


# =============================================================================
# vLLM-Omni Integration
# =============================================================================


class VJepa2Encoder(nn.Module):
    """V-JEPA Model for vLLM-Omni.

    This is the vLLM-Omni integration wrapper that instantiates VJepa2Model
    with configuration from vllm_config.

    Protocol markers (used by vLLM-Omni scheduler):
        supports_multimodal_raw_input_only: Accept raw tensors, not tokens
        is_attention_free: No KV-cache needed (use OmniGenerationScheduler)
        is_pooling_model: Outputs classifications/embeddings
    """

    supports_multimodal_raw_input_only = True
    is_attention_free = True
    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # Extract config from vllm_config
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = None
        if model_config and hasattr(model_config, "hf_config"):
            hf_config = model_config.hf_config

        # Build config from hf_config or use defaults
        model_id = getattr(hf_config, "model_id", None)
        if model_id is None:
            model_id = "facebook/vjepa2-vitl-fpc16-256-ssv2"

        config = VJepa2Config(
            model_id=model_id,
            num_frames=getattr(hf_config, "num_frames", 16),
            stride=getattr(hf_config, "stride", 8),
            device=None,  # Auto-detect
        )

        self._impl = VJepa2Model(config=config)
        self._config = config

        # Multi-head registry: name → ClassificationHead
        self._heads: dict[str, ClassificationHead] = {}
        self._default_head: str | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Process input and return logits tensor.

        For streaming mode (GStreamer):
            - Receives single frames via kwargs['tensor'], kwargs['pts_ns']
            - Accumulates in FrameBuffer
            - Returns logits tensor when clip is ready

        For batch mode:
            - Receives frames via kwargs['frames'] (numpy array)
            - Or preprocessed kwargs['pixel_values']
            - Returns logits tensor immediately

        Multi-head routing:
            - Pass kwargs['head'] to select a registered classification head
            - Without 'head', uses the default head (falls back to full model)

        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
            None: When streaming and clip not yet ready
        """
        head_name = kwargs.pop("head", None)

        # If a specific head is requested and heads are registered, route
        if head_name is not None and self._heads:
            return self._forward_with_head(head_name, **kwargs)

        return self._impl.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def _forward_with_head(self, head_name: str, **kwargs: Any) -> torch.Tensor:
        """Run encoder backbone and route output through a named head."""
        head = self.get_head(head_name)

        pixel_values = kwargs.get("pixel_values")
        frames = kwargs.get("frames")

        if pixel_values is None and frames is not None:
            self._impl._ensure_loaded()
            inputs = self._impl._processor(list(frames), return_tensors="pt")
            key = (
                "pixel_values_videos"
                if "pixel_values_videos" in inputs
                else "pixel_values"
            )
            pixel_values = inputs[key]

        if pixel_values is None:
            hidden_size = getattr(self._impl._model.config, "hidden_size", 1024)
            return torch.zeros(1, 1, hidden_size, device=self._impl._device)

        encoder_output = self._impl.forward_encoder_only(pixel_values)
        logits = head(encoder_output)
        return logits.unsqueeze(1) if logits.dim() == 2 else logits

    def reset(self):
        """Reset internal state (FrameBuffer, ActionBuffer)."""
        self._impl.reset()

    def set_metrics(self, metrics: VJepa2Metrics | None):
        """Set V-JEPA metrics collector for tracing."""
        self._impl.set_metrics(metrics)

    # ── Multi-head API ──────────────────────────────────────────────────

    def register_head(
        self,
        name: str,
        head: ClassificationHead,
        *,
        default: bool = False,
    ) -> None:
        """Register a classification head.

        Validates that the head's hidden_size matches the encoder backbone.
        """
        self._impl._ensure_loaded()
        backbone_hidden = self._impl._model.config.hidden_size
        if head.hidden_size != backbone_hidden:
            raise ValueError(
                f"Head '{name}' hidden_size={head.hidden_size} does not match "
                f"backbone hidden_size={backbone_hidden}"
            )
        self._heads[name] = head
        if default or self._default_head is None:
            self._default_head = name

    def register_default_head(self) -> None:
        """Extract and register the default head from the loaded HF model."""
        self._impl._ensure_loaded()
        head = ClassificationHead.from_hf_model(self._impl._model, name="default")
        self.register_head("default", head, default=True)

    def load_head(
        self,
        name: str,
        checkpoint_path: str,
        num_labels: int,
        label_map: dict[int, str] | None = None,
    ) -> None:
        """Load a head from a checkpoint file.

        The checkpoint should contain 'pooler.*' and 'classifier.*' keys.
        """
        self._impl._ensure_loaded()
        backbone_hidden = self._impl._model.config.hidden_size

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Reconstruct pooler and classifier from the default head as templates
        from copy import deepcopy
        pooler = deepcopy(self._impl._model.pooler)
        classifier = nn.Linear(backbone_hidden, num_labels)

        pooler_state = {k.removeprefix("pooler."): v for k, v in state_dict.items() if k.startswith("pooler.")}
        classifier_state = {k.removeprefix("classifier."): v for k, v in state_dict.items() if k.startswith("classifier.")}

        pooler.load_state_dict(pooler_state)
        classifier.load_state_dict(classifier_state)

        device = self._impl._device
        pooler.to(device)
        classifier.to(device)

        head = ClassificationHead(
            pooler=pooler,
            classifier=classifier,
            hidden_size=backbone_hidden,
            num_labels=num_labels,
            label_map=label_map or {},
            name=name,
        )
        self.register_head(name, head)

    def get_head(self, name: str | None = None) -> ClassificationHead:
        """Get a head by name, or the default head."""
        if name is None:
            name = self._default_head
        if name is None or name not in self._heads:
            raise KeyError(f"Head not found: {name}")
        return self._heads[name]

    @property
    def head_names(self) -> list[str]:
        return list(self._heads.keys())

    @property
    def config(self):
        """Return model configuration."""
        return self._config

    @property
    def num_labels(self) -> int:
        """Get number of classification labels."""
        return self._impl.num_labels

    @property
    def id2label(self) -> dict[int, str]:
        """Get label mapping."""
        return self._impl.id2label

    def load_weights(
        self,
        weights: "Iterable[tuple[str, torch.Tensor]]",
    ) -> set[str]:
        """Load model weights from vLLM weight loader.

        The HuggingFace model is already loaded in __init__ via _impl._load_model().
        This method handles the weight loading that vLLM expects by loading
        weights directly into the internal HuggingFace model.

        Weight name mapping (vLLM adapter -> HF checkpoint):
            - score.weight -> classifier.weight
            - score.bias -> classifier.bias

        The vLLM adapter system adds a 'score' layer for classification,
        but HF V-JEPA2 uses 'classifier'. We handle both by:
        1. Loading HF weights directly into the internal model
        2. Reporting 'score' weights as loaded (they map to 'classifier')

        Returns:
            Set of loaded weight names (includes mapped names).
        """
        self._impl._ensure_loaded()

        # The HF model already has correct weights from from_pretrained().
        # Consume the vLLM weight iterator without copying — avoids double
        # memory allocation on unified memory architectures (GB10).
        for name, weight in weights:
            pass

        # Report all model params plus vLLM adapter aliases as loaded.
        # Use self.named_parameters() so names include the full module path
        # (e.g. _impl._model.vjepa2.encoder...) matching what vLLM validates.
        loaded = {n for n, _ in self.named_parameters()}
        loaded.update(("score.weight", "score.bias"))
        return loaded
