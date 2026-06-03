# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol definitions for V-JEPA World Model API.

Pydantic models for request/response types used by /v1/world/* endpoints.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    CREATED = "created"  # Session created, not yet started
    STARTING = "starting"  # Pipeline starting
    RUNNING = "running"  # Actively processing frames
    PAUSED = "paused"  # Pipeline paused (future use)
    STOPPING = "stopping"  # Graceful shutdown in progress
    STOPPED = "stopped"  # Pipeline stopped
    ERROR = "error"  # Error state


class SourceType(str, Enum):
    """Video source types."""

    FILE = "file"
    RTSP = "rtsp"
    V4L2 = "v4l2"
    CSI = "csi"
    MOQ = "moq"  # Media over QUIC - WebTransport from browser camera


class SourceConfigRequest(BaseModel):
    """Video source configuration."""

    type: SourceType = Field(..., description="Source type: file, rtsp, v4l2, csi")
    uri: str | None = Field(None, description="URI for file or rtsp sources")
    device: str | None = Field(None, description="Device path for v4l2")
    sensor_id: int | None = Field(None, description="Sensor ID for csi")


class WorldSessionConfig(BaseModel):
    """Model configuration for world session."""

    num_frames: int = Field(default=16, description="Frames per prediction clip")
    stride: int = Field(default=8, description="Frame advance between predictions")
    ttl_seconds: int = Field(
        default=3600,
        description="Session time-to-live in seconds (0 = no expiry)",
    )


class WorldSessionCreateRequest(BaseModel):
    """Request to create a world model inference session.

    The model is determined at server startup via `vllm serve <model> --omni`.
    All sessions use the same model instance.
    """

    source: SourceConfigRequest
    config: WorldSessionConfig = Field(
        default_factory=WorldSessionConfig,
        description="Model configuration",
    )
    head: str | None = Field(
        default=None,
        description="Classification head to use (default: server's default head)",
    )


class WorldSessionCreateResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    status: SessionStatus
    moq_endpoint: str | None = Field(
        default=None,
        description="Deprecated: use moq_relay_url instead",
    )
    moq_relay_url: str | None = Field(
        default=None,
        description="MoQ relay WebTransport URL for browser publishing",
    )
    moq_broadcast_path: str | None = Field(
        default=None,
        description="MoQ broadcast path (session_id.hang) for pub/sub rendezvous",
    )


class WorldSessionStatusResponse(BaseModel):
    """Session status response."""

    session_id: str
    status: SessionStatus
    model: str
    config: WorldSessionConfig
    source: SourceConfigRequest
    frames_processed: int
    predictions_count: int
    created_at: float
    started_at: float | None = None
    error_message: str | None = None


class WorldSessionDeleteResponse(BaseModel):
    """Response from session deletion."""

    session_id: str
    status: str = "deleted"


class WorldSessionListResponse(BaseModel):
    """Response listing all sessions."""

    sessions: list[WorldSessionStatusResponse]
    total: int


class PredictionLabel(BaseModel):
    """A single prediction label with probability."""

    label: str
    probability: float


class WorldPrediction(BaseModel):
    """A single prediction result."""

    labels: list[PredictionLabel] = Field(description="Top predicted labels")
    frame_range: tuple[int, int] = Field(description="(start_frame, end_frame)")
    timestamp_ns: int = Field(description="Frame timestamp in nanoseconds")
    request_id: str = Field(description="Session ID")


class WorldPredictionEvent(BaseModel):
    """SSE event payload for predictions."""

    prediction: WorldPrediction | None = None
    status: str | None = None  # "ended" when stream completes
    error: str | None = None
    frames_processed: int | None = None


class WorldHeadInfo(BaseModel):
    """Information about a single classification head."""

    name: str = Field(..., description="Head name")
    num_labels: int = Field(..., description="Number of classification labels")
    is_default: bool = Field(default=False, description="Whether this is the default head")


class WorldHeadsResponse(BaseModel):
    """Response listing available classification heads."""

    heads: list[WorldHeadInfo]


class WorldHealthResponse(BaseModel):
    """Health check response for world model."""

    status: str
    model_loaded: bool
    gstreamer_available: bool


class WorldUploadResponse(BaseModel):
    """Response from video upload endpoint."""

    video_id: str = Field(..., description="Unique identifier for the uploaded video")
    uri: str = Field(..., description="File path URI for use in session creation")
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes")
    content_type: str = Field(..., description="MIME type of the uploaded file")
