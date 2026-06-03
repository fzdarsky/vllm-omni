# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for V-JEPA pipeline.

These tests verify the complete flow:
    VisionIOProcessor → OmniConnector → Stage 0 (VJepa2Model) → Predictions

Tests are structured to run with or without GPU, using appropriate fixtures.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest
import torch

from vllm_omni.io_processor import GST_AVAILABLE


# =============================================================================
# Stage 0 Model Tests (VJepa2Model)
# =============================================================================


class TestVJepa2Model:
    """Tests for VJepa2Model inference."""

    @pytest.fixture
    def model_config(self):
        """Configuration for test model."""
        from vllm_omni.model_executor.models.vjepa.encoder import VJepa2Config

        return VJepa2Config(
            model_id="facebook/vjepa2-vitl-fpc16-256-ssv2",
            num_frames=16,
            stride=8,
            device="cpu",  # Use CPU for tests
        )

    def test_frame_buffer_accumulation(self, model_config):
        """FrameBuffer accumulates frames correctly."""
        from vllm_omni.model_executor.models.vjepa.encoder import FrameBuffer

        buffer = FrameBuffer(num_frames=16, stride=8)

        # Add frames
        for i in range(16):
            frame = torch.randn(3, 256, 256)
            buffer.add_frame(frame, pts_ns=i * 33_000_000)

        assert buffer.ready()

        # Consume clip
        clip, actions = buffer.consume()
        assert clip.shape == (16, 3, 256, 256)

        # After stride=8, we should have 8 frames left
        assert len(buffer._frames) == 8
        assert not buffer.ready()

    def test_action_buffer_lookup(self):
        """ActionBuffer provides timestamp-based action lookup."""
        from vllm_omni.model_executor.models.vjepa.encoder import ActionBuffer

        buffer = ActionBuffer(capacity=100, tolerance_ns=1_000_000)

        # Add actions at specific timestamps
        for i in range(10):
            action = torch.tensor([i * 0.1] * 7)
            buffer.add(timestamp_ns=i * 33_000_000, action=action)

        # Look up actions for frame timestamps
        frame_pts = [0, 33_000_000, 66_000_000]
        actions = buffer.get_for_frames(frame_pts)

        assert actions.shape == (3, 7)
        assert torch.allclose(actions[0], torch.tensor([0.0] * 7))
        assert torch.allclose(actions[1], torch.tensor([0.1] * 7))
        assert torch.allclose(actions[2], torch.tensor([0.2] * 7))

    @pytest.mark.slow
    def test_model_batch_inference(self, model_config):
        """VJepa2Model runs batch inference correctly.

        Note: This test downloads the HuggingFace model on first run.
        """
        from vllm_omni.model_executor.models.vjepa.encoder import VJepa2Model

        model = VJepa2Model(config=model_config)

        # Create synthetic video frames (16 frames, 3 channels, 256x256)
        frames = torch.randint(0, 255, (16, 3, 256, 256), dtype=torch.uint8)

        # Run inference
        result = model.forward(frames=frames)

        assert result is not None
        assert "logits" in result
        assert result["logits"].shape[0] == 1  # Batch size 1


# =============================================================================
# Session-Based API Tests
# =============================================================================


class TestServingWorld:
    """Tests for OmniOpenAIServingWorld session management."""

    @pytest.fixture
    def serving_handler(self):
        """Create a serving handler with mock engine."""
        from vllm_omni.entrypoints.openai.serving_world import OmniOpenAIServingWorld

        mock_engine = MagicMock()
        handler = OmniOpenAIServingWorld(
            engine_client=mock_engine,
            model_name="test-vjepa",
        )
        return handler

    @pytest.mark.asyncio
    async def test_open_session(self, serving_handler):
        """Session creation returns valid response."""
        from vllm_omni.entrypoints.openai.protocol.world import (
            SourceConfigRequest,
            SourceType,
            WorldSessionConfig,
            WorldSessionCreateRequest,
        )

        request = WorldSessionCreateRequest(
            source=SourceConfigRequest(
                type=SourceType.FILE,
                uri="file:///test/video.mp4",
            ),
            config=WorldSessionConfig(num_frames=16, stride=8),
        )

        response = await serving_handler.open_session(request)

        assert response.session_id.startswith("sess_")
        assert response.status.value == "created"

    @pytest.mark.asyncio
    async def test_get_session(self, serving_handler):
        """Session status can be retrieved after creation."""
        from vllm_omni.entrypoints.openai.protocol.world import (
            SourceConfigRequest,
            SourceType,
            WorldSessionConfig,
            WorldSessionCreateRequest,
        )

        # Create session
        request = WorldSessionCreateRequest(
            source=SourceConfigRequest(
                type=SourceType.FILE,
                uri="file:///test/video.mp4",
            ),
            config=WorldSessionConfig(),
        )
        create_response = await serving_handler.open_session(request)

        # Get session
        status = await serving_handler.get_session(create_response.session_id)

        assert status is not None
        assert status.session_id == create_response.session_id
        assert status.model == "test-vjepa"

    @pytest.mark.asyncio
    async def test_list_sessions(self, serving_handler):
        """Session listing returns all sessions."""
        from vllm_omni.entrypoints.openai.protocol.world import (
            SourceConfigRequest,
            SourceType,
            WorldSessionConfig,
            WorldSessionCreateRequest,
        )

        # Create multiple sessions
        for _ in range(3):
            request = WorldSessionCreateRequest(
                source=SourceConfigRequest(type=SourceType.FILE, uri="file:///test.mp4"),
                config=WorldSessionConfig(),
            )
            await serving_handler.open_session(request)

        # List sessions
        response = await serving_handler.list_sessions()

        assert response.total == 3
        assert len(response.sessions) == 3

    @pytest.mark.asyncio
    async def test_close_session(self, serving_handler):
        """Session deletion removes session from registry."""
        from vllm_omni.entrypoints.openai.protocol.world import (
            SourceConfigRequest,
            SourceType,
            WorldSessionConfig,
            WorldSessionCreateRequest,
        )

        # Create session
        request = WorldSessionCreateRequest(
            source=SourceConfigRequest(type=SourceType.FILE, uri="file:///test.mp4"),
            config=WorldSessionConfig(),
        )
        create_response = await serving_handler.open_session(request)
        session_id = create_response.session_id

        # Delete session
        delete_response = await serving_handler.close_session(session_id)
        assert delete_response.session_id == session_id

        # Verify deletion
        status = await serving_handler.get_session(session_id)
        assert status is None

    @pytest.mark.asyncio
    async def test_frame_consumer_accumulation(self, serving_handler):
        """Frame consumer accumulates frames and triggers inference."""
        from vllm_omni.entrypoints.openai.protocol.world import (
            SessionStatus,
            SourceConfigRequest,
            SourceType,
            WorldSessionConfig,
            WorldSessionCreateRequest,
        )

        # Create mock model that returns logits
        mock_model = MagicMock()
        mock_model.forward.return_value = torch.randn(1, 400)  # Fake logits
        mock_model.id2label = {i: f"class_{i}" for i in range(400)}
        serving_handler._model = mock_model

        # Create session
        request = WorldSessionCreateRequest(
            source=SourceConfigRequest(type=SourceType.FILE, uri="file:///test.mp4"),
            config=WorldSessionConfig(num_frames=4, stride=2),  # Small for testing
        )
        create_response = await serving_handler.open_session(request)
        session = serving_handler._sessions[create_response.session_id]

        # Verify session has prediction queue
        assert session._prediction_queue is not None

        # Manually test queue_prediction
        test_logits = torch.randn(1, 400)
        serving_handler.queue_prediction(
            session=session,
            logits=test_logits,
            pts_ns=1000000,
            id2label=mock_model.id2label,
        )

        # Verify prediction was queued
        assert not session._prediction_queue.empty()
        prediction = await session._prediction_queue.get()
        assert prediction is not None
        assert len(prediction.labels) > 0
        assert session.predictions_count == 1


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


@pytest.mark.skipif(not GST_AVAILABLE, reason="GStreamer not available")
class TestE2EPipeline:
    """End-to-end pipeline tests (require GStreamer)."""

    @pytest.mark.asyncio
    async def test_video_to_frames_delivery(self, test_video_path: Path):
        """Video file is decoded and frames are delivered."""
        from vllm_omni.io_processor import (
            GStreamerPipeline,
            PipelineConfig,
            SourceConfig,
        )

        frames_received = []
        frame_event = asyncio.Event()

        def on_frame(tensor: torch.Tensor, pts_ns: int, timing: dict | None):
            frames_received.append({
                "tensor": tensor,
                "pts_ns": pts_ns,
                "timing": timing,
            })
            if len(frames_received) >= 16:
                frame_event.set()

        config = PipelineConfig(
            sources=[SourceConfig(type="file", path=str(test_video_path))],
        )
        pipeline = GStreamerPipeline(config, on_frame_callback=on_frame)
        pipeline.build()
        pipeline.start(blocking=False)

        try:
            # Wait for 16 frames (one V-JEPA clip)
            await asyncio.wait_for(frame_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        finally:
            pipeline.stop()

        assert len(frames_received) >= 16, f"Only received {len(frames_received)} frames"

        # Verify frame format
        frame = frames_received[0]
        assert frame["tensor"].shape == (3, 256, 256)
        assert frame["pts_ns"] >= 0
        assert frame["timing"] is not None
        assert "input_receive_ns" in frame["timing"]
        assert "input_decode_ns" in frame["timing"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_pipeline_inference(self, test_video_path: Path):
        """Full pipeline produces predictions from video input.

        This test verifies the complete flow:
        1. GStreamer decodes video
        2. Frames accumulate in FrameBuffer
        3. VJepa2Model processes clip
        4. Predictions are returned

        Note: Downloads HuggingFace model on first run.
        """
        from vllm_omni.io_processor import (
            GStreamerPipeline,
            PipelineConfig,
            SourceConfig,
        )
        from vllm_omni.model_executor.models.vjepa.encoder import (
            FrameBuffer,
            VJepa2Config,
            VJepa2Model,
        )

        # Setup FrameBuffer
        frame_buffer = FrameBuffer(num_frames=16, stride=8)
        predictions = []
        prediction_event = asyncio.Event()

        # Setup model (lazy load)
        model = None

        def on_frame(tensor: torch.Tensor, pts_ns: int, timing: dict | None):
            nonlocal model

            frame_buffer.add_frame(tensor, pts_ns)

            if frame_buffer.ready():
                # Load model on first clip (lazy)
                if model is None:
                    config = VJepa2Config(device="cpu")
                    model = VJepa2Model(config=config)

                # Process clip
                clip, actions = frame_buffer.consume()
                result = model.forward(frames=clip)

                if result is not None and "logits" in result:
                    predictions.append(result)
                    prediction_event.set()

        config = PipelineConfig(
            sources=[SourceConfig(type="file", path=str(test_video_path))],
        )
        pipeline = GStreamerPipeline(config, on_frame_callback=on_frame)
        pipeline.build()
        pipeline.start(blocking=False)

        try:
            # Wait for first prediction
            await asyncio.wait_for(prediction_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for prediction")
        finally:
            pipeline.stop()

        assert len(predictions) >= 1
        assert "logits" in predictions[0]


# =============================================================================
# V-JEPA Tracing Tests
# =============================================================================


class TestVJepa2Metrics:
    """Tests for V-JEPA tracing and metrics collection."""

    def test_span_timing(self):
        """SpanTiming calculates duration correctly."""
        from vllm_omni.model_executor.models.vjepa.encoder import SpanTiming

        span = SpanTiming(
            name="test_span",
            start_time_ns=1_000_000,
            end_time_ns=2_000_000,
        )

        assert span.duration_ms == 1.0

    def test_metrics_collection(self):
        """VJepa2Metrics collects and aggregates spans."""
        from vllm_omni.model_executor.models.vjepa.encoder import (
            SpanTiming,
            VJepa2Metrics,
        )

        metrics = VJepa2Metrics(request_id="test-001")

        # Add spans
        metrics.spans.append(SpanTiming("input_preprocess", 0, 10_000_000))
        metrics.spans.append(SpanTiming("jepa_encode", 10_000_000, 100_000_000))
        metrics.spans.append(SpanTiming("jepa_predict", 100_000_000, 120_000_000))
        metrics.spans.append(SpanTiming("jepa_pool", 120_000_000, 130_000_000))

        # Check L_comp calculation
        assert metrics.l_comp_ms == 130.0  # Total: 130ms

        # Check span retrieval
        encode_span = metrics.get_span("jepa_encode")
        assert encode_span is not None
        assert encode_span.duration_ms == 90.0

    def test_metrics_serialization(self):
        """VJepa2Metrics serializes to dict correctly."""
        from vllm_omni.model_executor.models.vjepa.encoder import (
            SpanTiming,
            VJepa2Metrics,
        )

        metrics = VJepa2Metrics(request_id="test-001")
        metrics.spans.append(SpanTiming("jepa_encode", 0, 100_000_000))

        data = metrics.to_dict()

        assert data["request_id"] == "test-001"
        assert data["l_comp_ms"] == 100.0
        assert "jepa_encode" in data["spans"]
        assert data["spans"]["jepa_encode"]["duration_ms"] == 100.0
