# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V-JEPA VisionIOProcessor and GStreamer pipeline.

These tests verify:
1. GStreamer pipeline builds correctly for different source types
2. Hardware detection works across platforms
3. VisionIOProcessor delivers frames via callback
4. Subprocess spawning works correctly
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.io_processor import (
    GST_AVAILABLE,
    HardwareTarget,
    PatchProfile,
    PipelineConfig,
    SourceConfig,
    VisionProcessorConfig,
)


# =============================================================================
# PatchProfile Tests
# =============================================================================


class TestPatchProfile:
    """Tests for PatchProfile configuration."""

    def test_vjepa_profile(self):
        """V-JEPA default profile has correct settings."""
        profile = PatchProfile.vjepa()
        assert profile.name == "vjepa"
        assert profile.temporal_size == 2
        assert profile.spatial_size == (16, 16)

    def test_vjepa_large_profile(self):
        """V-JEPA large model profile has correct settings."""
        profile = PatchProfile.vjepa_large()
        assert profile.name == "vjepa_large"
        assert profile.temporal_size == 2
        assert profile.spatial_size == (14, 14)

    def test_clip_profile(self):
        """CLIP profile has single-frame temporal size."""
        profile = PatchProfile.clip()
        assert profile.name == "clip"
        assert profile.temporal_size == 1
        assert profile.spatial_size == (14, 14)

    def test_from_name(self):
        """Profiles can be retrieved by name."""
        profile = PatchProfile.from_name("vjepa")
        assert profile.temporal_size == 2

    def test_from_name_invalid(self):
        """Invalid profile name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            PatchProfile.from_name("invalid")

    def test_frozen(self):
        """PatchProfile is immutable."""
        profile = PatchProfile.vjepa()
        with pytest.raises(AttributeError):
            profile.temporal_size = 4


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSourceConfig:
    """Tests for SourceConfig dataclass."""

    def test_file_source(self):
        """File source configuration."""
        config = SourceConfig(type="file", path="/path/to/video.mp4")
        assert config.type == "file"
        assert config.path == "/path/to/video.mp4"

    def test_rtsp_source(self):
        """RTSP source configuration."""
        config = SourceConfig(type="rtsp", uri="rtsp://camera.local/stream")
        assert config.type == "rtsp"
        assert config.uri == "rtsp://camera.local/stream"

    def test_v4l2_source(self):
        """V4L2 source configuration."""
        config = SourceConfig(type="v4l2", device="/dev/video0")
        assert config.type == "v4l2"
        assert config.device == "/dev/video0"

    def test_csi_source(self):
        """CSI source configuration."""
        config = SourceConfig(type="csi", sensor_id=0)
        assert config.type == "csi"
        assert config.sensor_id == 0


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_config(self):
        """Default configuration matches V-JEPA requirements."""
        config = PipelineConfig()
        assert config.target_width == 256
        assert config.target_height == 256
        assert config.target_fps == 30
        assert config.max_buffers == 2

    def test_custom_config(self):
        """Custom configuration is respected."""
        config = PipelineConfig(
            sources=[SourceConfig(type="file", path="/test.mp4")],
            target_width=512,
            target_height=512,
            target_fps=60,
        )
        assert config.target_width == 512
        assert len(config.sources) == 1


class TestVisionProcessorConfig:
    """Tests for VisionProcessorConfig dataclass."""

    def test_default_config(self):
        """Default VisionProcessorConfig."""
        config = VisionProcessorConfig()
        assert config.target_width == 256
        assert config.target_height == 256
        assert config.from_stage == "io_processor"
        assert config.to_stage == "stage_0"

    def test_sources_list(self):
        """Sources can be provided as list of dicts."""
        config = VisionProcessorConfig(
            sources=[{"type": "file", "path": "/test.mp4"}]
        )
        assert len(config.sources) == 1
        assert config.sources[0]["type"] == "file"


# =============================================================================
# Hardware Detection Tests
# =============================================================================


class TestHardwareDetection:
    """Tests for hardware auto-detection."""

    def test_hardware_target_enum(self):
        """HardwareTarget enum has expected values."""
        assert HardwareTarget.JETSON.value == "jetson"
        assert HardwareTarget.NVIDIA.value == "nvidia"
        assert HardwareTarget.INTEL.value == "intel"
        assert HardwareTarget.AMD.value == "amd"
        assert HardwareTarget.CPU.value == "cpu"

    @pytest.mark.skipif(not GST_AVAILABLE, reason="GStreamer not available")
    def test_hardware_detector_returns_valid_target(self):
        """HardwareDetector returns a valid HardwareTarget."""
        from vllm_omni.io_processor import HardwareDetector

        target = HardwareDetector.detect()
        assert isinstance(target, HardwareTarget)


# =============================================================================
# GStreamer Pipeline Tests (require GStreamer)
# =============================================================================


@pytest.mark.skipif(not GST_AVAILABLE, reason="GStreamer not available")
class TestGStreamerPipeline:
    """Tests for GStreamerPipeline (require GStreamer)."""

    def test_pipeline_build_file_source(self, test_video_path: Path):
        """Pipeline builds successfully with file source."""
        from vllm_omni.io_processor import GStreamerPipeline

        config = PipelineConfig(
            sources=[SourceConfig(type="file", path=str(test_video_path))],
        )
        pipeline = GStreamerPipeline(config)
        gst_pipeline = pipeline.build()

        assert gst_pipeline is not None
        assert pipeline.hw_target is not None

    def test_pipeline_frame_callback(self, test_video_path: Path):
        """Pipeline delivers frames via callback."""
        from vllm_omni.io_processor import GStreamerPipeline

        frames_received = []

        def on_frame(tensor: torch.Tensor, pts_ns: int, timing: dict | None):
            frames_received.append((tensor, pts_ns))

        config = PipelineConfig(
            sources=[SourceConfig(type="file", path=str(test_video_path))],
        )
        pipeline = GStreamerPipeline(config, on_frame_callback=on_frame)
        pipeline.build()

        # Start pipeline in background
        pipeline.start(blocking=False)

        # Wait for some frames (max 5 seconds)
        timeout = 5.0
        start = time.time()
        while len(frames_received) < 5 and (time.time() - start) < timeout:
            time.sleep(0.1)

        pipeline.stop()

        assert len(frames_received) > 0, "No frames received"

        # Check frame properties
        tensor, pts_ns = frames_received[0]
        assert tensor.shape[0] == 3  # RGB channels
        assert tensor.shape[1] == 256  # Height
        assert tensor.shape[2] == 256  # Width
        assert pts_ns >= 0

    def test_pipeline_hardware_selection(self):
        """Pipeline selects appropriate elements for detected hardware."""
        from vllm_omni.io_processor import ElementSelector, HardwareTarget

        # Test all hardware targets have valid element selections
        for target in HardwareTarget:
            decoder = ElementSelector.decoder(target)
            compositor = ElementSelector.compositor(target)
            convert = ElementSelector.video_convert(target)

            assert decoder is not None
            assert compositor is not None
            assert convert is not None


# =============================================================================
# VisionIOProcessor Tests
# =============================================================================


@pytest.mark.skipif(not GST_AVAILABLE, reason="GStreamer not available")
class TestVisionIOProcessor:
    """Tests for VisionIOProcessor."""

    def test_processor_init(self):
        """VisionIOProcessor initializes correctly."""
        from vllm_omni.io_processor import VisionIOProcessor

        config = VisionProcessorConfig()
        processor = VisionIOProcessor(config)

        assert processor.config == config
        assert processor.hw_target is not None

    def test_processor_configure(self, test_video_path: Path):
        """VisionIOProcessor configures with connector and session."""
        from vllm_omni.io_processor import VisionIOProcessor

        config = VisionProcessorConfig(
            sources=[{"type": "file", "path": str(test_video_path)}]
        )
        processor = VisionIOProcessor(config)

        # Mock connector
        mock_connector = MagicMock()
        mock_connector.put.return_value = (True, 0, {})

        processor.configure(mock_connector, "test_session")

        assert processor._connector == mock_connector
        assert processor._session_id == "test_session"
        assert processor._pipeline is not None

    def test_processor_frame_delivery(self, test_video_path: Path):
        """VisionIOProcessor delivers frames to connector."""
        from vllm_omni.io_processor import VisionIOProcessor

        config = VisionProcessorConfig(
            sources=[{"type": "file", "path": str(test_video_path)}]
        )
        processor = VisionIOProcessor(config)

        # Mock connector that records calls
        put_calls = []

        def mock_put(from_stage, to_stage, put_key, data):
            put_calls.append((from_stage, to_stage, put_key, data))
            return (True, 0, {})

        mock_connector = MagicMock()
        mock_connector.put = mock_put

        processor.configure(mock_connector, "test_session")
        processor.start(blocking=False)

        # Wait for some frames
        timeout = 5.0
        start = time.time()
        while len(put_calls) < 5 and (time.time() - start) < timeout:
            time.sleep(0.1)

        processor.stop()

        assert len(put_calls) > 0, "No frames delivered to connector"

        # Check delivery format
        from_stage, to_stage, put_key, data = put_calls[0]
        assert from_stage == "io_processor"
        assert to_stage == "stage_0"
        assert "test_session" in put_key
        assert "tensor" in data
        assert "pts_ns" in data
        assert "frame_idx" in data


# =============================================================================
# Subprocess Tests
# =============================================================================


class TestVisionProcessorHandle:
    """Tests for VisionProcessorHandle subprocess management."""

    @pytest.mark.skipif(not GST_AVAILABLE, reason="GStreamer not available")
    def test_spawn_and_stop(self, test_video_path: Path):
        """VisionIOProcessor can spawn in subprocess and stop cleanly."""
        from vllm_omni.io_processor import VisionIOProcessor, VisionProcessorConfig

        config = VisionProcessorConfig(
            sources=[{"type": "file", "path": str(test_video_path)}]
        )

        # Connector config for subprocess
        connector_config = {
            "name": "SharedMemoryConnector",
            "extra": {
                "shm_threshold_bytes": 65536,
            },
        }

        # This will fail if SharedMemoryConnector isn't available,
        # which is expected in unit tests. Skip gracefully.
        try:
            handle = VisionIOProcessor.spawn(
                config=config,
                connector_config=connector_config,
                session_id="test_subprocess",
            )
        except Exception as e:
            if "OmniConnectorFactory" in str(e) or "SharedMemory" in str(e):
                pytest.skip("SharedMemoryConnector not available in test env")
            raise

        assert handle.is_alive

        # Stop subprocess
        handle.stop(timeout=5.0)

        assert not handle.is_alive
