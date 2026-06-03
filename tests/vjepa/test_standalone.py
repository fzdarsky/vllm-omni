# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests that run without vllm dependency.

These tests verify basic functionality that doesn't require the full
vllm/vllm-omni stack. Useful for quick validation on dev machines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# =============================================================================
# Import Tests - verify modules load without vllm
# =============================================================================


class TestModuleImports:
    """Test that io_processor modules can be imported in isolation."""

    def test_patch_profile_standalone(self):
        """PatchProfile can be imported and used standalone."""
        # Import directly from the module file
        import importlib.util

        module_path = Path(__file__).parent.parent.parent / "vllm_omni" / "io_processor" / "patch_profile.py"
        spec = importlib.util.spec_from_file_location("patch_profile", module_path)
        patch_profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_profile)

        # Test functionality
        profile = patch_profile.PatchProfile.vjepa()
        assert profile.name == "vjepa"
        assert profile.temporal_size == 2
        assert profile.spatial_size == (16, 16)

    def test_gstreamer_enums_standalone(self):
        """GStreamer module enums work without full vllm stack."""
        import importlib.util

        module_path = Path(__file__).parent.parent.parent / "vllm_omni" / "io_processor" / "gstreamer.py"
        spec = importlib.util.spec_from_file_location("gstreamer", module_path)
        gstreamer = importlib.util.module_from_spec(spec)

        # Register module in sys.modules before exec (required for @dataclass)
        sys.modules["gstreamer"] = gstreamer

        try:
            spec.loader.exec_module(gstreamer)
        except ImportError:
            pytest.skip("GStreamer dependencies not available")
        finally:
            # Clean up
            sys.modules.pop("gstreamer", None)

        # Test HardwareTarget enum
        assert hasattr(gstreamer, "HardwareTarget")
        assert gstreamer.HardwareTarget.CPU.value == "cpu"
        assert gstreamer.HardwareTarget.NVIDIA.value == "nvidia"


# =============================================================================
# Configuration Tests - dataclass validation
# =============================================================================


class TestDataclassValidation:
    """Test dataclass definitions work correctly."""

    def test_source_config_fields(self):
        """SourceConfig has expected fields."""
        import importlib.util

        module_path = Path(__file__).parent.parent.parent / "vllm_omni" / "io_processor" / "gstreamer.py"
        spec = importlib.util.spec_from_file_location("gstreamer", module_path)
        gstreamer = importlib.util.module_from_spec(spec)

        # Register module in sys.modules before exec (required for @dataclass)
        sys.modules["gstreamer"] = gstreamer

        try:
            spec.loader.exec_module(gstreamer)
        except ImportError:
            pytest.skip("GStreamer dependencies not available")
        finally:
            sys.modules.pop("gstreamer", None)

        # Create SourceConfig
        config = gstreamer.SourceConfig(
            type="file",
            path="/test/video.mp4"
        )
        assert config.type == "file"
        assert config.path == "/test/video.mp4"
        assert config.uri is None
        assert config.device is None

    def test_pipeline_config_defaults(self):
        """PipelineConfig has correct defaults."""
        import importlib.util

        module_path = Path(__file__).parent.parent.parent / "vllm_omni" / "io_processor" / "gstreamer.py"
        spec = importlib.util.spec_from_file_location("gstreamer", module_path)
        gstreamer = importlib.util.module_from_spec(spec)

        # Register module in sys.modules before exec (required for @dataclass)
        sys.modules["gstreamer"] = gstreamer

        try:
            spec.loader.exec_module(gstreamer)
        except ImportError:
            pytest.skip("GStreamer dependencies not available")
        finally:
            sys.modules.pop("gstreamer", None)

        config = gstreamer.PipelineConfig()
        assert config.target_width == 256
        assert config.target_height == 256
        assert config.target_fps == 30
        assert config.max_buffers == 2
        assert config.sources == []


# =============================================================================
# Video Generation Tests
# =============================================================================


class TestVideoGeneration:
    """Test synthetic video generation for fixtures."""

    def test_opencv_available(self):
        """OpenCV is available for video generation."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("OpenCV not available")

        # Create a simple frame
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[100:150, 100:150] = [255, 0, 0]  # Blue square (BGR)

        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8

    def test_video_creation(self, tmp_path: Path):
        """Synthetic video can be created with OpenCV."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("OpenCV not available")

        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (256, 256))

        for i in range(16):
            frame = np.full((256, 256, 3), fill_value=i * 15, dtype=np.uint8)
            writer.write(frame)

        writer.release()

        assert video_path.exists()
        assert video_path.stat().st_size > 0

        # Verify video can be read back
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Note: frame count may vary due to codec behavior
        assert frame_count >= 10


# =============================================================================
# GStreamer Availability Test
# =============================================================================


class TestGStreamerAvailability:
    """Test GStreamer availability detection."""

    def test_gstreamer_check(self):
        """GST_AVAILABLE flag is set correctly."""
        import importlib.util

        module_path = Path(__file__).parent.parent.parent / "vllm_omni" / "io_processor" / "gstreamer.py"
        spec = importlib.util.spec_from_file_location("gstreamer", module_path)
        gstreamer = importlib.util.module_from_spec(spec)

        # Register module in sys.modules before exec (required for @dataclass)
        sys.modules["gstreamer"] = gstreamer

        try:
            spec.loader.exec_module(gstreamer)
        except ImportError:
            pytest.skip("Module dependencies not available")
        finally:
            sys.modules.pop("gstreamer", None)

        # GST_AVAILABLE should be a boolean
        assert isinstance(gstreamer.GST_AVAILABLE, bool)

        # If GStreamer is available, Gst should be importable
        if gstreamer.GST_AVAILABLE:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            assert Gst is not None
