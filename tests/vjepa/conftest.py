# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures for V-JEPA integration tests."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture
def test_video_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a synthetic test video with moving patterns.

    Generates a 2-second video at 30fps with 256x256 resolution.
    The video contains moving colored rectangles for visual testing.
    """
    video_path = tmp_path / "test_video.mp4"

    # Video parameters matching V-JEPA defaults
    width, height = 256, 256
    fps = 30
    duration_s = 2.0
    num_frames = int(fps * duration_s)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for i in range(num_frames):
            # Create frame with moving pattern
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Moving red rectangle
            x = int((i / num_frames) * (width - 50))
            cv2.rectangle(frame, (x, 50), (x + 50, 100), (0, 0, 255), -1)

            # Moving green rectangle (opposite direction)
            x2 = width - 50 - x
            cv2.rectangle(frame, (x2, 100), (x2 + 50, 150), (0, 255, 0), -1)

            # Static blue rectangle
            cv2.rectangle(frame, (100, 180), (150, 230), (255, 0, 0), -1)

            writer.write(frame)

        writer.release()
        yield video_path

    finally:
        if video_path.exists():
            video_path.unlink()


@pytest.fixture
def short_video_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a minimal test video (16 frames for one V-JEPA clip)."""
    video_path = tmp_path / "short_video.mp4"

    width, height = 256, 256
    fps = 30
    num_frames = 16  # Exactly one V-JEPA clip

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=i * 15, dtype=np.uint8)
            writer.write(frame)

        writer.release()
        yield video_path

    finally:
        if video_path.exists():
            video_path.unlink()


@pytest.fixture
def gstreamer_available() -> bool:
    """Check if GStreamer is available."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        return True
    except (ImportError, ValueError):
        return False
