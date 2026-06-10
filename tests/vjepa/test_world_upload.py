# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for World API video upload functionality."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from vllm_omni.entrypoints.openai.serving_world import OmniOpenAIServingWorld
from vllm_omni.entrypoints.openai.protocol.world import WorldUploadResponse


@pytest.fixture
def serving_world():
    """Create a mock serving world instance."""
    mock_engine = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        serving = OmniOpenAIServingWorld(
            engine_client=mock_engine,
            model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
            storage_path=tmpdir,
        )
        yield serving


@pytest.mark.asyncio
async def test_upload_video_success(serving_world):
    """Test successful video upload."""
    video_content = b"fake video content for testing"
    filename = "test_video.mp4"
    content_type = "video/mp4"

    response = await serving_world.upload_video(
        content=video_content,
        filename=filename,
        content_type=content_type,
    )

    assert isinstance(response, WorldUploadResponse)
    assert response.video_id.startswith("vid_")
    assert response.uri.startswith("file://")
    assert response.size_bytes == len(video_content)
    assert response.content_type == content_type

    # Verify file was written
    uri_path = response.uri.replace("file://", "")
    assert Path(uri_path).exists()
    assert Path(uri_path).read_bytes() == video_content


@pytest.mark.asyncio
async def test_upload_video_generates_unique_ids(serving_world):
    """Test that each upload gets a unique video_id."""
    video_content = b"fake video content"

    response1 = await serving_world.upload_video(
        content=video_content,
        filename="video1.mp4",
        content_type="video/mp4",
    )
    response2 = await serving_world.upload_video(
        content=video_content,
        filename="video2.mp4",
        content_type="video/mp4",
    )

    assert response1.video_id != response2.video_id
    assert response1.uri != response2.uri
