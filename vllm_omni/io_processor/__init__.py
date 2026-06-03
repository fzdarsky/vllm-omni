# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision IO Processor for V-JEPA video ingest.

Components:
- VisionIOProcessor: GStreamer-based video ingest with OmniConnector delivery
- GStreamerPipeline: Hardware-accelerated video decode and compositing
- PatchProfile: Configuration for V-JEPA tubelet extraction

Architecture:
    VisionIOProcessor runs in a separate process to avoid Python GIL
    contention. It uses GStreamer for hardware-accelerated video decode
    and delivers frames via OmniConnector to Stage 0 (VJepa2Model).

    Stage 0 contains FrameBuffer for accumulation and patchification.

Usage:
    # In-process (for testing)
    processor = VisionIOProcessor(config)
    processor.configure(connector, session_id)
    processor.start()

    # Separate process (for production)
    handle = VisionIOProcessor.spawn(config, connector_config, session_id)
    # ... later ...
    handle.stop()
"""
from vllm_omni.io_processor.gstreamer import (
    GST_AVAILABLE,
    GStreamerPipeline,
    HardwareDetector,
    HardwareTarget,
    PipelineConfig,
    SourceConfig,
    create_file_source_pipeline,
)
from vllm_omni.io_processor.moq_transport import (
    MoQSession,
    PortPool,
    create_moq_session,
    get_port_pool,
)
from vllm_omni.io_processor.patch_profile import PatchProfile
from vllm_omni.io_processor.vision import (
    VisionIOProcessor,
    VisionProcessorConfig,
    VisionProcessorHandle,
    VisionWorkerPool,
)

__all__ = [
    # Main processor
    "VisionIOProcessor",
    "VisionProcessorConfig",
    "VisionProcessorHandle",
    "VisionWorkerPool",
    # GStreamer pipeline
    "GStreamerPipeline",
    "PipelineConfig",
    "SourceConfig",
    "HardwareDetector",
    "HardwareTarget",
    "GST_AVAILABLE",
    "create_file_source_pipeline",
    # MoQ transport (browser camera)
    "MoQSession",
    "PortPool",
    "create_moq_session",
    "get_port_pool",
    # Configuration
    "PatchProfile",
]
