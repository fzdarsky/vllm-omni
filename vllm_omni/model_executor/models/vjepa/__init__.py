# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V-JEPA World Model support for vLLM-Omni.

This package provides V-JEPA 2 integration for vLLM-Omni, including:
- VJepa2Model: HuggingFace V-JEPA2 wrapper with tracing
- VJepa2Encoder: vLLM-Omni integration wrapper
- FrameBuffer/ActionBuffer: Streaming support with NTP sync
- VJepa2Metrics: Latency measurement for benchmarking
"""
from vllm_omni.model_executor.models.vjepa.encoder import (
    ActionBuffer,
    FrameBuffer,
    SpanTiming,
    VJepa2Config,
    VJepa2Encoder,
    VJepa2Metrics,
    VJepa2Model,
)

__all__ = [
    "ActionBuffer",
    "FrameBuffer",
    "SpanTiming",
    "VJepa2Config",
    "VJepa2Encoder",
    "VJepa2Metrics",
    "VJepa2Model",
]
