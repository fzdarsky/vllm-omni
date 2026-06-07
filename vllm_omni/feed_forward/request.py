# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request types for FeedForwardEngine.

Aligned with DreamZero/OpenPI's request pattern: observations are passed
via sampling_params.extra_args, session state via session_id + reset flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from opentelemetry import context as otel_context
    _HAS_OTEL = True
except ImportError:
    otel_context = None  # type: ignore[assignment]
    _HAS_OTEL = False

DUMMY_FF_REQUEST_ID = "dummy_ff_req_id"


@dataclass
class FeedForwardSamplingParams:
    """Parameters for a feed-forward inference request.

    Follows the OpenPI pattern: model-specific data goes in extra_args,
    session lifecycle via session_id + reset.
    """

    session_id: str | None = None
    head: str | None = None
    top_k: int = 5
    reset: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedForwardRequest:
    """A request to the FeedForwardEngine.

    The pixel_values tensor is the primary input (video frames preprocessed
    by AutoVideoProcessor). For models that also accept text, prompt carries
    the text input.
    """

    request_id: str
    pixel_values: torch.Tensor | None = None
    prompt: str = ""
    sampling_params: FeedForwardSamplingParams = field(
        default_factory=FeedForwardSamplingParams,
    )
    trace_context: Any = None

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string.")
