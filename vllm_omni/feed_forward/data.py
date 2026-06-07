# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data types for FeedForwardEngine inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class FeedForwardOutput:
    """Output from a single feed-forward inference pass.

    Extensible: models can return embeddings, logits, latent predictions,
    or any combination. Consumer (serving layer) interprets based on model type.
    """

    logits: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    predictions: list[dict[str, Any]] | None = None
    custom_output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_cpu(self) -> FeedForwardOutput:
        """Move tensors to CPU for serialization."""
        return FeedForwardOutput(
            logits=self.logits.cpu() if self.logits is not None else None,
            embedding=self.embedding.cpu() if self.embedding is not None else None,
            predictions=self.predictions,
            custom_output=self.custom_output,
            error=self.error,
        )


@dataclass
class FeedForwardConfig:
    """Configuration for FeedForwardEngine.

    Minimal compared to OmniDiffusionConfig — no noise schedules,
    no timestep management, no latent space configuration.
    """

    model_class_name: str = ""
    device: str = "cuda"
    dtype: str = "float16"
    max_num_running_reqs: int = 1
    enable_cpu_offload: bool = False
