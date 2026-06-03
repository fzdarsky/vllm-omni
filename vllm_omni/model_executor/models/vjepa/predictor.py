# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
V-JEPA Predictor (Stage 1) for vLLM-Omni.

This module provides the predictor stage of the V-JEPA pipeline, handling:
- Latent prediction from encoder output
- Action conditioning (input_sources: [0, -1])
- Attention pooling and classification head

The actual implementation is in vllm_jepa.models.predictor. This file
provides the vLLM-Omni integration wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class PredictionOutput:
    """Output from V-JEPA predictor."""

    logits: torch.Tensor
    predicted_latents: torch.Tensor | None = None
    pooled: torch.Tensor | None = None


class VJepa2Predictor(nn.Module):
    """
    V-JEPA Predictor for vLLM-Omni Stage 1.

    Receives latents from encoder (Stage 0) and optional action tensors
    from external sources (input_sources: [0, -1]).

    Protocol markers:
        supports_multimodal_raw_input_only: Accept raw tensors
        is_attention_free: No KV-cache needed
    """

    supports_multimodal_raw_input_only = True
    is_attention_free = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # Import the actual implementation
        try:
            from vllm_jepa.models.predictor import VJepa2Predictor as VJepa2PredictorImpl
            from vllm_jepa.models.predictor import PredictorConfig

            # Extract config from vllm_config
            model_config = getattr(vllm_config, "model_config", None)
            if model_config and hasattr(model_config, "hf_config"):
                hf_config = model_config.hf_config
                config = PredictorConfig(
                    embed_dim=getattr(hf_config, "embed_dim", 1024),
                    num_heads=getattr(hf_config, "num_heads", 16),
                    num_layers=getattr(hf_config, "num_layers", 6),
                    action_dim=getattr(hf_config, "action_dim", 7),
                    num_classes=getattr(hf_config, "num_classes", 400),
                )
            else:
                config = PredictorConfig()

            self._impl = VJepa2PredictorImpl(config=config)
        except ImportError:
            # Fallback stub for testing without vllm_jepa installed
            self._impl = None
            self._num_classes = 400

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> PredictionOutput:
        """
        Generate predictions from encoder latents.

        Args:
            input_ids: Not used (non-AR model)
            positions: Not used
            intermediate_tensors: Not used
            inputs_embeds: Not used
            **kwargs: Must contain:
                - latents: Encoder output (B, N, D)
                - action: Optional action tensor (B, action_dim)
                  or actions: Per-frame actions (B, T, action_dim)

        Returns:
            PredictionOutput with logits and optional intermediate tensors.
        """
        if self._impl is not None:
            return self._impl.forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        # Stub: return mock predictions
        latents = kwargs.get("latents")
        batch_size = latents.shape[0] if latents is not None else 1
        return PredictionOutput(
            logits=torch.randn(batch_size, self._num_classes),
        )
