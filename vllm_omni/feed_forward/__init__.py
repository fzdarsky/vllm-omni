# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FeedForwardEngine — single-pass inference engine for non-iterative models.

Aligned with RFC #2727 (ReconstructionEngine). Handles models that produce
output in a single forward pass (V-JEPA, VGGT, MAE, DINO, etc.) without
the noise scheduling, timestep management, and iterative denoising of
DiffusionEngine.

Key design choices:
- No step scheduling — always dispatch immediately
- Session-aware — per-request state for streaming video / action-conditioned loops
- Extensible output — embeddings, logits, latent predictions
- Reuses DiffusionExecutor's RPC and worker infrastructure
"""

from vllm_omni.feed_forward.data import FeedForwardOutput
from vllm_omni.feed_forward.engine import FeedForwardEngine
from vllm_omni.feed_forward.request import FeedForwardRequest, FeedForwardSamplingParams

__all__ = [
    "FeedForwardEngine",
    "FeedForwardOutput",
    "FeedForwardRequest",
    "FeedForwardSamplingParams",
]
