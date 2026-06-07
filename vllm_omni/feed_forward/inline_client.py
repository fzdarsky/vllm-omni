# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline (same-process) client for FeedForwardEngine.

Loads the model once and provides it to OmniOpenAIServingWorld, avoiding
the duplicate model copy that occurs when StageEngineCoreProc also loads
the model in a subprocess.

Pattern follows InlineStageDiffusionClient from the diffusion module.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from vllm_omni.feed_forward.data import FeedForwardConfig
from vllm_omni.feed_forward.engine import FeedForwardEngine

logger = logging.getLogger(__name__)


class InlineFeedForwardClient:
    """Same-process client that owns the model and FeedForwardEngine.

    When passed as ``engine_client`` to OmniOpenAIServingWorld, the serving
    layer can use ``client.model`` and ``client.processor`` instead of
    loading its own copy.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        heads_config: list[dict] | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.stage_configs: list[Any] = []

        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load the HuggingFace model and processor once."""
        from transformers import AutoModelForVideoClassification, AutoVideoProcessor

        logger.info("InlineFeedForwardClient loading model: %s", model_name)
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train(False)

        self.processor = AutoVideoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        logger.info(
            "Model loaded: %s (%s labels)",
            type(self.model).__name__,
            getattr(self.model.config, "num_labels", "unknown"),
        )

    async def get_vllm_config(self) -> None:
        """Return None — FeedForward models don't use vLLM engine config."""
        return None
