# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V-JEPA 2 pipeline topology (frozen).

Single-stage world model for video understanding:
  Stage 0: V-JEPA encoder + predictor + pooler (non-AR generation)

The HuggingFace VJEPA2ForVideoClassification model provides:
  - Video encoder (ViT with patch embedding)
  - Predictor head
  - Attentive pooler
  - Classification head

For action-conditioned variants (V-JEPA 2-AC), a two-stage topology
with action injection can be added in a separate pipeline.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# Single-stage V-JEPA (video classification / embedding)
# model_arch must match HF architecture name (VJEPA2ForVideoClassification)
# which the registry maps to our VJepa2Encoder class
VJEPA2_PIPELINE = PipelineConfig(
    model_type="vjepa2",
    model_arch="VJEPA2ForVideoClassification",
    hf_architectures=("VJEPA2ForVideoClassification",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(),
            final_output=True,
            final_output_type="prediction",
            owns_tokenizer=False,  # V-JEPA doesn't use text tokenizer
            requires_multimodal_data=True,
            model_arch="VJEPA2ForVideoClassification",
            engine_output_type="prediction",
            extras={
                "num_frames": 16,
                "stride": 8,
            },
        ),
    ),
)
