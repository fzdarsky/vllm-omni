# SPDX-License-Identifier: Apache-2.0
"""Async chunk processor for V-JEPA encoder -> predictor streaming."""

from typing import Any, Optional
import torch


def encoder_to_predictor_chunk(
    transfer_manager: Any,
    pooling_output: Optional[dict[str, Any]],
    request: Any,
    is_finished: bool = False,
    stride: int = 8,
) -> Optional[dict[str, Any]]:
    """Accumulate encoder latents until stride threshold.

    Delivers predictions every ~94ms (stride=8 @ 12fps equivalent).

    Args:
        transfer_manager: OmniChunkTransferAdapter managing request payloads.
        pooling_output: Encoder output containing latents.
        request: Request object with external_req_id.
        is_finished: Whether this is the final chunk.
        stride: Number of frames to accumulate before yielding.

    Returns:
        Dict with latents batch, or None if still accumulating.
    """
    request_id = request.external_req_id

    if request_id not in transfer_manager.request_payload:
        transfer_manager.request_payload[request_id] = {
            "latent_buffer": [],
            "frames_accumulated": 0,
        }

    payload = transfer_manager.request_payload[request_id]

    if pooling_output and "latents" in pooling_output:
        payload["latent_buffer"].append(pooling_output["latents"])
        payload["frames_accumulated"] += 1

    if payload["frames_accumulated"] < stride and not is_finished:
        return None

    if not payload["latent_buffer"]:
        return None

    latents = torch.cat(payload["latent_buffer"], dim=0)
    payload["latent_buffer"].clear()
    payload["frames_accumulated"] = 0

    return {"latents": latents, "finished": is_finished}
