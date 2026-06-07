# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StageFeedForwardProc — subprocess for feed-forward model inference.

Follows the StageDiffusionProc pattern: ZMQ PULL/PUSH async loop in a
spawned subprocess. Owns the model and GPU — no duplicate model copy
in the API server process.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from multiprocessing.process import BaseProcess
from typing import Any

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)

FEED_FORWARD_PROC_DEAD = b"__FEED_FORWARD_PROC_DEAD__"


class StageFeedForwardProc:
    """Feed-forward inference subprocess.

    Loads model once at init. Receives pixel_values + params via ZMQ,
    runs GPU preprocessing + model forward, returns predictions.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._pp_resize = 256
        self._pp_crop = (256, 256)
        self._pp_mean = None
        self._pp_std = None

    def initialize(self) -> None:
        """Load model and processor, init GPU preprocessing params."""
        from transformers import AutoModelForVideoClassification, AutoVideoProcessor

        logger.info("Loading model: %s on %s", self.model_name, self.device)
        self._model = AutoModelForVideoClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self._model.train(False)

        self._processor = AutoVideoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

        # Extract GPU preprocessing params from processor config
        p = self._processor
        size = getattr(p, "size", {})
        self._pp_resize = size.get("shortest_edge", size.get("height", 256))
        crop = getattr(p, "crop_size", {})
        self._pp_crop = (crop.get("height", self._pp_resize), crop.get("width", self._pp_resize))
        self._pp_mean = torch.tensor(
            getattr(p, "image_mean", [0.485, 0.456, 0.406]),
            device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self._pp_std = torch.tensor(
            getattr(p, "image_std", [0.229, 0.224, 0.225]),
            device=self.device, dtype=torch.float32,
        ).view(1, 3, 1, 1)

        logger.info(
            "Model loaded: %s (%s labels), preprocess: resize=%d crop=%s",
            type(self._model).__name__,
            getattr(self._model.config, "num_labels", "?"),
            self._pp_resize, self._pp_crop,
        )

    def _preprocess_on_gpu(self, clip: torch.Tensor) -> torch.Tensor:
        """Preprocess clip on GPU. No CPU round-trip."""
        import torchvision.transforms.functional as F

        if clip.ndim == 4 and clip.shape[-1] == 3:
            clip = clip.permute(0, 3, 1, 2)
        clip = clip.to(device=self.device, dtype=torch.float32)
        if clip.max() > 1.0:
            clip = clip / 255.0
        clip = F.resize(clip, self._pp_resize, antialias=True)
        clip = F.center_crop(clip, list(self._pp_crop))
        clip = (clip - self._pp_mean) / self._pp_std
        return clip.unsqueeze(0)

    def _process_request(self, request_id: str, pixel_values: torch.Tensor,
                         params: dict) -> dict:
        """Run inference on a single clip."""
        head = params.get("head")
        top_k = params.get("top_k", 5)

        pixel_values = self._preprocess_on_gpu(pixel_values)

        with torch.no_grad():
            outputs = self._model(pixel_values)
        logits = outputs.logits

        # Build predictions
        id2label = getattr(self._model.config, "id2label", None)
        probs = torch.softmax(logits[0], dim=-1)
        topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        preds = []
        for score, idx in zip(topk.values, topk.indices):
            label = (id2label.get(idx.item(), f"class_{idx.item()}")
                     if id2label else f"class_{idx.item()}")
            preds.append({"label": label, "score": round(score.item(), 6)})

        return {
            "request_id": request_id,
            "predictions": preds,
        }

    async def run_loop(self, request_address: str, response_address: str) -> None:
        """ZMQ async event loop."""
        ctx = zmq.asyncio.Context()
        request_socket = ctx.socket(zmq.PULL)
        request_socket.bind(request_address)
        response_socket = ctx.socket(zmq.PUSH)
        response_socket.bind(response_address)

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder()

        logger.info("StageFeedForwardProc loop started")

        try:
            while True:
                raw = await request_socket.recv()
                msg = decoder.decode(raw)
                msg_type = msg.get("type") if isinstance(msg, dict) else None

                if msg_type == "add_request":
                    request_id = msg["request_id"]
                    # Reconstruct tensor from serialized data
                    pv_data = msg.get("pixel_values")
                    if isinstance(pv_data, dict) and "data" in pv_data:
                        pixel_values = torch.tensor(
                            pv_data["data"],
                            dtype=getattr(torch, pv_data.get("dtype", "float32")),
                        ).reshape(pv_data["shape"])
                    elif isinstance(pv_data, (list, tuple)):
                        pixel_values = torch.tensor(pv_data)
                    else:
                        pixel_values = torch.zeros(16, 3, 256, 256)

                    params = msg.get("params", {})

                    try:
                        result = self._process_request(request_id, pixel_values, params)
                        await response_socket.send(encoder.encode({
                            "type": "result", "output": result,
                        }))
                    except Exception as e:
                        logger.exception("Request %s failed: %s", request_id, e)
                        await response_socket.send(encoder.encode({
                            "type": "error", "request_id": request_id, "error": str(e),
                        }))

                elif msg_type == "abort":
                    pass  # No long-running ops to cancel

                elif msg_type == "shutdown":
                    break

        except Exception:
            logger.exception("StageFeedForwardProc loop error")
        finally:
            request_socket.close()
            response_socket.close()
            ctx.term()


def _run_stage_proc(model_name: str, device: str,
                    handshake_address: str,
                    request_address: str, response_address: str) -> None:
    """Entry point for the spawned subprocess."""
    import os
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    proc = StageFeedForwardProc(model_name, device)
    proc.initialize()

    # Handshake: send READY
    ctx = zmq.Context()
    hs = ctx.socket(zmq.DEALER)
    hs.connect(handshake_address)
    hs.send(msgspec.msgpack.encode({"status": "READY"}))
    hs.close()
    ctx.term()

    logger.info("StageFeedForwardProc READY, entering loop")

    # Run async event loop
    asyncio.run(proc.run_loop(request_address, response_address))


def spawn_feed_forward_proc(model_name: str, device: str = "cuda"):
    """Spawn a StageFeedForwardProc subprocess.

    Returns (proc, handshake_address, request_address, response_address).
    """
    from vllm.utils.network_utils import get_open_zmq_ipc_path
    from vllm.utils.system_utils import get_mp_context

    handshake_address = get_open_zmq_ipc_path()
    request_address = get_open_zmq_ipc_path()
    response_address = get_open_zmq_ipc_path()

    mp_ctx = get_mp_context()
    proc = mp_ctx.Process(
        target=_run_stage_proc,
        name="StageFeedForwardProc",
        args=(model_name, device, handshake_address, request_address, response_address),
    )
    proc.start()
    return proc, handshake_address, request_address, response_address


def complete_feed_forward_handshake(proc: BaseProcess, handshake_address: str,
                                    timeout: int = 600) -> None:
    """Wait for READY from the subprocess."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(handshake_address)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    deadline = timeout * 1000
    events = dict(poller.poll(timeout=deadline))
    if sock not in events:
        proc.kill()
        raise TimeoutError(f"StageFeedForwardProc did not send READY within {timeout}s")

    identity, raw = sock.recv_multipart()
    msg = msgspec.msgpack.decode(raw)
    if msg.get("status") != "READY":
        raise RuntimeError(f"Expected READY, got: {msg}")

    sock.close()
    ctx.term()
    logger.info("StageFeedForwardProc handshake complete")
