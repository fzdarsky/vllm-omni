# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StageFeedForwardClient — ZMQ client for StageFeedForwardProc.

Spawns the subprocess, sends inference requests, receives predictions.
The model lives in the subprocess — no duplicate in the API server process.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from threading import Thread
from typing import Any

import msgspec
import torch
import zmq

from vllm_omni.feed_forward.data import FeedForwardOutput
from vllm_omni.feed_forward.stage_proc import (
    FEED_FORWARD_PROC_DEAD,
    complete_feed_forward_handshake,
    spawn_feed_forward_proc,
)

logger = logging.getLogger(__name__)


class StageFeedForwardClient:
    """Client for StageFeedForwardProc subprocess.

    Can be used as ``engine_client`` in OmniOpenAIServingWorld.
    Provides an async ``step()`` method matching FeedForwardEngine's API.
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 stage_init_timeout: int = 600):
        self.model_name = model_name
        self.device = device
        self._engine_dead = False
        self._shutting_down = False

        # Spawn subprocess
        self._proc, hs_addr, self._req_addr, self._resp_addr = (
            spawn_feed_forward_proc(model_name, device)
        )
        complete_feed_forward_handshake(self._proc, hs_addr, stage_init_timeout)

        # Connect ZMQ sockets
        self._ctx = zmq.Context()
        self._request_socket = self._ctx.socket(zmq.PUSH)
        self._request_socket.connect(self._req_addr)
        self._response_socket = self._ctx.socket(zmq.PULL)
        self._response_socket.connect(self._resp_addr)

        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder()

        # Pending results
        self._pending: dict[str, asyncio.Future] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # Monitor subprocess health
        self._start_proc_monitor()

        # Stage configs stub (api_server checks for this)
        self.stage_configs: list[Any] = []

        logger.info("StageFeedForwardClient connected to subprocess (PID %s)", self._proc.pid)

    def _start_proc_monitor(self) -> None:
        """Watch for subprocess death."""
        import weakref
        proc = self._proc
        self_ref = weakref.ref(self)

        def _monitor():
            try:
                multiprocessing.connection.wait([proc.sentinel])
            except Exception:
                return
            client = self_ref()
            if client is None or client._shutting_down or client._engine_dead:
                return
            client._engine_dead = True
            logger.error("StageFeedForwardProc died (exit code %s)", proc.exitcode)

        Thread(target=_monitor, daemon=True, name="FFProcMonitor").start()

    def _drain_responses(self) -> list[dict]:
        """Non-blocking drain of all available responses."""
        results = []
        while True:
            try:
                raw = self._response_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw == FEED_FORWARD_PROC_DEAD:
                self._engine_dead = True
                break

            msg = self._decoder.decode(raw)
            results.append(msg)
        return results

    async def step(self, request) -> FeedForwardOutput:
        """Submit a request and wait for the result.

        Args:
            request: FeedForwardRequest with pixel_values and sampling_params.

        Returns:
            FeedForwardOutput with predictions.
        """
        if self._engine_dead:
            raise RuntimeError("StageFeedForwardProc is dead")

        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()

        request_id = request.request_id
        future = self._main_loop.create_future()
        self._pending[request_id] = future

        # Serialize pixel_values for ZMQ transport
        pv = request.pixel_values
        if pv is not None and isinstance(pv, torch.Tensor):
            pv_data = {
                "data": pv.cpu().tolist(),
                "shape": list(pv.shape),
                "dtype": str(pv.dtype).replace("torch.", ""),
            }
        else:
            pv_data = None

        params = {}
        if request.sampling_params:
            sp = request.sampling_params
            params = {
                "head": sp.head,
                "top_k": sp.top_k,
                "session_id": sp.session_id,
                "extra_args": sp.extra_args,
            }

        self._request_socket.send(self._encoder.encode({
            "type": "add_request",
            "request_id": request_id,
            "pixel_values": pv_data,
            "params": params,
        }))

        # Poll for response
        while not future.done():
            responses = self._drain_responses()
            for resp in responses:
                if resp.get("type") == "result":
                    output = resp.get("output", {})
                    rid = output.get("request_id", "")
                    pending = self._pending.pop(rid, None)
                    if pending and not pending.done():
                        pending.set_result(FeedForwardOutput(
                            predictions=output.get("predictions"),
                        ))
                elif resp.get("type") == "error":
                    rid = resp.get("request_id", "")
                    pending = self._pending.pop(rid, None)
                    if pending and not pending.done():
                        pending.set_exception(RuntimeError(resp.get("error", "unknown")))

            if not future.done():
                await asyncio.sleep(0.001)

        self._pending.pop(request_id, None)
        return future.result()

    def close(self) -> None:
        """Shut down the subprocess."""
        self._shutting_down = True
        try:
            self._request_socket.send(self._encoder.encode({"type": "shutdown"}))
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
        self._request_socket.close()
        self._response_socket.close()
        self._ctx.term()
        logger.info("StageFeedForwardClient closed")

    async def get_vllm_config(self) -> None:
        return None
