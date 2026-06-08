# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FeedForwardEngine — single-pass inference engine.

Simplified version of DiffusionEngine that removes all iterative denoising
logic. Each request = one forward pass. No step scheduling, no noise
schedules, no trajectory tracking.

Reuses DiffusionExecutor's RPC and worker infrastructure for GPU dispatch.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

try:
    from opentelemetry import context as otel_context
    _HAS_OTEL = True
except ImportError:
    otel_context = None  # type: ignore[assignment]
    _HAS_OTEL = False

from vllm_omni.feed_forward.data import FeedForwardConfig, FeedForwardOutput
from vllm_omni.feed_forward.request import FeedForwardRequest
from vllm_omni.utils.cpu_affinity import pin_to_performance_cores

logger = init_logger(__name__)


class FeedForwardEngine:
    """Single-pass inference engine for non-iterative models.

    Each request is dispatched immediately — no step scheduling.
    The busy loop pattern matches DiffusionEngine for compatibility
    with vLLM-Omni's stage infrastructure.
    """

    def __init__(
        self,
        config: FeedForwardConfig,
        model_forward_fn: Any | None = None,
    ):
        self.config = config
        self._model_forward_fn = model_forward_fn

        self.main_loop: asyncio.AbstractEventLoop | None = None
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self._loop_started = False
        self._init_lock = asyncio.Lock()

        self._request_queue: queue.Queue[FeedForwardRequest] = queue.Queue()
        self._out_queue: dict[str, asyncio.Future] = {}
        self._abort_queue: queue.Queue[str] = queue.Queue()
        self._cv = threading.Condition()
        self._closed = False

    async def _check_and_start_background_loop(self):
        if self._closed:
            raise RuntimeError("FeedForwardEngine is closed.")
        if self._loop_started:
            return

        async with self._init_lock:
            if self._loop_started:
                return
            self.main_loop = asyncio.get_running_loop()
            self.worker_thread = threading.Thread(
                target=self._busy_loop, daemon=True, name="FeedForwardEngine",
            )
            self.worker_thread.start()
            self._loop_started = True

    async def step(self, request: FeedForwardRequest) -> FeedForwardOutput:
        """Submit a request and wait for the result."""
        await self._check_and_start_background_loop()

        if _HAS_OTEL:
            request.trace_context = otel_context.get_current()

        future = self.main_loop.create_future()
        self._out_queue[request.request_id] = future

        self._request_queue.put(request)
        with self._cv:
            self._cv.notify()

        try:
            return await future
        finally:
            self._out_queue.pop(request.request_id, None)

    async def abort(self, request_id: str) -> None:
        """Abort a pending request."""
        self._abort_queue.put(request_id)
        with self._cv:
            self._cv.notify()

    def _busy_loop(self):
        """Main processing loop — dispatch requests as they arrive."""
        # On heterogeneous ARM (big.LITTLE), pin to performance cores to
        # avoid bimodal latency from scheduler migration between core types.
        pin_to_performance_cores()

        while not self.stop_event.is_set():
            self._process_aborts()

            with self._cv:
                while (
                    self._request_queue.empty()
                    and self._abort_queue.empty()
                    and not self.stop_event.is_set()
                ):
                    self._cv.wait(timeout=1.0)

                if self.stop_event.is_set():
                    break

            if self._request_queue.empty():
                continue

            try:
                request = self._request_queue.get_nowait()
            except queue.Empty:
                continue

            try:
                result = self._execute(request)
                self._complete_request(request.request_id, result)
            except Exception as exc:
                logger.error(
                    "FeedForward execution failed for %s: %s",
                    request.request_id, exc, exc_info=True,
                )
                self._complete_request(
                    request.request_id,
                    FeedForwardOutput(error=str(exc)),
                )

    def _execute(self, request: FeedForwardRequest) -> FeedForwardOutput:
        """Execute a single forward pass.

        Override this in subclasses or set model_forward_fn for custom models.
        Default implementation calls self._model_forward_fn(request).
        """
        ctx = request.trace_context
        token = otel_context.attach(ctx) if ctx is not None else None
        try:
            if self._model_forward_fn is not None:
                return self._model_forward_fn(request)

            raise NotImplementedError(
                "FeedForwardEngine requires either model_forward_fn or _execute override."
            )
        finally:
            if token is not None:
                otel_context.detach(token)

    def _complete_request(self, request_id: str, result: FeedForwardOutput):
        """Deliver result to the waiting future on the main event loop."""
        future = self._out_queue.get(request_id)
        if future is None or future.done():
            return

        def _set_result():
            if not future.done():
                future.set_result(result)

        if self.main_loop is not None:
            self.main_loop.call_soon_threadsafe(_set_result)

    def _process_aborts(self):
        """Cancel pending requests that have been aborted."""
        while True:
            try:
                request_id = self._abort_queue.get_nowait()
            except queue.Empty:
                return

            future = self._out_queue.pop(request_id, None)
            if future is not None and not future.done():
                def _abort(f=future):
                    if not f.done():
                        f.set_exception(
                            RuntimeError(f"Request {request_id} aborted.")
                        )
                if self.main_loop is not None:
                    self.main_loop.call_soon_threadsafe(_abort)

    def close(self):
        """Shut down the engine."""
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        with self._cv:
            self._cv.notify_all()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
        logger.info("FeedForwardEngine closed.")
