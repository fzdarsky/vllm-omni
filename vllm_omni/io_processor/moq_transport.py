# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Media over QUIC (MoQ) transport for browser camera streaming.

Provides WebTransport server that receives H.264 frames from browser
and bridges them to GStreamer via appsrc for zero-copy decode.

Architecture:
    Browser (WebCodecs H.264) --WebTransport--> MoQServer --appsrc--> GStreamer

Port allocation uses a simple pool from range 5004-5099.
"""
from __future__ import annotations

import asyncio
import logging
import ssl
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gi.repository import Gst

logger = logging.getLogger(__name__)

# Port range for MoQ WebTransport servers
MOQ_PORT_START = 5004
MOQ_PORT_END = 5099


class PortPool:
    """Thread-safe port allocator for MoQ sessions.

    Allocates ports from a fixed range (5004-5099) and tracks
    which ports are in use. Ports are released when sessions end.
    """

    def __init__(self, start: int = MOQ_PORT_START, end: int = MOQ_PORT_END):
        self._available = set(range(start, end + 1))
        self._in_use: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self) -> int:
        """Allocate a port from the pool.

        Returns:
            Available port number.

        Raises:
            RuntimeError: If no ports available.
        """
        with self._lock:
            if not self._available:
                raise RuntimeError("No MoQ ports available")
            port = self._available.pop()
            self._in_use.add(port)
            logger.debug("Allocated MoQ port %d (%d remaining)", port, len(self._available))
            return port

    def release(self, port: int) -> None:
        """Release a port back to the pool.

        Args:
            port: Port number to release.
        """
        with self._lock:
            if port in self._in_use:
                self._in_use.remove(port)
                self._available.add(port)
                logger.debug("Released MoQ port %d", port)

    @property
    def available_count(self) -> int:
        """Number of available ports."""
        with self._lock:
            return len(self._available)


# Global port pool instance
_port_pool: PortPool | None = None


def get_port_pool() -> PortPool:
    """Get the global port pool instance."""
    global _port_pool
    if _port_pool is None:
        _port_pool = PortPool()
    return _port_pool


@dataclass
class MoQSession:
    """Active MoQ WebTransport session.

    Manages the WebTransport server and GStreamer bridge for a single
    browser camera session.
    """
    session_id: str
    port: int
    host: str = "0.0.0.0"
    _server: asyncio.Server | None = field(default=None, repr=False)
    _appsrc: "Gst.Element | None" = field(default=None, repr=False)
    _on_frame: Callable[[bytes, int], None] | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)

    @property
    def endpoint(self) -> str:
        """WebTransport endpoint in host:port format."""
        # Use localhost for local connections, actual host for remote
        return f"localhost:{self.port}"

    async def start(
        self,
        on_frame: Callable[[bytes, int], None],
        cert_path: Path | None = None,
        key_path: Path | None = None,
    ) -> None:
        """Start the WebTransport server.

        Args:
            on_frame: Callback for received H.264 frames (data, pts_ns).
            cert_path: TLS certificate path (required for WebTransport).
            key_path: TLS key path (required for WebTransport).
        """
        self._on_frame = on_frame
        self._running = True

        # For now, use a simple TCP server as a placeholder
        # Full WebTransport requires aioquic integration
        # TODO: Integrate aioquic WebTransport protocol handler

        try:
            self._server = await asyncio.start_server(
                self._handle_connection,
                self.host,
                self.port,
            )
            logger.info(
                "MoQ server started: session=%s, endpoint=%s",
                self.session_id,
                self.endpoint,
            )
        except Exception as e:
            self._running = False
            raise RuntimeError(f"Failed to start MoQ server: {e}") from e

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming WebTransport connection.

        Protocol (simplified for initial implementation):
        - 8 bytes: PTS timestamp (nanoseconds, big-endian)
        - 4 bytes: Frame length (big-endian)
        - N bytes: H.264 NAL unit data
        """
        peer = writer.get_extra_info("peername")
        logger.info("MoQ client connected: %s", peer)

        try:
            while self._running:
                # Read header: 8 bytes PTS + 4 bytes length
                header = await reader.readexactly(12)
                pts_ns = int.from_bytes(header[:8], "big")
                length = int.from_bytes(header[8:12], "big")

                # Read frame data
                data = await reader.readexactly(length)

                # Deliver to callback
                if self._on_frame:
                    self._on_frame(data, pts_ns)

        except asyncio.IncompleteReadError:
            logger.info("MoQ client disconnected: %s", peer)
        except Exception as e:
            logger.exception("MoQ connection error: %s", e)
        finally:
            writer.close()
            await writer.wait_closed()

    async def stop(self) -> None:
        """Stop the WebTransport server and release port."""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Release port back to pool
        get_port_pool().release(self.port)

        logger.info("MoQ server stopped: session=%s", self.session_id)


async def create_moq_session(
    session_id: str,
    on_frame: Callable[[bytes, int], None],
) -> MoQSession:
    """Create and start a new MoQ WebTransport session.

    Args:
        session_id: Unique session identifier.
        on_frame: Callback for received H.264 frames.

    Returns:
        Started MoQ session with allocated port.

    Raises:
        RuntimeError: If no ports available or server fails to start.
    """
    port = get_port_pool().allocate()

    session = MoQSession(
        session_id=session_id,
        port=port,
    )

    try:
        await session.start(on_frame)
        return session
    except Exception:
        # Release port if start fails
        get_port_pool().release(port)
        raise
