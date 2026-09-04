# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transparent HTTP proxy that captures the first LLM request body."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

LOG = logging.getLogger(__name__)

_LLM_ENDPOINT_SUFFIXES = ("/messages", "/chat/completions", "/responses")


class FirstRequestCaptureProxy:
    """Forward HTTP traffic and persist the first JSON inference request unchanged."""

    def __init__(self, upstream_base_url: str, output_file: Path):
        self.upstream = urlsplit(upstream_base_url)
        if self.upstream.scheme not in {"http", "https"} or not self.upstream.hostname:
            raise ValueError(f"Unsupported upstream URL: {upstream_base_url}")
        self.output_file = output_file
        self.server: asyncio.AbstractServer | None = None
        self._capture_lock = asyncio.Lock()
        self._captured = False
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> str:
        self.output_file.unlink(missing_ok=True)
        self.server = await asyncio.start_server(self._handle_connection, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        return urlunsplit(("http", f"127.0.0.1:{port}", self.upstream.path, self.upstream.query, ""))

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for writer in list(self._writers):
            writer.close()
        if self._writers:
            await asyncio.gather(*(writer.wait_closed() for writer in list(self._writers)), return_exceptions=True)
        self._writers.clear()

    async def _capture(self, request_target: str, body: bytes) -> None:
        request_path = urlsplit(request_target).path
        if not request_path.endswith(_LLM_ENDPOINT_SUFFIXES):
            return
        try:
            json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        async with self._capture_lock:
            if self._captured:
                return
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_bytes(body)
            self._captured = True

    def _rewrite_headers(self, header_lines: list[bytes]) -> bytes:
        upstream_host = self.upstream.hostname
        default_port = 443 if self.upstream.scheme == "https" else 80
        upstream_port = self.upstream.port or default_port
        host_value = upstream_host if upstream_port == default_port else f"{upstream_host}:{upstream_port}"

        rewritten = []
        host_seen = False
        for line in header_lines:
            name, separator, _ = line.partition(b":")
            if separator and name.strip().lower() == b"host":
                rewritten.append(f"Host: {host_value}".encode())
                host_seen = True
            else:
                rewritten.append(line)
        if not host_seen:
            rewritten.append(f"Host: {host_value}".encode())
        return b"\r\n".join(rewritten)

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while chunk := await reader.read(64 * 1024):
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    async def _read_chunked_body(self, reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
        """Read one chunked request body, returning wire bytes and decoded content."""
        wire_body = bytearray()
        decoded_body = bytearray()
        while True:
            size_line = await reader.readuntil(b"\r\n")
            wire_body.extend(size_line)
            size = int(size_line.split(b";", maxsplit=1)[0].strip(), 16)
            if size == 0:
                while True:
                    trailer_line = await reader.readuntil(b"\r\n")
                    wire_body.extend(trailer_line)
                    if trailer_line == b"\r\n":
                        return bytes(wire_body), bytes(decoded_body)
            chunk = await reader.readexactly(size + 2)
            if not chunk.endswith(b"\r\n"):
                raise ValueError("Malformed chunked HTTP request")
            wire_body.extend(chunk)
            decoded_body.extend(chunk[:-2])

    async def _handle_connection(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        upstream_writer = None
        self._writers.add(client_writer)
        try:
            raw_headers = await client_reader.readuntil(b"\r\n\r\n")
            header_block = raw_headers[:-4]
            lines = header_block.split(b"\r\n")
            request_line = lines[0]
            request_parts = request_line.split(b" ", maxsplit=2)
            if len(request_parts) != 3:
                raise ValueError("Malformed HTTP request line")
            method, raw_target, version = request_parts

            content_length = 0
            is_chunked = False
            for line in lines[1:]:
                name, separator, value = line.partition(b":")
                if not separator:
                    continue
                normalized_name = name.strip().lower()
                if normalized_name == b"content-length":
                    content_length = int(value.strip())
                elif normalized_name == b"transfer-encoding" and b"chunked" in value.lower():
                    is_chunked = True
            if is_chunked:
                wire_body, decoded_body = await self._read_chunked_body(client_reader)
            else:
                wire_body = await client_reader.readexactly(content_length) if content_length else b""
                decoded_body = wire_body
            await self._capture(raw_target.decode("ascii", errors="replace"), decoded_body)

            ssl_context = ssl.create_default_context() if self.upstream.scheme == "https" else None
            upstream_port = self.upstream.port or (443 if self.upstream.scheme == "https" else 80)
            upstream_reader, upstream_writer = await asyncio.open_connection(
                self.upstream.hostname,
                upstream_port,
                ssl=ssl_context,
                server_hostname=self.upstream.hostname if ssl_context else None,
            )
            self._writers.add(upstream_writer)

            rewritten_headers = self._rewrite_headers(lines[1:])
            upstream_writer.write(b" ".join((method, raw_target, version)) + b"\r\n")
            upstream_writer.write(rewritten_headers + b"\r\n\r\n" + wire_body)
            await upstream_writer.drain()

            await asyncio.gather(
                self._relay(client_reader, upstream_writer),
                self._relay(upstream_reader, client_writer),
            )
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError) as error:
            LOG.warning("First-request capture proxy connection failed: %s", error)
        finally:
            client_writer.close()
            self._writers.discard(client_writer)
            if upstream_writer is not None:
                upstream_writer.close()
                self._writers.discard(upstream_writer)


@asynccontextmanager
async def capture_first_llm_request(upstream_base_url: str, output_file: Path):
    """Yield a local proxy URL and close all proxy resources afterward."""
    proxy = FirstRequestCaptureProxy(upstream_base_url, output_file)
    proxy_base_url = await proxy.start()
    try:
        yield proxy_base_url
    finally:
        await proxy.close()
