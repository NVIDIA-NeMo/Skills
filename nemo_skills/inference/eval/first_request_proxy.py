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

"""HTTP proxy that captures the first LLM request and can rewrite request bodies."""

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


def _contains_string(value, substring: str) -> bool:
    if isinstance(value, str):
        return substring in value
    if isinstance(value, list):
        return any(_contains_string(item, substring) for item in value)
    if isinstance(value, dict):
        return any(_contains_string(item, substring) for item in value.values())
    return False


class FirstRequestCaptureProxy:
    def __init__(
        self,
        upstream_base_url: str,
        output_file: Path,
        *,
        skip_body_substrings: tuple[str, ...] = (),
        served_model_name: str | None = None,
    ):
        self.upstream = urlsplit(upstream_base_url)
        if self.upstream.scheme not in {"http", "https"} or not self.upstream.hostname:
            raise ValueError(f"Unsupported upstream URL: {upstream_base_url}")
        self.output_file = output_file
        self.skip_body_substrings = skip_body_substrings
        self.served_model_name = served_model_name
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
        await asyncio.gather(*(writer.wait_closed() for writer in list(self._writers)), return_exceptions=True)
        self._writers.clear()

    async def _capture(self, request_target: str, body: bytes) -> None:
        if not urlsplit(request_target).path.endswith(_LLM_ENDPOINT_SUFFIXES):
            return
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if any(_contains_string(request, substring) for substring in self.skip_body_substrings):
            return
        async with self._capture_lock:
            if self._captured:
                return
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_bytes(body)
            self._captured = True

    def _edit_request(self, body: bytes) -> bytes:
        try:
            request = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body
        if not isinstance(request, dict):
            return body

        # Any edits to the request body go here.
        if self.served_model_name is not None:
            request["model"] = self.served_model_name

        return json.dumps(request, ensure_ascii=False).encode()

    def _rewrite_headers(self, header_lines: list[bytes], content_length: int) -> bytes:
        default_port = 443 if self.upstream.scheme == "https" else 80
        port = self.upstream.port or default_port
        host = self.upstream.hostname if port == default_port else f"{self.upstream.hostname}:{port}"
        rewritten = [f"Host: {host}".encode(), f"Content-Length: {content_length}".encode()]
        for line in header_lines:
            if line.partition(b":")[0].strip().lower() not in {b"host", b"content-length", b"transfer-encoding"}:
                rewritten.append(line)
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

    async def _read_body(self, reader: asyncio.StreamReader, header_lines: list[bytes]) -> bytes:
        content_length = 0
        chunked = False
        for line in header_lines:
            name, _, value = line.partition(b":")
            name = name.strip().lower()
            if name == b"content-length":
                content_length = int(value.strip())
            elif name == b"transfer-encoding" and b"chunked" in value.lower():
                chunked = True
        if not chunked:
            return await reader.readexactly(content_length) if content_length else b""
        body = bytearray()
        while True:
            size = int((await reader.readuntil(b"\r\n")).split(b";", 1)[0].strip(), 16)
            if size == 0:
                while await reader.readuntil(b"\r\n") != b"\r\n":
                    pass
                return bytes(body)
            body.extend(await reader.readexactly(size))
            await reader.readexactly(2)

    async def _handle_connection(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        upstream_writer = None
        response_relay = None
        self._writers.add(client_writer)
        try:
            while True:
                raw_headers = await client_reader.readuntil(b"\r\n\r\n")
                lines = raw_headers[:-4].split(b"\r\n")
                method, raw_target, version = lines[0].split(b" ", maxsplit=2)
                body = self._edit_request(await self._read_body(client_reader, lines[1:]))
                await self._capture(raw_target.decode("ascii", errors="replace"), body)

                if upstream_writer is None:
                    ssl_context = ssl.create_default_context() if self.upstream.scheme == "https" else None
                    upstream_port = self.upstream.port or (443 if self.upstream.scheme == "https" else 80)
                    upstream_reader, upstream_writer = await asyncio.open_connection(
                        self.upstream.hostname,
                        upstream_port,
                        ssl=ssl_context,
                        server_hostname=self.upstream.hostname if ssl_context else None,
                    )
                    self._writers.add(upstream_writer)
                    response_relay = asyncio.create_task(self._relay(upstream_reader, client_writer))

                upstream_writer.write(b" ".join((method, raw_target, version)) + b"\r\n")
                upstream_writer.write(self._rewrite_headers(lines[1:], len(body)) + b"\r\n\r\n" + body)
                await upstream_writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError) as error:
            if not isinstance(error, asyncio.IncompleteReadError):
                LOG.warning("First-request capture proxy connection failed: %s", error)
        finally:
            if upstream_writer is not None:
                upstream_writer.close()
                self._writers.discard(upstream_writer)
            if response_relay is not None:
                await asyncio.gather(response_relay, return_exceptions=True)
            client_writer.close()
            self._writers.discard(client_writer)


@asynccontextmanager
async def capture_first_llm_request(
    upstream_base_url: str,
    output_file: Path,
    *,
    skip_body_substrings: tuple[str, ...] = (),
    served_model_name: str | None = None,
):
    """Yield a local proxy URL and close all proxy resources afterward."""
    proxy = FirstRequestCaptureProxy(
        upstream_base_url,
        output_file,
        skip_body_substrings=skip_body_substrings,
        served_model_name=served_model_name,
    )
    proxy_base_url = await proxy.start()
    try:
        yield proxy_base_url
    finally:
        await proxy.close()
