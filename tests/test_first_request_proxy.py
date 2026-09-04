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

import asyncio
import json
from urllib.parse import urlsplit

from nemo_skills.inference.eval.first_request_proxy import capture_first_llm_request


async def _send_json_request(base_url, endpoint, body):
    parsed = urlsplit(base_url)
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    target = f"{parsed.path.rstrip('/')}/{endpoint.lstrip('/')}"
    request = (
        f"POST {target} HTTP/1.1\r\n"
        f"Host: {parsed.netloc}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode() + body
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


def test_proxy_captures_first_llm_request_and_forwards_it(tmp_path):
    async def run_test():
        received = []

        async def upstream_handler(reader, writer):
            raw_headers = await reader.readuntil(b"\r\n\r\n")
            lines = raw_headers[:-4].split(b"\r\n")
            content_length = next(
                int(line.split(b":", maxsplit=1)[1].strip())
                for line in lines[1:]
                if line.lower().startswith(b"content-length:")
            )
            body = await reader.readexactly(content_length)
            received.append((lines[0], body))
            response_body = b'{"ok":true}'
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(response_body)}\r\n".encode()
                + b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
                + response_body
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]
        capture_file = tmp_path / "first-llm-request.json"
        first_body = json.dumps(
            {
                "model": "test-model",
                "system": [{"type": "text", "text": "Exact system prompt"}],
                "messages": [{"role": "user", "content": "Fix the issue"}],
            },
            ensure_ascii=False,
        ).encode()
        second_body = b'{"model":"test-model","messages":[{"role":"user","content":"second turn"}]}'

        try:
            async with capture_first_llm_request(
                f"http://127.0.0.1:{upstream_port}/v1",
                capture_file,
            ) as proxy_base:
                first_response = await _send_json_request(proxy_base, "messages", first_body)
                second_response = await _send_json_request(proxy_base, "messages", second_body)
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert b"200 OK" in first_response
        assert b"200 OK" in second_response
        assert received == [
            (b"POST /v1/messages HTTP/1.1", first_body),
            (b"POST /v1/messages HTTP/1.1", second_body),
        ]
        assert capture_file.read_bytes() == first_body

    asyncio.run(run_test())


def test_proxy_ignores_non_inference_requests(tmp_path):
    async def run_test():
        async def upstream_handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        upstream_port = upstream.sockets[0].getsockname()[1]
        capture_file = tmp_path / "first-llm-request.json"
        try:
            async with capture_first_llm_request(
                f"http://127.0.0.1:{upstream_port}/v1",
                capture_file,
            ) as proxy_base:
                await _send_json_request(proxy_base, "models", b"{}")
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert not capture_file.exists()

    asyncio.run(run_test())
