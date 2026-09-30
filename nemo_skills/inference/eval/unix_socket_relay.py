#!/usr/bin/env python3
"""Relay loopback TCP connections to a mounted Unix-domain socket."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(64 * 1024):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    socket_path: str,
) -> None:
    upstream_writer = None
    try:
        upstream_reader, upstream_writer = await asyncio.open_unix_connection(socket_path)
        await asyncio.gather(
            relay(client_reader, upstream_writer),
            relay(upstream_reader, client_writer),
        )
    except (ConnectionError, OSError):
        client_writer.close()
    finally:
        if upstream_writer is not None:
            upstream_writer.close()


async def main_async(port: int, socket_path: str, ready_file: str | None = None) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(reader, writer, socket_path),
        "127.0.0.1",
        port,
    )
    if ready_file:
        Path(ready_file).touch()
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    asyncio.run(main_async(args.port, args.socket, args.ready_file))


if __name__ == "__main__":
    main()
