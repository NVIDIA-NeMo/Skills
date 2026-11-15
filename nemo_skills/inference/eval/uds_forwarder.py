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

"""Forwards TCP connections on 127.0.0.1:<port> to a Unix socket.

Runs inside network-isolated agent containers, so it must only use the standard library.
Usage: python uds_forwarder.py <socket_path> <port>
"""

import asyncio
import sys


async def pipe(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()
    writer.close()


async def handle(reader, writer):
    upstream_reader, upstream_writer = await asyncio.open_unix_connection(sys.argv[1])
    await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))


async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", int(sys.argv[2]))
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
