/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Network blocking library for sandbox code execution
 *
 * This library intercepts socket() calls and blocks IPv4/IPv6 sockets
 * while allowing Unix domain sockets (needed for local IPC).
 *
 * Enabled by setting NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1 at container runtime.
 * The startup script adds this library to /etc/ld.so.preload AFTER the API
 * server starts, ensuring the API can still accept connections while all
 * user code execution has network access blocked.
 *
 * Using /etc/ld.so.preload (vs LD_PRELOAD env var) ensures this cannot be
 * bypassed by user code clearing environment variables or spawning
 * subprocesses with env={}.
 *
 * Build: gcc -shared -fPIC -o libblock_network.so block_network.c -ldl
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <dlfcn.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>

/* Override socket() to block internet sockets */
int socket(int domain, int type, int protocol) {
    /* Get the real socket function */
    static int (*real_socket)(int, int, int) = NULL;
    if (!real_socket) {
        real_socket = dlsym(RTLD_NEXT, "socket");
    }

    /* Allow Unix domain sockets (needed for local IPC, uwsgi, etc.) */
    if (domain == AF_UNIX || domain == AF_LOCAL) {
        return real_socket(domain, type, protocol);
    }

    /* Block IPv4 and IPv6 internet sockets */
    if (domain == AF_INET || domain == AF_INET6) {
        errno = ENETUNREACH;  /* Network is unreachable */
        return -1;
    }

    /* Allow other socket types (netlink, packet, etc.) */
    return real_socket(domain, type, protocol);
}
