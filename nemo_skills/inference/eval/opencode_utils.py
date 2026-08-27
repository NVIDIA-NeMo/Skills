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

import copy
import json
import shlex
from pathlib import Path

OPENCODE_DEFAULT_OUTPUT_TOKEN_MAX = 131072
OPENCODE_DEFAULT_VERSION = "1.17.11"
OPENCODE_NODE_VERSION = "22.15.0"
OPENCODE_NPM_PACKAGE = "opencode-ai"
OPENCODE_PROVIDER_ID = "nemo"


def build_opencode_install_command(version: str) -> str:
    """Build an installer that supports glibc and Alpine/musl on x86_64 and ARM64."""
    version = shlex.quote(version)
    return (
        "rm -rf /root/opencode /root/node && "
        "mkdir -p /root/opencode && "
        "if [[ $(uname -m) == 'aarch64' || $(uname -m) == 'arm64' ]]; then "
        "    export OPENCODE_ARCH=arm64 OPENCODE_MUSL_ARCH=aarch64 && "
        "    export OPENCODE_MUSL_PACKAGE=opencode-linux-arm64-musl; "
        "else "
        "    export OPENCODE_ARCH=x64 OPENCODE_MUSL_ARCH=x86_64 && "
        "    export OPENCODE_MUSL_PACKAGE=opencode-linux-x64-baseline-musl; "
        "fi && "
        "if [ -f /etc/alpine-release ]; then "
        "    apk add --no-cache nodejs npm && "
        f"   npm install -g --prefix /root/opencode {OPENCODE_NPM_PACKAGE}@{version} "
        f"${{OPENCODE_MUSL_PACKAGE}}@{version} && "
        "    cp /root/opencode/lib/node_modules/${OPENCODE_MUSL_PACKAGE}/bin/opencode "
        "        /root/opencode/bin/opencode-native && "
        "    chmod 755 /root/opencode/bin/opencode-native && "
        "    ln -sf opencode-native /root/opencode/bin/opencode && "
        "    mkdir -p /root/opencode/lib && "
        "    cp -L /lib/ld-musl-${OPENCODE_MUSL_ARCH}.so.1 /root/opencode/lib/ && "
        "    cp -L /lib/ld-musl-${OPENCODE_MUSL_ARCH}.so.1 "
        "        /root/opencode/lib/libc.musl-${OPENCODE_MUSL_ARCH}.so.1 && "
        "    cp -L /usr/lib/libstdc++.so.6 /usr/lib/libgcc_s.so.1 /root/opencode/lib/ && "
        "    touch /root/opencode/.musl; "
        "else "
        f"   export NODE_VERSION={OPENCODE_NODE_VERSION} && "
        "    export NODE_ARCH=linux-${OPENCODE_ARCH} && "
        "    mkdir -p /root/node && "
        "    curl -Lf "
        '        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-${NODE_ARCH}.tar.gz" '
        "        -o /tmp/node.tar.gz && "
        "    tar -xzf /tmp/node.tar.gz -C /root/node --strip-components=1 && "
        "    export PATH=/root/node/bin:$PATH && "
        f"   npm install -g --prefix /root/opencode {OPENCODE_NPM_PACKAGE}@{version}; "
        "fi && "
        "export PATH=/root/opencode/bin:/root/node/bin:$PATH && "
        "opencode --version"
    )


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Merge override into base in place, recursing into nested dictionaries."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def build_opencode_config(
    agent_config: dict,
    api_base: str,
    model: str,
    context_window: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    extra_body: dict,
    agent_max_turns: int,
    tokens_to_generate: int | None = None,
) -> dict:
    """Build an OpenCode config for a NeMo-Skills OpenAI-compatible server."""
    if extra_body is None:
        raise ValueError("OpenCode inference.extra_body cannot be null; omit it or set it to {}.")
    if not isinstance(extra_body, dict):
        raise ValueError("OpenCode inference.extra_body must be a dictionary.")

    config = _deep_merge_dicts({}, copy.deepcopy(agent_config) if agent_config else {})
    config.setdefault(
        "permission",
        {
            "bash": "allow",
            "edit": "allow",
            "webfetch": "allow",
        },
    )

    providers = config.setdefault("provider", {})
    nemo = providers.setdefault(
        OPENCODE_PROVIDER_ID,
        {
            "npm": "@ai-sdk/openai-compatible",
            "name": "NeMo-Skills LLM server",
        },
    )
    nemo.setdefault("npm", "@ai-sdk/openai-compatible")
    nemo.setdefault("options", {}).update(
        {
            "baseURL": api_base,
            "apiKey": "EMPTY",
        }
    )

    models = nemo.setdefault("models", {})
    model_entry = models.get(model, {}) if isinstance(models.get(model), dict) else {}
    _deep_merge_dicts(
        model_entry,
        {
            "id": model,
            "name": model,
            "temperature": True,
            "tool_call": True,
            "limit": {
                "context": context_window,
                "output": (
                    tokens_to_generate
                    if tokens_to_generate is not None
                    else min(OPENCODE_DEFAULT_OUTPUT_TOKEN_MAX, context_window)
                ),
            },
        },
    )
    if extra_body or top_k is not None:
        model_options = model_entry.setdefault("options", {})
        if not isinstance(model_options, dict):
            raise ValueError("OpenCode model options must be a dictionary.")
        _deep_merge_dicts(model_options, copy.deepcopy(extra_body))
        if top_k is not None:
            model_options["top_k"] = top_k
    models[model] = model_entry

    agents = config.setdefault("agent", {})
    agent_name = config.get("default_agent") or "build"
    primary_agent = agents.get(agent_name, {}) if isinstance(agents.get(agent_name), dict) else {}
    primary_agent.update(
        {
            "temperature": temperature,
            "top_p": top_p,
            "steps": agent_max_turns,
        }
    )
    agents[agent_name] = primary_agent
    return config


def extract_final_assistant_text(session: dict) -> str:
    """Return the last non-empty assistant text from an OpenCode session export."""
    messages = session.get("messages")
    if not isinstance(messages, list):
        return ""

    for allow_tool_calls in (False, True):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            info = message.get("info")
            if not isinstance(info, dict) or info.get("role") != "assistant":
                continue
            if not allow_tool_calls and info.get("finish") in {"tool-calls", "tool_calls"}:
                continue
            parts = message.get("parts")
            if not isinstance(parts, list):
                continue
            text = "\n".join(
                str(part["text"])
                for part in parts
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
            ).strip()
            if text:
                return text
    return ""


def extract_final_assistant_text_from_jsonl(path: str | Path) -> str:
    """Best-effort fallback for OpenCode's `run --format=json` event stream."""
    messages: dict[str, dict[str, str]] = {}
    last_message_id = None
    last_text = ""
    with open(path, encoding="utf-8") as input_file:
        for event_index, line in enumerate(input_file):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "text":
                continue
            part = event.get("part")
            text = part.get("text") if isinstance(part, dict) else event.get("text")
            if text:
                last_text = str(text).strip()
                if isinstance(part, dict) and part.get("messageID"):
                    last_message_id = str(part["messageID"])
                    part_id = str(part.get("id", event_index))
                    messages.setdefault(last_message_id, {})[part_id] = last_text

    if last_message_id is not None:
        return "\n".join(messages[last_message_id].values()).strip()
    return last_text
