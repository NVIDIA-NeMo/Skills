"""HIL-Bench litellm shim — coerce empty assistant message ``content`` to a non-empty string.

WHY THIS EXISTS
---------------
Some OpenAI-compatible backends behind an inference gateway (notably certain vLLM-served
MoE model routes) REJECT a
request whose message history contains an assistant tool-call message with empty ``content``
("message content cannot be empty" / HTTP 400, and in the minimal repro a 504 hang) — even though
the OpenAI spec allows ``content: null`` for tool-call messages, and other routes (Azure/OpenAI
gpt-5.4) accept it. SWE-agent relays the model's own (empty-content) tool-call messages back
verbatim on the next turn, so every multi-turn run against such a backend dies after ~2 turns.

Isolated repro (hil_bench_eval/empty_content_probe.py) confirmed: on the affected route an empty
``""`` content fails, while ``null`` / omitted / a single space / any text all return 200; gpt-5.4
accepts everything. So coercing empty assistant content to a single space is a safe, model-agnostic
fix.

HOW IT WORKS (no vendored edit)
-------------------------------
This file is a ``sitecustomize`` module: Python auto-imports it at interpreter startup from any
directory on ``sys.path``. ``hilbench.py`` puts *only this directory* on the agent's PYTHONPATH, so
it loads before ``python -m sweagent run`` issues any model call. It then wraps
``litellm.completion`` / ``litellm.acompletion`` in place (the litellm package files and SWE-agent
source are untouched) and rewrites empty/whitespace assistant ``content`` to a placeholder before
the request is sent. Disable by setting ``HIL_EMPTY_CONTENT_SHIM=0``; tune the placeholder via
``HIL_EMPTY_CONTENT_PLACEHOLDER`` (default a single space).
"""

import os
import sys

# MUST be NON-WHITESPACE: a single space (" ") gets normalized back to "" by litellm before the
# request hits the wire, so the gateway still 400s "content cannot be empty" (observed: shim
# fixed=4 yet the call kept failing + retrying). A visible token survives normalization.
_PLACEHOLDER = os.environ.get("HIL_EMPTY_CONTENT_PLACEHOLDER", "(tool call)") or "(tool call)"


def _is_empty(content) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        # Multimodal content list: empty list, or only blank text parts.
        if not content:
            return True
        return all(
            isinstance(p, dict) and p.get("type") == "text" and not (p.get("text") or "").strip()
            for p in content
        )
    return False


def _msg_get(m, key):
    """Read a field from a message that may be a dict OR an object (litellm Message/pydantic)."""
    if isinstance(m, dict):
        return m.get(key)
    return getattr(m, key, None)


def _msg_set_content(m, value):
    if isinstance(m, dict):
        m["content"] = value
    else:
        try:
            setattr(m, "content", value)
        except Exception:
            pass


def _sanitize_messages(messages) -> int:
    """In-place: give empty-content assistant messages a non-empty placeholder. Returns #fixed.

    Handles both dict messages and object messages (litellm Message / pydantic).
    """
    if not isinstance(messages, (list, tuple)):
        return 0
    fixed = 0
    for m in messages:
        if _msg_get(m, "role") == "assistant" and _is_empty(_msg_get(m, "content")):
            _msg_set_content(m, _PLACEHOLDER)
            fixed += 1
    return fixed


_LOG_CALLS = int(os.environ.get("HIL_SHIM_LOG_CALLS", "10"))
_state = {"calls": 0}


def _log(msg: str) -> None:
    print(f"[hil-shim] {msg}", file=sys.stderr, flush=True)


def _find_messages(args, kwargs):
    m = kwargs.get("messages")
    if isinstance(m, (list, tuple)):
        return m
    for a in args:  # positional fallback: first arg that looks like a message list
        if isinstance(a, (list, tuple)) and a and _msg_get(a[0], "role") is not None:
            return a
    return None


def _wrap(orig_fn, fname):
    def wrapper(*args, **kwargs):
        msgs = _find_messages(args, kwargs)
        do_log = _state["calls"] < _LOG_CALLS
        if msgs is None:
            if do_log:
                _state["calls"] += 1
                _log(f"{fname}: NO messages list found (nargs={len(args)} kwkeys={list(kwargs)[:8]})")
            return orig_fn(*args, **kwargs)
        empty_roles = [str(_msg_get(x, "role")) for x in msgs if _is_empty(_msg_get(x, "content"))]
        fixed = _sanitize_messages(msgs)
        if do_log:
            _state["calls"] += 1
            _log(f"{fname}: nmsgs={len(msgs)} empty_by_role={empty_roles} fixed={fixed}")
        return orig_fn(*args, **kwargs)

    wrapper._hil_empty_content_wrapped = True
    return wrapper


def _install() -> None:
    if os.environ.get("HIL_EMPTY_CONTENT_SHIM", "1") == "0":
        return
    try:
        import litellm
    except Exception:  # litellm not importable in this interpreter -> nothing to do
        return
    wrapped = []
    for name in ("completion", "acompletion", "completion_with_retries"):
        orig = getattr(litellm, name, None)
        if orig is None or getattr(orig, "_hil_empty_content_wrapped", False):
            continue
        setattr(litellm, name, _wrap(orig, name))
        wrapped.append(name)
    ver = getattr(litellm, "__version__", "?")
    _log(f"active placeholder={_PLACEHOLDER!r} litellm={ver} wrapped={wrapped or 'NONE'}")


_install()
