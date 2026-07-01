import argparse
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired, TypedDict

import litellm
from flask import Flask, jsonify, request
from pydantic import BaseModel, model_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

litellm.drop_params = True

app = Flask(__name__)

CANT_ANSWER = "can't answer (perhaps transient hiccup)"
IRRELEVANT_QUESTION = "irrelevant question"


class _CircuitBreaker:
    """Fail-fast guard around the judge LLM.

    When the judge is offline/slow, each ask would otherwise re-pay the full
    EVAL_MAX_RETRIES x EVAL_TIMEOUT_S budget (~9 min by default), stalling the agent. After
    ``fail_threshold`` consecutive judge-transport failures the breaker OPENS for ``cooldown_s``
    seconds, during which asks fast-fail to "can't answer" immediately (which the generation
    module then classifies as an infra hiccup -> excluded + rerun). A success closes it.
    Set ASK_HUMAN_BREAKER_FAILS=0 to disable. State is per-process (per chunk job), so a rerun
    in a fresh job always starts closed.
    """

    def __init__(self, fail_threshold: int, cooldown_s: float):
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self.consecutive_failures = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        if self.fail_threshold <= 0:
            return True
        with self._lock:
            return not (self.open_until and time.time() < self.open_until)

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive_failures += 1
            if self.fail_threshold > 0 and self.consecutive_failures >= self.fail_threshold:
                self.open_until = time.time() + self.cooldown_s
                logger.warning(
                    "ask_human circuit breaker OPEN for %.0fs after %d consecutive judge failures",
                    self.cooldown_s,
                    self.consecutive_failures,
                )


# DISABLED by default (fail_threshold=0): a benchmark needs the judge to actually answer, and an
# aggressive breaker fast-fails a slow-but-working judge into "can't answer" (which inflates the
# hiccup rate and triggers agent re-ask spirals). Genuine judge outages are handled out-of-band by
# the orchestrator (judge health-check + restart) and the per-task timeout. Set ASK_HUMAN_BREAKER_FAILS>0
# only if you specifically want per-process fast-fail against a judge you expect to be down.
_JUDGE_BREAKER = _CircuitBreaker(
    fail_threshold=int(os.getenv("ASK_HUMAN_BREAKER_FAILS", "0")),
    cooldown_s=float(os.getenv("ASK_HUMAN_BREAKER_COOLDOWN_S", "120")),
)


class BlockerEntry(BaseModel):
    """Represents a blocker entry with description, resolution, and example questions."""

    id: str
    description: str
    resolution: str
    example_questions: list[str] = []
    type: str | None = None  # missing_parameter, ambiguous_requirement, etc.

    @model_validator(mode="before")
    @classmethod
    def normalize_question_field(cls, data: dict) -> dict:
        """Support 'acceptable_questions', 'example_questions', and 'trigger_questions' as aliases."""
        if isinstance(data, dict):
            # Check for alternative field names and normalize to "example_questions"
            for alt_name in ("acceptable_questions", "trigger_questions"):
                if alt_name in data and "example_questions" not in data:
                    data["example_questions"] = data.pop(alt_name)
                elif alt_name in data:
                    # If both exist, example_questions takes precedence
                    data.pop(alt_name)
        return data


class BlockerRegistry(BaseModel):
    """Container for a list of blockers, used for parsing JSON files."""

    blockers: list[BlockerEntry]

    def to_dict(self) -> dict[str, BlockerEntry]:
        """Convert to {blocker_id: BlockerEntry} format."""
        return {blocker.id: blocker for blocker in self.blockers}


# Global state set at server startup
TASKS_DIR: Path | None = None
# Maps instance_id -> {blocker_id -> BlockerEntry}
CACHED_BLOCKERS: dict[str, dict[str, BlockerEntry]] | None = None

# Max number of answered questions per blocker for each instance.
# The 3rd and later matched question for the same blocker is forced to IRRELEVANT_QUESTION.
MAX_ANSWERED_QUESTIONS_PER_BLOCKER = 2


class LogEntry(TypedDict):
    question: str
    response: str
    blocker_name: str | None


class Log(TypedDict):
    questions: list[LogEntry]
    n_blockers: int
    blockers: dict[str, bool]
    precision: NotRequired[float]
    recall: NotRequired[float]
    ask_f1: NotRequired[float]


# Track all interactions across tool requests
GLOBAL_LOGS: dict[str, Log] = {}


class AskHuman:
    """LLM-equipped blocker registry for semantic question matching."""

    def __init__(self, blocker_registry: dict[str, BlockerEntry]):
        """
        Initialize Ask_Human tool with pre-parsed blocker registry.

        Args:
            blocker_registry: Dict mapping blocker_id to BlockerEntry.
        """
        self.provider = self._resolve_provider()
        self.hosting_type = os.getenv("ASK_HUMAN_HOSTING_TYPE", self.provider)

        self.api_key = os.getenv("LITELLM_API_KEY")
        self.api_base = (os.getenv("ASK_HUMAN_LITELLM_BASE_URL") or "").strip()
        self.model = os.getenv("ASK_HUMAN_MODEL", "casperhansen/llama-3.3-70b-instruct-awq")

        # Self-hosted OpenAI-compatible judge configuration
        self.self_hosted_base_url = (os.getenv("ASK_HUMAN_SELF_HOSTED_BASE_URL") or "").strip()
        self.self_hosted_api_key = (os.getenv("ASK_HUMAN_SELF_HOSTED_API_KEY") or "").strip()
        self.self_hosted_model = (os.getenv("ASK_HUMAN_SELF_HOSTED_MODEL") or self.model).strip()

        if self.provider == "litellm" and not self.api_base:
            raise ValueError(
                "ASK_HUMAN_PROVIDER=litellm requires ASK_HUMAN_LITELLM_BASE_URL "
                "(set via judge config hosting.type=litellm_proxy)"
            )
        if self.provider == "self_hosted" and not self.self_hosted_base_url:
            raise ValueError(
                "ASK_HUMAN_PROVIDER=self_hosted requires ASK_HUMAN_SELF_HOSTED_BASE_URL "
                "(set via judge config hosting.type=self_hosted)"
            )
        if self.provider == "litellm":
            logger.info(
                "ask_human judge hosting=%s backend=litellm model=%s api_base=%s",
                self.hosting_type,
                self.model,
                self.api_base or "(default)",
            )
        else:
            logger.info(
                "ask_human judge hosting=%s backend=%s model=%s base_url=%s",
                self.hosting_type,
                self.provider,
                self.self_hosted_model,
                self.self_hosted_base_url,
            )

        if self.provider == "litellm" and not self.api_key:
            logger.warning("LITELLM_API_KEY not set in server environment")

        if self.api_key:
            litellm.api_key = self.api_key
        if self.api_base:
            litellm.api_base = self.api_base

        self.blocker_registry = blocker_registry
        self.log: Log = {
            "questions": [],
            "n_blockers": len(blocker_registry),
            "blockers": {key: False for key in blocker_registry},
        }

    @staticmethod
    def _resolve_provider() -> str:
        """Resolve ask_human judge backend from environment variables.

        Supported values:
          - litellm
          - self_hosted (default)
          - openai_compatible
        """
        raw = os.getenv("ASK_HUMAN_PROVIDER", "self_hosted").strip().lower().replace("-", "_")
        if raw in {"litellm", "self_hosted", "openai_compatible"}:
            return raw
        logger.warning("Unknown ASK_HUMAN_PROVIDER=%r; defaulting to self_hosted", raw)
        return "self_hosted"

    @staticmethod
    def _normalize_openai_compatible_base_url(base_url: str) -> str:
        """Normalize a base URL so chat completions live at /v1/chat/completions."""
        base = base_url.strip().rstrip("/")
        if not base:
            return base
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _create_comprehensive_prompt(self, agent_question: str) -> str:
        """Create comprehensive prompt for single LLM call evaluation."""
        blockers_info = []
        for key, blocker in self.blocker_registry.items():
            blockers_info.append(
                f"""
BLOCKER: {key}
DESCRIPTION: {blocker.description}
EXAMPLE QUESTIONS: {', '.join(blocker.example_questions)}"""
            )

        return f"""You are an EXTREMELY PRECISE semantic similarity evaluator for a blocker registry system.

AVAILABLE BLOCKERS:
{chr(10).join(blockers_info)}

AGENT MESSAGE: "{agent_question}"

First, determine if the agent's message is valid. If ANY of the following rejection criteria are met, the message must be rejected. **Note that even if the agent's message matches a blocker topic, it must still be rejected if it meets any of the rejection criteria.**

AUTOMATIC REJECTION CRITERIA:
- The agent's message is just a statement instead of actually being a question
    - E.g. "Columns x, y, and z discuss metrics." or "Columns x, y, and z are too similar." are just statements.
    - E.g. "Which of the x, y, or z columns should I use?" or "What are the differences between the x, y, and z columns?" are actual questions.
- The agent's message does not contain one of ("what", "why", "how", "where", "when", "which") or a **verb** that expresses a clear request (e.g. "define", "explain", "describe", "differentiate").
    - E.g. "Makeup brands?" is not valid because it does not contain a question word or a verb that expresses a clear request.
    - No exceptions. Questions that state an assumption and ask for confirmation (e.g. "I'm assuming X. Is that correct?") are NOT valid because they embed the agent's guess into the question.
- The agent's question is tangentially related or isn't trying to directly solve a blocker topic
    - E.g. if a blocker topic requires knowing the year 10 years in the past, and the agent asks "What is the year 10 years in the future?", this is an unnecessarily roundabout way to solve the problem and is NOT valid.
- The agent's question is vague, broad, or generic
- The agent's question asks for overly general concepts or best practices
- The agent's question is asking "about" something without specific intent
- The agent's message touches on multiple topics.
    - E.g. "What columns should I use to determine car color? What info does the description field tell me?" is not valid. Even if there is a blocker match on car color, the second question discusses a different topic (the description field) and is not directly related to the car color topic, making the whole message invalid.
- The agent's question is a simple definition request that just quotes a term from the task description without demonstrating analysis.
    - E.g. "What does 'high-quality' mean?" or "How is 'peak career' defined?" are too simple — the agent is just parroting back an ambiguous term.
    - A valid question must show the agent has done SOME work: e.g. "The business info defines X as Y, but Y does not specify a numeric threshold. What threshold should I use?"

If any of the above criteria are met, the agent's message must be rejected. Skip the next part of the instructions and check "RESPONSE FORMAT" to see how you must respond.

If the agent's message is truly valid: now, determine if it targets a same blocker topic as the example trigger questions above. **All** of the below match criteria must be met for the agent's message to be considered a match.

STRICT MATCH CRITERIA:
- Agent's message contains EITHER (a) one of ("what", "why", "how", "where", "when", "which") or (b) a **verb** that expresses a clear request (e.g. "define", "explain", "describe", "differentiate").
- Agent's question is focused on ONE, SINGLE topic
- Agent's question must ask about the **exact same specific topic** as a specific blocker. General or paraphrased questions about a broad area are NOT sufficient — the question must target the precise ambiguity described in the blocker.
- Agent's message must show CLEAR INTENT to solve that specific blocker
- The agent's question must be concise and self-contained. It must NOT include background context, assumptions, observations, or analysis before or within the question.

CONFIDENCE THRESHOLD: Must be 95%+ confident for MATCH. When in doubt, reject.

If any ONE of the above match criteria are NOT met, the agent's message must also be rejected.

RESPONSE FORMAT:
You **MUST** respond in JSON format. Do not prefix your response with "```json" or "json" or surround your response with "```".

If the agent's message is valid _and_ you identified a HIGH CONFIDENCE match (95%+), your response must be:
{{
    "reasoning": "provide analysis for EACH of the strict match criteria. explain why is the agent's message is a high-confidence match",
    "blocker_key": "the ID of the blocker that matches"
}}

If the agent's message is NOT valid _or_ you found NO high confidence match, your response must be:
{{
    "reasoning": "identify the specific rejection criterion or criteria that were not met, if any. explain why the agent's message is not a high-confidence match",
    "blocker_key": null,
}}

Think carefully and thoroughly. Make sure you analyze all criteria.

Your response:"""

    # Env-tunable so external gateways (seconds-scale latency) can fail fast instead of waiting the
    # default ~9 min worst case (3 x 180s). Worst case must stay below the ask_human tool client's
    # timeout and SWE-agent's tool execution_timeout.
    EVAL_TIMEOUT_S = int(os.getenv("ASK_HUMAN_EVAL_TIMEOUT_S", "180"))
    EVAL_MAX_RETRIES = int(os.getenv("ASK_HUMAN_EVAL_MAX_RETRIES", "3"))

    @staticmethod
    def _run_with_hard_timeout(func, timeout_s):
        """Run a synchronous callable in a daemon thread with a hard wall-clock
        timeout.  Returns (result, None) on success, (None, exception) on
        failure, or raises TimeoutError if the thread doesn't finish in time.

        This is necessary because litellm's ``timeout`` parameter only covers
        the HTTP response phase and cannot interrupt DNS, TCP-connect, or
        SSL-handshake hangs.
        """
        result_holder = []
        error_holder = []

        def _target():
            try:
                result_holder.append(func())
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            raise TimeoutError(f"LLM call exceeded hard timeout of {timeout_s}s")
        if error_holder:
            raise error_holder[0]
        return result_holder[0]

    def _evaluate_single_call(self, agent_question: str) -> str:
        """Evaluate semantic similarity against all blockers.

        Retries up to ``EVAL_MAX_RETRIES`` times with a hard per-attempt
        wall-clock timeout of ``EVAL_TIMEOUT_S`` seconds.  On total failure
        returns ``"NO_MATCH"`` so the caller falls back to "can't answer".
        """
        prompt = self._create_comprehensive_prompt(agent_question)

        # Fast-fail while the breaker is open (judge presumed down) so the agent isn't stalled
        # ~EVAL_MAX_RETRIES x EVAL_TIMEOUT_S per question. "NO_MATCH" -> "can't answer" -> the
        # generation module flags the instance infra (hiccup) -> excluded + rerun.
        if not _JUDGE_BREAKER.allow():
            logger.warning("ask_human breaker OPEN; fast-failing question (judge presumed offline).")
            return "NO_MATCH"

        for attempt in range(1, self.EVAL_MAX_RETRIES + 1):
            try:
                text = self._run_with_hard_timeout(
                    lambda: self._evaluate_prompt(prompt),
                    timeout_s=self.EVAL_TIMEOUT_S,
                )
                logger.debug(f"LLM evaluation: '{agent_question}' -> {text}")
                _JUDGE_BREAKER.record_success()
                return text
            except Exception as e:
                logger.error(
                    f"LLM evaluation attempt {attempt}/{self.EVAL_MAX_RETRIES} " f"failed: {e}"
                )

        _JUDGE_BREAKER.record_failure()
        return "NO_MATCH"

    def _evaluate_prompt(self, prompt: str) -> str:
        """Dispatch prompt evaluation to selected backend."""
        if self.provider == "litellm":
            # Pass api_key/api_base EXPLICITLY. litellm's hosted_vllm provider does NOT read the
            # global litellm.api_key, so relying on the global made it send its "fake-key" default
            # and the gateway returned 401 ("LiteLLM Virtual Key expected"). Forward the real bearer
            # key (and base) on every call instead.
            result = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                timeout=self.EVAL_TIMEOUT_S,
                num_retries=0,
                api_base=self.api_base or None,
                api_key=self.api_key or None,
            )
            return result.choices[0].message.content.strip()
        return self._evaluate_prompt_self_hosted(prompt)

    def _evaluate_prompt_self_hosted(self, prompt: str) -> str:
        """Call a self-hosted OpenAI-compatible endpoint directly."""
        base_url = self._normalize_openai_compatible_base_url(self.self_hosted_base_url)
        if not base_url:
            raise ValueError("Self-hosted ask_human backend requires a base URL")
        url = f"{base_url}/chat/completions"

        payload = {
            "model": self.self_hosted_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.05,
        }
        headers = {"Content-Type": "application/json"}
        if self.self_hosted_api_key:
            headers["Authorization"] = f"Bearer {self.self_hosted_api_key}"

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.EVAL_TIMEOUT_S) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"Self-hosted judge HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Self-hosted judge request failed: {exc}") from exc

        parsed = json.loads(body)
        try:
            return parsed["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected self-hosted response shape: {parsed}") from exc

    def _log_entry(self, question: str, response: str, blocker_name: str | None):
        """Helper to add entry to log."""
        self.log["questions"].append(
            {"question": question, "response": response, "blocker_name": blocker_name}
        )
        # Mark blocker as resolved if a valid blocker was matched
        if blocker_name is not None and blocker_name in self.log["blockers"]:
            self.log["blockers"][blocker_name] = True

    def maybe_extract_json(self, response: str) -> str:
        # Drop any <think>...</think> reasoning some models emit before the JSON.
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        # Otherwise grab the first {...} object if there's surrounding prose.
        brace = re.search(r"\{.*\}", response, re.DOTALL)
        return brace.group(0).strip() if brace else response.strip()

    @staticmethod
    def _single_line(value: str | None, max_chars: int = 2000) -> str:
        if value is None:
            return "<none>"
        compact = " ".join(value.split())
        if len(compact) > max_chars:
            return f"{compact[:max_chars]}...<truncated>"
        return compact

    def _log_judge_failure(
        self,
        *,
        agent_question: str,
        raw_judge_response: str | None,
        surfaced_response: str,
        error: Exception | None = None,
    ) -> None:
        error_text = f" error={self._single_line(str(error), max_chars=500)}" if error else ""
        logger.error(
            "ask_human judge failure question: %s\nraw_judge_response: %s%s\nresponse surfaced to agent: %s",
            self._single_line(agent_question, max_chars=600),
            self._single_line(raw_judge_response),
            error_text,
            surfaced_response,
        )

    def ask_human(self, agent_question: str) -> str:
        """
        Main entry point for agent questions.

        Returns resolution if direct match found, otherwise "irrelevant question".
        Returns "can't answer" if no blocker registry is available.
        """
        # Check if blocker registry is available
        if not hasattr(self, "blocker_registry") or not self.blocker_registry:
            logger.error("No blocker registry available")
            response = CANT_ANSWER
            self._log_judge_failure(
                agent_question=agent_question,
                raw_judge_response=None,
                surfaced_response=response,
            )
            self._log_entry(agent_question, response, None)
            return response

        # Basic validation
        if not agent_question or not agent_question.strip() or len(agent_question.strip()) < 3:
            logger.info("Rejected: Empty or too short question")
            response = IRRELEVANT_QUESTION
            self._log_entry(agent_question, response, None)
            return response

        agent_question = agent_question.strip()

        # Single LLM call to evaluate against all blockers
        raw_judge_response = None
        try:
            raw_judge_response = self._evaluate_single_call(agent_question)
            result = json.loads(self.maybe_extract_json(raw_judge_response))
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            response = CANT_ANSWER
            self._log_judge_failure(
                agent_question=agent_question,
                raw_judge_response=raw_judge_response,
                surfaced_response=response,
                error=e,
            )
            self._log_entry(agent_question, response, None)
            return response

        blocker_key = result["blocker_key"]
        if blocker_key is not None:
            if blocker_key in self.blocker_registry:
                resolution = self.blocker_registry[blocker_key].resolution

                self._log_entry(agent_question, resolution, blocker_key)
                return resolution
            else:
                logger.warning(f"LLM returned invalid blocker key: {blocker_key}")

        response = IRRELEVANT_QUESTION
        self._log_entry(agent_question, response, None)
        return response


def normalize_instance_id(instance_id: str) -> list[str]:
    """
    Generate possible lookup keys for an instance ID.

    Returns a list of possible keys to try, in order of preference.

    Handles expanded instance IDs from the mixed-mode batch runner, which have the format:
        {original_id}__{model_safe}__{mode}__pass_{n}

    For example:
        "ansible__callback-host-label__openai_llmengine_qwen3__ask_human__pass_1"
        -> tries: "ansible__callback-host-label__openai_llmengine_qwen3__ask_human",
                  "ansible__callback-host-label", "ansible"
    """
    lookup_id = instance_id

    # Strip pass-specific suffix if present
    if "__pass_" in lookup_id:
        lookup_id = lookup_id.rsplit("__pass_", 1)[0]

    candidates = [lookup_id]

    # Strip known mode suffixes that the batch runner appends
    for mode_suffix in ("__ask_human", "__baseline", "__full_info", "__standard"):
        if lookup_id.endswith(mode_suffix):
            without_mode = lookup_id[: -len(mode_suffix)]
            if without_mode not in candidates:
                candidates.append(without_mode)
            break

    # Try progressively shorter prefixes by stripping from the right on "__"
    # This handles model name suffixes like "__openai_llmengine_qwen3..."
    if CACHED_BLOCKERS is not None:
        parts = lookup_id.split("__")
        for i in range(len(parts) - 1, 0, -1):
            prefix = "__".join(parts[:i])
            if prefix in CACHED_BLOCKERS and prefix not in candidates:
                candidates.append(prefix)
                break

    # Fallback: try just the first segment
    if "__" in lookup_id:
        folder_name = lookup_id.split("__")[0]
        if folder_name not in candidates:
            candidates.append(folder_name)

    return candidates


def get_blockers_for_instance(instance_id: str) -> dict[str, BlockerEntry] | None:
    """
    Get blocker registry for a given instance ID from CACHED_BLOCKERS.

    Tries multiple normalized forms of the instance_id.
    """
    if CACHED_BLOCKERS is None:
        return None

    # Try each normalized form of the instance_id
    for candidate in normalize_instance_id(instance_id):
        if candidate in CACHED_BLOCKERS:
            return CACHED_BLOCKERS[candidate]

    logger.warning(
        f"Could not find blocker registry for instance: {instance_id} "
        f"(tried: {normalize_instance_id(instance_id)})"
    )
    return None


def parse_blocker_registry(data: dict) -> dict[str, BlockerEntry]:
    """
    Parse blocker registry data into {blocker_id: BlockerEntry} format.

    Expected input format:
        {"blockers": [{"id": ..., "description": ..., "resolution": ..., "example_questions": [...]}]}

    Raises ValidationError if the data doesn't match the expected format.
    """
    registry = BlockerRegistry.model_validate(data)
    return registry.to_dict()


def load_blocker_registry(path: Path) -> dict[str, BlockerEntry]:
    """Load, validate, and parse a blocker registry from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    return parse_blocker_registry(data)


def _instance_id_aliases_for_task_dir(task_dir: Path) -> list[str]:
    """Build lookup aliases for a task directory.

    Uses both task directory name and metadata.json.instance_id (if present).
    """
    aliases: list[str] = [task_dir.name]
    metadata_path = task_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            metadata_instance_id = metadata.get("instance_id")
            if isinstance(metadata_instance_id, str) and metadata_instance_id.strip():
                normalized = metadata_instance_id.strip()
                if normalized not in aliases:
                    aliases.append(normalized)
        except Exception as exc:
            logger.warning(f"Failed to parse metadata from {metadata_path}: {exc}")
    return aliases


@app.route("/get_logs", methods=["POST"])
def get_logs():
    return jsonify(GLOBAL_LOGS)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        question = data.get("question")
        instance_id = data.get("instance_id")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        if not instance_id:
            logger.warning("Missing instance_id")
            return jsonify({"response": CANT_ANSWER})

        blockers = get_blockers_for_instance(instance_id)

        if not blockers:
            logger.warning(f"No blocker registry found for instance {instance_id}")
            return jsonify({"response": CANT_ANSWER})

        tool = AskHuman(blockers)
        response = tool.ask_human(question)

        if instance_id:
            if instance_id not in GLOBAL_LOGS:
                GLOBAL_LOGS[instance_id] = {
                    "questions": [],
                    "n_blockers": tool.log["n_blockers"],
                    "blockers": tool.log["blockers"].copy(),
                }

            # Enforce per-blocker cap: allow only two answered hits per blocker.
            # On the 3rd+ matched question for the same blocker, force irrelevant.
            latest_entry = tool.log["questions"][-1] if tool.log["questions"] else None
            if latest_entry and latest_entry.get("blocker_name") is not None:
                blocker_name = latest_entry["blocker_name"]
                prior_hits = sum(
                    1
                    for q in GLOBAL_LOGS[instance_id]["questions"]
                    if q.get("blocker_name") == blocker_name
                )
                if prior_hits >= MAX_ANSWERED_QUESTIONS_PER_BLOCKER:
                    logger.info(
                        "Per-blocker cap reached for %s blocker=%s (%d/%d)",
                        instance_id,
                        blocker_name,
                        prior_hits,
                        MAX_ANSWERED_QUESTIONS_PER_BLOCKER,
                    )
                    latest_entry["response"] = IRRELEVANT_QUESTION
                    latest_entry["blocker_name"] = None
                    response = IRRELEVANT_QUESTION

            GLOBAL_LOGS[instance_id]["questions"].extend(tool.log["questions"])
            # Update blockers status (merge True values)
            for blocker_key, resolved in tool.log["blockers"].items():
                if resolved:
                    GLOBAL_LOGS[instance_id]["blockers"][blocker_key] = True

        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({"response": CANT_ANSWER}), 500


def main():
    global TASKS_DIR, CACHED_BLOCKERS

    parser = argparse.ArgumentParser(description="Ask Human Server")
    parser.add_argument(
        "--tasks-dir", type=str, help="Directory containing tasks (or single task folder)"
    )
    parser.add_argument(
        "--blockers-json",
        type=str,
        help="JSON string mapping instance_id to blocker registry: "
        '{instance_id: {"blockers": [...]}, ...} (takes precedence over --tasks-dir)',
    )
    parser.add_argument(
        "--blockers-file",
        type=str,
        help="Path to a JSON file with the same content as --blockers-json. Use this instead of "
        "--blockers-json when the registry is large, to avoid the OS argument-length limit.",
    )
    parser.add_argument("--port", type=int, default=9521, help="Port to run on")
    args = parser.parse_args()

    # Load blockers from a file (avoids 'Argument list too long' for large registries).
    if args.blockers_file and not args.blockers_json:
        with open(args.blockers_file) as bf:
            args.blockers_json = bf.read()

    # Load blockers from JSON string if provided
    # Expected format: {instance_id: {"blockers": [...]}, ...}
    if args.blockers_json:
        try:
            raw_data = json.loads(args.blockers_json)
            if not isinstance(raw_data, dict):
                raise ValueError("Expected dict mapping instance_id to blocker registry")
            # Parse each instance's blocker registry
            CACHED_BLOCKERS = {}
            for instance_id, registry in raw_data.items():
                CACHED_BLOCKERS[instance_id] = parse_blocker_registry(registry)
            logger.info(
                f"Loaded blockers for {len(CACHED_BLOCKERS)} instance(s) from --blockers-json"
            )
        except Exception as e:
            logger.error(f"Failed to parse --blockers-json: {e}")
            raise

    # Set tasks directory if provided - load all blockers at startup
    if args.tasks_dir:
        TASKS_DIR = Path(args.tasks_dir)
        if not TASKS_DIR.exists():
            logger.warning(f"Tasks directory does not exist: {TASKS_DIR}")
        else:
            logger.info(f"Using tasks directory: {TASKS_DIR}")
            # Load all blockers from tasks directory if not already loaded from JSON
            if CACHED_BLOCKERS is None:
                CACHED_BLOCKERS = {}

                # Check if TASKS_DIR itself is a single task folder
                # Look for blocker_registry.json or *_registry.json
                single_registry = TASKS_DIR / "blocker_registry.json"
                if not single_registry.exists():
                    # SQL tasks use {task_id}_registry.json naming
                    registry_candidates = list(TASKS_DIR.glob("*_registry.json"))
                    if registry_candidates:
                        single_registry = registry_candidates[0]
                if single_registry.exists():
                    try:
                        registry = load_blocker_registry(single_registry)
                        aliases = _instance_id_aliases_for_task_dir(TASKS_DIR)
                        for instance_id in aliases:
                            CACHED_BLOCKERS[instance_id] = registry
                        logger.info(
                            f"Loaded {len(registry)} blocker(s) for instance aliases "
                            f"{aliases} from {single_registry}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load registry from {single_registry}: {e}")
                else:
                    # Multi-task directory: iterate through subdirectories
                    for subdir in TASKS_DIR.iterdir():
                        if subdir.is_dir():
                            registry_path = subdir / "blocker_registry.json"
                            if not registry_path.exists():
                                # SQL tasks use {task_id}_registry.json
                                candidates = list(subdir.glob("*_registry.json"))
                                if candidates:
                                    registry_path = candidates[0]
                            if registry_path.exists():
                                try:
                                    registry = load_blocker_registry(registry_path)
                                    aliases = _instance_id_aliases_for_task_dir(subdir)
                                    for instance_id in aliases:
                                        CACHED_BLOCKERS[instance_id] = registry
                                    logger.debug(
                                        f"Loaded {len(registry)} blocker(s) for aliases {aliases}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to load registry from {registry_path}: {e}"
                                    )

                    logger.info(
                        f"Loaded blockers for {len(CACHED_BLOCKERS)} instance(s) from {TASKS_DIR}"
                    )

    if not CACHED_BLOCKERS:
        logger.warning(
            "No blockers loaded. Provide --blockers-json or --tasks-dir with blocker registries"
        )

    logger.info(f"Starting ask_human server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
