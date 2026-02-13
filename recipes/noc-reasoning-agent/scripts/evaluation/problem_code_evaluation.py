import argparse
import json
import re

from tqdm import tqdm

# Parse arguments for input JSONL path
parser = argparse.ArgumentParser(description="Evaluation Pipeline for Agent Responses")
parser.add_argument("input_jsonl", help="Path to agent_responses.jsonl containing expected_answer and agent_response")
args = parser.parse_args()

# Map free-form / synonym phrases (lowercase) to canonical close codes (lowercase) for matching.
# Expected is from ground truth (e.g. "Resolved", "Issue Corrected"); we check if response
# contains the expected phrase OR any synonym that maps to the same meaning.
CLOSE_CODE_SYNONYMS = {
    "resolved": [
        "resolved",
        "commercial power restored",
        "power restored",
        "cold reboot",
        "incident closed",
        "issue resolved",
        "activity completed",
    ],
    "issue corrected": [
        "issue corrected",
        "solved remotely",
        "solved remotely (permanently)",
        "cleared in testing",
        "configuration corrected",
        "software fix",
        "network fix",
        "performance improvement",
        "activity completed",
    ],
    "activity completed": ["activity completed", "cleared in testing", "resolved"],
    "ru reset": ["ru reset", "reset ru"],
    "other": ["other"],
}


def normalize_close_code(s: str) -> str:
    """Return lowercase, stripped; empty if missing."""
    if not s or not isinstance(s, str):
        return ""
    return s.strip().lower()


def _acceptable_phrases_for_expected(expected_norm: str):
    """Return list of phrases (lowercase) that count as a match for this expected close code."""
    if expected_norm in CLOSE_CODE_SYNONYMS:
        return [expected_norm] + list(CLOSE_CODE_SYNONYMS[expected_norm])
    for canonical, synonyms in CLOSE_CODE_SYNONYMS.items():
        if expected_norm == canonical or expected_norm in synonyms:
            return [canonical] + list(synonyms)
    return [expected_norm]


def response_matches_expected(response_lower: str, expected_close_code: str) -> bool:
    """True if response contains expected close code or an accepted synonym."""
    expected_norm = normalize_close_code(expected_close_code)
    if not expected_norm:
        return False
    acceptable = _acceptable_phrases_for_expected(expected_norm)
    return any(phrase in response_lower for phrase in acceptable)


print(f"Loading input JSONL: {args.input_jsonl}")

# Counters
correct = 0
incorrect = 0
failed = 0
total = 0

# Count total lines (for tqdm)
with open(args.input_jsonl, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# Process JSONL line-by-line
with open(args.input_jsonl, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines):
        try:
            row = json.loads(line)

            agent_response = row.get("agent_response")
            expected = row.get("expected", "")

            if agent_response is None or not isinstance(agent_response, list) or len(agent_response) == 0:
                print("Error: missing or empty 'agent_response'")
                failed += 1
                total += 1
                continue

            # Extract close code from expected
            m = re.search(r"Close Code:\s*\[(.*?)\]", expected) if expected else None
            close_code = m.group(1).strip() if m else None

            if not close_code:
                failed += 1
                total += 1
                continue

            # Take the model's last message
            last_msg = agent_response[-1]
            content = last_msg.get("content") if isinstance(last_msg, dict) else str(last_msg)
            response = (content or "").lower()

            # Slice from "close code..." if present
            idx = response.rfind("close code")
            if idx >= 0:
                response = response[idx:]

            if response_matches_expected(response, close_code):
                print(f"✅ Real Close code: {close_code}, Response: {response[:120]}...")
                correct += 1
            else:
                incorrect += 1
                print(f"❌ Real Close code: {close_code}, Response: {response[:120]}...")

        except Exception as e:
            print("Error:", e)
            failed += 1

        total += 1

print(f"Total: {total}, correct: {correct}, failed: {failed}, incorrect: {incorrect}")
