# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
    # RAN codes
    "ran-001: cell service interruption": ["ran-001", "cell service interruption", "cell down", "site down"],
    "ran-002: cell administratively disabled": ["ran-002", "cell administratively disabled", "cell locked"],
    "ran-005: rrc setup success rate degraded": ["ran-005", "rrc setup", "rrc degraded"],
    "ran-006: data radio bearer degradation": ["ran-006", "data radio bearer", "drb degradation"],
    "ran-007: cell not radiating": ["ran-007", "cell not radiating", "no radiation"],
    "ran-008: dormant cell detected": ["ran-008", "dormant cell"],
    "ran-009: tx array fault": ["ran-009", "tx array fault", "antenna fault"],
    "ran-011: remote radio unit alarm": ["ran-011", "remote radio unit", "rru alarm", "ru alarm"],
    "ran-013: site communication failure": ["ran-013", "site communication failure"],
    "ran-014: csr unreachable": ["ran-014", "csr unreachable"],
    "ran-015: fronthaul link down": ["ran-015", "fronthaul link down", "fronthaul down"],
    "ran-018: link flapping detected": ["ran-018", "link flapping"],
    "ran-019: ptp synchronization failure": ["ran-019", "ptp synchronization", "ptp failure"],
    # Power codes
    "pwr-001: ac power failure": ["pwr-001", "ac power failure", "power failure"],
    "pwr-002: dc rectifier failure": ["pwr-002", "dc rectifier failure", "rectifier failure"],
    "pwr-003: battery discharge alert": ["pwr-003", "battery discharge"],
    # Compute codes
    "cmp-002: pod container creating": ["cmp-002", "pod container creating", "containercreating"],
    "cmp-003: pod pending or evicted": ["cmp-003", "pod pending", "pod evicted"],
    "cmp-004: pod crashloopbackoff": ["cmp-004", "crashloopbackoff", "crash loop"],
    "cmp-005: pod terminating stuck": ["cmp-005", "pod terminating", "terminating stuck"],
    "cmp-008: du function pod restart": ["cmp-008", "du function pod", "du pod restart"],
    "cmp-010: site not scrolling": ["cmp-010", "site not scrolling"],
    # Environment codes
    "env-001: high temperature alert": ["env-001", "high temperature"],
    "env-002: hvac system fault": ["env-002", "hvac system fault", "hvac fault"],
    "env-005: cabinet intrusion detected": ["env-005", "cabinet intrusion"],
    "env-006: battery high temperature": ["env-006", "battery high temperature"],
    # Signaling codes
    "sig-001: n2 interface down": ["sig-001", "n2 interface down"],
    "sig-003: sctp association failure": ["sig-003", "sctp association", "sctp failure"],
    "sig-009: e2 interface errors": ["sig-009", "e2 interface"],
    "sig-010: cu communication failure": ["sig-010", "cu communication failure"],
    # Service codes
    "svc-002: data throughput degradation": ["svc-002", "data throughput degradation", "throughput degradation"],
    "svc-003: call drop rate elevated": ["svc-003", "call drop rate"],
    "svc-005: service accessibility degraded": ["svc-005", "service accessibility"],
    # Transport codes
    "trn-004: fiber path degradation": ["trn-004", "fiber path degradation"],
    "trn-007: packet loss threshold exceeded": ["trn-007", "packet loss threshold"],
    "trn-008: latency sla violation": ["trn-008", "latency sla violation"],
    # Free-form codes
    "access instability": ["access instability"],
    "bgp issue": ["bgp issue", "bgp"],
    "node not functional": ["node not functional"],
    "problematic vm": ["problematic vm"],
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
