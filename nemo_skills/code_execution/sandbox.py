# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import abc
import asyncio
import glob
import json
import logging
import os
import re
import traceback
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import httpx
import tqdm

from nemo_skills.code_execution.utils import clean_formal_generation
from nemo_skills.dataset.utils import get_lean4_header
from nemo_skills.utils import get_logger_name, python_doc_to_cmd_help

LOG = logging.getLogger(get_logger_name(__file__))


def unroll_files(input_files):
    for manifest_pattern in input_files:
        for manifest in sorted(glob.glob(manifest_pattern, recursive=True)):
            yield manifest


def extract_proof_only(lean_code: str) -> str:
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_start_pattern = re.compile(r"^\s*(theorem|example)\b")
    header_start_idx = None

    # 1. Find where the theorem starts
    for i, line in enumerate(lines):
        if header_start_pattern.match(line):
            header_start_idx = i
            break

    if header_start_idx is None:
        return lean_code.strip()

    # 2. Find where ':=' occurs, starting from the header
    header_end_idx = None
    for i in range(header_start_idx, len(lines)):
        if ":=" in lines[i]:
            header_end_idx = i
            break

    if header_end_idx is None:
        return lean_code.strip()

    # 3. Extract the line after ':='
    header_line, after = lines[header_end_idx].split(":=", 1)
    proof_first_line = after.strip()

    # 4. Collect proof lines
    if proof_first_line:
        proof_lines = [proof_first_line] + lines[header_end_idx + 1 :]
    else:
        proof_lines = lines[header_end_idx + 1 :]

    # 5. Remove leading 'by' (with or without indentation)
    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]  # Strip 'by '

    return "\n".join(proof_lines).rstrip()


class Sandbox(abc.ABC):
    """Code execution sandbox.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_HOST env var.
        port: Optional[str] = '5000' - Port of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_PORT env var.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access.
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        # Create async HTTP client with high limits
        self.http_session = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=2048, max_connections=2048),
        )
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)
        self.session_histories = defaultdict(list)  # session_id -> list of generated_code

    async def close(self):
        """Close the HTTP session."""
        await self.http_session.aclose()

    async def _send_request(self, request, timeout):
        session_id = request.pop("session_id", None)
        extra_headers = {}
        if session_id is not None:
            extra_headers["X-Session-ID"] = str(session_id)

        if self.ssh_server and self.ssh_key_path:
            # For SSH tunneling, use threads since there's no async version
            import sshtunnel_requests

            def ssh_request():
                sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
                return sshtunnel_request.post(
                    url=self._get_execute_url(),
                    data=json.dumps(request),
                    timeout=timeout,
                    headers={"Content-Type": "application/json", **extra_headers},
                )

            # Native async requires more lines of code, so we use to_thread
            # Should be ok since this is a debug mode
            output = await asyncio.to_thread(ssh_request)
        else:
            output = await self.http_session.post(
                url=self._get_execute_url(),
                content=json.dumps(request),
                timeout=timeout,
                headers={"Content-Type": "application/json", **extra_headers},
            )
        # retrying 502 errors
        if output.status_code == 502:
            raise httpx.TimeoutException("502 error")
        return self._parse_request_output(output)

    @abc.abstractmethod
    def _parse_request_output(self, output):
        pass

    @abc.abstractmethod
    def _get_execute_url(self):
        pass

    @abc.abstractmethod
    def _prepare_request(
        self,
        generated_code,
        timeout,
        language="ipython",
        std_input="",
        max_output_characters=1000,
        traceback_verbosity="Plain",
    ):
        pass

    @abc.abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a remote execution session if supported by the backend."""
        pass

    async def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        language: str = "ipython",
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
        traceback_verbosity="plain",  # could be plain, context, verbose, or minimal
    ) -> Tuple[Dict, str]:
        traceback_verbosity = traceback_verbosity.capitalize()
        if language in ["python", "pypy3", "python3", "lean4", "shell"] and session_id is not None:
            raise RuntimeError(
                f"Stateful execution for {language} is not supported. session_id is {session_id} but should be None"
            )
        if language not in ["ipython", "python", "pypy3", "python3", "lean4", "shell"]:
            raise ValueError(f"Unsupported language: {language}")
        if language != "ipython" and traceback_verbosity != "Plain":
            raise ValueError("Configurable traceback_verbosity is only supported for ipython")

        request_session_id = session_id
        if request_session_id is None and language == "ipython":  # creating a new session with empty state
            request_session_id = uuid.uuid4()

        TO_EXECUTE = generated_code
        request = self._prepare_request(
            TO_EXECUTE, timeout, language, std_input, max_output_characters, traceback_verbosity
        )
        request["session_id"] = request_session_id if request_session_id is None else str(request_session_id)
        try:
            output = await self._send_request(request, timeout)
        except httpx.TimeoutException:
            output = {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}
        new_session_created = output.pop("new_session_created", False)

        # Rebuild state by executing concatenated history
        if session_id is not None and new_session_created:
            # TODO: this history restoration is not ideal, we are returning output that is not the actual output of the new code
            # and the execution of the concatenation may not be the same as the execution of the respective cells
            # see details in this thread: https://github.com/NVIDIA/NeMo-Skills/pull/803#discussion_r2338240505
            history = self.session_histories.get(session_id, [])
            combined_code = "\n".join(history) + ("\n" if history else "") + generated_code
            request = self._prepare_request(
                combined_code, timeout, language, std_input, max_output_characters, traceback_verbosity
            )
            request["session_id"] = request_session_id if request_session_id is None else str(request_session_id)
            try:
                output = await self._send_request(request, timeout)
            except httpx.TimeoutException:
                output = {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}

        # Append to history if successful execution (process_status == 'completed')
        if output.get("process_status") == "completed":
            self.session_histories[request_session_id].append(generated_code)

        output.pop("new_session_created", None)

        return output, request_session_id

    async def is_proof_correct(self, pred_output, timeout=30.0):
        TO_EXECUTE = pred_output

        request = self._prepare_request(TO_EXECUTE, timeout, "lean4")
        try:
            output = await self._send_request(request, timeout)
        except httpx.TimeoutException:
            return "timeout"
        if output["process_status"] == "completed":
            stdout = output["stdout"].lower()
            stderr = output["stderr"].lower()
            combined = stdout + "\n" + stderr
            if re.search(r"\bsorry\b", combined) is not None:
                return "has_sorry"
            return "completed"
        return output["process_status"]

    async def batch_evaluate_results(
        self,
        input_files: List[str],
        num_parallel_requests=10,
        timeout=30.0,
        answer_format="lean4-proof",
        use_predicted_proof_key: bool = False,
        final_answer_key: str = "**FINAL ANSWER**",
        restate_formal_statement: bool = True,
        strip_theorem_from_proof: bool = True,
        extract_code_mode: str = "last",
    ):
        """Evaluate results and write back to original files."""

        semaphore = asyncio.Semaphore(num_parallel_requests)

        async def process_line(line_data):
            """Process a single line and return updated line data."""
            if not line_data or not line_data.strip():
                return line_data

            line_dict = json.loads(line_data)
            if not line_dict:
                return line_data

            # Prepare predicted_proof based on format
            if answer_format == "lean4-proof":
                if not use_predicted_proof_key:
                    generation = clean_formal_generation(
                        line_dict["generation"], final_answer_key=final_answer_key, extract_code_mode=extract_code_mode
                    )
                    line_dict["predicted_proof"] = (
                        line_dict["header"]
                        + (line_dict["formal_statement"] if restate_formal_statement else "")
                        + extract_proof_only(generation)
                        if strip_theorem_from_proof
                        else generation
                    )
                else:
                    if "predicted_proof" not in line_dict:
                        raise ValueError(
                            "predicted_proof key not found in the line_dict. "
                            "Set use_predicted_proof_key=False to re-combine"
                        )
            elif answer_format == "lean4-statement":
                if not use_predicted_proof_key:
                    generation = clean_formal_generation(line_dict["generation"], extract_code_mode=extract_code_mode)
                    header = get_lean4_header()
                    line_dict["predicted_proof"] = header + generation + "\n sorry"
                else:
                    if "predicted_proof" not in line_dict:
                        raise ValueError(
                            "predicted_proof key not found in the line_dict. "
                            "Set use_predicted_proof_key=False to re-combine"
                        )
            else:
                raise ValueError(f"Unknown answer_format: {answer_format}")

            # Evaluate proof with concurrency control
            async with semaphore:
                proof_status = await self.is_proof_correct(line_dict["predicted_proof"], timeout=timeout)
                line_dict["proof_status"] = proof_status

            return json.dumps(line_dict)

        # Process each file
        for input_file in unroll_files(input_files):
            # Read all lines
            with open(input_file, "rt", encoding="utf-8") as f:
                lines = f.readlines()

            # Process lines concurrently with progress bar
            print(f"Processing {input_file}...")
            processed_lines = []
            for line in tqdm.tqdm(lines):
                result = await process_line(line.rstrip("\n"))
                processed_lines.append(result)

            # Write to temp file then replace original
            temp_file = input_file + "-tmp"
            with open(temp_file, "wt", encoding="utf-8") as f:
                for line in processed_lines:
                    f.write(line + "\n")

            # Replace original with temp file
            os.replace(temp_file, input_file)


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def _get_execute_url(self):
        return f"http://{self.host}:{self.port}/execute"

    def _parse_request_output(self, output):
        try:
            return output.json()
        except json.JSONDecodeError:
            LOG.error("Error during parsing output: %s", output.text)
            return {"process_status": "error", "stdout": "", "stderr": "Unknown error"}

    def _prepare_request(
        self,
        generated_code,
        timeout,
        language="ipython",
        std_input="",
        max_output_characters=1000,
        traceback_verbosity="Plain",
    ):
        return {
            "generated_code": generated_code,
            "std_input": std_input,
            "timeout": timeout,
            "language": language,
            "max_output_characters": max_output_characters,
            "traceback_verbosity": traceback_verbosity,
        }

    async def delete_session(self, session_id: str) -> None:
        """Delete an IPython session on the local sandbox server."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = await self.http_session.delete(
                    url=f"http://{self.host}:{self.port}/sessions/{session_id}",
                    timeout=10.0,
                    headers={"X-Session-ID": session_id},
                )
                if response.status_code == 200:  # Success
                    if session_id in self.session_histories:
                        del self.session_histories[session_id]
                    return
                if response.status_code == 404:  # We were routed to a different worker
                    LOG.warning(f"Session {session_id} not found (already deleted?). Treating as success.")
                    if session_id in self.session_histories:
                        del self.session_histories[session_id]
                    return
                response.raise_for_status()
            except (
                httpx.ReadTimeout,  # retry for other communication errors and statuses
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as e:
                LOG.warning("Retry %d/%d deleting session %s – %s", attempt + 1, max_retries, session_id, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    LOG.warning(f"Failed to delete session {session_id} after {max_retries} attempts. ")
            except Exception as e:
                LOG.warning(
                    "Failed to delete session %s: %s (type: %s, repr: %r)\nTraceback:\n%s",
                    session_id,
                    e,
                    type(e).__name__,
                    e,
                    traceback.format_exc(),
                )
                raise  # Re-raise unexpected exceptions


sandboxes = {
    "local": LocalSandbox,
}


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)


def sandbox_params():
    """Returns sandbox documentation (to include in cmd help)."""
    prefix = f"\n        sandbox_type: str = MISSING - Choices: {list(sandboxes.keys())}"
    return python_doc_to_cmd_help(Sandbox, docs_prefix=prefix, arg_prefix="sandbox.")
