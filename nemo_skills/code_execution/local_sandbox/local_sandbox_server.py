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


import io
import logging
import multiprocessing as mp
import os
import re
import resource
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

import psutil
from flask import Flask, request
from IPython.terminal.interactiveshell import TerminalInteractiveShell

app = Flask(__name__)

# Identify worker and configure logging so messages are visible per-worker
worker_id = os.getenv("WORKER_NUM", "unknown")
logging.basicConfig(
    level=logging.INFO,
    format=f"[worker {worker_id}] %(asctime)s %(levelname)s: %(message)s",
)

# =============================================================================
# Network Blocking Configuration (Defense in Depth)
# =============================================================================
# When NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1, we use TWO layers of protection
# to prevent LLM-generated code from making network requests:
#
# LAYER 1: libblock_network.so (via /etc/ld.so.preload, set by start-with-nginx.sh)
#   - Intercepts socket() syscalls at the C library level
#   - Blocks any NEW PROCESS that exec()'s (curl, wget, python3 subprocess, etc.)
#   - System-enforced by the dynamic linker - user code CANNOT bypass it
#   - Limitation: Doesn't affect forked processes (no exec = no linker = no preload)
#
# LAYER 2: Python socket patch (below, in shell_worker function)
#   - Patches socket.socket and _socket.socket at Python level
#   - Blocks code running in the IPython shell_worker (which is forked, not exec'd)
#   - Limitation: Only works for Python code in the same process
#
# WHY BOTH ARE NEEDED:
#   | Attack Vector                              | Python Patch | ld.so.preload |
#   |--------------------------------------------|--------------|---------------|
#   | socket.socket() in IPython                 |  Blocked     |    (forked)   |
#   | import _socket; _socket.socket() in IPython|  Blocked     |    (forked)   |
#   | subprocess.run(["curl", url])              |  Can't help  |    Blocked    |
#   | subprocess.run(["python3",...], env={})    |  Can't help  |    Blocked    |
#   | requests.get() in IPython                  |  Blocked     |    (forked)   |
#
# Neither layer alone is sufficient - together they cover all adversarial scenarios.
# =============================================================================
BLOCK_NETWORK = os.getenv("NEMO_SKILLS_SANDBOX_BLOCK_NETWORK", "0") == "1"
if BLOCK_NETWORK:
    logging.info("Network blocking mode enabled")


# Worker that runs inside the shell process and owns a TerminalInteractiveShell()
def shell_worker(conn):
    # LAYER 2: Python-level socket blocking for IPython sessions
    # The shell_worker is forked (not exec'd) from the uWSGI worker, so it does NOT
    # get the ld.so.preload library loaded. We must patch Python's socket module directly.
    # This blocks: socket.socket(), _socket.socket(), requests.get(), urllib, etc.
    if BLOCK_NETWORK:
        import _socket as _socket_module  # Low-level C extension
        import socket as socket_module  # High-level module

        class BlockedSocket(_socket_module.socket):
            def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
                # Allow Unix domain sockets (needed for IPC)
                # Check both AF_UNIX and AF_LOCAL for consistency with C implementation
                af_local = getattr(_socket_module, "AF_LOCAL", _socket_module.AF_UNIX)
                if family in (_socket_module.AF_UNIX, af_local):
                    super().__init__(family, type, proto, fileno)
                # Block IPv4/IPv6
                elif family in (_socket_module.AF_INET, _socket_module.AF_INET6):
                    raise OSError(101, "Network is unreachable (blocked by sandbox)")
                else:
                    super().__init__(family, type, proto, fileno)

        # Patch BOTH modules to prevent bypass attempts
        _socket_module.socket = BlockedSocket  # Blocks: import _socket; _socket.socket()
        socket_module.socket = BlockedSocket  # Blocks: import socket; socket.socket()

    shell = TerminalInteractiveShell()
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            if not isinstance(msg, dict):
                continue
            cmd = msg.get("cmd")
            if cmd == "exec":
                code = msg.get("code", "")
                exec_id = msg.get("id")
                traceback_verbosity = msg.get("traceback_verbosity", "Plain")

                # Set traceback verbosity for this execution
                shell.InteractiveTB.set_mode(mode=traceback_verbosity)

                stdout_buf = io.StringIO()
                stderr_buf = io.StringIO()
                try:
                    # Capture stdout/stderr so we can send back outputs even if the caller times out
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        res = shell.run_cell(code)
                    conn.send(
                        {
                            "status": "ok",
                            "id": exec_id,
                            "result_repr": repr(getattr(res, "result", None)),
                            "stdout": stdout_buf.getvalue(),
                            "stderr": stderr_buf.getvalue(),
                            "has_error": res.error_before_exec or res.error_in_exec,
                        }
                    )
                except KeyboardInterrupt:
                    conn.send(
                        {
                            "status": "interrupted",
                            "id": exec_id,
                            "stdout": stdout_buf.getvalue(),
                            "stderr": stderr_buf.getvalue(),
                        }
                    )
                except Exception:
                    conn.send(
                        {
                            "status": "error",
                            "id": exec_id,
                            "traceback": traceback.format_exc(),
                            "stdout": stdout_buf.getvalue(),
                            "stderr": stderr_buf.getvalue(),
                        }
                    )
            elif cmd == "shutdown":
                break
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ShellManager:
    def __init__(self):
        """
        Manages IPython shell processes with proper timeout and cancellation support.
        """
        self.shells = {}  # shell_id -> dict(proc, conn, lock, created, last_used, restart_pending)
        self.manager_lock = threading.Lock()  # Protects shells dict

    def start_shell(self, shell_id):
        parent_conn, child_conn = mp.Pipe(duplex=True)
        proc = mp.Process(target=shell_worker, args=(child_conn,), daemon=True)
        proc.start()
        current_time = time.time()
        with self.manager_lock:
            self.shells[shell_id] = {
                "proc": proc,
                "conn": parent_conn,
                "lock": threading.Lock(),
                "created": current_time,
                "last_used": current_time,
                "restart_pending": False,  # Flag to indicate this shell was restarted and needs new_session_created=True
            }

    def stop_shell(self, shell_id):
        with self.manager_lock:
            entry = self.shells.pop(shell_id, None)

        if not entry:
            return
        proc, conn = entry["proc"], entry["conn"]
        try:
            conn.send({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        proc.terminate()
        proc.join(timeout=2.0)

    def run_cell(self, shell_id, code, timeout=1.0, grace=0.5, traceback_verbosity="Plain"):
        """
        Execute `code` on shell `shell_id`.
        - timeout: seconds to wait for normal completion
        - grace: seconds to wait after sending SIGINT
        - traceback_verbosity: IPython traceback verbosity mode
        Returns a dict with status and outputs. If it times out and we kill the shell,
        status will be 'timeout_killed' and the shell will be restarted (memory dropped).
        """
        current_time = time.time()
        with self.manager_lock:
            entry = self.shells.get(shell_id)
            shell_was_created = False

            if entry is None:
                shell_was_created = True
            else:
                # Update last_used timestamp
                entry["last_used"] = current_time

        # Check if this shell has a restart pending and clear the flag
        shell_was_recently_restarted = False
        if not shell_was_created and shell_id in self.shells:
            with self.manager_lock:
                if self.shells[shell_id].get("restart_pending", False):
                    shell_was_recently_restarted = True
                    self.shells[shell_id]["restart_pending"] = False

        # Create shell outside the lock to avoid blocking other operations
        if shell_was_created:
            self.start_shell(shell_id)
            with self.manager_lock:
                entry = self.shells[shell_id]

        proc = entry["proc"]
        conn = entry["conn"]
        lock = entry["lock"]

        exec_id = time.time_ns()
        with lock:
            # send execution request
            try:
                conn.send({"cmd": "exec", "id": exec_id, "code": code, "traceback_verbosity": traceback_verbosity})
            except Exception as exc:
                return {
                    "status": "error",
                    "msg": f"send failed: {exc}",
                    "shell_was_created": shell_was_created,
                    "shell_was_recently_restarted": shell_was_recently_restarted,
                }

            # wait for the result up to `timeout`
            if conn.poll(timeout):
                try:
                    result = conn.recv()
                    result["shell_was_created"] = shell_was_created
                    result["shell_was_recently_restarted"] = shell_was_recently_restarted
                    return result
                except EOFError:
                    # Connection closed - shell process died, need to restart
                    logging.warning(f"Shell process for {shell_id} died during execution, restarting")
                    with self.manager_lock:
                        self.shells.pop(shell_id, None)
                    self.start_shell(shell_id)

                    # Mark the new shell as having a restart pending
                    with self.manager_lock:
                        if shell_id in self.shells:
                            self.shells[shell_id]["restart_pending"] = True

                    return {
                        "status": "error",
                        "msg": "connection closed",
                        "shell_was_created": shell_was_created,
                        "shell_was_restarted": True,
                        "shell_was_recently_restarted": shell_was_recently_restarted,
                    }

            # no reply yet -> try gentle interrupt (SIGINT)
            try:
                # Process.send_signal exists on Unix; fallback to os.kill if necessary
                try:
                    proc.send_signal(signal.SIGINT)
                except AttributeError:
                    os.kill(proc.pid, signal.SIGINT)
            except Exception:
                # If we couldn't send SIGINT, fall through to termination after grace
                pass

            # wait short grace period for the shell to handle the interrupt
            if conn.poll(grace):
                try:
                    result = conn.recv()
                    result["shell_was_created"] = shell_was_created
                    result["shell_was_recently_restarted"] = shell_was_recently_restarted
                    return result
                except EOFError:
                    # Connection closed - shell process died, need to restart
                    logging.warning(f"Shell process for {shell_id} died during interrupt, restarting")
                    with self.manager_lock:
                        self.shells.pop(shell_id, None)
                    self.start_shell(shell_id)

                    # Mark the new shell as having a restart pending
                    with self.manager_lock:
                        if shell_id in self.shells:
                            self.shells[shell_id]["restart_pending"] = True

                    return {
                        "status": "interrupted",
                        "msg": "connection closed after interrupt",
                        "shell_was_created": shell_was_created,
                        "shell_was_restarted": True,
                        "shell_was_recently_restarted": shell_was_recently_restarted,
                    }

            # still stuck -> terminate the shell and restart it (drop memory)
            try:
                proc.terminate()
            except Exception:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            proc.join(timeout=2.0)

            # close old connection (best-effort)
            try:
                conn.close()
            except Exception:
                pass

            # remove and restart a fresh shell for this id
            with self.manager_lock:
                self.shells.pop(shell_id, None)
            self.start_shell(shell_id)

            # Mark the new shell as having a restart pending
            with self.manager_lock:
                if shell_id in self.shells:
                    self.shells[shell_id]["restart_pending"] = True

            return {
                "status": "timeout_killed",
                "id": exec_id,
                "shell_was_restarted": True,
                "shell_was_recently_restarted": shell_was_recently_restarted,
            }


def log_session_count(prefix: str = "") -> None:
    try:
        with shell_manager.manager_lock:
            session_count = len(shell_manager.shells)
        logging.info("%sactive_sessions=%d", prefix, session_count)
    except Exception:
        pass


# Global shell manager for IPython sessions
shell_manager = ShellManager()
SESSION_TIMEOUT = float(os.getenv("NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT", 0))  # disabled by default


def cleanup_expired_sessions():
    """Remove IPython sessions that haven't been used recently"""
    if SESSION_TIMEOUT <= 0:
        return  # Session expiration disabled

    current_time = time.time()
    expired_sessions = []

    with shell_manager.manager_lock:
        for session_id, session_data in shell_manager.shells.items():
            if current_time - session_data["last_used"] > SESSION_TIMEOUT:
                expired_sessions.append(session_id)

    for session_id in expired_sessions:
        try:
            shell_manager.stop_shell(session_id)
            logging.info(f"Cleaned up expired session: {session_id}")
        except Exception as e:
            logging.warning(f"Error cleaning up session {session_id}: {e}")


def postprocess_output(output, traceback_verbosity):
    if traceback_verbosity not in ["Minimal", "Plain"]:
        return output

    def strip_ansi_codes(text):
        ansi_escape = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    output = strip_ansi_codes(output)
    lines = []
    for line in output.split("\n"):
        if line.strip().startswith("File <ipython-"):
            continue
        line = re.sub(r"^Out\[\d+\]:\s*", "", line)
        lines.append(line)

    # Remove leading blank lines introduced by displayhook
    while lines and lines[0] == "":
        lines.pop(0)

    output = "\n".join(lines).rstrip()
    return output + ("\n" if output else "")


def cleanup_session(session_id):
    """Clean up and remove a specific session"""
    shell_manager.stop_shell(session_id)
    logging.info(f"Cleaned up session: {session_id}")


def execute_ipython_session(generated_code, session_id, timeout=30, traceback_verbosity="Plain"):
    """Execute Python code in a persistent IPython session with proper timeout handling"""
    try:
        # Clean up expired sessions periodically
        if SESSION_TIMEOUT > 0:
            cleanup_expired_sessions()

        # Execute code using ShellManager with proper timeout and cancellation
        result = shell_manager.run_cell(
            session_id, generated_code, timeout=timeout, grace=0.5, traceback_verbosity=traceback_verbosity
        )

        # Determine if this is effectively a new session
        new_session_created = (
            result.get("shell_was_created", False)  # Shell was created for this request
            or result.get("shell_was_restarted", False)  # Shell was restarted during this request
            or result.get("shell_was_recently_restarted", False)  # Shell was restarted in a previous request
        )

        # Map ShellManager results to expected format
        if result["status"] == "ok":
            process_status = "error" if result.get("has_error", False) else "completed"
            return {
                "process_status": process_status,
                "stdout": postprocess_output(result.get("stdout", ""), traceback_verbosity),
                "stderr": postprocess_output(result.get("stderr", ""), traceback_verbosity),
                "new_session_created": new_session_created,
            }
        elif result["status"] == "timeout_killed":
            logging.warning(f"IPython session {session_id} timed out after {timeout}s, shell was restarted")
            return {
                "process_status": "timeout",
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds\n",
                "new_session_created": True,  # Shell was restarted, so it's effectively new
            }
        elif result["status"] == "interrupted":
            # For timeout scenarios, treat interruption as timeout since it happened due to timeout
            # Check if shell was restarted during interruption
            if result.get("shell_was_restarted", False):
                new_session_created = True
                logging.warning(f"IPython session {session_id} was interrupted and restarted after {timeout}s timeout")
            else:
                logging.warning(f"IPython session {session_id} was interrupted after {timeout}s timeout")
            return {
                "process_status": "timeout",
                "stdout": postprocess_output(result.get("stdout", ""), traceback_verbosity),
                "stderr": postprocess_output(
                    result.get("stderr", "") + f"Execution timed out after {timeout} seconds\n", traceback_verbosity
                ),
                "new_session_created": new_session_created,
            }
        else:  # error status
            # Check if shell was restarted during error
            if result.get("shell_was_restarted", False):
                new_session_created = True
                logging.warning(f"IPython session {session_id} had an error and was restarted")
            error_msg = result.get("traceback", result.get("msg", "Unknown error"))
            return {
                "process_status": "error",
                "stdout": postprocess_output(result.get("stdout", ""), traceback_verbosity),
                "stderr": postprocess_output(result.get("stderr", "") + f"\n{error_msg}\n", traceback_verbosity),
                "new_session_created": new_session_created,
            }

    except Exception as e:
        logging.error(f"Error in execute_ipython_session for session {session_id}: {e}")
        return {
            "process_status": "error",
            "stdout": "",
            "stderr": f"Session error: {e}\n",
            "new_session_created": False,
        }


# Log per-request session count after each response
@app.after_request
def _after_log_session_count(response):
    log_session_count()
    return response


MEM_LIMIT_BYTES = int(os.environ.get("NEMO_SKILLS_SANDBOX_MEM_LIMIT", 50 * 1024**3))  # 50 GiB default

# Set per-worker memory limit for ipython session
resource.setrlimit(resource.RLIMIT_AS, (2 * MEM_LIMIT_BYTES, 2 * MEM_LIMIT_BYTES))
resource.setrlimit(resource.RLIMIT_DATA, (2 * MEM_LIMIT_BYTES, 2 * MEM_LIMIT_BYTES))
logging.info("Applied worker memory limit (RLIMIT_AS/RLIMIT_DATA): %d bytes", 2 * MEM_LIMIT_BYTES)


# Code to kill the process tree for lean4 code execution
def kill_process_tree(proc):
    """
    Safely and aggressively kills a process and all its descendants.
    This is the recommended approach for ensuring cleanup.
    """
    try:
        parent = psutil.Process(proc.pid)
        # Get all children of the process, recursively.
        children = parent.children(recursive=True)
        # Add the parent to the list of processes to be killed.
        all_processes = children + [parent]

        # Kill all processes in the tree.
        for p in all_processes:
            try:
                # SIGKILL is a forceful, non-ignorable kill signal.
                p.kill()
            except psutil.NoSuchProcess:
                # The process might have already died, which is fine.
                pass

        # Wait for all processes to be terminated.
        gone, alive = psutil.wait_procs(all_processes, timeout=3)
        if alive:
            # If any processes are still alive, they are likely zombies
            # or in an unkillable state. This is a last resort.
            for p in alive:
                logging.warning("Process %s could not be killed.", p.pid)
    except psutil.NoSuchProcess:
        # The main process already died before we could kill it.
        pass
    except Exception as e:
        logging.error("Error in kill_process_tree: %s", e)


def set_limits(mem_bytes: int = MEM_LIMIT_BYTES) -> None:
    """
    Apply RLIMITs and start a new session for the child process.

    Called via `preexec_fn` (subprocess) or directly in a forked worker.
    """
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
    os.setsid()  # isolate PGID / signals


def execute_python(generated_code, std_input, timeout, language):
    execution_command = [language, "-c", generated_code]
    try:
        process = subprocess.Popen(
            execution_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            preexec_fn=set_limits,
        )
        stdout, stderr = process.communicate(input=std_input, timeout=timeout)
        return {"process_status": "completed", "stdout": stdout, "stderr": stderr}
    except subprocess.TimeoutExpired:
        try:
            # kill the whole process group
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)  # reap, no extra timeout needed
        return {"process_status": "timeout", "stdout": "", "stderr": f"Execution timed out after {timeout} seconds\n"}


def execute_lean4(generated_code, timeout):
    temp_file_name = None
    proc = None  # <-- Keep track of the process object
    try:
        project_path = "/lean4/my_project"
        # Use a with statement for the temp file to ensure it's closed
        with tempfile.NamedTemporaryFile(dir=project_path, delete=False, suffix=".lean") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(generated_code.encode("utf-8"))
            temp_file.flush()  # Ensure data is written to disk

        # Use subprocess.Popen for more control
        proc = subprocess.Popen(
            ["lake", "env", "--dir", project_path, "lean", temp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_path,
            preexec_fn=os.setsid,
        )

        # Communicate with the process, which waits for it to finish
        # This will raise TimeoutExpired if the timeout is reached
        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode == 0:
            process_status = "completed"
        else:
            process_status = "failed"

        return {
            "process_status": process_status,
            "stdout": stdout.decode("utf-8"),
            "stderr": stderr.decode("utf-8"),
        }

    except subprocess.TimeoutExpired:
        # kill the process tree
        kill_process_tree(proc)
        # Now we can safely get any output that was generated before the kill.
        stdout, stderr = proc.communicate()

        final_stderr = stderr.decode("utf-8") + f"Execution timed out after {timeout} seconds\n"
        return {
            "process_status": "timeout",
            "stdout": stdout.decode("utf-8"),
            "stderr": final_stderr,
        }

    except Exception as e:
        logging.error("Error executing Lean4 code: %s", e)
        return {"process_status": "error", "stdout": "", "stderr": str(e) + "\n"}
    finally:
        # Safely remove the temporary file if it was created
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


def execute_shell(command, timeout):
    tmp_path = None
    try:
        # Write the full script to a temp file so /bin/bash can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sh", mode="w") as tmp:
            tmp.write(command)
            tmp_path = tmp.name
        os.chmod(tmp_path, 0o755)

        result = subprocess.run(
            ["/bin/bash", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            preexec_fn=set_limits,
        )
        process_status = "completed" if result.returncode == 0 else "error"
        return {"process_status": process_status, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"process_status": "timeout", "stdout": "", "stderr": f"Execution timed out after {timeout} seconds\n"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    generated_code = request.json["generated_code"]
    timeout = request.json["timeout"]
    language = request.json.get("language", "ipython")
    std_input = request.json.get("std_input", "")
    max_output_characters = request.json.get("max_output_characters", 1000)
    traceback_verbosity = request.json.get("traceback_verbosity", "Plain")

    session_id = request.headers.get("X-Session-ID")

    if language == "ipython":
        if session_id is None:
            return {"error": "X-Session-ID header required for ipython sessions"}, 400
        result = execute_ipython_session(generated_code, session_id, timeout, traceback_verbosity)
    elif language == "lean4":
        result = execute_lean4(generated_code, timeout)
    elif language == "shell":
        result = execute_shell(generated_code, timeout)
    else:
        result = execute_python(generated_code, std_input, timeout, language)

    if len(result["stdout"]) > max_output_characters:
        result["stdout"] = result["stdout"][:max_output_characters] + "<output cut>"
    if len(result["stderr"]) > max_output_characters:
        result["stderr"] = result["stderr"][:max_output_characters] + "<output cut>"

    return result


# Session management endpoints
@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active IPython sessions"""
    try:
        session_info = {}

        # Get sessions from ShellManager with proper locking
        with shell_manager.manager_lock:
            for session_id, session_data in shell_manager.shells.items():
                session_info[session_id] = {
                    "backend": "ipython",
                    "created": session_data["created"],
                    "last_used": session_data["last_used"],
                    "alive": True,  # All shells in manager are alive
                }
        return {"sessions": session_info, "backend": "ipython"}
    except Exception as e:
        import traceback

        error_msg = f"Error in list_sessions: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return {"error": error_msg}, 500


@app.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a specific IPython session"""
    try:
        with shell_manager.manager_lock:
            session_exists = session_id in shell_manager.shells

        if session_exists:
            shell_manager.stop_shell(session_id)
            return {"message": f"IPython session {session_id} deleted successfully"}
        else:
            return {"error": f"IPython session {session_id} not found"}, 404
    except Exception as e:
        return {"error": f"Error deleting IPython session {session_id}: {e}"}, 500


@app.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "worker": os.environ.get("WORKER_NUM", "unknown")}


if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)
    app.run(port=6000)
