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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, soft_assert  # noqa: E402


def check_server_startup(workspace: str, server_timeout: int):
    """Check that the server started successfully within timeout."""
    server_logs_dir = Path(workspace) / "server-logs"

    # Check if server logs directory exists
    soft_assert(server_logs_dir.exists(), f"Server logs directory not found: {server_logs_dir}")

    if not server_logs_dir.exists():
        return

    # Look for log files
    log_files = list(server_logs_dir.glob("*.log")) + list(server_logs_dir.glob("*.out"))
    soft_assert(len(log_files) > 0, f"No log files found in {server_logs_dir}")

    if not log_files:
        return

    # Check logs for successful startup indicators
    server_started = False
    startup_time = None

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()

                # Look for common server startup success indicators
                if any(
                    indicator in content
                    for indicator in [
                        "Server started",
                        "gRPC server",
                        "Riva",
                        "TTS NIM Server started",
                        "Application startup complete",
                    ]
                ):
                    server_started = True

                # Try to extract startup time if available
                time_match = re.search(r"startup.*?(\d+\.?\d*)\s*(?:seconds|s)", content, re.IGNORECASE)
                if time_match:
                    startup_time = float(time_match.group(1))
        except Exception as e:
            print(f"Warning: Could not read log file {log_file}: {e}")

    soft_assert(server_started, "Server startup indicators not found in logs")

    if startup_time is not None:
        soft_assert(
            startup_time <= server_timeout,
            f"Server startup took {startup_time}s, exceeds timeout of {server_timeout}s",
        )
        print(f"✓ Server started successfully in {startup_time}s")
    else:
        print("✓ Server startup indicators found in logs")


def check_generation_results(workspace: str):
    """Check that audio files were generated for all input samples."""
    output_dir = Path(workspace) / "tts_outputs"
    audio_dir = output_dir / "audio_outputs"

    # Check output directory exists
    soft_assert(output_dir.exists(), f"Output directory not found: {output_dir}")

    # Check for generated outputs JSONL
    output_files = list(output_dir.glob("output*.jsonl")) + list(output_dir.glob("output-rs*.jsonl"))
    soft_assert(len(output_files) > 0, f"No output JSONL files found in {output_dir}")

    if not output_files:
        return

    # Read all output files
    all_outputs = []
    for output_file in output_files:
        try:
            with open(output_file, "r") as f:
                for line in f:
                    if line.strip():
                        all_outputs.append(json.loads(line))
        except Exception as e:
            soft_assert(False, f"Failed to read output file {output_file}: {e}")

    # Check we have expected number of outputs (6 samples)
    expected_samples = 6
    soft_assert(len(all_outputs) == expected_samples, f"Expected {expected_samples} outputs, found {len(all_outputs)}")

    # Check each output has audio_path and verify the file exists
    missing_audio = []
    empty_audio = []

    for idx, output in enumerate(all_outputs):
        sample_id = output.get("id", f"sample_{idx}")

        # Check for audio_path in output
        if "audio_path" not in output and "result" not in output:
            missing_audio.append(sample_id)
            continue

        audio_path = output.get("audio_path") or output.get("result")

        if not audio_path:
            missing_audio.append(sample_id)
            continue

        # Check if it's an absolute path or relative
        audio_file = Path(audio_path) if Path(audio_path).is_absolute() else audio_dir / Path(audio_path).name

        # Check file exists
        if not audio_file.exists():
            missing_audio.append(f"{sample_id} (file not found: {audio_path})")
            continue

        # Check file is not empty
        file_size = audio_file.stat().st_size
        if file_size == 0:
            empty_audio.append(f"{sample_id} (file: {audio_path})")

    soft_assert(len(missing_audio) == 0, f"Missing audio files for samples: {', '.join(missing_audio)}")

    soft_assert(len(empty_audio) == 0, f"Empty audio files for samples: {', '.join(empty_audio)}")

    if not missing_audio and not empty_audio:
        print(f"✓ All {len(all_outputs)} audio files generated successfully")

        # Print some statistics
        total_size = sum(
            (
                Path(output.get("audio_path") or output.get("result"))
                if Path(output.get("audio_path") or output.get("result", "")).is_absolute()
                else audio_dir / Path(output.get("audio_path") or output.get("result", "")).name
            )
            .stat()
            .st_size
            for output in all_outputs
            if (output.get("audio_path") or output.get("result"))
        )
        avg_size = total_size / len(all_outputs) if all_outputs else 0
        print(f"  Total audio size: {total_size / 1024:.1f} KB")
        print(f"  Average file size: {avg_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing results")
    parser.add_argument(
        "--mode", choices=["server", "generation", "full"], default="full", help="Test mode to validate"
    )
    parser.add_argument("--server_timeout", type=int, default=180, help="Maximum seconds allowed for server startup")

    args = parser.parse_args()

    print(f"\nChecking TTS NIM test results (mode: {args.mode})...")
    print(f"Workspace: {args.workspace}\n")

    # Check based on mode
    if args.mode == "server":
        check_server_startup(args.workspace, args.server_timeout)
    elif args.mode in ["generation", "full"]:
        check_generation_results(args.workspace)

    assert_all()


if __name__ == "__main__":
    main()
