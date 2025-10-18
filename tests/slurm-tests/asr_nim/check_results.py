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


def normalize_text(text):
    """Normalize text by lowercasing and removing punctuation.

    Args:
        text: Input text string

    Returns:
        str: Normalized text with only lowercase letters and spaces
    """
    # Lowercase
    text = text.lower()
    # Replace hyphens with spaces (so "text-to-speech" becomes "text to speech")
    text = text.replace("-", " ")
    # Remove all punctuation, keep only letters, numbers, and spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


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
                        "ASR NIM Server started",
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


def load_references(workspace: str):
    """Load reference transcripts from the original test file.

    The test file is part of the packaged code and available at:
    /nemo_run/code/tests/slurm-tests/asr_nim/asr.test

    Args:
        workspace: Workspace directory path (not used, kept for compatibility)

    Returns:
        dict: Mapping from audio_path to reference transcript
    """
    # The original test file is packaged with the code
    test_file = Path("/nemo_run/code/tests/slurm-tests/asr_nim/asr.test")

    references = {}
    if test_file.exists():
        try:
            with open(test_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "_reference" in data and "audio_path" in data:
                            # Use the audio filename as key (since paths may differ)
                            audio_filename = Path(data["audio_path"]).name
                            references[audio_filename] = data["_reference"]
        except Exception as e:
            print(f"Warning: Could not load references from {test_file}: {e}")

    return references


def check_generation_results(workspace: str):
    """Check that transcripts were generated and validate against references."""
    output_dir = Path(workspace) / "asr_outputs"

    # Check output directory exists
    soft_assert(output_dir.exists(), f"Output directory not found: {output_dir}")

    # Check for generated outputs JSONL
    output_files = list(output_dir.glob("output*.jsonl")) + list(output_dir.glob("output-rs*.jsonl"))
    soft_assert(len(output_files) > 0, f"No output JSONL files found in {output_dir}")

    if not output_files:
        return

    # Load reference transcripts
    references = load_references(workspace)

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

    # Check we have expected number of outputs (2 samples)
    expected_samples = 2
    soft_assert(len(all_outputs) == expected_samples, f"Expected {expected_samples} outputs, found {len(all_outputs)}")

    # Check each output has non-empty transcript
    empty_transcripts = []
    transcript_stats = []
    word_match_results = []

    for idx, output in enumerate(all_outputs):
        sample_id = output.get("id", f"sample_{idx}")

        # Check for transcript in output (ASR outputs use 'pred_text' inside 'result')
        transcript = None
        if "result" in output and isinstance(output["result"], dict):
            transcript = output["result"].get("pred_text")

        if not transcript or not transcript.strip():
            empty_transcripts.append(sample_id)
            continue

        transcript_stats.append({"id": sample_id, "transcript": transcript, "length": len(transcript)})

        # If we have a reference, validate word matching
        audio_path = output.get("audio_path", "")
        if audio_path:
            audio_filename = Path(audio_path).name
            if audio_filename in references:
                reference = references[audio_filename]

                # Normalize both texts
                normalized_reference = normalize_text(reference)
                normalized_transcript = normalize_text(transcript)

                # Check that all reference words are in the transcript
                reference_words = set(normalized_reference.split())
                transcript_words = set(normalized_transcript.split())

                missing_words = reference_words - transcript_words

                word_match_results.append(
                    {
                        "id": sample_id,
                        "audio": audio_filename,
                        "reference": reference,
                        "transcript": transcript,
                        "normalized_reference": normalized_reference,
                        "normalized_transcript": normalized_transcript,
                        "all_words_present": len(missing_words) == 0,
                        "missing_words": list(missing_words),
                    }
                )

    soft_assert(len(empty_transcripts) == 0, f"Empty transcripts for samples: {', '.join(empty_transcripts)}")

    if not empty_transcripts:
        print(f"✓ All {len(all_outputs)} transcripts generated successfully")

        # Print transcript statistics
        if transcript_stats:
            avg_length = sum(s["length"] for s in transcript_stats) / len(transcript_stats)
            print(f"  Average transcript length: {avg_length:.1f} characters")

            for stat in transcript_stats:
                print(f"  - {stat['id']}: {stat['length']} chars")

        # Print word matching results
        if word_match_results:
            print("\n✓ Word matching validation:")
            passed_count = sum(1 for r in word_match_results if r["all_words_present"])
            print(f"  {passed_count}/{len(word_match_results)} samples passed word matching")

            for result in word_match_results:
                if result["all_words_present"]:
                    print(f"  ✓ {result['audio']}: All reference words present")
                else:
                    print(f"  ✗ {result['audio']}: Missing words: {', '.join(result['missing_words'])}")
                    print(f"    Reference (original): {result['reference']}")
                    print(f"    Reference (normalized): {result['normalized_reference']}")
                    print(f"    Transcript (original): {result['transcript']}")
                    print(f"    Transcript (normalized): {result['normalized_transcript']}")

            # Soft assert that all word matches passed
            for result in word_match_results:
                soft_assert(
                    result["all_words_present"],
                    f"Sample {result['audio']}: Missing words in transcript: {', '.join(result['missing_words'])}",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing results")
    parser.add_argument(
        "--mode", choices=["server", "generation", "full"], default="full", help="Test mode to validate"
    )
    parser.add_argument("--server_timeout", type=int, default=180, help="Maximum seconds allowed for server startup")

    args = parser.parse_args()

    print(f"\nChecking ASR NIM test results (mode: {args.mode})...")
    print(f"Workspace: {args.workspace}\n")

    # Check based on mode
    if args.mode == "server":
        check_server_startup(args.workspace, args.server_timeout)
    elif args.mode in ["generation", "full"]:
        check_generation_results(args.workspace)

    assert_all()


if __name__ == "__main__":
    main()
