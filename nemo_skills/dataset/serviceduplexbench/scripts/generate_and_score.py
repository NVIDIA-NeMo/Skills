# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Generate ServiceDuplexBench responses using nemo-skills and score with LLM judge.

Usage:
    python generate_and_score.py --config config.yaml
"""

import argparse
from pathlib import Path

import yaml

from nemo_skills.pipeline.cli import eval as nemo_eval
from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_score_command(config: dict) -> str:
    """Build the scoring command to run via run_cmd."""
    eval_results_dir = f"{config['output_dir']}/eval-results/serviceduplexbench"
    scoring_script = "nemo_skills/dataset/serviceduplexbench/scripts/score_serviceduplexbench.py"

    cmd_args = [
        f"python {scoring_script}",
        f"--eval_results_dir {eval_results_dir}",
    ]

    if config.get("api_type"):
        cmd_args.append(f"--api_type {config['api_type']}")
    if config.get("nvidia_model"):
        cmd_args.append(f"--nvidia_model {config['nvidia_model']}")

    return " ".join(cmd_args)


def build_agent_audio_asr_command(config: dict) -> str:
    """Build the agent-audio ASR + WER/CER command to run via run_cmd."""
    eval_results_dir = f"{config['output_dir']}/eval-results/serviceduplexbench"
    asr_script = "nemo_skills/dataset/voicebench/scripts/run_agent_audio_asr_metrics.py"
    asr_model = config.get("agent_audio_asr_model", "nvidia/parakeet-tdt-0.6b-v2")

    cmd_args = [
        f"python {asr_script}",
        f"--eval_results_dir {eval_results_dir}",
        "--subtest serviceduplexbench",
        "--input_jsonl output.jsonl",
        "--output_jsonl output_asr.jsonl",
        f"--asr_model {asr_model}",
    ]
    if config.get("agent_audio_force", False):
        cmd_args.append("--force")

    return " ".join(cmd_args)


def run_serviceduplexbench_eval(config: dict):
    """Run ServiceDuplexBench evaluation."""

    generation_only = config.get("generation_only", False)
    scoring_only = config.get("scoring_only", False)
    dry_run = config.get("dry_run", False)

    agent_audio_stage_enabled = config.get("agent_audio_stage_enabled")
    if agent_audio_stage_enabled is None:
        agent_audio_stage_enabled = "--decode_audio" in (config.get("server_args") or "")

    expname = config.get("expname", "serviceduplexbench")
    benchmark = "serviceduplexbench"
    eval_results_path = f"{config['output_dir']}/eval-results/serviceduplexbench"
    eval_dir = Path(eval_results_path)
    output_jsonl = eval_dir / "output.jsonl"
    output_jsonl_done = eval_dir / "output.jsonl.done"

    print(f"Output directory: {config['output_dir']}")

    base_extra_args = ["++eval_type=null"]
    if config.get("max_samples"):
        base_extra_args.append(f"++max_samples={config['max_samples']}")
    if config.get("server_server_type"):
        base_extra_args.append(f"++server.server_type={config['server_server_type']}")
    if config.get("api_key_env_var"):
        base_extra_args.append(f"++server.api_key_env_var={config['api_key_env_var']}")

    extra_args_str = " ".join(base_extra_args)

    num_chunks = int(config.get("num_chunks", 1) or 1)
    if num_chunks > 1:
        chunk_done_ok = all((eval_dir / f"output_chunk_{i}.jsonl.done").exists() for i in range(num_chunks))
    else:
        chunk_done_ok = (eval_dir / "output_chunk_0.jsonl.done").exists() or output_jsonl_done.exists()
    generation_complete = output_jsonl.exists() and chunk_done_ok
    generation_submitted = False

    # Generation phase
    if not scoring_only:
        if generation_complete:
            print(f"\n--- Skipping generation (found {output_jsonl} and done markers) ---")
        else:
            print("\n--- Running generation ---")
            server_gpus = config.get("server_gpus", 1)
            partition = config.get("cpu_partition") if server_gpus == 0 else config.get("partition")
            gen_exp = nemo_eval(
                ctx=wrap_arguments(extra_args_str),
                cluster=config["cluster"],
                output_dir=config["output_dir"],
                benchmarks=benchmark,
                model=config["model"],
                server_type=config.get("server_type", "vllm"),
                server_gpus=server_gpus,
                server_address=config.get("server_address"),
                num_chunks=config.get("num_chunks", 1),
                server_container=config.get("server_container"),
                server_entrypoint=config.get("server_entrypoint"),
                data_dir=config.get("data_dir"),
                server_args=config.get("server_args", ""),
                installation_command=config.get("installation_command"),
                partition=partition,
                expname=expname,
                auto_summarize_results=False,
                dry_run=dry_run,
            )
            generation_submitted = gen_exp is not None

    # Agent-audio ASR + WER/CER phase (optional)
    agent_audio_expname = f"{expname}_agent_audio_asr"
    if not generation_only and agent_audio_stage_enabled:
        print("\n--- Running agent-audio ASR + WER/CER ---")
        asr_command = build_agent_audio_asr_command(config)
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=config["cluster"],
            command=asr_command,
            container=config.get("server_container") or "nemo-skills",
            partition=config.get("partition"),
            num_gpus=1,
            run_after=[expname] if generation_submitted else None,
            expname=agent_audio_expname,
            installation_command=config.get("agent_audio_installation_command"),
            log_dir=f"{eval_results_path}/summarized-results",
            dry_run=dry_run,
        )

    # Scoring phase: LLM judge on generated text
    if not generation_only:
        print("\n--- Running LLM judge scoring (generated text) ---")
        score_command = f"{build_score_command(config)} --input_jsonl output.jsonl --metrics_variant generated"
        score_expname = f"{expname}_score_generated"
        run_after = [expname] if generation_submitted else None
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=config["cluster"],
            command=score_command,
            partition=config.get("cpu_partition") or config.get("partition"),
            run_after=run_after,
            expname=score_expname,
            installation_command=config.get("scoring_installation_command"),
            log_dir=f"{eval_results_path}/summarized-results",
            dry_run=dry_run,
        )

        # Scoring on ASR transcript (if ASR stage is enabled)
        if agent_audio_stage_enabled:
            print("\n--- Running LLM judge scoring (agent ASR) ---")
            score_command_asr = f"{build_score_command(config)} --input_jsonl output_asr.jsonl --metrics_variant asr"
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=config["cluster"],
                command=score_command_asr,
                partition=config.get("cpu_partition") or config.get("partition"),
                run_after=[agent_audio_expname, score_expname],
                expname=f"{expname}_score_asr",
                installation_command=config.get("scoring_installation_command"),
                log_dir=f"{eval_results_path}/summarized-results",
                dry_run=dry_run,
            )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="ServiceDuplexBench evaluation with LLM judge scoring")
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    parser.add_argument("--cluster", help="Override cluster")
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--generation_only", action="store_true", help="Only run generation")
    parser.add_argument("--scoring_only", action="store_true", help="Only run scoring")

    args = parser.parse_args()
    config = load_config(args.config)

    for key in ["cluster", "partition", "model", "output_dir", "max_samples"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.dry_run:
        config["dry_run"] = True
    if args.generation_only:
        config["generation_only"] = True
    if args.scoring_only:
        config["scoring_only"] = True

    run_serviceduplexbench_eval(config)


if __name__ == "__main__":
    main()
