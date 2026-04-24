"""Evaluate DeepSeek-V4-{Flash,Pro} on Apex Shortlist and IMOAnswerBench.

Target cluster: aws-iad. Weights live at
/lustre/fsw/portfolios/nemotron/users/igitman/hf_models/DeepSeek-V4-{Flash,Pro}
which is mounted inside the vllm container as /hf_models/...

The script reproduces the per-mode settings from the DeepSeek-V4 tech report
(April 2026), Table 2 / Table 3 / §5.3.1:

  - Sampling:   temperature=1.0, top_p=1.0 (all modes)
  - Context:    128K tokens for "Think High", 384K tokens for "Think Max"
  - Prompt:     math template "{problem}\\nPlease reason step by step, and
                put your final answer within \\boxed{{}}."  (both models, both
                modes). Pro-Max swaps in the proof-oriented wrapper from
                §5.3.1 as the user message.
  - Think Max:  NOT injected manually -- DeepSeek's own encoding
                (encoding/encoding_dsv4.py in the HF repo) auto-prepends the
                Table-3 instruction whenever reasoning_effort='max' and
                thinking_mode='thinking'. We just pass
                ++inference.reasoning_effort={high,max}; the V4-aware vllm
                container routes that through to the encoder.

H100 topology is sized for FP8 native weights (both checkpoints ship as
block-wise FP8, see config.json -> quantization_config.quant_method):

  - Flash  (149 GB weights): 1 node, 8x H100, TP=8, EP enabled.
  - Pro    (806 GB weights): 2 nodes, 16x H100 for High; 4 nodes, 32x H100
    for Max (larger KV-cache budget at 384K ctx). Edit PRO_NODES_* below
    to change.

DeepSeek-V4 architecture is new (`model_type: deepseek_v4`,
`DeepseekV4ForCausalLM`). It requires --trust-remote-code and a vllm build
that recognizes the arch. If the nemo-skills vllm-latest image predates V4
support, bump the `server_container` override at the bottom of this file.
"""

from nemo_skills.pipeline.cli import eval, wrap_arguments

# ---------------------------------------------------------------------------
# Knobs -- comment rows out of VARIANTS to skip them.
# ---------------------------------------------------------------------------

# DEBUG=True: submit just flash-high on apex-shortlist, no dependent-job chain,
# no email on failure -- keeps noise down while we shake out bugs. Flip to
# False once a single variant works end-to-end.
DEBUG = True

VARIANTS = (
    ["flash-high"]
    if DEBUG
    else [
        "flash-high",
        "flash-max",
        "pro-high",
        "pro-max",
    ]
)
BENCHMARKS = "apex-shortlist:1" if DEBUG else "apex-shortlist:4,imo-answerbench:4"

# 4 rollouts/problem -> Pass@1 is averaged; matches the usual Avg@k convention
# for stochastic sampling at T=1.0. Bump for tighter error bars.
NUM_REPEATS = 4

CLUSTER = "aws-iad-dsv4"  # drops /workspace mount so the container's
# built-in /workspace/sglang (v4-aware) isn't shadowed by lustre.
ACCOUNT = "nemotron_reason_math"
PARTITION = "pool0"

# Container paths on aws-iad-dsv4 (see cluster_configs/aws-iad-dsv4.yaml).
MODEL_DIR = "/hf_models"
# /workspace is no longer mounted; write outputs straight to lustre.
OUT_DIR = "/lustre/fsw/portfolios/nemotron/users/igitman/claude-dsv4-eval"

# DeepSeek-V4 on H100: vllm does not officially support Hopper for V4
#   (recipes target H200+). The default DeepSeek checkpoint stores MoE
#   experts as FP4, which H100 has no native compute for. The combination
#   of the -cu130 vllm image + either the default FP4 checkpoint or the
#   sgl-project FP8 re-pack hit either Inductor crashes or weight-loader
#   shape mismatches on our cluster. Officially-supported path for H-series
#   is SGLang's `deepseek-v4-hopper` image fed the `sgl-project/*-FP8`
#   checkpoint -- that's what we use here.
V4_SGLANG_CONTAINER = (
    "/lustre/fsw/portfolios/nemotron/users/igitman/images/claude-containers/nemo-skills-sglang-dsv4-hopper.sqsh"
)
# Preserved for reference -- keep around in case vllm lands H100 support.
V4_VLLM_CONTAINER = (
    "/lustre/fsw/portfolios/nemotron/users/igitman/images/claude-containers/nemo-skills-vllm-dsv4-cu130-ray.sqsh"
)

# Local judge for IMOAnswerBench -- avoids needing a Gemini API key on the
# cluster. The dataset's default judge is gemini-2.5-pro; using a capable
# open model as a judge is known to degrade scores slightly -- keep in mind
# when comparing to the published numbers.
JUDGE_MODEL = "/hf_models/gpt-oss-120b"
JUDGE_SERVER_TYPE = "sglang"
JUDGE_GPUS = 4

# ---------------------------------------------------------------------------
# Per-variant deployment plan.
# ---------------------------------------------------------------------------

# Per-mode eval-time context budget (from tech report §5.3.1).
MODE_CTX_LEN = {"high": 128 * 1024, "max": 384 * 1024}
MODE_GEN_TOKS = {"high": 64 * 1024, "max": 256 * 1024}

# (server_gpus, server_nodes) for each (model, mode) on H100.
# Flash fits comfortably on 1 node for both modes.
# Pro needs >= 2 nodes for weights alone; Max mode bumps to 4 nodes to fit
# the 384K-token KV cache at decent throughput.
PRO_NODES_HIGH = 2
PRO_NODES_MAX = 4

TOPOLOGY = {
    ("flash", "high"): (8, 1),
    ("flash", "max"): (8, 1),
    ("pro", "high"): (8, PRO_NODES_HIGH),
    ("pro", "max"): (8, PRO_NODES_MAX),
}


def pick_prompt_config(model_short: str, mode: str) -> str:
    """Pick the user-message wrapper. The Table-3 system instruction is
    injected by DeepSeek's own encoder when reasoning_effort='max', so these
    configs only need to carry the user template.

    - All High variants and Flash-Max: standard "\\boxed{}" wrapper.
    - Pro-Max: proof-oriented wrapper (§5.3.1).
    """
    if model_short == "pro" and mode == "max":
        return "generic/deepseek-v4-math-max"
    return "generic/deepseek-v4-math"


def build_server_args(max_model_len: int, tp_size: int) -> str:
    # SGLang DSv4-hopper flags. The cookbook uses TP across all GPUs +
    # implicit expert parallelism (no --enable-ep-moe in this sglang
    # version; was removed/renamed). --reasoning-parser deepseek-r1
    # handles V4's <think>...</think>.
    return " ".join(
        [
            "--trust-remote-code",
            f"--tp-size {tp_size}",
            f"--context-length {max_model_len}",
            "--mem-fraction-static 0.9",
            "--reasoning-parser deepseek-r1",
        ]
    )


def build_ctx(prompt_config: str, tokens_to_generate: int, mode: str):
    # reasoning_effort={high,max} activates DeepSeek-V4's in-encoder logic:
    # thinking blocks are emitted for both; the Table-3 system instruction is
    # automatically prepended only when mode='max'. Non-think mode would use
    # reasoning_effort=None (currently not exposed here since the paper's
    # eval numbers for Flash-Max/Pro-Max both use thinking modes).
    return wrap_arguments(
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        f"++inference.tokens_to_generate={tokens_to_generate} "
        f"++inference.reasoning_effort={mode} "
        f"++prompt_config={prompt_config} "
    )


# ---------------------------------------------------------------------------
# Launch loop.
# ---------------------------------------------------------------------------


def run():
    for variant in VARIANTS:
        model_short, mode = variant.split("-")
        # sgl-project/DeepSeek-V4-{Flash,Pro}-FP8 is the H-series-targeted
        # checkpoint: the default DeepSeek checkpoint stores MoE experts as
        # FP4 (Blackwell-native, UE8M0 scales). H100 has no native FP4, so
        # running the default checkpoint crashes during Inductor lowering and
        # -- when it does run eager -- produces garbage outputs. The FP8
        # variant is a lossless re-pack of the same weights (paper §3.3).
        model_path = f"{MODEL_DIR}/DeepSeek-V4-{'Pro' if model_short == 'pro' else 'Flash'}-FP8"
        server_gpus, server_nodes = TOPOLOGY[(model_short, mode)]
        tp_size = server_gpus * server_nodes
        max_model_len = MODE_CTX_LEN[mode]
        tokens_to_generate = MODE_GEN_TOKS[mode]
        prompt_config = pick_prompt_config(model_short, mode)

        # --switches 1 asks slurm to pack the allocation onto one switch
        # (meaningful for multi-node jobs; a no-op for single-node). Always
        # non-empty because parse_kwargs({}) returns None, which trips
        # `'segment' in sbatch_kwargs` in eval.py:93.
        # In DEBUG mode we also silence FAIL emails while shaking out bugs.
        sbatch_kwargs = {"switches": 1}
        if DEBUG:
            sbatch_kwargs["mail_type"] = "NONE"

        eval(
            ctx=build_ctx(prompt_config, tokens_to_generate, mode),
            cluster=CLUSTER,
            expname=f"eval-dsv4-{variant}",
            model=model_path,
            server_type="sglang",
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            benchmarks=BENCHMARKS,
            # num_jobs=1 keeps all of a variant's work (both benchmarks, all
            # seeds) on a single vllm server -- sticks to the 64-GPU budget.
            # Going above 1 multiplies the server count per variant.
            num_jobs=1,
            # In DEBUG we skip the job chain -- a single failed attempt is
            # enough to diagnose from logs.
            dependent_jobs=0 if DEBUG else 2,
            server_args=build_server_args(max_model_len, tp_size),
            # server_container override removed -- use whatever the
            # aws-iad-dsv4 cluster config provides (sglang container path
            # is set there).
            output_dir=f"{OUT_DIR}/{variant}",
            # Local gpt-oss-120b judge for imo-answerbench.
            judge_model=JUDGE_MODEL,
            judge_server_type=JUDGE_SERVER_TYPE,
            judge_server_gpus=JUDGE_GPUS,
            extra_judge_args="++inference.reasoning_effort=medium",
            partition=PARTITION,
            account=ACCOUNT,
            sbatch_kwargs=sbatch_kwargs,
            # Uncomment once a baseline run has seeded /nemo-run/.../code on
            # aws-iad -- speeds up subsequent launches by avoiding rsync.
            # reuse_code_exp="eval-dsv4-flash-high",
        )


if __name__ == "__main__":
    run()
