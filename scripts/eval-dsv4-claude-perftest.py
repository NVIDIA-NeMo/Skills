"""Side-by-side perf test for DSV4-Flash-High on apex-shortlist.

All runs share the same evaluation (1 variant, 1 benchmark, 1 seed) so the
only variable is the inference stack. Lets us pick the fastest working
config before scaling up to full eval matrix.

Three variants launched in parallel:

  - v1-compile-lvl1 : vllm, --compilation-config '{"level":1}' (piecewise,
                     fallback to eager for unsupported subgraphs; typically
                     dodges the decompose_triton_kernel_wrapper_functional
                     Inductor crash we hit on full-graph compile).
  - v2-aot-eager   : vllm, full-graph CUDA-graph capture but Inductor off --
                     level 3 + use_inductor=false.
  - v3-sglang      : sglang on lmsysorg/sglang:deepseek-v4-hopper. Separate
                     compile stack, so immune to the vllm Inductor bug.

Baseline (--enforce-eager) is whatever's already running from the main
eval-dsv4-claude.py script -- compare wallclocks after all four finish.
"""

from nemo_skills.pipeline.cli import eval, wrap_arguments

CLUSTER = "aws-iad"
ACCOUNT = "nemotron_reason_math"
PARTITION = "pool0"
MODEL = "/hf_models/DeepSeek-V4-Flash"
MODEL_LOADER_EXTRA = '\'{"enable_multithread_load":true,"num_threads":96}\''
CONTAINERS_DIR = "/lustre/fsw/portfolios/nemotron/users/igitman/images/claude-containers"
VLLM_CONTAINER = f"{CONTAINERS_DIR}/nemo-skills-vllm-dsv4-cu130-ray.sqsh"
SGLANG_CONTAINER = f"{CONTAINERS_DIR}/nemo-skills-sglang-dsv4-hopper.sqsh"
OUT_BASE = "/workspace/claude-dsv4-eval/perftest"

COMMON_CTX = (
    "++inference.temperature=1.0 "
    "++inference.top_p=1.0 "
    "++inference.tokens_to_generate=65536 "
    "++inference.reasoning_effort=high "
    "++prompt_config=generic/deepseek-v4-math "
)


def _base_vllm_args(extra):
    return " ".join(
        [
            "--trust-remote-code",
            "--load-format auto",
            "--tensor-parallel-size 8",
            "--enable-expert-parallel",
            "--distributed-executor-backend ray",
            "--max-model-len 131072",
            "--gpu-memory-utilization 0.92",
            "--kv-cache-dtype fp8",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens 8192",
            "--max-num-seqs 32",
            "--reasoning-parser deepseek_r1",
            f"--model-loader-extra-config {MODEL_LOADER_EXTRA}",
            extra,
        ]
    )


# CompilationConfig schema (from in-container introspection):
#   mode: NONE=0, STOCK_TORCH_COMPILE=1, DYNAMO_TRACE_ONCE=2, VLLM_COMPILE=3
#   cudagraph_mode: NONE=0, PIECEWISE=1, FULL=2, ...
#   backend: 'inductor' (default), 'aot_eager', 'eager', ...
# The known crash on V4 is in Inductor post-grad passes, so both configs
# below avoid Inductor entirely but keep enough compile work to let CUDA
# graphs capture the forward pass.
RUNS = [
    ("v1-dyn-trace", "vllm", VLLM_CONTAINER, _base_vllm_args("""--compilation-config '{"mode":2}'""")),
    (
        "v2-aot-eager",
        "vllm",
        VLLM_CONTAINER,
        _base_vllm_args("""--compilation-config '{"mode":1,"backend":"aot_eager"}'"""),
    ),
]


def run():
    for name, server_type, container, server_args in RUNS:
        eval(
            ctx=wrap_arguments(COMMON_CTX),
            cluster=CLUSTER,
            expname=f"eval-dsv4-perftest-{name}",
            model=MODEL,
            server_type=server_type,
            server_gpus=8,
            server_nodes=1,
            benchmarks="apex-shortlist:1",
            num_jobs=1,
            dependent_jobs=0,
            server_args=server_args,
            server_container=container,
            output_dir=f"{OUT_BASE}/{name}",
            partition=PARTITION,
            account=ACCOUNT,
            # silence FAIL emails during debug
            sbatch_kwargs={"switches": 1, "mail_type": "NONE"},
        )


if __name__ == "__main__":
    run()
