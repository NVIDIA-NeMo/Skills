# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

# settings that define how evaluation should be done by default (all can be changed from cmdline)
GENERATION_ARGS = "++prompt_config=generic/default ++eval_type=ccc"
METRICS_TYPE = "ccc"

# environment variables required by this benchmark
SANDBOX_ENV_VARS = [
    "UWSGI_PROCESSES=1024",
    "UWSGI_CPU_AFFINITY=8",
    "UWSGI_CHEAPER=1023",
    "NUM_WORKERS=1",
    "STATEFUL_SANDBOX=0",
]
