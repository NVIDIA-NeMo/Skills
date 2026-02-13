from pathlib import Path

from nemo_skills.pipeline.utils.packager import (
    RepoMetadata,
    register_external_repo,
)

# Register repo so it gets packaged inside containers.
# ignore_if_registered avoids errors when the module is imported more than once.
register_external_repo(
    RepoMetadata(name="my_benchmarks", path=Path(__file__).parents[2]),
    ignore_if_registered=True,
)

# Metrics class - use module::Class format for custom metrics
METRICS_TYPE = "my_benchmarks.metrics.word_count::WordCountMetrics"

# Default generation arguments
# prompt_config ending in .yaml triggers absolute-path resolution;
# /nemo_run/code/ is the root where code is extracted inside the container
GENERATION_ARGS = (
    "++prompt_config=/nemo_run/code/my_benchmarks/prompt/eval/word_count/default.yaml "
    "++eval_type=my_benchmarks.evaluation.word_count::WordCountEvaluator"
)

# Custom generation module (optional - remove this line to use the default)
GENERATION_MODULE = "my_benchmarks.inference.word_count"
