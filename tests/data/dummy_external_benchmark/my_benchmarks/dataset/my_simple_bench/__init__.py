from pathlib import Path

from nemo_skills.pipeline.utils.packager import (
    RepoMetadata,
    register_external_repo,
)

# Register repo so it gets packaged inside containers.
register_external_repo(
    RepoMetadata(name="my_benchmarks", path=Path(__file__).parents[2]),
    ignore_if_registered=True,
)

METRICS_TYPE = "math"
GENERATION_ARGS = "++prompt_config=generic/math ++eval_type=math"
