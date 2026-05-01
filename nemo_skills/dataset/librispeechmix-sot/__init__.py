# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""LibriSpeechMix SOT benchmark group."""

IS_BENCHMARK_GROUP = True
REQUIRES_DATA_DIR = True
SCORE_MODULE = "nemo_skills.dataset.librispeechmix-sot.librispeechmix_sot_score"

_DURATION_SPLITS = ("under20s", "over20s")
_LSM_SPLITS = ("test-clean", "dev-clean")
_MIXES = ("1mix", "2mix", "3mix")

BENCHMARKS = {
    f"librispeechmix-sot.{duration_split}-{lsm_split}-{mix}": {}
    for duration_split in _DURATION_SPLITS
    for lsm_split in _LSM_SPLITS
    for mix in _MIXES
}
