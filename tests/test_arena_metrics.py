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

import random

from nemo_skills.evaluation.metrics.arena_metrics import ArenaMetrics


def _make_prediction(gen_base_score, base_gen_score, category=None):
    """Helper to create a prediction dict with judgment scores."""
    pred = {
        "judgement-gen-base": f"[[{gen_base_score}]]",
        "judgement-base-gen": f"[[{base_gen_score}]]",
    }
    if category is not None:
        pred["category"] = category
    return pred


def test_arena_metrics_per_category_scoring_v2():
    """Test that arena-hard-v2 with multiple categories produces per-category scores."""
    m = ArenaMetrics()

    random.seed(42)
    scores_pool = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B"), ("A>>B", "B>>A"), ("B>>A", "A>>B")]

    # 50 hard_prompt entries
    for _ in range(50):
        score = random.choice(scores_pool)
        m.update([_make_prediction(score[0], score[1], category="hard_prompt")])

    # 25 creative_writing entries
    for _ in range(25):
        score = random.choice(scores_pool)
        m.update([_make_prediction(score[0], score[1], category="creative_writing")])

    assert m.total == 75
    assert set(m.categories) == {"hard_prompt", "creative_writing"}

    metrics = m.get_metrics()

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert metrics["pass@1"]["num_entries"] == 75
    assert "score" in metrics["pass@1"]
    assert "95_CI" in metrics["pass@1"]

    # Check per-category metrics exist
    assert "category_hard_prompt" in metrics["pass@1"]
    assert metrics["pass@1"]["category_hard_prompt"]["num_entries"] == 50
    assert "score" in metrics["pass@1"]["category_hard_prompt"]

    assert "category_creative_writing" in metrics["pass@1"]
    assert metrics["pass@1"]["category_creative_writing"]["num_entries"] == 25
    assert "score" in metrics["pass@1"]["category_creative_writing"]


def test_arena_metrics_single_category_v1():
    """Test that arena-hard-v1 with single category does not produce per-category breakdown."""
    m = ArenaMetrics()

    random.seed(42)
    scores_pool = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B"), ("A>>B", "B>>A"), ("B>>A", "A>>B")]

    # All entries have same category (v1 scenario)
    for _ in range(50):
        score = random.choice(scores_pool)
        m.update([_make_prediction(score[0], score[1], category="arena-hard-v0.1")])

    assert m.total == 50
    assert set(m.categories) == {"arena-hard-v0.1"}

    metrics = m.get_metrics()

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert metrics["pass@1"]["num_entries"] == 50
    assert "score" in metrics["pass@1"]

    # Check no per-category breakdown for single category
    has_category_keys = any(k.startswith("category_") for k in metrics["pass@1"].keys())
    assert not has_category_keys


def test_arena_metrics_legacy_data_no_category():
    """Test that legacy data without category field works correctly."""
    m = ArenaMetrics()

    random.seed(42)
    scores_pool = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B")]

    # Data without category field
    for _ in range(30):
        score = random.choice(scores_pool)
        m.update([_make_prediction(score[0], score[1])])  # No category

    assert m.total == 30
    assert set(m.categories) == {None}

    metrics = m.get_metrics()

    # Check overall metrics exist
    assert "pass@1" in metrics
    assert metrics["pass@1"]["num_entries"] == 30
    assert "score" in metrics["pass@1"]

    # Check no per-category breakdown
    has_category_keys = any(k.startswith("category_") for k in metrics["pass@1"].keys())
    assert not has_category_keys


def test_arena_metrics_score_parsing():
    """Test that judge scores are correctly parsed."""
    m = ArenaMetrics()

    # Test various score formats
    test_cases = [
        ("A>>B", "A>>B"),
        ("A>B", "A>B"),
        ("A=B", "A=B"),
        ("B>A", "B>A"),
        ("B>>A", "B>>A"),
    ]

    for gen_base, base_gen in test_cases:
        m.reset()
        m.update([_make_prediction(gen_base, base_gen, category="test")])
        # per_prompt_scores stores the full per-prediction pair list (one tuple per
        # prediction). For N=1 it's a single-element list.
        assert m.per_prompt_scores[0] == [(gen_base, base_gen)]


def test_arena_metrics_invalid_score_handling():
    """Test that invalid scores are handled correctly."""
    m = ArenaMetrics()

    # Invalid score format
    pred = {
        "judgement-gen-base": "No valid score here",
        "judgement-base-gen": "Also invalid",
        "category": "test",
    }
    m.update([pred])

    assert m.per_prompt_scores[0] == [(None, None)]


def test_arena_metrics_pass_at_k_with_repeats():
    """num_repeats > 1 should emit pass@N (best-of-N) and pass@1[avg-of-N] (avg-of-N),
    including per-category breakdowns when more than one category is present."""
    m = ArenaMetrics()
    random.seed(42)
    scores_pool = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B"), ("A>>B", "B>>A"), ("B>>A", "A>>B")]

    # 100 prompts (50 per category) — enough samples for the Elo bootstrap inside each
    # category to see >1 outcome class.
    n_prompts, n_repeats = 100, 5
    for i in range(n_prompts):
        # Alternate categories to exercise the per-category emission path under N > 1.
        category = "hard_prompt" if i % 2 == 0 else "creative_writing"
        preds = [_make_prediction(*random.choice(scores_pool), category=category) for _ in range(n_repeats)]
        m.update(preds)

    metrics = m.get_metrics()

    pass_at_n = f"pass@{n_repeats}"
    avg_of_n = f"pass@1[avg-of-{n_repeats}]"
    assert set(metrics.keys()) == {pass_at_n, avg_of_n}, (
        f"Expected exactly {{{pass_at_n}, {avg_of_n}}}, got {set(metrics.keys())}"
    )

    assert metrics[pass_at_n]["num_entries"] == n_prompts
    assert metrics[avg_of_n]["num_entries"] == n_prompts

    # Per-category sub-aggregations should appear under both keys.
    for metric_name in (pass_at_n, avg_of_n):
        assert "category_hard_prompt" in metrics[metric_name]
        assert "category_creative_writing" in metrics[metric_name]
        assert metrics[metric_name]["category_hard_prompt"]["num_entries"] == n_prompts // 2
        assert metrics[metric_name]["category_creative_writing"]["num_entries"] == n_prompts // 2

    # best-of-N can never score worse than avg-of-N: picking the best out of N
    # generations is at least as candidate-favorable as the average single-shot.
    assert metrics[pass_at_n]["score"] >= metrics[avg_of_n]["score"], (
        f"best-of-N ({metrics[pass_at_n]['score']}) < avg-of-N ({metrics[avg_of_n]['score']})"
    )
