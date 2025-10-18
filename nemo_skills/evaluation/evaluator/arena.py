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

# adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py

import inspect
import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


JUDGE_MODEL = "gpt-4-1106-preview"
JUDGE_SERVER = "openai"


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if "baseline" in models.index:
        elo_scores += 1000 - elo_scores[models["baseline"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    kwargs = {}
    if "baseline" in inspect.signature(func_compute_elo).parameters:
        kwargs["baseline"] = "baseline"
    for _ in range(num_round):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), **kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.nan for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table["baseline"].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgment(scores, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()
    num_invalid = 0

    for score in scores:
        # game 1
        output = {"model_a": "candidate", "model_b": "baseline"}

        assert len(score) == 2
        cur_score = score[0]

        weight = 1
        if cur_score == "A=B":
            output["winner"] = "tie"
        elif cur_score == "A>B":
            output["winner"] = "model_a"
        elif cur_score == "A>>B":
            output["winner"] = "model_a"
            weight = WEIGHT
        elif cur_score == "B>A":
            output["winner"] = "model_b"
        elif cur_score == "B>>A":
            output["winner"] = "model_b"
            weight = WEIGHT
        else:
            num_invalid += 1
            weight = 0

        if weight:
            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

        # game 2
        output = {"model_a": "candidate", "model_b": "baseline"}

        cur_score = score[1]

        weight = 1
        if cur_score == "A=B":
            output["winner"] = "tie"
        elif cur_score == "A>B":
            output["winner"] = "model_b"
        elif cur_score == "A>>B":
            output["winner"] = "model_b"
            weight = WEIGHT
        elif cur_score == "B>A":
            output["winner"] = "model_a"
        elif cur_score == "B>>A":
            output["winner"] = "model_a"
            weight = WEIGHT
        else:
            num_invalid += 1
            weight = 0

        if weight:
            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
    return arena_hard_battles, num_invalid


def get_aggregate_score(scores, weight=3):
    battles, num_invalid = get_battles_from_judgment(scores, weight)
    bootstrap_online_elo = compute_mle_elo(battles)

    np.random.seed(42)
    num_rounds = 100
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, num_rounds)

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()

    stats.sort_values(by="model", inplace=True)
    stats["score"] = get_win_rate_column(stats, "score").tolist()
    stats["lower"] = get_win_rate_column(stats, "lower").tolist()
    stats["upper"] = get_win_rate_column(stats, "upper").tolist()

    candidate_stats = stats[stats["model"] == "candidate"]
    interval = (
        float(round((candidate_stats["lower"] - candidate_stats["score"]).iloc[0], 2)),
        float(round((candidate_stats["upper"] - candidate_stats["score"]).iloc[0], 2)),
    )
    metrics = {
        "score": candidate_stats["score"].iloc[0],
        "95_CI": interval,
        "invalid_scores": num_invalid,
    }
    return metrics
