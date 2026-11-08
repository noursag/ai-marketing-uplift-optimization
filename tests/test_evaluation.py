"""Unit tests for the evaluation and simulation functions on tiny examples whose answers
can be checked by hand. Run from the project root:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (auuc, bootstrap, diff_in_means, policy_incremental_at_k, qini_coefficient,
                            uplift_at_k, uplift_by_decile, uplift_curves)
from src.simulation import add_economics, budget_cap

# 8 customers, ranked 1..8 by score. Treated/control alternate.
# Only the two treated customers in the top half respond -> perfect ranking.
SCORE = np.array([8, 7, 6, 5, 4, 3, 2, 1], float)
T = np.array([1, 0, 1, 0, 1, 0, 1, 0])
Y = np.array([1, 0, 1, 0, 0, 0, 0, 0], float)


def test_diff_in_means():
    assert diff_in_means([1, 0], [0, 0])["diff"] == pytest.approx(0.5)


def test_uplift_at_k_known_values():
    assert uplift_at_k(Y, T, SCORE, 0.5) == pytest.approx(1.0)   # top 4: treated 2/2, control 0/2
    assert uplift_at_k(Y, T, SCORE, 1.0) == pytest.approx(0.5)   # all: treated 2/4, control 0/4


def test_uplift_at_k_needs_both_groups():
    assert np.isnan(uplift_at_k(Y, T, SCORE, 1 / 8))             # top 1 has no control customer


def test_qini_curve_values():
    c = uplift_curves(Y, T, SCORE)
    assert list(c["qini"]) == pytest.approx([0, 1, 1, 2, 2, 2, 2, 2, 2])
    assert c["uplift"].iloc[-1] == pytest.approx(0.5 * 8)       # (2/4 - 0/4) x 8


def test_qini_coefficient_hand_computed():
    # trapezoid area of qini/n over frac = 0.203125, random line area = 0.5 x 2/8 = 0.125
    assert qini_coefficient(Y, T, SCORE) == pytest.approx(0.078125)


def test_good_ranking_beats_reversed_and_random_is_zero_at_end():
    assert qini_coefficient(Y, T, SCORE) > qini_coefficient(Y, T, -SCORE)
    assert auuc(Y, T, SCORE) > auuc(Y, T, -SCORE)


def test_auuc_absolute_vs_relative():
    assert auuc(Y, T, SCORE, relative=False) - auuc(Y, T, SCORE) == pytest.approx(0.5 * 0.5)


def test_deciles():
    d = uplift_by_decile(Y, T, SCORE, n_bins=2)
    assert list(d["uplift"]) == pytest.approx([1.0, 0.0])
    assert d["n_treated"].sum() + d["n_control"].sum() == 8


def test_bootstrap_contains_estimate():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 2000)
    y = (rng.random(2000) < 0.1 + 0.1 * t).astype(float)        # true uplift = 0.10
    b = bootstrap(uplift_at_k, y, t, rng.random(2000), n_boot=200, k=1.0)
    assert b["lo"] <= b["estimate"] <= b["hi"]
    assert b["lo"] < 0.10 < b["hi"]



def test_policy_incremental_matches_recommended_arm():
    arm = np.array([1, 0, 2, 0, 1, 0, 2, 0])

    policies = [
        np.ones(8, dtype=int),
        np.array([1, 1, 2, 2, 1, 1, 2, 2])
    ]

    probabilities = np.bincount(arm, minlength=3) / len(arm)

    selected = np.argsort(-SCORE, kind="stable")[:4]

    for pa in policies:
        expected = np.mean([
            (
                (Y[i] / probabilities[arm[i]] if arm[i] == pa[i] else 0.0)
                -
                (Y[i] / probabilities[0] if arm[i] == 0 else 0.0)
            )
            for i in selected
        ])

        result = policy_incremental_at_k(
            Y, arm, pa, SCORE, 0.5
        )

        assert result == pytest.approx(expected)


def test_economics_and_budget():
    t = pd.DataFrame({"k": [0.5], "inc_spend_per_targeted": [2.0]})
    e = add_economics(t, population=100, cost=0.5, margin=0.3)
    assert e["incremental_profit"].iloc[0] == pytest.approx(2.0 * 50 * 0.3 - 0.5 * 50)
    assert budget_cap(0.5, budget=100, cost=1.0, population=1000) == pytest.approx(0.1)


def test_paired_bootstrap_identical_scores_is_zero():
    from src.evaluation import bootstrap_diff
    d = bootstrap_diff(qini_coefficient, Y, T, SCORE, SCORE, n_boot=50)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_policy_diff_of_same_policy_is_zero():
    from src.evaluation import bootstrap_policy_diff
    arm = np.array([1, 0, 2, 0, 1, 0, 2, 0])
    d = bootstrap_policy_diff(Y, arm, np.ones(8), SCORE, np.ones(8), SCORE, 0.5, n_boot=30)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_random_strategy_uses_population_effect():
    from src.simulation import strategy_table
    df = pd.DataFrame({"visit": Y, "arm": np.array([1, 0, 1, 0, 1, 0, 1, 0])})
    tab = strategy_table(df, {"Random": (SCORE, np.ones(8))}, outcomes=("visit",), k_grid=[0.5, 1.0])
    assert list(tab["inc_visit_per_targeted"]) == pytest.approx([0.5, 0.5])


def test_best_threshold_sends_nothing_when_all_negative():
    from src.simulation import best_threshold
    e = pd.DataFrame({"strategy": ["A", "A"], "k": [0.5, 1.0], "n_contacted": [5, 10], "inc_spend": [0, 0],
                      "campaign_cost": [1, 2], "incremental_profit": [-1.0, -2.0]})
    assert best_threshold(e)["k"].iloc[0] == 0 and best_threshold(e)["incremental_profit"].iloc[0] == 0
