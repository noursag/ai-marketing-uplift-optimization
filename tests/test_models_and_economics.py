"""Unit tests for the honest uplift tree, the new evaluation metric, the visit-value economics
and the messaging helpers. Small synthetic examples with known answers."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import _order, share_of_lift_at_k, uplift_at_k
from src.messaging import _satisfies, describe_customer, draft_message
from src.models import HonestUpliftTree, UpliftTreeLearner, recommend
from src.simulation import add_visit_economics, break_even_cost, decision_map


def planted_data(n=8000, seed=0):
    """Binary feature `flag` doubles the uplift: 0.02 without, 0.12 with. `noise` is irrelevant."""
    rng = np.random.default_rng(seed)
    flag = rng.integers(0, 2, n)
    X = pd.DataFrame({"flag": flag, "noise": rng.random(n)})
    t = rng.integers(0, 2, n)
    p = 0.10 + t * np.where(flag == 1, 0.12, 0.02)
    y = (rng.random(n) < p).astype(float)
    return X, y, t


def test_tree_recovers_planted_split_and_estimates_honestly():
    X, y, t = planted_data()
    tree = HonestUpliftTree(max_depth=2, min_leaf=300, min_t=2.0).fit(X, y, t)
    rules = tree.rules()
    assert len(rules) == 2 and set(rules["rule"]) == {"flag = yes", "flag = no"}   # one split, on the right feature
    hi, lo = rules.set_index("rule").loc["flag = yes", "uplift"], rules.set_index("rule").loc["flag = no", "uplift"]
    assert hi == pytest.approx(0.12, abs=0.03) and lo == pytest.approx(0.02, abs=0.03)
    # predictions are the leaf estimates, so the top half of the ranking is exactly the flag=1 group
    pred = tree.predict(X)
    assert np.all((pred > pred.min()) == (X.flag.to_numpy() == 1))


def test_tree_gate_blocks_noise_splits():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.random(4000), "b": rng.random(4000)})
    t = rng.integers(0, 2, 4000)
    y = (rng.random(4000) < 0.1 + 0.05 * t).astype(float)          # uniform uplift: nothing to find
    gated = HonestUpliftTree(max_depth=3, min_leaf=200, min_t=2.5).fit(X, y, t)
    assert len(gated.rules()) <= 2                                  # at most one (chance) split


def test_multi_arm_tree_learner_interface():
    X, y, t = planted_data(3000)
    arm = np.where(t == 1, 2, 0)                                    # women's email vs control only
    arm[:600] = 1                                                   # a few men's-email customers too
    learner = UpliftTreeLearner(params={1: {"max_depth": 1, "min_leaf": 100, "min_leaf_arm": 20}, 2: {"max_depth": 2, "min_leaf": 200}}).fit(X, y, arm)
    u = learner.predict_uplift(X)
    assert list(u.columns) == ["uplift_mens", "uplift_womens"] and len(u) == len(X)
    r = recommend(u)
    assert set(r["recommended_action"]).issubset({"Mens E-Mail", "Womens E-Mail", "No E-Mail"})


def test_ties_are_broken_at_random_not_by_row_order():
    score = np.array([1, 1, 1, 1, 0, 0, 0, 0], float)
    o = _order(score)
    assert set(o[:4]) == {0, 1, 2, 3}                               # the tied top group stays on top ...
    assert list(o[:4]) != [0, 1, 2, 3] or list(_order(score, seed=5)[:4]) != [0, 1, 2, 3]   # ... but not in row order


def test_share_of_lift_perfect_and_random():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0], float); t = np.array([1, 0, 1, 0, 1, 0, 1, 0]); s = np.arange(8, 0, -1, dtype=float)
    assert share_of_lift_at_k(y, t, s, 0.5) == pytest.approx(1.0)   # all responders are in the top half
    assert uplift_at_k(y, t, s, 0.5) == pytest.approx(1.0)


def test_visit_economics_and_break_even():
    tab = pd.DataFrame({"strategy": ["Random", "Random", "A"], "k": [0.5, 1.0, 0.5], "inc_visit_per_targeted": [0.08, 0.08, 0.10]})
    e = add_visit_economics(tab, population=1000, value=7.0, margin=0.3, cost=0.15)
    a = e[e.strategy == "A"].iloc[0]
    assert a["inc_visits"] == pytest.approx(0.10 * 500)
    assert a["incremental_profit"] == pytest.approx(0.10 * 500 * 7.0 * 0.3 - 0.15 * 500)
    assert break_even_cost(0.10, 7.0, 0.3) == pytest.approx(0.21)
    d = decision_map(tab, 1000, 7.0, 0.3, costs=[0.05, 0.20, 0.50])
    assert d.loc[0, "best_strategy"] == "Random" and d.loc[0, "best_k"] == 1.0   # cheap: email everyone (same per-head effect, more heads)
    assert d.loc[1, "best_strategy"] == "A"                                      # dearer: only the targeted half pays
    assert d.loc[2, "best_strategy"] == "No campaign" and d.loc[2, "best_profit"] == 0


def test_messaging_helpers():
    r = pd.Series({"recency": 3, "history": 420.5, "mens": 1, "womens": 1, "newbie": 0, "zip_code": "Urban", "channel": "Web",
                   "T_uplift_mens": 0.09, "T_uplift_womens": 0.04, "T_action": "Mens E-Mail"})
    assert "men's and women's merchandise" in describe_customer(r)
    m = draft_message(r, "Mens E-Mail")
    assert m["source"] == "template" and "men's" in m["subject"] and len(m["body"]) > 50
    assert draft_message(r, "No E-Mail")["subject"] == "(no email)"
    feats = {"womens": 1, "recency": 3.0, "zip_code_Urban": 1}
    assert _satisfies("womens = yes and recency <= 9", feats) and not _satisfies("womens = no", feats)
    assert _satisfies("zip_code_Urban = yes", feats) and not _satisfies("zip_code_Rural = yes", feats)
