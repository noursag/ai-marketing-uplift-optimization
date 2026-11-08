"""Uplift evaluation: Qini / uplift curves, AUUC, uplift@K, deciles, bootstrap CIs, policy value.

Key idea: we never see the same customer both with and without the email, so we
cannot score individual predictions. Instead we score RANKINGS: take the top-k
customers by predicted uplift, and compare the treated and control customers
inside that group. Because the email was assigned at random, the two groups are
comparable, so their difference is an unbiased estimate of the uplift of the group.

Conventions: `t` is 1 for treated (the email arm being evaluated) and 0 for control.
Scores are sorted from highest to lowest. Ties are broken at random with a fixed seed:
a model with few distinct scores (the uplift tree has one value per leaf) must not get
its tied customers ordered by their row position, which would be arbitrary and, inside a
bootstrap resample (treated rows first), systematically biased.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _order(score, seed: int = 0) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    tiebreak = np.random.default_rng(seed).random(len(s))
    return np.lexsort((tiebreak, -s))          # sort by -score, then by the random tie-break


def diff_in_means(y_t, y_c) -> dict:
    """Difference of two means with a normal-approximation 95% CI (Welch standard error)."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    if len(y_t) == 0 or len(y_c) == 0:
        return {"diff": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan}
    d = y_t.mean() - y_c.mean()
    se = math.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c)) if min(len(y_t), len(y_c)) > 1 else np.nan
    return {"diff": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


def uplift_curves(y, t, score) -> pd.DataFrame:
    """Cumulative curves when targeting customers in score order.

    For the top k customers (n_t treated, n_c control, with r_t and r_c responders):
      qini(k)   = r_t - r_c * n_t / n_c        (incremental responders, control scaled to treated size)
      uplift(k) = (r_t/n_t - r_c/n_c) * k      (incremental responders if all k were treated)
    A row for k = 0 (nobody targeted, value 0) is included so curves start at the origin.
    """
    o = _order(score)
    y, t = np.asarray(y, float)[o], np.asarray(t, int)[o]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    r_t, r_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    k = np.arange(1, len(y) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = np.where(n_c > 0, r_t - r_c * n_t / n_c, r_t)
        upl = np.where((n_t > 0) & (n_c > 0), (r_t / n_t - r_c / n_c) * k, 0.0)
    return pd.DataFrame({"k": np.r_[0, k], "frac": np.r_[0, k / len(y)],
                         "qini": np.r_[0, qini], "uplift": np.r_[0, upl]})


def qini_coefficient(y, t, score) -> float:
    """Area between the Qini curve and the random-targeting line, both divided by n.

    0 = no better than random; larger = better ranking. Units: incremental responses
    per customer (area over the 0-1 'share targeted' axis).
    """
    c = uplift_curves(y, t, score)
    q = c["qini"].to_numpy() / (len(c) - 1)
    return float(np.trapezoid(q, c["frac"]) - 0.5 * q[-1])


def auuc(y, t, score, relative: bool = True) -> float:
    """Area Under the Uplift Curve (divided by n). With relative=True the area of the
    random line (straight line to the final value) is subtracted, so 0 = random."""
    c = uplift_curves(y, t, score)
    u = c["uplift"].to_numpy() / (len(c) - 1)
    area = float(np.trapezoid(u, c["frac"]))
    return area - 0.5 * u[-1] if relative else area


def uplift_at_k(y, t, score, k: float) -> float:
    """Uplift inside the top k share (0 < k <= 1): treated rate minus control rate."""
    o = _order(score)[: max(1, math.ceil(k * len(y)))]
    y, t = np.asarray(y, float)[o], np.asarray(t, int)[o]
    if t.sum() == 0 or (1 - t).sum() == 0:
        return np.nan
    return float(y[t == 1].mean() - y[t == 0].mean())


def share_of_lift_at_k(y, t, score, k: float) -> float:
    """Share of ALL incremental responders (if everyone were treated) that the top-k share
    captures. Business reading: 'contact 50% of the base, keep 75% of the campaign's effect'.
    Random targeting captures k on average; a perfect ranking captures 1 as soon as all
    persuadables are included."""
    c = uplift_curves(y, t, score)
    total = c["qini"].iloc[-1]
    if total == 0:
        return np.nan
    i = min(len(c) - 1, int(round(k * (len(c) - 1))))
    return float(c["qini"].iloc[i] / total)


def uplift_by_decile(y, t, score, n_bins: int = 10) -> pd.DataFrame:
    """Observed uplift (with 95% CI) in each equal-size bin of the score, bin 1 = highest scores."""
    o = _order(score)
    y, t = np.asarray(y, float)[o], np.asarray(t, int)[o]
    bins = np.arange(len(y)) * n_bins // len(y)
    rows = []
    for b in range(n_bins):
        m = bins == b
        d = diff_in_means(y[m & (t == 1)], y[m & (t == 0)])
        rows.append({"decile": b + 1, "n_treated": int((m & (t == 1)).sum()), "n_control": int((m & (t == 0)).sum()),
                     "rate_treated": y[m & (t == 1)].mean(), "rate_control": y[m & (t == 0)].mean(),
                     "uplift": d["diff"], "ci_lo": d["lo"], "ci_hi": d["hi"]})
    return pd.DataFrame(rows)


def bootstrap(fn, y, t, score, n_boot: int = 300, seed: int = 42, **kw) -> dict:
    """Percentile bootstrap 95% CI for any metric fn(y, t, score, **kw).

    Resamples customers with replacement WITHIN treated and control separately,
    so every resample keeps the experiment's group sizes. Needed because spend is
    mostly zeros with a long tail, so normal formulas are unreliable.
    """
    rng = np.random.default_rng(seed)
    y, t, s = np.asarray(y, float), np.asarray(t, int), np.asarray(score, float)
    it, ic = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(it, len(it)), rng.choice(ic, len(ic))]
        vals.append(fn(y[idx], t[idx], s[idx], **kw))
    vals = np.asarray(vals, float)
    return {"estimate": float(fn(y, t, s, **kw)), "lo": float(np.nanpercentile(vals, 2.5)),
            "hi": float(np.nanpercentile(vals, 97.5)), "se": float(np.nanstd(vals))}


# ---------------------------------------------------------------- multi-arm policy evaluation

def policy_incremental_at_k(y, arm, policy_arm, score, k: float) -> float:
    """IPW estimate of incremental outcome per selected customer.

    The top k fraction follows its recommended arm (0, 1 or 2).
    All other customers receive no email.

    Requires randomized experimental assignment.
    """
    y = np.asarray(y, dtype=float)
    arm = np.asarray(arm, dtype=int)
    pa = np.asarray(policy_arm, dtype=int)
    score = np.asarray(score, dtype=float)

    if not (len(y) == len(arm) == len(pa) == len(score)):
        raise ValueError("Input lengths must match.")
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1].")
    if not np.isin(arm, [0, 1, 2]).all():
        raise ValueError("Invalid experimental arm.")
    if not np.isin(pa, [0, 1, 2]).all():
        raise ValueError("Invalid policy arm.")

    probabilities = np.bincount(arm, minlength=3) / len(arm)
    chosen = _order(score)[:max(1, math.ceil(k * len(y)))]
    recommended = pa[chosen]
    required_arms = np.unique(np.concatenate(([0], recommended)))

    if np.any(probabilities[required_arms] <= 0):
        raise ValueError(
           "Control and every recommended treatment arm must "
           "be represented in the experiment."
        )

    yy = y[chosen]
    aa = arm[chosen]
    recommended = pa[chosen]

    # Value of the recommended policy minus no email.
    policy_value = np.mean(
        (aa == recommended) * yy / probabilities[aa]
    )
    control_value = np.mean(
        (aa == 0) * yy / probabilities[0]
    )

    return float(policy_value - control_value)


def bootstrap_policy(y, arm, policy_arm, score, k: float,
                     n_boot: int = 300, seed: int = 42) -> dict:
    """Arm-stratified bootstrap confidence interval for IPW policy uplift."""
    rng = np.random.default_rng(seed)
    y, arm, pa, score = (np.asarray(v) for v in
                         (y, arm, policy_arm, score))

    groups = [np.flatnonzero(arm == a) for a in (0, 1, 2)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("All three arms must be represented.")

    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(g, len(g), replace=True) for g in groups
        ])
        vals.append(
            policy_incremental_at_k(
                y[idx], arm[idx], pa[idx], score[idx], k
            )
        )

    estimate = policy_incremental_at_k(y, arm, pa, score, k)
    return {
        "estimate": float(estimate),
        "lo": float(np.nanpercentile(vals, 2.5)),
        "hi": float(np.nanpercentile(vals, 97.5)),
    }

def bootstrap_diff(fn, y, t, score_a, score_b, n_boot: int = 300, seed: int = 42, **kw) -> dict:
    """Paired bootstrap of fn(score_a) - fn(score_b) on the SAME resamples.

    Answers 'is model A really better than model B, or is the gap just noise?'.
    If the 95% CI excludes 0, the difference is statistically convincing.
    """
    rng = np.random.default_rng(seed)
    y, t = np.asarray(y, float), np.asarray(t, int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    it, ic = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(it, len(it)), rng.choice(ic, len(ic))]
        vals.append(fn(y[idx], t[idx], a[idx], **kw) - fn(y[idx], t[idx], b[idx], **kw))
    vals = np.asarray(vals, float)
    return {"diff": float(fn(y, t, a, **kw) - fn(y, t, b, **kw)), "lo": float(np.nanpercentile(vals, 2.5)),
            "hi": float(np.nanpercentile(vals, 97.5)), "p_a_better": float(np.mean(vals > 0))}


def bootstrap_policy_diff(
    y, arm, parm_a, score_a, parm_b, score_b,
    k: float, k_b: float | None = None,
    n_boot: int = 300, seed: int = 42
) -> dict:
    """Paired bootstrap: difference between two IPW policy effects."""
    if k_b is None:
        k_b = k

    rng = np.random.default_rng(seed)
    y, arm, pa, sa, pb, sb = (
        np.asarray(v) for v in
        (y, arm, parm_a, score_a, parm_b, score_b)
    )

    groups = [np.flatnonzero(arm == a) for a in (0, 1, 2)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("All three arms must be represented.")

    def difference(idx):
        a = policy_incremental_at_k(
            y[idx], arm[idx], pa[idx], sa[idx], k
        )
        b = policy_incremental_at_k(
            y[idx], arm[idx], pb[idx], sb[idx], k_b
        )
        return a - b

    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(g, len(g), replace=True) for g in groups
        ])
        vals.append(difference(idx))

    vals = np.asarray(vals, dtype=float)

    return {
        "diff": float(difference(np.arange(len(y)))),
        "lo": float(np.nanpercentile(vals, 2.5)),
        "hi": float(np.nanpercentile(vals, 97.5)),
        "p_a_better": float(np.nanmean(vals > 0)),
    }



def evaluate_policy_ipw(y, actual_arm, recommended_arm, assignment_probs):
    """
    Evaluate a fixed marketing policy against no email.

    y: observed outcome (visit, conversion, or spend)
    actual_arm: randomized treatment (0, 1, 2)
    recommended_arm: policy recommendation (0, 1, 2)
    assignment_probs: experimental probabilities for each arm
                      e.g. {0: p0, 1: p1, 2: p2}

    Returns incremental outcome per eligible customer.
    """
    y = np.asarray(y, dtype=float)
    actual_arm = np.asarray(actual_arm, dtype=int)
    recommended_arm = np.asarray(recommended_arm, dtype=int)

    if not (len(y) == len(actual_arm) == len(recommended_arm)):
        raise ValueError("Input lengths must match.")

    if not np.isin(actual_arm, [0, 1, 2]).all():
        raise ValueError("Invalid actual treatment arm.")
    if not np.isin(recommended_arm, [0, 1, 2]).all():
        raise ValueError("Invalid recommended treatment arm.")

    p = np.array([assignment_probs[int(a)] for a in actual_arm])
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Assignment probabilities must be positive.")

    # Value of following the recommended policy
    matches = (actual_arm == recommended_arm)
    policy_value = np.mean(matches * y / p)

    # Value of sending no emails
    control_value = np.mean(
        (actual_arm == 0) * y / p
    )

    return float(policy_value - control_value)
