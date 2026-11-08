"""Cross-fitting: honest uplift scores for EVERY customer, and uplift-aware model selection.

Why this exists. Tuning an uplift model by the log-loss of its outcome models (notebook 03)
answers "do the probabilities fit?", not "does the ranking find the persuadables?". A model
can predict visits well and still rank uplift badly, because the uplift is a small
difference between two noisy predictions. So, on the TRAINING set only, we also compare
candidate models by the metric we actually care about: the cross-fitted Qini coefficient.

Cross-fitting = K-fold: each fold is scored by a model trained on the other folds, so every
customer gets an out-of-sample score. On the full 64,000 customers (notebook 05) this gives
an evaluation set 3.3x larger than the 30% test set, hence much tighter confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ARM_SHORT, SEED, TREATED_ARMS
from .evaluation import qini_coefficient
from .preprocessing import FEATURES, build_preprocessor


def cross_fit_uplift(data: pd.DataFrame, make_learner, outcome: str = "visit", n_folds: int = 5,
                     seed: int = SEED, fit_kw: dict | None = None) -> pd.DataFrame:
    """Out-of-fold uplift scores (uplift_mens, uplift_womens) for every row of `data`.

    make_learner() must return an object with fit(X, y, arm, **fit_kw) and predict_uplift(X).
    The one-hot encoder is re-fitted inside each fold on the fitting part only (no leakage).
    Folds are stratified on arm x outcome so every fold keeps the experiment's structure.
    """
    fit_kw = fit_kw or {}
    strata = data["arm"].astype(str) + "_" + data[outcome].astype(str)
    out = pd.DataFrame(np.nan, index=data.index, columns=[f"uplift_{ARM_SHORT[a]}" for a in TREATED_ARMS])
    for fit_idx, pred_idx in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(data, strata):
        fit_part, pred_part = data.iloc[fit_idx], data.iloc[pred_idx]
        pre = build_preprocessor().fit(fit_part[FEATURES])
        cols = pre.get_feature_names_out()
        X_fit = pd.DataFrame(pre.transform(fit_part[FEATURES]), index=fit_part.index, columns=cols)
        X_pred = pd.DataFrame(pre.transform(pred_part[FEATURES]), index=pred_part.index, columns=cols)
        learner = make_learner().fit(X_fit, fit_part[outcome].to_numpy(), fit_part["arm"].to_numpy(), **fit_kw)
        out.loc[pred_part.index] = learner.predict_uplift(X_pred).to_numpy()
    return out


def cross_fit_single(data: pd.DataFrame, make_model, target: str = "visit", n_folds: int = 5, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predicted probability of `target` from a plain classifier (the propensity baseline)."""
    strata = data["arm"].astype(str) + "_" + data[target].astype(str)
    out = np.full(len(data), np.nan)
    for fit_idx, pred_idx in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(data, strata):
        fit_part, pred_part = data.iloc[fit_idx], data.iloc[pred_idx]
        pre = build_preprocessor().fit(fit_part[FEATURES])
        model = make_model().fit(pre.transform(fit_part[FEATURES]), fit_part[target].to_numpy())
        out[pred_idx] = model.predict_proba(pre.transform(pred_part[FEATURES]))[:, 1]
    return out


def qini_per_email(data: pd.DataFrame, scores: dict, outcome: str = "visit") -> pd.DataFrame:
    """Qini coefficient of several score columns, for each email vs control.

    scores = {model name: DataFrame with uplift_mens/uplift_womens, or a 1-d array used for both emails}.
    """
    rows = []
    arm, y = data["arm"].to_numpy(), data[outcome].to_numpy()
    for a in TREATED_ARMS:
        m = np.isin(arm, [0, a])
        t = (arm[m] == a).astype(int)
        for name, s in scores.items():
            col = s[f"uplift_{ARM_SHORT[a]}"].to_numpy() if isinstance(s, pd.DataFrame) else np.asarray(s)
            rows.append({"email": ARM_SHORT[a], "model": name, "qini": qini_coefficient(y[m], t, col[m])})
    return pd.DataFrame(rows).pivot(index="model", columns="email", values="qini").loc[list(scores)]


def _simplicity(params: dict) -> tuple:
    """Sort key: simplest tree first (strict significance gate, shallow, big leaves)."""
    return (-params.get("min_t", 0.0), params.get("max_depth", 0), -params.get("min_leaf", 0))


def select_tree_params(train: pd.DataFrame, grid: list[dict], n_folds: int = 5, seed: int = SEED,
                       n_boot: int = 100, outcome: str = "visit") -> tuple[dict, pd.DataFrame]:
    """Uplift-aware tuning of the honest uplift tree, separately for each email.

    For every candidate setting: 5-fold cross-fitted scores on the training set and the Qini
    coefficient they achieve. Then the **one-standard-error rule** (as in CART pruning): among
    the settings whose Qini is within one bootstrap standard error of the best, take the
    simplest (strictest significance gate, then shallowest, then largest leaves). This avoids
    picking a complex tree whose tiny edge is noise.

    Returns ({arm: params}, table with one row per setting and the Qini per email).
    """
    from .evaluation import bootstrap
    from .models import UpliftTreeLearner

    arm, y = train["arm"].to_numpy(), train[outcome].to_numpy()
    rows, oofs = [], []
    for params in grid:
        oof = cross_fit_uplift(train, lambda p=params: UpliftTreeLearner(**p), n_folds=n_folds, seed=seed, outcome=outcome)
        oofs.append(oof)
        q = qini_per_email(train, {"tree": oof}, outcome=outcome).iloc[0]
        rows.append({**params, "qini_mens": q["mens"], "qini_womens": q["womens"]})
    table = pd.DataFrame(rows)
    chosen = {}
    for a in TREATED_ARMS:
        col = f"qini_{ARM_SHORT[a]}"
        i_best = int(table[col].idxmax())
        m = np.isin(arm, [0, a])
        se = bootstrap(qini_coefficient, y[m], (arm[m] == a).astype(int),
                       oofs[i_best][f"uplift_{ARM_SHORT[a]}"].to_numpy()[m], n_boot=n_boot, seed=seed)["se"]
        ok = table.index[table[col] >= table.loc[i_best, col] - se]
        i_pick = min(ok, key=lambda i: _simplicity(grid[i]))
        chosen[a] = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in grid[i_pick].items()}
        table[f"within_1se_{ARM_SHORT[a]}"] = table.index.isin(ok)
        table[f"chosen_{ARM_SHORT[a]}"] = table.index == i_pick
    return chosen, table
