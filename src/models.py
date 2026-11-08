"""Propensity baseline, uplift meta-learners (T-, X-, S-learner) and an honest uplift tree, written by hand.

Why hand-written: the multi-treatment T- and X-learner are each ~30 lines and the
honest uplift tree ~80, which keeps the logic visible for the defense and avoids heavy
dependencies (causalml / econml) that often fail to install on Windows.

All learners share the same base model: scikit-learn's HistGradientBoosting,
a fast gradient-boosted tree model that handles mixed features without scaling.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import ARMS, ARM_SHORT, SEED, TREATED_ARMS

# Small grid: "light tuning". Each combination is scored with 3-fold CV on TRAIN only.
PARAM_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [50, 200],
}


def base_classifier(**params) -> HistGradientBoostingClassifier:
    """Gradient boosting with early stopping (stops adding trees when validation loss stalls)."""
    return HistGradientBoostingClassifier(max_iter=300, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=SEED, **params)


def base_regressor(**params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=SEED, **params)


def tune_classifier(X, y, cv: int = 3) -> tuple[HistGradientBoostingClassifier, dict]:
    """Pick hyper-parameters by cross-validated log-loss (a calibration-aware score:
    uplift = difference of two probabilities, so the probabilities must be good,
    not just the ranking)."""
    search = GridSearchCV(base_classifier(), PARAM_GRID, scoring="neg_log_loss",
                          cv=StratifiedKFold(cv, shuffle=True, random_state=SEED), n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# ------------------------------------------------------------------ baseline
class PropensityModel:
    """Classic 'response' model: P(visit | features), ignoring who got an email.

    It answers 'who is likely to visit?', not 'whose visit is CAUSED by the email?'.
    That difference is the whole point of the project.
    """

    def fit(self, X, y, tune: bool = True):
        self.model_, self.params_ = tune_classifier(X, y) if tune else (base_classifier().fit(X, y), {})
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ T-learner
class TLearner:
    """Multi-treatment T-learner ("two-model" approach, here three models).

    Fit one outcome model per arm on that arm's customers only:
        mu_0(x) = P(y=1 | x, no email), mu_1(x) = P(y=1 | x, mens), mu_2(x) = P(y=1 | x, womens)
    Estimated uplift of email m for customer x:  tau_m(x) = mu_m(x) - mu_0(x).
    """

    def __init__(self, params: dict | None = None):
        """params: optional {arm: hyper-parameters} fixed in advance (then fit() does not tune)."""
        self.params = params

    def fit(self, X, y, arm, tune: bool = True):
        X, y, arm = pd.DataFrame(X), np.asarray(y), np.asarray(arm)
        self.models_, self.params_ = {}, {}
        for a in ARMS:
            m = arm == a
            if self.params is not None:
                self.params_[a] = self.params[a]
                self.models_[a] = base_classifier(**self.params[a]).fit(X[m], y[m])
            elif tune:
                self.models_[a], self.params_[a] = tune_classifier(X[m], y[m])
            else:
                self.models_[a] = base_classifier().fit(X[m], y[m])
        return self

    def predict_outcomes(self, X) -> pd.DataFrame:
        """Predicted response probability under each arm."""
        names = {0: "p_control", 1: "p_mens", 2: "p_womens"}
        return pd.DataFrame({names[a]: self.models_[a].predict_proba(X)[:, 1] for a in ARMS},
                            index=getattr(X, "index", None))

    def predict_uplift(self, X) -> pd.DataFrame:
        p0 = self.models_[0].predict_proba(X)[:, 1]
        return pd.DataFrame({f"uplift_{ARM_SHORT[a]}": self.models_[a].predict_proba(X)[:, 1] - p0 for a in TREATED_ARMS},
                            index=getattr(X, "index", None))


# ------------------------------------------------------------------ X-learner
class XLearner:
    """Multi-treatment X-learner (Kuenzel et al., 2019), one treated arm vs control at a time.

    Stage 1: the same per-arm outcome models as the T-learner.
    Stage 2: impute each customer's individual effect using the OTHER arm's model:
        treated in arm m:  D = y - mu_0(x)      (what they did minus what control-model predicts)
        control:           D = mu_m(x) - y      (what arm-m model predicts minus what they did)
      then fit a regressor tau_m^T on treated D's and tau_m^C on control D's.
    Stage 3: blend: tau_m(x) = g * tau_m^C(x) + (1 - g) * tau_m^T(x),
      with g = share of arm m among (arm m + control). Randomization makes g known (~0.5).
    Stage 2 smooths the effect directly, which helps when the effect is simpler than the outcome.
    """

    def __init__(self, stage1: TLearner | None = None, reg_params: dict | None = None):
        self.stage1 = stage1
        self.reg_params = reg_params or {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200}

    def fit(self, X, y, arm, tune: bool = True):
        X, y, arm = pd.DataFrame(X), np.asarray(y, dtype=float), np.asarray(arm)
        if self.stage1 is None:
            self.stage1 = TLearner().fit(X, y, arm, tune=tune)
        mu = {a: self.stage1.models_[a] for a in ARMS}
        self.tau_t_, self.tau_c_, self.g_ = {}, {}, {}
        for a in TREATED_ARMS:
            t, c = arm == a, arm == 0
            d_t = y[t] - mu[0].predict_proba(X[t])[:, 1]
            d_c = mu[a].predict_proba(X[c])[:, 1] - y[c]
            self.tau_t_[a] = base_regressor(**self.reg_params).fit(X[t], d_t)
            self.tau_c_[a] = base_regressor(**self.reg_params).fit(X[c], d_c)
            self.g_[a] = t.sum() / (t.sum() + c.sum())
        return self

    def predict_uplift(self, X) -> pd.DataFrame:
        out = {}
        for a in TREATED_ARMS:
            g = self.g_[a]
            out[f"uplift_{ARM_SHORT[a]}"] = g * self.tau_c_[a].predict(X) + (1 - g) * self.tau_t_[a].predict(X)
        return pd.DataFrame(out, index=getattr(X, "index", None))


# ------------------------------------------------------------------ S-learner
class SLearner:
    """Single model with the arm as an extra feature; uplift = f(x, arm=m) - f(x, arm=0).

    Simple, but trees can ignore the arm feature when its effect is small,
    which shrinks uplift towards zero. Included only as a comparison.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def _with_arm(self, X, a):
        Z = pd.DataFrame(X).copy()
        for b in TREATED_ARMS:
            Z[f"arm_{b}"] = float(a == b)
        return Z

    def fit(self, X, y, arm):
        Z = pd.DataFrame(X).copy()
        for b in TREATED_ARMS:
            Z[f"arm_{b}"] = (np.asarray(arm) == b).astype(float)
        self.model_ = base_classifier(**self.params).fit(Z, y)
        return self

    def predict_uplift(self, X) -> pd.DataFrame:
        p0 = self.model_.predict_proba(self._with_arm(X, 0))[:, 1]
        return pd.DataFrame({f"uplift_{ARM_SHORT[a]}": self.model_.predict_proba(self._with_arm(X, a))[:, 1] - p0
                             for a in TREATED_ARMS}, index=getattr(X, "index", None))


# ------------------------------------------------------------------ honest uplift tree
class HonestUpliftTree:
    """Hand-written uplift tree for ONE email vs control (the 'causal tree' stretch goal).

    Idea (Rzepakowski & Jaroszewicz 2012; Athey & Imbens 2016): instead of predicting
    the outcome and subtracting, split the customers directly where the *uplift*
    (treated visit rate − control visit rate) differs most between the two children:

        gain(split) = n_left · n_right / n · (uplift_left − uplift_right)²

    Three safeguards against chasing noise, which is the main risk in uplift modeling:
      * **Honesty**: one random half of the training data chooses the splits, the OTHER
        half estimates the uplift in each leaf. A leaf's estimate is then an ordinary
        difference of means on data that did not pick the split.
      * **Large leaves**: `min_leaf` customers per leaf and `min_leaf_arm` per arm.
      * **Significance gate**: a split is accepted only if the two children's uplifts differ
        by at least `min_t` standard errors (default 2, i.e. roughly a 95% test). With
        min_t = 0 the tree splits whenever the gain is positive, like a classic uplift tree.

    The result is a handful of plain-language rules ("past women's buyers: +7 pts,
    others: +2 pts") that a marketing team can read, check and act on.
    """

    def __init__(self, max_depth: int = 3, min_leaf: int = 1000, min_leaf_arm: int = 200, min_t: float = 2.0,
                 n_thresholds: int = 16, honest: bool = True, random_state: int = SEED):
        self.max_depth, self.min_leaf, self.min_leaf_arm, self.min_t = max_depth, min_leaf, min_leaf_arm, min_t
        self.n_thresholds, self.honest, self.random_state = n_thresholds, honest, random_state

    @staticmethod
    def _uplift(y, t) -> float:
        n_t, n_c = t.sum(), len(t) - t.sum()
        return (y[t == 1].mean() if n_t else 0.0) - (y[t == 0].mean() if n_c else 0.0)

    @staticmethod
    def _se2(y, t) -> float:
        """Squared standard error of the uplift (Welch): var_t/n_t + var_c/n_c."""
        y_t, y_c = y[t == 1], y[t == 0]
        return y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c)

    def _best_split(self, X, y, t):
        n, best = len(y), (0.0, None, None)
        for j in range(X.shape[1]):
            x = X[:, j]
            for thr in np.unique(np.quantile(x, np.linspace(0, 1, self.n_thresholds + 2)[1:-1])):
                left = x <= thr
                n_l = int(left.sum())
                if min(n_l, n - n_l) < self.min_leaf:
                    continue
                y_l, t_l, y_r, t_r = y[left], t[left], y[~left], t[~left]
                if min(t_l.sum(), len(t_l) - t_l.sum(), t_r.sum(), len(t_r) - t_r.sum()) < self.min_leaf_arm:
                    continue
                diff = self._uplift(y_l, t_l) - self._uplift(y_r, t_r)
                if self.min_t > 0 and abs(diff) < self.min_t * math.sqrt(self._se2(y_l, t_l) + self._se2(y_r, t_r)):
                    continue
                gain = n_l * (n - n_l) / n * diff ** 2
                if gain > best[0]:
                    best = (gain, j, float(thr))
        return best

    def _grow(self, X, y, t, depth) -> dict:
        node = {"uplift": self._uplift(y, t), "n": int(len(y))}
        if depth < self.max_depth:
            _, j, thr = self._best_split(X, y, t)
            if j is not None:
                left = X[:, j] <= thr
                node.update(feature=int(j), thr=thr, left=self._grow(X[left], y[left], t[left], depth + 1),
                            right=self._grow(X[~left], y[~left], t[~left], depth + 1))
        return node

    def _estimate(self, node, X, y, t, idx):
        """Honest step: re-estimate every node's uplift on the estimation half."""
        if len(idx):
            node["uplift"], node["n"] = self._uplift(y[idx], t[idx]), int(len(idx))
        if "feature" in node:
            left = X[idx, node["feature"]] <= node["thr"]
            self._estimate(node["left"], X, y, t, idx[left])
            self._estimate(node["right"], X, y, t, idx[~left])

    def fit(self, X, y, t):
        self.feature_names_ = list(getattr(X, "columns", [f"x{i}" for i in range(np.shape(X)[1])]))
        X, y, t = np.asarray(X, float), np.asarray(y, float), np.asarray(t, int)
        self.binary_ = {n for j, n in enumerate(self.feature_names_) if set(np.unique(X[:, j])) <= {0.0, 1.0}}
        if self.honest:
            idx = np.random.default_rng(self.random_state).permutation(len(y))
            split_half, est_half = idx[: len(y) // 2], idx[len(y) // 2:]
            self.tree_ = self._grow(X[split_half], y[split_half], t[split_half], 0)
            self._estimate(self.tree_, X, y, t, est_half)
        else:
            self.tree_ = self._grow(X, y, t, 0)
        return self

    def predict(self, X) -> np.ndarray:
        X, out = np.asarray(X, float), np.empty(np.shape(X)[0])

        def walk(node, idx):
            if "feature" not in node or len(idx) == 0:
                out[idx] = node["uplift"]
                return
            left = X[idx, node["feature"]] <= node["thr"]
            walk(node["left"], idx[left])
            walk(node["right"], idx[~left])

        walk(self.tree_, np.arange(len(X)))
        return out

    def rules(self) -> pd.DataFrame:
        """One row per leaf: plain-language condition, estimated uplift, customers in the leaf."""
        rows = []

        def cond(name, thr, is_left):
            if name in self.binary_:
                return f"{name} = {'no' if is_left else 'yes'}"
            return f"{name} <= {thr:g}" if is_left else f"{name} > {thr:g}"

        def walk(node, path):
            if "feature" not in node:
                rows.append({"rule": " and ".join(path) or "everyone", "uplift": node["uplift"], "n": node["n"]})
                return
            name = self.feature_names_[node["feature"]]
            walk(node["left"], path + [cond(name, node["thr"], True)])
            walk(node["right"], path + [cond(name, node["thr"], False)])

        walk(self.tree_, [])
        return pd.DataFrame(rows).sort_values("uplift", ascending=False).reset_index(drop=True)


class UpliftTreeLearner:
    """One honest uplift tree per email (men's vs control, women's vs control); same interface
    as the meta-learners (predict_uplift -> uplift_mens, uplift_womens)."""

    def __init__(self, params: dict | None = None, **tree_kw):
        """params: optional per-arm settings {1: {...}, 2: {...}} (from uplift-aware tuning);
        tree_kw: defaults for every arm."""
        self.params, self.tree_kw = params or {}, tree_kw

    def fit(self, X, y, arm):
        X, y, arm = pd.DataFrame(X), np.asarray(y, float), np.asarray(arm)
        self.trees_ = {}
        for a in TREATED_ARMS:
            m = np.isin(arm, [0, a])
            kw = {**self.tree_kw, **self.params.get(a, self.params.get(str(a), {}))}
            self.trees_[a] = HonestUpliftTree(**kw).fit(X[m], y[m], (arm[m] == a).astype(int))
        return self

    def predict_uplift(self, X) -> pd.DataFrame:
        return pd.DataFrame({f"uplift_{ARM_SHORT[a]}": self.trees_[a].predict(X) for a in TREATED_ARMS},
                            index=getattr(X, "index", None))

    def rules(self) -> dict:
        return {ARMS[a]: self.trees_[a].rules() for a in TREATED_ARMS}


# ------------------------------------------------------------------ decisions
def recommend(uplift: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Best email per customer = the arm with the larger estimated uplift; 'No E-Mail'
    if neither beats `threshold`. Also returns the policy score (= best uplift) used
    to rank customers for 'target the top K%'."""
    u = uplift[["uplift_mens", "uplift_womens"]].to_numpy()
    best = np.where(u[:, 0] >= u[:, 1], 1, 2)
    score = u.max(axis=1)
    action_arm = np.where(score > threshold, best, 0)
    return pd.DataFrame({"policy_score": score, "best_arm": best, "action_arm": action_arm,
                         "recommended_action": [ARMS[a] for a in action_arm]}, index=uplift.index)


def model_inferred_segment(p_control: np.ndarray, uplift: np.ndarray, base_rate: float, eps: float = 0.01) -> np.ndarray:
    """MODEL-INFERRED targeting categories (never observed truth):
       persuadable  - estimated uplift > eps
       sleeping dog - estimated uplift < -eps
       sure thing   - small uplift, likely to visit anyway (p_control >= base rate)
       lost cause   - small uplift, unlikely to visit either way
    """
    seg = np.where(uplift > eps, "persuadable",
          np.where(uplift < -eps, "sleeping dog",
          np.where(p_control >= base_rate, "sure thing", "lost cause")))
    return seg
