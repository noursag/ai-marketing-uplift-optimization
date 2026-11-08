"""Campaign economics: turn held-out incremental estimates into money under SCENARIO ASSUMPTIONS.

incremental profit = incremental spend x margin rate  -  cost per contact x customers contacted

Incremental spend comes from the held-out experiment (see evaluation.policy_incremental_at_k);
margin rate and cost per contact are assumptions (src/config.py), not data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import policy_incremental_at_k

K_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def strategy_table(test: pd.DataFrame, strategies: dict, outcomes=("visit", "conversion", "spend"),
                   k_grid=K_GRID, expected_random: bool = True) -> pd.DataFrame:
    """For each strategy and share targeted k: incremental outcome per targeted customer.

    strategies = {name: (score, policy_arm)}; score ranks customers, policy_arm says
    which email each one would get (1 mens, 2 womens).
    """
    rows = []
    for name, (score, parm) in strategies.items():
        for k in k_grid:
            # Random targeting: in expectation, any random K% has the same per-customer effect
            # as the whole population, so we use the k = 100% estimate for every k. Using one
            # particular random order would only add noise to the curve.
            k_eval = 1.0 if (name == "Random" and expected_random) else k
            row = {"strategy": name, "k": float(k)}
            for out in outcomes:
                row[f"inc_{out}_per_targeted"] = policy_incremental_at_k(test[out], test["arm"], parm, score, k_eval)
            rows.append(row)
    return pd.DataFrame(rows)


def add_economics(table: pd.DataFrame, population: int, cost: float, margin: float) -> pd.DataFrame:
    """Scale per-customer estimates to a population and compute profit."""
    t = table.copy()
    t["n_contacted"] = np.ceil(t["k"] * population)
    for out in ("visit", "conversion", "spend"):
        if f"inc_{out}_per_targeted" in t:
            t[f"inc_{out}"] = t[f"inc_{out}_per_targeted"] * t["n_contacted"]
    t["campaign_cost"] = cost * t["n_contacted"]
    t["incremental_profit"] = t["inc_spend"] * margin - t["campaign_cost"]
    return t




def best_threshold(econ: pd.DataFrame) -> pd.DataFrame:
    """Find the most profitable targeting percentage for each strategy.

    Works with both observed-spend economics and visit-value economics.
    If all percentages lose money, select no campaign.
    """
    cols = [
        "strategy",
        "k",
        "n_contacted",
        "campaign_cost",
        "incremental_profit",
    ]

    if "inc_spend" in econ.columns:
        cols.append("inc_spend")

    if "inc_visits" in econ.columns:
        cols.append("inc_visits")

    best = econ.loc[
        econ.groupby("strategy")["incremental_profit"].idxmax(),
        cols
    ].reset_index(drop=True)

    losing = best["incremental_profit"] <= 0

    numeric_cols = [
        c for c in cols if c != "strategy"
    ]

    best.loc[losing, numeric_cols] = 0.0

    return best


def budget_cap(k: float, budget: float, cost: float, population: int) -> float:
    """Largest share we can afford: min(chosen share, budget / (cost x population))."""
    if cost <= 0:
        return k
    return float(min(k, budget / (cost * population)))


# ---------------------------------------------------------------------------------
# Visit-value bridge: a less noisy road from visits to money.
#
# Spend is $0 for 99% of customers, so "incremental spend inside a targeted group" is
# extremely noisy. A standard marketing shortcut is to value an incremental VISIT at the
# average spend per visit observed in the data, i.e. assume an email-induced visitor is
# worth about as much as an ordinary visitor. The data lets us check that assumption
# (spend per visitor is similar in every arm) and the number is conservative: observed
# incremental spend per customer is higher than incremental visits x value per visit.
# ---------------------------------------------------------------------------------
def value_per_visit(data: pd.DataFrame) -> dict:
    """Average spend per visitor, overall and per arm (a data-derived constant, not a model)."""
    v = data[data["visit"] == 1]
    out = {"pooled": float(v["spend"].mean())}
    for seg, g in v.groupby("segment"):
        out[seg] = float(g["spend"].mean())
    return out


def profit_from_visits(inc_visit_per_targeted, n_contacted, value: float, margin: float, cost: float):
    """Incremental profit = extra visits x value per visit x margin − cost x contacts."""
    return np.asarray(inc_visit_per_targeted) * np.asarray(n_contacted) * value * margin - cost * np.asarray(n_contacted)


def break_even_cost(inc_visit_per_targeted, value: float, margin: float):
    """Cost per contact at which a strategy stops making money: uplift x value x margin."""
    return np.asarray(inc_visit_per_targeted) * value * margin


def add_visit_economics(table: pd.DataFrame, population: int, value: float, margin: float, cost: float) -> pd.DataFrame:
    """Like add_economics, but money comes from incremental VISITS via the value-per-visit bridge."""
    t = table.copy()
    t["n_contacted"] = np.ceil(t["k"] * population)
    t["inc_visits"] = t["inc_visit_per_targeted"] * t["n_contacted"]
    t["campaign_cost"] = cost * t["n_contacted"]
    t["gross_profit"] = t["inc_visits"] * value * margin
    t["incremental_profit"] = t["gross_profit"] - t["campaign_cost"]
    t["break_even_cost"] = break_even_cost(t["inc_visit_per_targeted"], value, margin)
    t["cost_per_inc_visit"] = np.where(t["inc_visits"] > 0, t["campaign_cost"] / t["inc_visits"], np.nan)
    return t


def decision_map(table: pd.DataFrame, population: int, value: float, margin: float, costs) -> pd.DataFrame:
    """For each cost per contact: the best (strategy, share) and its profit, next to 'email everyone'.
    Shows WHEN targeting pays: cheap contacts -> mail everyone; expensive contacts -> target."""
    rows = []
    for c in costs:
        e = add_visit_economics(table, population, value, margin, c)
        best = e.loc[e["incremental_profit"].idxmax()]
        everyone = e[(e["strategy"] == "Random") & (e["k"] == 1.0)].iloc[0]
        profitable = best["incremental_profit"] > 0
        rows.append({"cost_per_contact": float(c), "best_strategy": best["strategy"] if profitable else "No campaign",
                     "best_k": float(best["k"]) if profitable else 0.0,
                     "best_profit": float(max(best["incremental_profit"], 0.0)),
                     "everyone_profit": float(everyone["incremental_profit"]),
                     "gain_vs_everyone": float(max(best["incremental_profit"], 0.0) - everyone["incremental_profit"])})
    return pd.DataFrame(rows)
