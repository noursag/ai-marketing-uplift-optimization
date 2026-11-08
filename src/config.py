"""Project-wide settings: paths, seed, arm names and the economic SCENARIO ASSUMPTIONS.

Everything that is a choice (not data) lives here, so it is easy to find and change.
"""
from pathlib import Path

SEED = 42  # fixed seed everywhere -> every run gives the same split, models and numbers

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# Treatment arms. Integer codes make the maths simple; 0 is always the control group.
ARMS = {0: "No E-Mail", 1: "Mens E-Mail", 2: "Womens E-Mail"}
ARM_CODES = {v: k for k, v in ARMS.items()}
TREATED_ARMS = [1, 2]
ARM_SHORT = {1: "mens", 2: "womens"}

PRIMARY_OUTCOME = "visit"          # see README / guide: conversion is too sparse (~0.9%)
SECONDARY_OUTCOMES = ["conversion", "spend"]
TEST_SIZE = 0.30
N_BOOT = 300                        # bootstrap resamples for confidence intervals

# ---------------------------------------------------------------------------
# SCENARIO ASSUMPTIONS (NOT DATA). The Hillstrom data has no costs or margins.
# These defaults were fixed before looking at test results; the dashboard and
# the sensitivity table let you change them.
# ---------------------------------------------------------------------------
COST_PER_CONTACT = 0.15   # $ per email sent (production + platform + small incentive)
MARGIN_RATE = 0.30        # gross margin as a share of incremental spend
COST_GRID = [0.05, 0.15, 0.30, 0.50]  # used for the sensitivity table
