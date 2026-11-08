"""Data loading, fixed train/validation/test split, feature preparation, and randomization checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .config import ARM_CODES, ARMS, PROCESSED_DIR, RAW_DIR, SEED

RAW_FILE = RAW_DIR / "hillstrom.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
VAL_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]
FEATURES = NUMERIC + CATEGORICAL
OUTCOMES = ["visit", "conversion", "spend"]

_FALLBACK_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"


def load_hillstrom(refresh: bool = False) -> pd.DataFrame:
    """Load the Hillstrom experiment, caching the original data locally."""
    if RAW_FILE.exists() and not refresh:
        return pd.read_csv(RAW_FILE)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from sklift.datasets import fetch_hillstrom
        parts = {t: fetch_hillstrom(target_col=t) for t in OUTCOMES}
        df = parts["visit"].data.copy()
        df["segment"] = parts["visit"].treatment.values
        for t in OUTCOMES:
            df[t] = parts[t].target.values
    except ImportError:
        df = pd.read_csv(_FALLBACK_URL)
    df = df[FEATURES + ["history_segment", "segment"] + OUTCOMES]
    df.to_csv(RAW_FILE, index=False)
    return df


def add_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Add treatment codes: 0 = control, 1 = men's email, 2 = women's email."""
    df = df.copy()
    df["arm"] = df["segment"].map(ARM_CODES).astype(int)
    return df


def make_split(df: pd.DataFrame | None = None, refresh: bool = False):
    """Return a fixed 60/20/20 train/validation/test split.

    Stratify on treatment arm and visit outcome. Cache all three sets so every
    notebook uses identical rows. Keep final test outcomes out of model and
    targeting-threshold selection.

    Returns (train, validation, test).
    """
    if not refresh and all(p.exists() for p in (TRAIN_FILE, VAL_FILE, TEST_FILE)):
        return tuple(
            pd.read_csv(p, index_col="customer_id")
            for p in (TRAIN_FILE, VAL_FILE, TEST_FILE)
        )

    if df is None:
        df = load_hillstrom()
    df = add_arm(df)
    df.index.name = "customer_id"  # Original row index; not a real customer ID.

    strata = df["arm"].astype(str) + "_" + df["visit"].astype(str)
    train, remainder = train_test_split(
        df, test_size=0.40, stratify=strata, random_state=SEED
    )
    remainder_strata = (
        remainder["arm"].astype(str) + "_" + remainder["visit"].astype(str)
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder_strata,
        random_state=SEED,
    )

    train, validation, test = (
        part.sort_index() for part in (train, validation, test)
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_FILE)
    validation.to_csv(VAL_FILE)
    test.to_csv(TEST_FILE)
    return train, validation, test


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categorical features; pass numeric features through."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def fit_transform_features(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
):
    """Fit encoder on training data only and transform all three datasets.

    Returns (X_train, X_validation, X_test, preprocessor).
    """
    pre = build_preprocessor().fit(train[FEATURES])
    cols = pre.get_feature_names_out()

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            pre.transform(frame[FEATURES]), index=frame.index, columns=cols
        )

    return transform(train), transform(validation), transform(test), pre


# --------------------------------------------------------------------- randomization check
def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    """SMD = (mean_a - mean_b) / pooled SD; |SMD| < 0.1 is a common balance rule."""
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """SMD of each feature for each email arm versus control."""
    d = add_arm(df) if "arm" not in df else df
    X = pd.get_dummies(d[FEATURES + ["history_segment"]], dtype=float)
    rows = []
    for col in X.columns:
        row = {"feature": col}
        for arm in (1, 2):
            row[f"SMD {ARMS[arm]} vs control"] = standardized_mean_difference(
                X.loc[d.arm == arm, col], X.loc[d.arm == 0, col]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def chi_square_table(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square tests of feature independence from randomized treatment arm."""
    d = add_arm(df) if "arm" not in df else df
    rows = []
    for col in ["history_segment", "zip_code", "channel", "mens", "womens", "newbie"]:
        chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(d[col], d["arm"]))
        rows.append({"feature": col, "chi2": chi2, "dof": dof, "p_value": p})
    return pd.DataFrame(rows)
