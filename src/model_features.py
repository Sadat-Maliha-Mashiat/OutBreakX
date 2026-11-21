"""Shared feature engineering and risk-tier helpers for the dengue risk model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "dengue_cases"

NUMERIC_FEATURES = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "avg_temp_c_lag_2w",
    "rainfall_mm_lag_2w",
    "relative_humidity_pct_lag_2w",
    "avg_temp_c_lag_3w",
    "rainfall_mm_lag_3w",
    "relative_humidity_pct_lag_3w",
    "avg_temp_c_lag_4w",
    "rainfall_mm_lag_4w",
    "relative_humidity_pct_lag_4w",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
]

CATEGORICAL_FEATURES = ["ward_id", "city"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TIER_ORDER = ["low", "medium", "high", "critical"]
DEFAULT_TIERS = ("q75", "q90", "q97")


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical month and ISO-week features from the ``date`` column."""
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    month = dates.dt.month.astype(float)
    week = dates.dt.isocalendar().week.astype(float)

    result["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    result["week_sin"] = np.sin(2 * np.pi * week / 52.18)
    result["week_cos"] = np.cos(2 * np.pi * week / 52.18)
    return result


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calendar features and validate model inputs exist."""
    result = add_calendar_features(frame)
    missing = sorted(set(FEATURE_COLUMNS).difference(result.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    return result


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a prepared weekly CSV with parsed dates and calendar features."""
    frame = pd.read_csv(path, parse_dates=["date"])
    return build_features(frame)


def compute_ward_cutoffs(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-ward case-count quantiles used as dynamic risk-tier boundaries."""
    quantiles = frame.groupby("ward_id")[TARGET_COLUMN].quantile(
        [0.75, 0.90, 0.97]
    ).unstack()
    quantiles.columns = list(DEFAULT_TIERS)
    # Keep the boundaries monotonic even for wards with heavily tied counts.
    quantiles = quantiles.cummax(axis=1)
    return quantiles


def resolve_boundaries(
    ward_ids: pd.Series,
    cutoffs: pd.DataFrame,
    global_cutoffs: dict[str, float],
) -> pd.DataFrame:
    """Return the q75/q90/q97 boundaries for each row's ward."""
    wards = pd.Series(ward_ids).reset_index(drop=True)
    boundaries = cutoffs.reindex(wards).reset_index(drop=True)
    for name in DEFAULT_TIERS:
        boundaries[name] = boundaries[name].fillna(global_cutoffs[name])
    return boundaries


def assign_risk_tiers(
    predicted_cases: np.ndarray | pd.Series,
    ward_ids: pd.Series,
    cutoffs: pd.DataFrame,
    global_cutoffs: dict[str, float],
) -> pd.Series:
    """Map predicted case counts to low/medium/high/critical tiers per ward.

    Boundaries are ward-specific quantiles learned from the training data, so
    the same predicted count can mean different risk in different wards.
    """
    wards = pd.Series(ward_ids).reset_index(drop=True)
    predictions = pd.Series(np.asarray(predicted_cases), index=wards.index)
    boundaries = resolve_boundaries(wards, cutoffs, global_cutoffs)

    tiers = pd.Series("low", index=wards.index, dtype="object")
    tiers[predictions >= boundaries["q75"]] = "medium"
    tiers[predictions >= boundaries["q90"]] = "high"
    tiers[predictions >= boundaries["q97"]] = "critical"
    return tiers
