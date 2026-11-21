"""Run dynamic dengue risk inference for selected wards and time windows.

Predictions come from the trained gradient-boosting model, and each week is
mapped to low/medium/high/critical using ward-specific boundaries learned from
that ward's observed case distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from src.model_features import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        TIER_ORDER,
        assign_risk_tiers,
        load_dataset,
        resolve_boundaries,
    )
except ImportError:  # allow `python src/predict_risk.py`
    from model_features import (  # type: ignore[no-redef]
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        TIER_ORDER,
        assign_risk_tiers,
        load_dataset,
        resolve_boundaries,
    )


DEFAULT_INPUT = "data/bangladesh_dengue_weather_lags.csv"
DEFAULT_MODEL_DIR = "models"


def load_artifacts(model_dir: str | Path) -> tuple[object, dict, pd.DataFrame, dict[str, float]]:
    """Load the model, metadata, and per-ward risk boundaries."""
    directory = Path(model_dir)
    model_path = directory / "dengue_risk_model.joblib"
    metadata_path = directory / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {directory}. Run `python -m src.train_model` first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cutoffs = pd.DataFrame.from_dict(metadata["cutoffs"], orient="index")
    cutoffs.index.name = "ward_id"
    cutoffs = cutoffs[["q75", "q90", "q97"]].astype(float)
    return model, metadata, cutoffs, metadata["global_cutoffs"]


def highest_tier(tiers: pd.Series) -> str:
    """Return the most severe tier present in a window."""
    ranks = {tier: rank for rank, tier in enumerate(TIER_ORDER)}
    return max(tiers, key=lambda tier: ranks[tier])


def predict_window(
    input_path: str | Path,
    model_dir: str | Path,
    wards: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Score every week in the requested ward/window with the trained model."""
    model, metadata, cutoffs, global_cutoffs = load_artifacts(model_dir)
    frame = load_dataset(input_path)

    known = set(frame["ward_id"].unique())
    unknown = sorted(set(wards).difference(known))
    if unknown:
        available = ", ".join(sorted(known)[:5])
        raise ValueError(
            f"Unknown ward(s): {', '.join(unknown)}. Example valid wards: {available}"
        )

    selected = frame[frame["ward_id"].isin(wards)].copy()

    if start is not None:
        selected = selected[selected["date"] >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected["date"] <= pd.Timestamp(end)]
    if selected.empty:
        raise ValueError("No rows remain after applying the requested time window")

    selected = selected.sort_values(["ward_id", "date"]).reset_index(drop=True)
    predictions = np.clip(
        model.predict(selected[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0, None
    )
    boundaries = resolve_boundaries(selected["ward_id"], cutoffs, global_cutoffs)

    result = pd.DataFrame(
        {
            "date": selected["date"].dt.strftime("%Y-%m-%d"),
            "ward_id": selected["ward_id"],
            "city": selected["city"],
            "predicted_cases": np.round(predictions, 2),
            "risk_ratio": np.round(predictions / boundaries["q90"].to_numpy(), 2),
            "risk_tier": assign_risk_tiers(
                predictions, selected["ward_id"], cutoffs, global_cutoffs
            ),
        }
    )
    if TARGET_COLUMN in selected.columns:
        result["actual_cases"] = selected[TARGET_COLUMN].to_numpy()
        result["actual_tier"] = assign_risk_tiers(
            selected[TARGET_COLUMN].to_numpy(dtype=float),
            selected["ward_id"],
            cutoffs,
            global_cutoffs,
        )
    return result


def print_summary(result: pd.DataFrame) -> None:
    """Print per-ward window summaries and the detailed weekly table."""
    print("Per-ward window summary")
    print("-" * 78)
    for ward, group in result.groupby("ward_id", sort=True):
        counts = group["risk_tier"].value_counts()
        breakdown = "  ".join(
            f"{tier}={int(counts.get(tier, 0))}" for tier in TIER_ORDER
        )
        print(
            f"{ward:14s} window_tier={highest_tier(group['risk_tier']):8s} "
            f"max_pred={group['predicted_cases'].max():6.2f} "
            f"mean_pred={group['predicted_cases'].mean():6.2f}  {breakdown}"
        )

    print("\nWeekly predictions")
    print("-" * 78)
    print(result.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ward", nargs="+", required=True, help="One or more ward_id values"
    )
    parser.add_argument("--start", help="Window start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Window end date (YYYY-MM-DD)")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", type=Path, help="Optional CSV for weekly results")
    args = parser.parse_args()

    result = predict_window(
        input_path=args.input,
        model_dir=args.model_dir,
        wards=args.ward,
        start=args.start,
        end=args.end,
    )
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"\nWrote {len(result):,} weekly predictions to {args.output}")


if __name__ == "__main__":
    main()
