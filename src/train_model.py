"""Train a dengue risk model and save per-ward dynamic risk-tier boundaries.

The model predicts weekly dengue cases from static ward characteristics, current
weather, and 2-4 week weather lags. Risk tiers are not fixed global thresholds:
each ward's low/medium/high/critical boundaries are learned from that ward's
observed case distribution.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.model_features import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        TIER_ORDER,
        assign_risk_tiers,
        compute_ward_cutoffs,
        load_dataset,
    )
except ImportError:  # allow `python src/train_model.py`
    from model_features import (  # type: ignore[no-redef]
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET_COLUMN,
        TIER_ORDER,
        assign_risk_tiers,
        compute_ward_cutoffs,
        load_dataset,
    )


DEFAULT_INPUT = "data/bangladesh_dengue_weather_lags.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_TEST_START = "2024-01-01"


def build_model(random_state: int = 42) -> TransformedTargetRegressor:
    """Create the preprocessing + gradient-boosting pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate_holdout(
    model: TransformedTargetRegressor,
    test: pd.DataFrame,
    cutoffs: pd.DataFrame,
    global_cutoffs: dict[str, float],
) -> dict:
    """Score the model and its tier assignments on a time-based holdout."""
    actual = test[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = np.clip(model.predict(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0, None)

    predicted_tiers = assign_risk_tiers(predicted, test["ward_id"], cutoffs, global_cutoffs)
    actual_tiers = assign_risk_tiers(actual, test["ward_id"], cutoffs, global_cutoffs)

    q90 = test["ward_id"].map(
        lambda ward: cutoffs.at[ward, "q90"] if ward in cutoffs.index else global_cutoffs["q90"]
    )
    actual_high = (actual >= q90.to_numpy()).astype(int)
    predicted_high = predicted_tiers.isin(["high", "critical"]).astype(int)

    binary_metrics: dict[str, float | None] = {
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
    }
    if actual_high.sum() > 0 and actual_high.sum() < len(actual_high):
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_high, predicted_high, average="binary", zero_division=0
        )
        binary_metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc_score(actual_high, predicted)),
        }

    spearman = spearmanr(actual, predicted)
    tier_counts = predicted_tiers.value_counts().reindex(TIER_ORDER, fill_value=0)

    return {
        "rows": int(len(test)),
        "mae_cases": float(mean_absolute_error(actual, predicted)),
        "rmse_cases": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2_cases": float(r2_score(actual, predicted)),
        "spearman_rho": float(spearman.statistic),
        "tier_accuracy": float(accuracy_score(actual_tiers, predicted_tiers)),
        "tier_macro_f1": float(
            f1_score(actual_tiers, predicted_tiers, labels=TIER_ORDER, average="macro", zero_division=0)
        ),
        "predicted_tier_counts": {tier: int(tier_counts[tier]) for tier in TIER_ORDER},
        "high_risk_detection": binary_metrics,
    }


def save_artifacts(
    model: TransformedTargetRegressor,
    cutoffs: pd.DataFrame,
    global_cutoffs: dict[str, float],
    metadata: dict,
    model_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist the trained pipeline and its metadata."""
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "dengue_risk_model.joblib"
    metadata_path = destination / "model_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        **metadata,
        "cutoffs": {
            ward: {name: float(row[name]) for name in ("q75", "q90", "q97")}
            for ward, row in cutoffs.iterrows()
        },
        "global_cutoffs": global_cutoffs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def train(
    input_path: str | Path = DEFAULT_INPUT,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    test_start: str = DEFAULT_TEST_START,
    random_state: int = 42,
) -> dict:
    """Train, evaluate on a holdout, then refit on all data for deployment."""
    frame = load_dataset(input_path)
    cutoff_date = pd.Timestamp(test_start)
    train_frame = frame[frame["date"] < cutoff_date].reset_index(drop=True)
    test_frame = frame[frame["date"] >= cutoff_date].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError(
            f"Time split at {test_start} left {len(train_frame)} train and "
            f"{len(test_frame)} holdout rows"
        )

    train_cutoffs = compute_ward_cutoffs(train_frame)
    global_cutoffs = {
        "q75": float(train_frame[TARGET_COLUMN].quantile(0.75)),
        "q90": float(train_frame[TARGET_COLUMN].quantile(0.90)),
        "q97": float(train_frame[TARGET_COLUMN].quantile(0.97)),
    }

    print(
        f"Training on {len(train_frame):,} rows "
        f"({train_frame['date'].min():%Y-%m-%d} to {train_frame['date'].max():%Y-%m-%d})"
    )
    print(
        f"Holdout: {len(test_frame):,} rows "
        f"({test_frame['date'].min():%Y-%m-%d} to {test_frame['date'].max():%Y-%m-%d})"
    )

    evaluation_model = build_model(random_state=random_state)
    evaluation_model.fit(
        train_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_frame[TARGET_COLUMN]
    )
    metrics = evaluate_holdout(evaluation_model, test_frame, train_cutoffs, global_cutoffs)

    # Refit on all rows so the deployed model uses the most recent seasons.
    final_model = build_model(random_state=random_state)
    final_model.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame[TARGET_COLUMN])
    deployment_cutoffs = compute_ward_cutoffs(frame)
    deployment_global = {
        "q75": float(frame[TARGET_COLUMN].quantile(0.75)),
        "q90": float(frame[TARGET_COLUMN].quantile(0.90)),
        "q97": float(frame[TARGET_COLUMN].quantile(0.97)),
    }

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "HistGradientBoostingRegressor on log1p(dengue_cases)",
        "target": TARGET_COLUMN,
        "tier_boundaries": "per-ward q75/q90/q97 of observed dengue_cases",
        "tier_order": TIER_ORDER,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "rows": {
            "total": int(len(frame)),
            "train": int(len(train_frame)),
            "holdout": int(len(test_frame)),
        },
        "train_window": [
            str(train_frame["date"].min().date()),
            str(train_frame["date"].max().date()),
        ],
        "holdout_window": [
            str(test_frame["date"].min().date()),
            str(test_frame["date"].max().date()),
        ],
        "metrics": metrics,
    }

    model_path, metadata_path = save_artifacts(
        final_model, deployment_cutoffs, deployment_global, metadata, model_dir
    )

    print("\nHoldout metrics (cases scale):")
    print(f"  MAE:            {metrics['mae_cases']:.3f}")
    print(f"  RMSE:           {metrics['rmse_cases']:.3f}")
    print(f"  R2:             {metrics['r2_cases']:.3f}")
    print(f"  Spearman rho:   {metrics['spearman_rho']:.3f}")
    print(f"  Tier accuracy:  {metrics['tier_accuracy']:.3f}")
    print(f"  Tier macro F1:  {metrics['tier_macro_f1']:.3f}")
    detection = metrics["high_risk_detection"]
    if detection["roc_auc"] is not None:
        print(
            "  High-risk (>= q90): "
            f"precision={detection['precision']:.3f} "
            f"recall={detection['recall']:.3f} "
            f"ROC-AUC={detection['roc_auc']:.3f}"
        )
    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(
        input_path=args.input,
        model_dir=args.model_dir,
        test_start=args.test_start,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
