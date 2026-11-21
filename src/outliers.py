"""Detect and optionally handle outliers in weekly ward-level dengue data.

Outliers are judged per ward and calendar month, so monsoon rainfall and
seasonal temperature swings are not treated as errors. Dengue is zero-inflated
count data, so it is scored on a log scale and only high values are flagged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = ("avg_temp_c", "rainfall_mm", "relative_humidity_pct")
# Rainfall is scored on the log1p scale, so its floor is in log units.
MAD_FLOORS = {"avg_temp_c": 1.0, "rainfall_mm": 0.2, "relative_humidity_pct": 2.0}
Z_THRESHOLD = 4.0


def check_sanity(frame: pd.DataFrame) -> dict[str, int]:
    """Count values that are impossible regardless of season or ward."""
    return {
        "negative_rainfall": int((frame["rainfall_mm"] < 0).sum()),
        "humidity_out_of_range": int(
            (~frame["relative_humidity_pct"].between(0, 100)).sum()
        ),
        "max_below_min_temp": int((frame["max_temp_c"] < frame["min_temp_c"]).sum()),
        "avg_temp_outside_min_max": int(
            (
                (frame["avg_temp_c"] < frame["min_temp_c"])
                | (frame["avg_temp_c"] > frame["max_temp_c"])
            ).sum()
        ),
        "negative_dengue": int((frame["dengue_cases"] < 0).sum()),
    }


def seasonal_robust_z(
    frame: pd.DataFrame, column: str, floor: float, log: bool = False
) -> pd.Series:
    """Robust z-score within each ward and calendar month, with a scale floor.

    The floor prevents a zero MAD (common in dry-season rainfall) from turning
    ordinary values into huge scores.
    """
    values = np.log1p(frame[column]) if log else frame[column]
    keys = [frame["ward_id"], frame["date"].dt.month]
    median = values.groupby(keys).transform("median")
    mad = (values - median).abs().groupby(keys).transform("median")
    scale = (1.4826 * mad).clip(lower=floor)
    return (values - median) / scale


def flag_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean frame of outlier flags aligned to ``frame``."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    flags = pd.DataFrame(index=frame.index)

    for column in WEATHER_COLUMNS:
        z = seasonal_robust_z(
            frame,
            column,
            floor=MAD_FLOORS[column],
            log=column == "rainfall_mm",
        )
        flags[f"{column}_outlier"] = z.abs() > Z_THRESHOLD

    # Dengue: zero-inflated counts, so flag only unusually high values.
    dengue_z = seasonal_robust_z(frame, "dengue_cases", floor=0.5, log=True)
    flags["dengue_cases_outlier"] = dengue_z > Z_THRESHOLD

    # Isolated spikes: a high value that reverts on both neighbouring weeks.
    ordered = frame.sort_values(["ward_id", "date"])
    grouped = ordered.groupby("ward_id")
    prev = grouped["rainfall_mm"].shift(1)
    nxt = grouped["rainfall_mm"].shift(-1)
    flags.loc[ordered.index, "rainfall_jump_outlier"] = (
        (ordered["rainfall_mm"] > 100)
        & (ordered["rainfall_mm"] > 2 * prev)
        & (ordered["rainfall_mm"] > 2 * nxt)
    )

    prev = grouped["dengue_cases"].shift(1)
    nxt = grouped["dengue_cases"].shift(-1)
    flags.loc[ordered.index, "dengue_jump_outlier"] = (
        (ordered["dengue_cases"] >= 10)
        & (ordered["dengue_cases"] > 3 * prev.clip(lower=1))
        & (ordered["dengue_cases"] > 3 * nxt.clip(lower=1))
    )

    return flags.fillna(False).astype(bool)


def handle_outliers(
    frame: pd.DataFrame, flags: pd.DataFrame, winsorize: bool = False
) -> pd.DataFrame:
    """Add integer flag columns and optionally cap seasonal rainfall peaks.

    Dengue peaks are never removed because they are the outbreak signal.
    When rainfall is capped, weather lags should be regenerated afterwards.
    """
    result = frame.copy()
    for column in flags.columns:
        result[column] = flags[column].astype(int)

    if winsorize:
        keys = [result["ward_id"], pd.to_datetime(result["date"]).dt.month]
        cap = result.groupby(keys)["rainfall_mm"].transform(
            lambda series: series.quantile(0.99)
        )
        result["rainfall_mm"] = np.minimum(result["rainfall_mm"], cap)

    return result


def summarize(flags: pd.DataFrame) -> dict[str, int]:
    """Count flagged rows per check."""
    return {column: int(flags[column].sum()) for column in flags.columns}


def run(
    input_path: str | Path,
    output_path: str | Path | None = None,
    winsorize: bool = False,
) -> pd.DataFrame:
    """Read a CSV, flag outliers, and optionally write the annotated result."""
    frame = pd.read_csv(input_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    issues = check_sanity(frame)
    flags = flag_outliers(frame)
    result = handle_outliers(frame, flags, winsorize=winsorize)

    print(f"Rows: {len(result):,}")
    print("Sanity issues:", issues)
    print("Flagged rows:")
    for name, count in summarize(flags).items():
        print(f"  {name:32s} {count:6d} ({count / len(result) * 100:5.2f}%)")

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = result.copy()
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
        output.to_csv(destination, index=False)
        print(f"Wrote annotated data to {destination}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input weekly CSV")
    parser.add_argument(
        "output", type=Path, nargs="?", help="Optional annotated output CSV"
    )
    parser.add_argument(
        "--winsorize",
        action="store_true",
        help="Cap rainfall at the per-ward/month 99th percentile",
    )
    args = parser.parse_args()
    run(args.input, args.output, winsorize=args.winsorize)


if __name__ == "__main__":
    main()
