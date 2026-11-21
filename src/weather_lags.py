"""Ingest weekly dengue weather data and add ward-level weather lags."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WEATHER_COLUMNS = ("avg_temp_c", "rainfall_mm", "relative_humidity_pct")
LAG_WEEKS = (2, 3, 4)
LAG_FEATURES = tuple(
    f"{column}_lag_{weeks}w" for weeks in LAG_WEEKS for column in WEATHER_COLUMNS
)


def add_weather_lags(data: pd.DataFrame) -> pd.DataFrame:
    """Add exact calendar-week weather lags, independently for each ward.

    A lag is missing when that ward has no observation on the corresponding
    date; observations from other wards or nearby weeks are never substituted.
    """
    frame = data.copy()
    # Normalize column names so avg_temp_C and avg_temp_c are both accepted.
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    required = {"date", "ward_id", *WEATHER_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["ward_id"].isna().any():
        raise ValueError("ward_id cannot be empty")
    if frame.duplicated(["ward_id", "date"]).any():
        raise ValueError("Each ward_id/date pair must be unique to calculate lags")

    frame = frame.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)
    weather = frame.set_index(["ward_id", "date"])[list(WEATHER_COLUMNS)]

    for weeks in LAG_WEEKS:
        lag_dates = pd.MultiIndex.from_arrays(
            [frame["ward_id"], frame["date"] - pd.Timedelta(weeks=weeks)],
            names=["ward_id", "date"],
        )
        lagged = weather.reindex(lag_dates)
        for column in WEATHER_COLUMNS:
            frame[f"{column}_lag_{weeks}w"] = lagged[column].to_numpy()

    # Keep only rows with a complete 2–4 week history for every weather input.
    return frame.dropna(subset=list(LAG_FEATURES)).reset_index(drop=True)


def ingest_weather_csv(input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    """Read a CSV, add 2–4 week weather lags, and optionally save the result."""
    result = add_weather_lags(pd.read_csv(input_path))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input weekly CSV")
    parser.add_argument("output", type=Path, help="Output CSV with weather lag features")
    args = parser.parse_args()
    result = ingest_weather_csv(args.input, args.output)
    print(f"Wrote {len(result):,} rows with {len(WEATHER_COLUMNS) * len(LAG_WEEKS)} weather lag features to {args.output}")


if __name__ == "__main__":
    main()
