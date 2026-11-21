# Data Preparation Architecture

## Overview

`src/weather_lags.py` prepares the weekly dengue dataset for modeling. It creates
weather features for the previous 2, 3, and 4 calendar weeks, matched by
`ward_id`, and drops rows without a complete lag history. It does not train or
evaluate a model.

## Inputs and outputs

The pipeline reads `date`, `ward_id`, `avg_temp_c`, `rainfall_mm`, and
`relative_humidity_pct`. Column names are trimmed and lowercased, so headers such
as `avg_temp_C` are accepted. Other source columns are retained in the output.

The generated features are:

- `avg_temp_c_lag_2w`, `avg_temp_c_lag_3w`, `avg_temp_c_lag_4w`
- `rainfall_mm_lag_2w`, `rainfall_mm_lag_3w`, `rainfall_mm_lag_4w`
- `relative_humidity_pct_lag_2w`, `relative_humidity_pct_lag_3w`,
  `relative_humidity_pct_lag_4w`

The current output is `data/bangladesh_dengue_weather_lags.csv`, generated from
`data/hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv`.

## Processing and missing values

The pipeline parses dates, validates required columns, and rejects missing ward
IDs or duplicate `(ward_id, date)` records. It sorts records by ward and date,
then looks up weather values at the exact date 2, 3, or 4 weeks earlier for the
same ward. It does not substitute nearby dates or values from other wards.

If any of the nine lag values is missing, the entire row is removed. Other
missing fields do not trigger row removal. No lag values are imputed. For the
current dataset, 31,080 of 31,320 rows remain; 240 rows lack complete lag history.

## Scope

The implementation expects weekly observations but does not enforce a specific
weekday. It performs exact calendar-date lookups and does not create data splits,
train a model, or calculate evaluation metrics.
