# Architecture

## System overview

The project has three layers, all driven by the same prepared dataset:

```
data/hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv
        │  src/weather_lags.py  →  src/outliers.py
        ▼
data/bangladesh_dengue_weather_lags.csv
        │
        ├── src/train_model.py  →  models/dengue_risk_model.joblib + model_metadata.json
        │
        ├── src/predict_risk.py          (batch CLI scoring)
        │
        └── app.py (Streamlit)           (interactive dashboard)
```

`src/weather_lags.py` prepares the weekly dataset, `src/outliers.py` flags
suspicious values, `src/train_model.py` fits and persists the model, and
`src/predict_risk.py` / `app.py` serve predictions from the saved artifacts.

## Data preparation

`src/weather_lags.py` prepares the weekly dengue dataset for modeling. It creates
weather features for the previous 2, 3, and 4 calendar weeks, matched by
`ward_id`, and drops rows without a complete lag history. `src/outliers.py` then
checks and annotates outliers. Neither script trains or evaluates a model.

### Inputs and outputs

The pipeline reads `date`, `ward_id`, `avg_temp_c`, `rainfall_mm`, and
`relative_humidity_pct`. Column names are trimmed and lowercased, so headers such
as `avg_temp_C` are accepted. Other source columns are retained in the output.

The generated features are:

- `avg_temp_c_lag_2w`, `avg_temp_c_lag_3w`, `avg_temp_c_lag_4w`
- `rainfall_mm_lag_2w`, `rainfall_mm_lag_3w`, `rainfall_mm_lag_4w`
- `relative_humidity_pct_lag_2w`, `relative_humidity_pct_lag_3w`,
  `relative_humidity_pct_lag_4w`

The current output is `data/bangladesh_dengue_weather_lags.csv`, generated from
`data/hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv`: 31,080
weekly ward rows spanning 2016-02-01 to 2025-12-29 across 60 wards in 5 city
corporations (Chattogram, Dhaka North, Dhaka South, Gazipur, Narayanganj).

### Processing and missing values

The pipeline parses dates, validates required columns, and rejects missing ward
IDs or duplicate `(ward_id, date)` records. It sorts records by ward and date,
then looks up weather values at the exact date 2, 3, or 4 weeks earlier for the
same ward. It does not substitute nearby dates or values from other wards.

If any of the nine lag values is missing, the entire row is removed. Other
missing fields do not trigger row removal. No lag values are imputed. For the
current dataset, 31,080 of 31,320 rows remain; 240 rows lack complete lag history.

### Outlier checks

`src/outliers.py` flags, but does not delete, suspicious values. Sanity checks
count impossible records: negative rainfall or dengue, humidity outside 0–100,
`max_temp_c` below `min_temp_c`, and average temperature outside the min/max
range. None occur in the current dataset.

Statistical checks are seasonal and per ward, so monsoon rainfall is not
mistaken for an error:

- Weather: robust z-score within each `ward_id` and calendar month, using the
  median and MAD with a scale floor. Rainfall is scored on the `log1p` scale;
  flags are raised at `|z| > 4`.
- Dengue: the same seasonal score on `log1p`, high side only, because counts are
  zero-inflated.
- Isolated spikes: rainfall above 100 mm, or dengue of at least 10 cases, that is
  more than 2–3× both neighbouring weeks.
- Static per-ward features (population density, water-logging risk, sanitation)
  have no within-ward variance and are not checked.

Flags are written as integer columns named `<column>_outlier`. With
`--winsorize`, rainfall is capped at the per-ward/month 99th percentile; dengue
peaks are kept because they are the outbreak signal. If rainfall is capped,
regenerate weather lags afterwards. Current flags: 1,263 rainfall, 677 rainfall
jumps, 177 dengue seasonal, 68 dengue jumps, 264 humidity, and 16 temperature.

## Feature engineering

`src/model_features.py` holds the shared definitions used by training, batch
inference, and the dashboard:

- 22 numeric features: 3 static ward characteristics (`population_density_sqkm`,
  `water_logging_risk_index`, `sanitation_score`), 5 current-weather readings,
  9 weather lags (2–4 weeks for temperature, rainfall, humidity), and 4 cyclical
  calendar encodings (`month_sin`/`month_cos`, `week_sin`/`week_cos`) derived
  from the `date` column.
- 2 categorical features: `ward_id`, `city`.
- Risk-tier helpers. `compute_ward_cutoffs` derives per-ward `q75`/`q90`/`q97`
  quantiles of observed `dengue_cases` as dynamic tier boundaries, and
  `assign_risk_tiers` maps any case count to low/medium/high/critical against
  those boundaries. `resolve_boundaries` falls back to global cutoffs for wards
  without stored per-ward boundaries (the dataset's 60 wards vs 55 stored
  cutoffs). The same predicted count can therefore mean different risk in
  different wards.

## Model training

`src/train_model.py` trains, evaluates, and persists the model:

- Pipeline: median imputation for numerics, most-frequent imputation +
  one-hot encoding for categoricals (`ColumnTransformer`), then
  `HistGradientBoostingRegressor`, all wrapped in a `TransformedTargetRegressor`
  that fits on `log1p(dengue_cases)` and predicts back on the case scale.
- Split: strict time holdout at 2024-01-01 — 24,780 train rows (2016–2023),
  6,300 holdout rows (2024–2025).
- Evaluation on the holdout: MAE 0.821 and RMSE 1.625 weekly cases, R² 0.823,
  Spearman ρ 0.850, tier accuracy 87.4%, macro F1 0.532, and high-risk
  (≥ q90) detection with ROC-AUC 0.934 at 76.6% precision / 32.8% recall.
- Deployment: after evaluation, the pipeline refits on all 31,080 rows so the
  deployed model uses the most recent seasons, and stores per-ward cutoffs
  computed on the full dataset.
- Artifacts: `models/dengue_risk_model.joblib` and `models/model_metadata.json`
  (model spec, feature lists, split windows, metrics, per-ward and global
  cutoffs). `src/predict_risk.py::load_artifacts` is the single loader used by
  both serving paths.

## Batch inference

`src/predict_risk.py` scores a ward/time window from the prepared CSV and saved
artifacts (`python -m src.predict_risk --ward CCC_Ward_01 --start ... --end ...`).
It validates wards against the dataset, clips predictions at zero, attaches the
per-ward tier and a `risk_ratio` (prediction ÷ ward q90), and reports actual
cases/tiers where the dataset has them.

## Dashboard

`app.py` is a Streamlit application (`streamlit run app.py`) with five tabs,
built on the same artifacts and tier helpers. It caches the dataset, artifact
loading, and every derived table with `st.cache_data`/`st.cache_resource`.

- **Predict & Drivers**: scores one ward-week with the deployed model. Inputs
  pre-fill from observed data and are editable for what-if analysis. The driver
  breakdown approximates each feature's contribution as the prediction minus the
  prediction with that feature set to the ward's historical median; the values
  are directional, not an exact decomposition, because features interact.
- **Risk Heatmap**: ward × week matrix of predicted tier / predicted / observed
  cases, pivoting the fully scored dataset (31,080 predictions, computed once).
- **Multi-Week Forecast**: 1–12 weeks ahead of an anchor week. Future weather is
  assumed from per-ward per-ISO-week climatology or the last observed week, with
  optional temperature/rainfall anomalies; lag features are resolved by exact
  calendar-date lookup on the combined observed-plus-synthetic series, mirroring
  `weather_lags.py`. Intervals are empirical quantiles of 2024–25 holdout
  residuals, taken per predicted risk tier (sparse tiers fall back to pooled
  residuals) at 50/80/95% levels.
- **Model Validation**: chronological MAE/RMSE and tier diagnostics on the
  holdout. Because the deployed model was refit on all rows, scoring 2024–25 with
  it would leak; the dashboard therefore retrains the evaluation model on
  2016–2023 at first load (~2s, cached) so every validation number is genuinely
  out-of-sample. Global driver importance is permutation importance of each input
  against holdout MAE.
- **Data Explorer**: statistical views of the dataset (city trends, seasonality,
  Spearman correlations, distributions, outlier-flag counts).

The dashboard adds no new dependencies beyond the training stack: `streamlit`
and `plotly` (both listed in `requirements.txt`).

## Scope

The data pipeline expects weekly observations but does not enforce a specific
weekday. It performs exact calendar-date lookups and does not create data splits,
train a model, or calculate evaluation metrics. The model serves weekly ward-level
risk estimates only; it is not an individual clinical predictor, and multi-week
forecasts inherit the uncertainty of their assumed weather inputs.
