# Architecture

## System overview

OutbreakX is a weekly, ward-level dengue early-warning system with one shared
data pipeline for training, batch inference, and the interactive dashboard.

```
data/hackathon_preli_dataset - bangladesh_dengue_large_dataset.csv
        │
        │  src/weather_lags.py  →  src/outliers.py
        ▼
data/bangladesh_dengue_weather_lags.csv
        │
        ├── src/train_model.py
        │      └── models/dengue_risk_model.joblib + model_metadata.json
        ├── src/predict_risk.py  (batch predictions for wards/date windows)
        └── app.py               (Streamlit dashboard)
```

The system runs locally without external APIs. Dataset loading, model artifacts,
and derived dashboard tables are cached with Streamlit to avoid repeated file
reads, model loading, and scoring.

## Data preparation and features

`src/weather_lags.py` reads `date`, `ward_id`, `avg_temp_c`, `rainfall_mm`, and
`relative_humidity_pct`. Headers are trimmed and lowercased, so `avg_temp_C` is
also accepted. Dates are parsed, missing ward IDs and duplicate ward/date pairs
are rejected, and records are sorted by ward and date.

For each ward, exact calendar-date lookups attach weather from 2, 3, and 4 weeks
earlier. Values are never borrowed from another ward or nearby date. This
produces nine lag features: temperature, rainfall, and humidity at each lag.
Rows without all nine values are removed rather than imputed. The prepared data
contains 31,080 weekly rows from 2016-02-01 to 2025-12-29 across 60 wards and
five city corporations; 240 source rows lack complete lag history.

`src/outliers.py` annotates suspicious values without deleting outbreak peaks.
It checks impossible values and calculates seasonal robust z-scores within each
ward and calendar month. Rainfall is scored on a `log1p` scale; dengue counts
are flagged only on the high side. Optional rainfall winsorization caps values
at the ward/month 99th percentile, after which lags must be regenerated.

`src/model_features.py` defines the inputs shared by every training and serving
path:

- Three static ward features: population density, water-logging risk, and
  sanitation score.
- Five current-weather features and nine 2–4 week weather lags.
- Four cyclical calendar features: month and ISO-week sine/cosine encodings.
- Two categorical features: `ward_id` and `city`.

## Model selection rationale

The model is a `HistGradientBoostingRegressor` trained on
`log1p(dengue_cases)`. Gradient-boosted trees were selected because dengue risk
is nonlinear: rainfall, temperature, humidity, seasonality, and ward conditions
interact, and their effects need not follow a fixed linear relationship. The
histogram-based implementation is efficient enough for repeated local training,
while native handling of nonlinear splits avoids manually specifying many
interaction terms.

The target is log-transformed because case counts are non-negative, skewed, and
contain outbreak peaks; this reduces the dominance of extreme counts during
training. Predictions are converted back to the case scale and clipped at zero.
A regularized histogram booster provides a stronger fit for these tabular,
interacting inputs than a simple linear baseline without the deployment cost or
data requirements of a deep-learning model. Model choice is assessed on a later
time period rather than a random split, which better matches operational use.

Preprocessing is packaged with the model: median imputation for numeric inputs,
most-frequent imputation and one-hot encoding for categoricals, and the target
transform. Persisting the complete pipeline prevents training/serving feature
mismatches.

## Chronological validation and risk tiers

Training uses a strict time split at **2024-01-01**:

| Partition | Rows | Date range | Purpose |
|---|---:|---|---|
| Training | 24,780 | 2016-02-01 to 2023-12-25 | Fit the evaluation model and training-only tier boundaries |
| Holdout | 6,300 | 2024-01-01 to 2025-12-29 | Evaluate entirely later, unseen seasons |

Holdout results are MAE **0.821**, RMSE **1.625**, R² **0.823**, Spearman ρ
**0.850**, and tier accuracy **87.4%**. High-risk detection (at or above ward
q90) has ROC-AUC **0.934**, precision **0.766**, and recall **0.328**. These
metrics describe retrospective weekly case prediction, not clinical diagnosis.

Risk tiers use each ward's observed-case quantiles: medium at q75, high at q90,
and critical at q97. Ward-specific boundaries allow the same predicted count to
represent different relative risk in different wards; global boundaries are a
fallback for unseen wards. After evaluation, the deployment model is refit on
all rows, and deployment boundaries are recomputed from the full dataset.

## Inference and dashboard

`src/predict_risk.py` loads the saved pipeline and metadata, filters the prepared
data by one or more wards and an optional date window, and returns weekly case
predictions, risk tiers, and risk ratios. Both serving paths use the same loader
and risk-tier helpers.

The Streamlit dashboard provides single-week prediction and weather-driver
analysis, a ward/week heatmap, 1–12 week forecasts, chronological validation,
and a data explorer. Interactive speed comes from two rules: model predictions
are always issued as one batched `predict` call per operation (a single
pipeline pass costs ~7 ms regardless of row count, so per-feature and
per-week loops would dominate response time), and per-tab figures/derived
tables are cached with `st.cache_data` on hashable inputs, so a widget change
only recomputes the outputs that depend on it. Forecast weather is explicitly
assumed from ward climatology or the latest observed week; it is not a live
weather forecast.
Forecast intervals use empirical 2024–2025 holdout residuals. Because the
deployed model was refit on all data, the dashboard separately trains and caches
a pre-2024 evaluation model for genuinely out-of-sample validation charts.

The main limitation is that predictions depend on historical relationships and
assumed future weather. They should support—not replace—public-health judgment,
surveillance, and field investigation.
