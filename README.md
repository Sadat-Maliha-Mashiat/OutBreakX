# OutbreakX — Dengue Ward Risk Prediction

Ward-level dengue early-warning system for Bangladeshi city corporations.
A gradient-boosting model (`HistGradientBoostingRegressor` on `log1p(dengue_cases)`)
predicts weekly dengue cases per ward from static ward characteristics, current
weather, and 2–4 week weather lags. Predictions are mapped to per-ward risk tiers
(low / medium / high / critical) using ward-specific quantile boundaries (q75/q90/q97).

See `ARCHITECTURE.md` for the data-pipeline design.

## Interactive dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tabs:

- **Predict & Drivers** — pick a ward and week, adjust weather inputs, and see the
  predicted cases, risk tier, empirical interval, and a driver breakdown vs the
  ward's historical normals.
- **Risk Heatmap** — ward × week risk matrix with per-city tier mix.
- **Multi-Week Forecast** — 1–12 week ahead forecast with empirical confidence
  intervals (from 2024–25 holdout residuals) and assumed-weather inputs.
- **Model Validation** — chronological MAE/RMSE on the time holdout, per-ward
  error, tier confusion matrix, and global permutation importance.
- **Data Explorer** — statistical visualizations of the dataset.

## Training and batch prediction

```bash
python -m src.train_model                 # trains, evaluates, saves artifacts to models/
python -m src.predict_risk --ward CCC_Ward_01 --start 2024-01-01 --end 2024-12-31
```

Artifacts: `models/dengue_risk_model.joblib`, `models/model_metadata.json`
(holdout MAE 0.82, RMSE 1.63, R² 0.82 on 2024–25).
