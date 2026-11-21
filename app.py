"""Interactive dengue risk dashboard for ward-level early warning.

Run with:  streamlit run app.py

Tabs: ward prediction + driver breakdown, ward-by-week risk heatmap,
multi-week forecast with empirical confidence intervals, chronological
holdout validation (MAE/RMSE), and a statistical data explorer.

Performance notes: the scikit-learn pipeline costs ~7 ms per ``predict`` call
regardless of row count, so predictions are always issued as ONE batched call
per operation (a single ward-week driver breakdown, or all forecast weeks and
their per-feature perturbations at once). Per-tab figures and derived tables
are cached with ``st.cache_data`` keyed on plain scalars, so changing a widget
only recomputes the outputs that actually depend on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance

from src.model_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TIER_ORDER,
    add_calendar_features,
    assign_risk_tiers,
    load_dataset,
    resolve_boundaries,
)
from src.predict_risk import load_artifacts

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bangladesh_dengue_weather_lags.csv"
MODEL_DIR = ROOT / "models"
HOLDOUT_START = "2024-01-01"

WEATHER_COLS = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]

TIER_RANK = {tier: rank for rank, tier in enumerate(TIER_ORDER)}
TIER_COLORS = {
    "low": "#2ca02c",
    "medium": "#f1c40f",
    "high": "#e67e22",
    "critical": "#d62728",
}

FRIENDLY = {
    "population_density_sqkm": "Population density (per km²)",
    "water_logging_risk_index": "Water-logging risk index",
    "sanitation_score": "Sanitation score",
    "avg_temp_c": "Avg temperature °C",
    "max_temp_c": "Max temperature °C",
    "min_temp_c": "Min temperature °C",
    "rainfall_mm": "Rainfall (mm)",
    "relative_humidity_pct": "Relative humidity (%)",
    "avg_temp_c_lag_2w": "Avg temperature °C (lag 2w)",
    "rainfall_mm_lag_2w": "Rainfall mm (lag 2w)",
    "relative_humidity_pct_lag_2w": "Humidity % (lag 2w)",
    "avg_temp_c_lag_3w": "Avg temperature °C (lag 3w)",
    "rainfall_mm_lag_3w": "Rainfall mm (lag 3w)",
    "relative_humidity_pct_lag_3w": "Humidity % (lag 3w)",
    "avg_temp_c_lag_4w": "Avg temperature °C (lag 4w)",
    "rainfall_mm_lag_4w": "Rainfall mm (lag 4w)",
    "relative_humidity_pct_lag_4w": "Humidity % (lag 4w)",
    "month_sin": "Seasonal cycle (month, sin)",
    "month_cos": "Seasonal cycle (month, cos)",
    "week_sin": "Seasonal cycle (week, sin)",
    "week_cos": "Seasonal cycle (week, cos)",
    "ward_id": "Ward identity",
    "city": "City",
}

st.set_page_config(page_title="Dengue Ward Risk Intelligence", page_icon="🦟", layout="wide")


# --------------------------------------------------------------------------- #
# Cached data / model layer
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_artifacts():
    return load_artifacts(MODEL_DIR)


@st.cache_resource(show_spinner="Training the evaluation model (2016–2023) for honest holdout scoring…")
def get_eval_model():
    """Model trained only on pre-2024 data, so 2024–25 scores are truly out-of-sample."""
    from src.train_model import build_model

    frame = get_data()
    train = frame[frame["date"] < HOLDOUT_START]
    model = build_model(random_state=42)
    model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET_COLUMN])
    return model


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@st.cache_data
def score_history() -> pd.DataFrame:
    """Score every ward-week in the dataset with the deployed model."""
    model, _metadata, cutoffs, global_cutoffs = get_artifacts()
    frame = get_data()
    pred = np.clip(model.predict(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0, None)
    q90 = resolve_boundaries(frame["ward_id"], cutoffs, global_cutoffs)["q90"].to_numpy()
    result = pd.DataFrame(
        {
            "date": frame["date"],
            "ward_id": frame["ward_id"],
            "city": frame["city"],
            "predicted_cases": pred,
            "actual_cases": frame[TARGET_COLUMN].astype(float).to_numpy(),
        }
    )
    result["risk_tier"] = assign_risk_tiers(pred, frame["ward_id"], cutoffs, global_cutoffs).to_numpy()
    result["actual_tier"] = assign_risk_tiers(
        result["actual_cases"].to_numpy(), frame["ward_id"], cutoffs, global_cutoffs
    ).to_numpy()
    result["risk_rank"] = result["risk_tier"].map(TIER_RANK)
    result["residual"] = result["actual_cases"] - result["predicted_cases"]
    result["abs_error"] = result["residual"].abs()
    result["risk_ratio"] = pred / q90
    result["iso_week"] = frame["date"].dt.isocalendar().week.astype(int)
    return result


@st.cache_data
def holdout_frame() -> pd.DataFrame:
    frame = get_data()
    return frame[frame["date"] >= HOLDOUT_START].reset_index(drop=True)


@st.cache_data
def holdout_scores() -> pd.DataFrame:
    """Out-of-sample model scores on the chronological holdout (2024 onward)."""
    model = get_eval_model()
    _deployed, _metadata, cutoffs, global_cutoffs = get_artifacts()
    test = holdout_frame()
    pred = np.clip(model.predict(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0, None)
    out = pd.DataFrame(
        {
            "date": test["date"],
            "ward_id": test["ward_id"],
            "city": test["city"],
            "actual_cases": test[TARGET_COLUMN].astype(float).to_numpy(),
            "predicted_cases": pred,
        }
    )
    out["risk_tier"] = assign_risk_tiers(pred, test["ward_id"], cutoffs, global_cutoffs).to_numpy()
    out["actual_tier"] = assign_risk_tiers(
        out["actual_cases"].to_numpy(), test["ward_id"], cutoffs, global_cutoffs
    ).to_numpy()
    out["residual"] = out["actual_cases"] - out["predicted_cases"]
    out["abs_error"] = out["residual"].abs()
    return out


@st.cache_data(show_spinner="Computing global driver importance (permutation on holdout)…")
def global_importance() -> pd.DataFrame:
    """Permutation importance of every model input, measured on holdout MAE."""
    model = get_eval_model()
    _deployed, _metadata, _cutoffs, _global = get_artifacts()
    test = holdout_frame()
    sample = test.sample(n=min(2500, len(test)), random_state=0)
    X = sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = sample[TARGET_COLUMN].astype(float)
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error"
    )
    rows = [
        {"feature": col, "label": FRIENDLY.get(col, col), "importance": -mean, "std": std}
        for col, mean, std in zip(X.columns, result.importances_mean, result.importances_std)
    ]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


@st.cache_data
def residual_bands() -> dict:
    """Empirical forecast intervals from 2024-25 holdout residuals, per predicted tier."""
    hs = holdout_scores()
    resid_all = hs["actual_cases"] - hs["predicted_cases"]
    qs = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]
    names = ["lo95", "lo80", "lo50", "med", "hi50", "hi80", "hi95"]
    bands = {}
    for tier in TIER_ORDER:
        resid = hs.loc[hs["risk_tier"] == tier, "residual"]
        if len(resid) < 30:  # sparse tiers (e.g. critical) fall back to pooled residuals
            resid = resid_all
        bands[tier] = dict(zip(names, resid.quantile(qs).to_numpy()))
    bands["_global"] = dict(zip(names, resid_all.quantile(qs).to_numpy()))
    return bands


@st.cache_data
def ward_normals(anchor: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-ward climatology using only observations available by the anchor."""
    frame = get_data()
    tmp = frame.copy()
    if anchor is not None:
        tmp = tmp[tmp["date"] <= anchor]
    tmp["iso_week"] = tmp["date"].dt.isocalendar().week.astype(int)
    normals = tmp.groupby(["ward_id", "iso_week"])[WEATHER_COLS].mean().reset_index()
    static = tmp[["ward_id", "city"] + STATIC_COLS].drop_duplicates("ward_id")
    return normals.merge(static, on="ward_id")


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict_rows(rows: pd.DataFrame) -> np.ndarray:
    model, _metadata, _cutoffs, _global = get_artifacts()
    return np.clip(model.predict(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]), 0, None)


def _batch_predict(feats: list[dict], ward_id: str, city: str) -> np.ndarray:
    """Score many feature dicts in ONE pipeline pass.

    A ``predict`` call on this pipeline costs ~7 ms no matter the row count,
    so batching ~25 rows per call instead of issuing one call per row is the
    difference between ~200 ms and ~8 ms.
    """
    batch = pd.DataFrame(feats)
    batch["ward_id"] = ward_id
    batch["city"] = city
    return predict_rows(batch)


def ward_medians(ward_id: str) -> dict:
    """Historical median of every numeric feature for one ward (its 'normal')."""
    history = get_data()[get_data()["ward_id"] == ward_id]
    return {c: float(history[c].median()) for c in NUMERIC_FEATURES}


def _driver_calc(ward_id: str, city: str, feat: dict) -> tuple[pd.DataFrame, float, float]:
    """Approximate per-feature contribution to one prediction.

    Contribution of a feature = prediction with the real value minus prediction
    with that feature set to the ward's historical median (its "normal"). The
    values do not sum exactly (features interact), so they are shown as the
    top directional drivers rather than an exact decomposition. All candidate
    rows are scored with a single batched ``predict`` call.
    """
    normals = ward_medians(ward_id)
    base_feat = {c: normals[c] for c in NUMERIC_FEATURES}
    changed = [c for c in NUMERIC_FEATURES if abs(float(feat[c]) - normals[c]) >= 1e-9]
    preds = _batch_predict(
        [feat, base_feat] + [{**feat, col: normals[col]} for col in changed], ward_id, city
    )
    full_pred = float(preds[0])
    base_pred = float(preds[1])
    rows = [
        {
            "feature": col,
            "label": FRIENDLY.get(col, col),
            "actual": float(feat[col]),
            "normal": normals[col],
            "delta": full_pred - float(perturbed_pred),
        }
        for col, perturbed_pred in zip(changed, preds[2:])
    ]
    drivers = pd.DataFrame(rows)
    if not drivers.empty:
        drivers["abs_delta"] = drivers["delta"].abs()
        drivers = drivers.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    return drivers, full_pred, base_pred


@st.cache_data(show_spinner=False)
def compute_drivers(
    ward_id: str, city: str, values: tuple[float, ...]
) -> tuple[pd.DataFrame, float, float]:
    """Cached driver breakdown for one ward-week (hashable scalar inputs)."""
    return _driver_calc(ward_id, city, dict(zip(NUMERIC_FEATURES, values)))


def tier_for(ward_id: str, pred: float) -> str:
    _model, _metadata, cutoffs, global_cutoffs = get_artifacts()
    tiers = assign_risk_tiers(np.array([pred]), pd.Series([ward_id]), cutoffs, global_cutoffs)
    return tiers.iloc[0]


def ward_cutoffs(ward_id: str) -> dict:
    _model, _metadata, cutoffs, global_cutoffs = get_artifacts()
    if ward_id in cutoffs.index:
        row = cutoffs.loc[ward_id]
        return {name: float(row[name]) for name in ("q75", "q90", "q97")}
    return dict(global_cutoffs)


def tier_interval(pred: float, tier: str, level: int) -> tuple[float, float]:
    bands = residual_bands()
    band = bands.get(tier, bands["_global"])
    low_key, high_key = {50: ("lo50", "hi50"), 80: ("lo80", "hi80"), 95: ("lo95", "hi95")}[level]
    return max(0.0, pred + band[low_key]), max(0.0, pred + band[high_key])


def build_forecast(
    ward_id: str,
    anchor_date: pd.Timestamp,
    horizon: int,
    scenario: str,
    temp_delta: float,
    rain_mult: float,
) -> pd.DataFrame:
    """Assemble future feature rows using climatology or last-week weather.

    Weather series = observed history up to the anchor, then synthetic weeks.
    Lag features are resolved by exact calendar-date lookup on the combined
    series, mirroring ``src/weather_lags.py``.
    """
    frame = get_data()
    hist = frame[
        (frame["ward_id"] == ward_id) & (frame["date"] <= anchor_date)
    ].sort_values("date").reset_index(drop=True)
    if hist.empty:
        raise ValueError(f"No observations for {ward_id} on or before {anchor_date:%Y-%m-%d}")
    normals = ward_normals(anchor_date)
    ward_normal = normals[normals["ward_id"] == ward_id]
    last = hist.iloc[-1]

    future_rows = []
    for k in range(1, horizon + 1):
        day = anchor_date + pd.Timedelta(weeks=k)
        iso_week = int(day.isocalendar().week)
        match = ward_normal[ward_normal["iso_week"] == iso_week]
        if scenario == "Repeat last observed week":
            vals = {c: float(last[c]) for c in WEATHER_COLS}
        elif match.empty:
            vals = {c: float(hist[c].mean()) for c in WEATHER_COLS}
        else:
            vals = {c: float(match[c].iloc[0]) for c in WEATHER_COLS}
        vals["avg_temp_c"] = max(0.0, vals["avg_temp_c"] + temp_delta)
        vals["max_temp_c"] = max(0.0, vals["max_temp_c"] + temp_delta)
        vals["min_temp_c"] = max(0.0, vals["min_temp_c"] + temp_delta)
        vals["rainfall_mm"] = max(0.0, vals["rainfall_mm"] * rain_mult)
        future_rows.append({"date": day, **vals})

    combined = pd.concat(
        [hist[["date"] + WEATHER_COLS], pd.DataFrame(future_rows)], ignore_index=True
    ).sort_values("date")
    lag_base = {"avg_temp_c": "avg_temp", "rainfall_mm": "rainfall", "relative_humidity_pct": "humidity"}
    for lag in (2, 3, 4):
        shifted = combined[["date"] + WEATHER_COLS].copy()
        shifted["date"] = shifted["date"] + pd.Timedelta(weeks=lag)
        renamed = {base: f"{base}_lag_{lag}w" for base in lag_base if base != "date"}
        shifted = shifted.rename(columns=renamed)
        combined = combined.merge(
            shifted[["date"] + list(renamed.values())], on="date", how="left"
        )

    feat = combined[combined["date"] > anchor_date].copy().reset_index(drop=True)
    for col in STATIC_COLS:
        feat[col] = float(last[col])
    feat["ward_id"] = ward_id
    feat["city"] = last["city"]
    feat = add_calendar_features(feat)
    return feat


def _forecast_top_drivers(
    ward_id: str, city: str, feat: pd.DataFrame, preds: np.ndarray
) -> list[str]:
    """Top driver label per forecast week, all perturbations in one predict call."""
    normals = ward_medians(ward_id)
    batch: list[dict] = []
    meta: list[tuple[int, str]] = []
    for i in range(len(feat)):
        row_vals = {c: float(feat.iloc[i][c]) for c in NUMERIC_FEATURES}
        for col in NUMERIC_FEATURES:
            if abs(row_vals[col] - normals[col]) < 1e-9:
                continue
            batch.append({**row_vals, col: normals[col]})
            meta.append((i, col))
    if not batch:
        return ["—"] * len(feat)
    perturbed_preds = _batch_predict(batch, ward_id, city)
    best: dict[int, tuple[str, float]] = {}
    for (i, col), perturbed_pred in zip(meta, perturbed_preds):
        delta = float(preds[i] - perturbed_pred)
        if i not in best or abs(delta) > abs(best[i][1]):
            best[i] = (col, delta)
    return [
        "—" if i not in best else f"{FRIENDLY.get(best[i][0], best[i][0])} ({best[i][1]:+.1f})"
        for i in range(len(feat))
    ]


@st.cache_data(show_spinner=False)
def compute_forecast(
    ward_id: str,
    anchor: str,
    horizon: int,
    scenario: str,
    temp_delta: float,
    rain_mult: float,
    level: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full forecast pipeline in one shot: one batched predict for the weekly
    predictions (plus one for all driver perturbations), vectorized tier
    assignment, and empirical intervals. Inputs are hashable scalars."""
    _model, _metadata, cutoffs, global_cutoffs = get_artifacts()
    feat = build_forecast(ward_id, pd.Timestamp(anchor), horizon, scenario, temp_delta, rain_mult)
    preds = predict_rows(feat)
    tiers = assign_risk_tiers(preds, feat["ward_id"], cutoffs, global_cutoffs).tolist()
    bands = residual_bands()
    low_key, high_key = {50: ("lo50", "hi50"), 80: ("lo80", "hi80"), 95: ("lo95", "hi95")}[level]
    lows, highs = [], []
    for p, t in zip(preds, tiers):
        band = bands.get(t, bands["_global"])
        lows.append(max(0.0, p + band[low_key]))
        highs.append(max(0.0, p + band[high_key]))
    table = pd.DataFrame(
        {
            "week": feat["date"],
            "predicted_cases": np.round(preds, 1),
            "tier": tiers,
            "interval_low": [round(v, 1) for v in lows],
            "interval_high": [round(v, 1) for v in highs],
            "top_driver": _forecast_top_drivers(ward_id, feat["city"].iloc[0], feat, preds),
        }
    )
    weather = feat[["date", "rainfall_mm", "avg_temp_c"]].reset_index(drop=True)
    return table, weather


# --------------------------------------------------------------------------- #
# Figure helpers (pure builders — cached wrappers live in the tab section)
# --------------------------------------------------------------------------- #
def tier_colorscale() -> list:
    c = [TIER_COLORS[t] for t in TIER_ORDER]
    return [
        [0.00, c[0]], [0.25, c[0]],
        [0.25, c[1]], [0.50, c[1]],
        [0.50, c[2]], [0.75, c[2]],
        [0.75, c[3]], [1.00, c[3]],
    ]


def risk_gauge(ratio: float, tier: str, cut: dict) -> go.Figure:
    top = max(2.0, ratio * 1.25, cut["q97"] / cut["q90"] * 1.1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=ratio,
            number={"suffix": "×", "valueformat": ".2f"},
            title={"text": "Risk ratio — prediction ÷ ward high-risk threshold"},
            gauge={
                "axis": {"range": [0, top], "tickformat": ".1f"},
                "bar": {"color": TIER_COLORS[tier]},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, cut["q75"] / cut["q90"]], "color": "#e8f8ee"},
                    {"range": [cut["q75"] / cut["q90"], 1.0], "color": "#fdf6dd"},
                    {"range": [1.0, cut["q97"] / cut["q90"]], "color": "#fdebd9"},
                    {"range": [cut["q97"] / cut["q90"], top], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#333333", "width": 2},
                    "value": 1.0,
                    "thickness": 0.75,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def drivers_chart(drivers: pd.DataFrame, top_n: int = 8) -> go.Figure:
    show = drivers.head(top_n).copy()
    show = show.sort_values("delta")
    show["direction"] = show["delta"].apply(lambda d: "raises risk" if d > 0 else "lowers risk")
    fig = px.bar(
        show,
        x="delta",
        y="label",
        color="direction",
        color_discrete_map={"raises risk": "#d62728", "lowers risk": "#2ca02c"},
        orientation="h",
        text=show["delta"].apply(lambda d: f"{d:+.2f}"),
        labels={"delta": "Contribution to predicted cases (vs ward normal)", "label": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=90 + 34 * len(show), margin=dict(l=10, r=20, t=20, b=10))
    return fig


def tier_badge(tier: str) -> str:
    color = TIER_COLORS[tier]
    return (
        f"<span style='background:{color};color:white;padding:3px 14px;"
        f"border-radius:12px;font-weight:600;text-transform:uppercase;'>{tier}</span>"
    )


# --------------------------------------------------------------------------- #
# Cached per-tab figures — keyed on plain scalars so unrelated widget changes
# are cache hits and only affected figures are rebuilt.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def gauge_fig(ratio: float, tier: str, q75: float, q90: float, q97: float) -> go.Figure:
    return risk_gauge(ratio, tier, {"q75": q75, "q90": q90, "q97": q97})


@st.cache_data(show_spinner=False)
def drivers_fig(drivers: pd.DataFrame) -> go.Figure:
    return drivers_chart(drivers)


@st.cache_data(show_spinner=False)
def heatmap_fig(year: int, cities: tuple[str, ...], metric: str) -> go.Figure | None:
    scores = score_history()
    data = scores[(scores["city"].isin(cities)) & (scores["date"].dt.year == year)]
    if data.empty:
        return None
    value_col = {"Predicted risk tier": "risk_rank", "Predicted cases": "predicted_cases",
                 "Observed cases": "actual_cases"}[metric]
    pivot = data.pivot_table(index="ward_id", columns="date", values=value_col, aggfunc="mean")
    hover = data.assign(
        txt=lambda d: "Predicted "
        + d["predicted_cases"].round(1).astype(str)
        + " cases · observed "
        + d["actual_cases"].round(1).astype(str)
        + " · tier "
        + d["risk_tier"].str.capitalize()
    ).pivot_table(index="ward_id", columns="date", values="txt", aggfunc="first")

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns,
            y=pivot.index,
            text=hover.reindex(index=pivot.index, columns=pivot.columns).to_numpy(),
            hovertemplate="%{y} · %{x|%Y-%m-%d}<br>%{text}<extra></extra>",
            hoverongaps=False,
            colorscale=tier_colorscale() if value_col == "risk_rank" else "YlOrRd",
            zmin=0 if value_col == "risk_rank" else 0,
            zmax=3 if value_col == "risk_rank" else float(np.nanmax(pivot.to_numpy())),
            colorbar=(
                dict(tickvals=[0.375, 1.125, 1.875, 2.625],
                     ticktext=[t.capitalize() for t in TIER_ORDER])
                if value_col == "risk_rank" else dict(title="cases")
            ),
        )
    )
    fig.update_layout(
        height=140 + 26 * len(pivot.index),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickformat="%d %b", dtick=28 * 24 * 3600 * 1000),
        yaxis=dict(autorange="reversed"),
    )
    return fig


@st.cache_data(show_spinner=False)
def tier_mix_fig(year: int, cities: tuple[str, ...]) -> go.Figure | None:
    scores = score_history()
    data = scores[(scores["city"].isin(cities)) & (scores["date"].dt.year == year)]
    if data.empty:
        return None
    share = (
        data.groupby(["city", "risk_tier"]).size().unstack(fill_value=0)
        .reindex(columns=TIER_ORDER, fill_value=0)
    )
    share_pct = share.div(share.sum(axis=1), axis=0) * 100
    fig = px.bar(
        share_pct.reset_index().melt(id_vars="city", var_name="tier", value_name="pct"),
        x="city", y="pct", color="tier", color_discrete_map=TIER_COLORS,
        category_orders={"tier": TIER_ORDER},
        labels={"pct": "% of ward-weeks", "city": "", "tier": "Risk tier"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    return fig


@st.cache_data(show_spinner=False)
def forecast_fig(
    ward_id: str,
    anchor: str,
    level: int,
    q90: float,
    q97: float,
    table: pd.DataFrame,
) -> go.Figure:
    dates = list(table["week"])
    preds = table["predicted_cases"]
    tiers = table["tier"]
    low = list(table["interval_low"])
    high = list(table["interval_high"])
    hist = get_data()
    hist = hist[(hist["ward_id"] == ward_id) & (hist["date"] <= pd.Timestamp(anchor))].tail(8)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=high + low[::-1],
            fill="toself",
            fillcolor="rgba(31,119,180,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{level}% interval",
            hovertemplate="%{y:.1f} cases<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=preds,
            mode="lines+markers+text",
            text=[f"{p:.1f}" for p in preds], textposition="top center",
            marker=dict(size=11, color=[TIER_COLORS[t] for t in tiers]),
            line=dict(color="#1f77b4", dash="dash"),
            name="Forecast",
        )
    )
    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["date"], y=hist[TARGET_COLUMN],
                mode="lines+markers", name="Observed (recent)",
                line=dict(color="#7f7f7f"),
            )
        )
    for q, name, color in ((q90, "High-risk threshold", "#e67e22"),
                           (q97, "Critical threshold", "#d62728")):
        fig.add_hline(
            y=q, line_dash="dot", line_color=color,
            annotation_text=name, annotation_position="top left",
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Weekly dengue cases",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


@st.cache_data(show_spinner=False)
def weather_fig(weather: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=weather["date"], y=weather["rainfall_mm"], name="Rainfall (mm)",
               marker_color="#7fb3d5", yaxis="y")
    )
    fig.add_trace(
        go.Scatter(x=weather["date"], y=weather["avg_temp_c"], name="Avg temp °C",
                   mode="lines+markers", line=dict(color="#c0392b"), yaxis="y2")
    )
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Rainfall (mm)"),
        yaxis2=dict(title="Avg temp °C", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15),
        title=dict(text="Assumed weather inputs", font=dict(size=14)),
    )
    return fig


@st.cache_data(show_spinner=False)
def error_over_time_fig() -> go.Figure:
    hs = holdout_scores()
    weekly = hs.groupby("date").agg(
        mae=("abs_error", "mean"),
        rmse=("residual", lambda s: float(np.sqrt((s**2).mean()))),
    ).reset_index()
    weekly["mae_4w"] = weekly["mae"].rolling(4, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["date"], y=weekly["rmse"], mode="lines",
                             name="Weekly RMSE", line=dict(color="#c0392b", width=1)))
    fig.add_trace(go.Scatter(x=weekly["date"], y=weekly["mae"], mode="lines",
                             name="Weekly MAE", line=dict(color="#2471a3", width=1)))
    fig.add_trace(go.Scatter(x=weekly["date"], y=weekly["mae_4w"], mode="lines",
                             name="MAE (4-week avg)", line=dict(color="#2471a3", width=3)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="Cases", title=dict(text="Error over time", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def hardest_wards_fig() -> go.Figure:
    hs = holdout_scores()
    ward_mae = hs.groupby(["ward_id", "city"])["abs_error"].mean().reset_index()
    ward_mae = ward_mae.sort_values("abs_error", ascending=False).head(15).iloc[::-1]
    fig = px.bar(
        ward_mae, x="abs_error", y="ward_id", color="city", orientation="h",
        labels={"abs_error": "Mean absolute error (cases)", "ward_id": "", "city": "City"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      title=dict(text="Hardest wards (holdout MAE)", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def obs_vs_pred_fig() -> go.Figure:
    hs = holdout_scores()
    sample = hs.sample(min(2500, len(hs)), random_state=0)
    fig = px.scatter(
        sample, x="actual_cases", y="predicted_cases", color="actual_tier",
        color_discrete_map=TIER_COLORS, category_orders={"actual_tier": TIER_ORDER},
        opacity=0.55,
        labels={"actual_cases": "Observed cases", "predicted_cases": "Predicted cases",
                "actual_tier": "Observed tier"},
    )
    lim = float(max(sample["actual_cases"].max(), sample["predicted_cases"].max())) * 1.05
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                  line=dict(color="black", dash="dot"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      title=dict(text="Observed vs predicted (holdout sample)", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def confusion_fig() -> go.Figure:
    hs = holdout_scores()
    conf = pd.crosstab(hs["actual_tier"], hs["risk_tier"])
    conf = conf.reindex(index=TIER_ORDER, columns=TIER_ORDER, fill_value=0)
    fig = go.Figure(
        go.Heatmap(
            z=conf.to_numpy(), x=[t.capitalize() for t in TIER_ORDER],
            y=[t.capitalize() for t in TIER_ORDER], text=conf.to_numpy(),
            texttemplate="%{text}", colorscale="Blues", showscale=False,
            hovertemplate="Observed %{y} → predicted %{x}: %{z} ward-weeks<extra></extra>",
        )
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Predicted tier", yaxis_title="Observed tier",
                      title=dict(text="Tier confusion matrix", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def ward_drill_fig(ward: str) -> go.Figure:
    hs = holdout_scores()
    sub = hs[hs["ward_id"] == ward].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["actual_cases"], name="Observed",
                             mode="lines+markers", line=dict(color="#7f7f7f")))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["predicted_cases"], name="Predicted",
                             mode="lines+markers", line=dict(color="#2471a3", dash="dash")))
    fig.add_trace(go.Bar(x=sub["date"], y=sub["abs_error"], name="Absolute error",
                         marker_color="#e6b0aa", yaxis="y2"))
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Cases"),
        yaxis2=dict(title="Abs error", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12),
        title=dict(text=f"{ward} — observed vs predicted over holdout", font=dict(size=15)),
    )
    return fig


@st.cache_data(show_spinner=False)
def importance_fig() -> go.Figure:
    imp = global_importance()
    show = imp.sort_values("importance").tail(14)
    fig = px.bar(
        show, x="importance", y="label", orientation="h",
        error_x=show["std"],
        labels={"importance": "MAE increase when shuffled (cases)", "label": ""},
    )
    fig.update_layout(height=480, margin=dict(l=10, r=20, t=20, b=10))
    return fig


@st.cache_data(show_spinner=False)
def city_weekly_fig() -> go.Figure:
    data = get_data()
    city_weekly = data.groupby(["date", "city"])[TARGET_COLUMN].sum().reset_index()
    fig = px.line(city_weekly, x="date", y=TARGET_COLUMN, color="city",
                  labels={TARGET_COLUMN: "Weekly cases", "date": "", "city": "City"})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      title=dict(text="Weekly cases by city", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def seasonality_fig() -> go.Figure:
    data = get_data()
    tmp = data.copy()
    tmp["iso_week"] = tmp["date"].dt.isocalendar().week.astype(int)
    seasonal = tmp.groupby("iso_week")[TARGET_COLUMN].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seasonal["iso_week"], y=seasonal["mean"] + seasonal["std"],
                             mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=seasonal["iso_week"], y=seasonal["mean"] - seasonal["std"],
                             mode="lines", fill="tonexty", fillcolor="rgba(36,113,163,0.15)",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=seasonal["iso_week"], y=seasonal["mean"], mode="lines+markers",
                             name="Mean cases", line=dict(color="#2471a3")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title="ISO week of year", yaxis_title="Weekly cases",
                      title=dict(text="Seasonality across the year (mean ± 1 SD)", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def corr_fig() -> go.Figure:
    data = get_data()
    corr_cols = WEATHER_COLS + [c for c in NUMERIC_FEATURES if "lag" in c] + [TARGET_COLUMN]
    corr = data[corr_cols].corr(method="spearman").round(2)
    corr.index = [FRIENDLY.get(c, c) for c in corr.index]
    corr.columns = [FRIENDLY.get(c, c) for c in corr.columns]
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="Spearman correlation (weather ↔ cases)", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def rain_scatter_fig() -> go.Figure:
    data = get_data()
    sample = data.sample(min(3000, len(data)), random_state=0)
    fig = px.scatter(
        sample, x="rainfall_mm", y=TARGET_COLUMN, color="city", opacity=0.4,
        labels={"rainfall_mm": "Weekly rainfall (mm)", TARGET_COLUMN: "Weekly cases"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      title=dict(text="Rainfall vs dengue cases", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def dist_fig() -> go.Figure:
    data = get_data()
    sample = data.sample(min(3000, len(data)), random_state=0)
    fig = px.histogram(
        sample, x=TARGET_COLUMN, color="city", barmode="overlay", nbins=40, log_y=True,
        labels={TARGET_COLUMN: "Weekly cases per ward", "count": "Ward-weeks (log)"},
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=20, b=10),
                      title=dict(text="Case-count distribution by city", font=dict(size=15)))
    return fig


@st.cache_data(show_spinner=False)
def outlier_fig() -> go.Figure | None:
    data = get_data()
    flag_cols = [c for c in data.columns if c.endswith("_outlier")]
    if not flag_cols:
        return None
    counts = data[flag_cols].sum().sort_values(ascending=True)
    labels = [c.replace("_outlier", "").replace("_", " ") for c in counts.index]
    fig = px.bar(
        x=counts.to_numpy(), y=labels, orientation="h",
        labels={"x": "Flagged records", "y": ""},
    )
    fig.update_layout(height=300, margin=dict(l=10, r=20, t=20, b=10),
                      title=dict(text="Outlier flags (sanity + seasonal robust z-scores)",
                                 font=dict(size=15)))
    return fig


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def ward_selector(data: pd.DataFrame, key_prefix: str) -> tuple[str, pd.DataFrame]:
    cities = sorted(data["city"].unique())
    city = st.selectbox("City corporation", cities, key=f"{key_prefix}_city")
    wards = sorted(data[data["city"] == city]["ward_id"].unique())
    ward = st.selectbox("Ward", wards, key=f"{key_prefix}_ward")
    return ward, data[data["ward_id"] == ward].sort_values("date").reset_index(drop=True)


def predict_tab(data: pd.DataFrame) -> None:
    st.subheader("Predict risk for one ward-week")
    st.caption(
        "Pick a ward and week — inputs refresh from that observation. Adjust weather to test "
        "what-if scenarios; lag weather can be overridden under *Advanced*."
    )
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        ward, ward_df = ward_selector(data, "pred")
        labels = [
            f"{d:%Y-%m-%d}  (week {int(pd.Timestamp(d).isocalendar().week):02d})"
            for d in ward_df["date"]
        ]
        ix = st.selectbox("Target week", range(len(labels)), index=len(labels) - 1,
                          format_func=lambda i: labels[i], key="pred_week")
        row = ward_df.iloc[ix]

        context = f"{ward}|{row['date']:%Y-%m-%d}"
        if st.session_state.get("pred_context") != context:
            input_values = {
                "pred_t": row["avg_temp_c"],
                "pred_r": row["rainfall_mm"],
                "pred_h": row["relative_humidity_pct"],
                "pred_maxt": row["max_temp_c"],
                "pred_mint": row["min_temp_c"],
            }
            for col in NUMERIC_FEATURES:
                if "lag" in col or col in STATIC_COLS:
                    input_values[f"pred_{col}"] = row[col]
            for key, value in input_values.items():
                st.session_state[key] = float(value)
            st.session_state["pred_context"] = context

        st.markdown("**Current-week weather**")
        c1, c2 = st.columns(2)
        temp = c1.slider("Avg temp °C", 5.0, 45.0, float(row["avg_temp_c"]), 0.1, key="pred_t")
        rain = c2.slider("Rainfall mm", 0.0, 400.0, float(row["rainfall_mm"]), 0.5, key="pred_r")
        hum = c1.slider("Humidity %", 0.0, 100.0, float(row["relative_humidity_pct"]), 0.5, key="pred_h")
        maxt = c2.slider("Max temp °C", 5.0, 50.0, float(row["max_temp_c"]), 0.1, key="pred_maxt")
        mint = st.slider("Min temp °C", -5.0, 40.0, float(row["min_temp_c"]), 0.1, key="pred_mint")

        overrides = {
            "avg_temp_c": temp, "rainfall_mm": rain, "relative_humidity_pct": hum,
            "max_temp_c": maxt, "min_temp_c": mint,
        }
        with st.expander("Advanced: lag weather & ward profile"):
            for col in NUMERIC_FEATURES:
                if "lag" not in col and col not in STATIC_COLS:
                    continue
                step = 0.01 if col in STATIC_COLS else 0.1
                overrides[col] = st.number_input(
                    FRIENDLY.get(col, col), value=float(row[col]), step=step, format="%.2f",
                    key=f"pred_{col}",
                )

    feat = {c: float(row[c]) for c in NUMERIC_FEATURES}
    feat.update({k: float(v) for k, v in overrides.items()})
    values = tuple(feat[c] for c in NUMERIC_FEATURES)

    drivers, pred, base_pred = compute_drivers(ward, row["city"], values)
    tier = tier_for(ward, pred)
    cut = ward_cutoffs(ward)
    lo, hi = tier_interval(pred, tier, 80)

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted cases", f"{pred:.1f}", delta=f"{pred - base_pred:+.1f} vs ward normal")
        m2.metric("Risk tier", tier.capitalize())
        m3.metric("80% interval", f"{lo:.1f} – {hi:.1f}")
        st.markdown(
            f"Ward thresholds: medium ≥ **{cut['q75']:.0f}**, high ≥ **{cut['q90']:.0f}**, "
            f"critical ≥ **{cut['q97']:.0f}** weekly cases",
            help="Per-ward quantiles (q75/q90/q97) of observed cases; the same count can mean "
                 "different risk in different wards.",
        )
        st.plotly_chart(
            gauge_fig(pred / cut["q90"], tier, cut["q75"], cut["q90"], cut["q97"]),
            use_container_width=True,
        )

        observed = float(row[TARGET_COLUMN])
        obs_tier = tier_for(ward, observed)
        st.info(
            f"Observed in {row['date']:%Y-%m-%d}: **{observed:.0f} cases** "
            f"({obs_tier}) — model {'over' if pred > observed else 'under'}-predicts by "
            f"{abs(observed - pred):.1f}."
        )

        st.markdown("**What is driving this prediction?**")
        if drivers.empty:
            st.caption("All inputs sit at the ward's historical median — no strong driver this week.")
        else:
            st.plotly_chart(drivers_fig(drivers), use_container_width=True)
            show = drivers.head(8)[["label", "actual", "normal", "delta"]].copy()
            show.columns = ["Driver", "This input", "Ward normal", "Δ cases"]
            show["This input"] = show["This input"].round(2)
            show["Ward normal"] = show["Ward normal"].round(2)
            show["Δ cases"] = show["Δ cases"].round(2)
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption(
                "Contribution = prediction with the real input minus prediction with that input "
                "set to the ward's median. Approximate — features interact, so values won't sum "
                "exactly to the total."
            )


def heatmap_tab() -> None:
    st.subheader("Ward-level risk heatmap")
    st.caption("Every ward-week at a glance — spot outbreaks spreading across wards and weeks.")
    scores = score_history()
    years = sorted(scores["date"].dt.year.unique())
    cities = sorted(scores["city"].unique())
    c1, c2, c3 = st.columns([1, 2, 2])
    year = c1.selectbox("Year", years, index=len(years) - 1)
    sel_cities = c2.multiselect("City corporations", cities, default=cities)
    metric = c3.radio("Colour by", ["Predicted risk tier", "Predicted cases", "Observed cases"],
                      horizontal=True)

    fig = heatmap_fig(year, tuple(sel_cities), metric)
    if fig is None:
        st.warning("No data for this selection.")
        return
    st.plotly_chart(fig, use_container_width=True)

    fig2 = tier_mix_fig(year, tuple(sel_cities))
    if fig2 is not None:
        st.markdown("**Tier mix per city corporation** (share of ward-weeks)")
        st.plotly_chart(fig2, use_container_width=True)


def forecast_tab(data: pd.DataFrame) -> None:
    st.subheader("Multi-week forecast with confidence intervals")
    st.caption(
        "Forecast ahead from an anchor week using only observations available at that time. "
        "Future weather is assumed from ward climatology or the anchor week's weather. "
        "Bands are empirical intervals from 2024–25 holdout residuals."
    )
    left, right = st.columns([1, 1.9], gap="large")
    with left:
        ward, ward_df = ward_selector(data, "fc")
        labels = [f"{d:%Y-%m-%d}" for d in ward_df["date"]]
        anchor_ix = st.selectbox("Anchor week", range(len(labels)), index=len(labels) - 1,
                                 format_func=lambda i: labels[i], key="fc_anchor")
        anchor = str(pd.Timestamp(ward_df.iloc[anchor_ix]["date"]).date())
        horizon = st.slider("Horizon (weeks)", 1, 12, 6)
        scenario = st.radio(
            "Future weather",
            ["Ward climatology (typical for that week)", "Repeat last observed week"],
        )
        c1, c2 = st.columns(2)
        temp_delta = c1.slider("Temp anomaly °C", -3.0, 3.0, 0.0, 0.5)
        rain_mult = c2.selectbox("Rainfall multiplier", [0.5, 0.75, 1.0, 1.5, 2.0], index=2)
        level = st.selectbox("Interval level", [50, 80, 95], index=1)

    table, weather = compute_forecast(
        ward, anchor, horizon, scenario, temp_delta, rain_mult, level
    )
    cut = ward_cutoffs(ward)

    with right:
        st.plotly_chart(
            forecast_fig(ward, anchor, level, cut["q90"], cut["q97"], table),
            use_container_width=True,
        )
        st.plotly_chart(weather_fig(weather), use_container_width=True)

    show = table.copy()
    show["week"] = show["week"].dt.strftime("%Y-%m-%d")
    show.columns = ["Week", "Predicted", "Tier", "Interval low", "Interval high", "Top driver (Δ cases)"]
    st.dataframe(show, hide_index=True, use_container_width=True)
    csv = table.assign(week=table["week"].dt.strftime("%Y-%m-%d")).to_csv(index=False)
    st.download_button("Download forecast CSV", csv, file_name=f"forecast_{ward}.csv",
                       mime="text/csv")


def validation_tab() -> None:
    st.subheader("Chronological validation — holdout 2024–2025")
    st.caption(
        "At first load the dashboard retrains an evaluation model on 2016–2023 only, so every "
        "number below is out-of-sample (the deployed model refits on all data). Errors are in "
        "weekly-case units."
    )
    _model, metadata, _cutoffs, _global = get_artifacts()
    m = metadata["metrics"]
    det = m["high_risk_detection"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("MAE", f"{m['mae_cases']:.2f}", help="Mean absolute error, weekly cases")
    c2.metric("RMSE", f"{m['rmse_cases']:.2f}", help="Root mean squared error, weekly cases")
    c3.metric("R²", f"{m['r2_cases']:.3f}")
    c4.metric("Spearman ρ", f"{m['spearman_rho']:.3f}",
              help="Rank correlation between predicted and observed cases")
    c5.metric("Tier accuracy", f"{m['tier_accuracy']:.1%}")
    c6.metric("High-risk recall", f"{det['recall']:.1%}",
              help="Share of true high-risk ward-weeks (≥ ward q90) that the model flags")

    st.plotly_chart(error_over_time_fig(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(hardest_wards_fig(), use_container_width=True)
    with right:
        st.plotly_chart(obs_vs_pred_fig(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_fig(), use_container_width=True)
    with right:
        hs = holdout_scores()
        ward_mae_all = hs.groupby("ward_id")["abs_error"].mean()
        wards_sorted = sorted(hs["ward_id"].unique())
        ward = st.selectbox("Drill into ward", wards_sorted,
                            index=wards_sorted.index(ward_mae_all.idxmax()), key="val_ward")
        st.plotly_chart(ward_drill_fig(ward), use_container_width=True)

    st.markdown("**What drives the model overall?** (permutation importance on holdout MAE)")
    st.plotly_chart(importance_fig(), use_container_width=True)
    st.caption(
        f"Holdout: {m['rows']:,} ward-weeks · {metadata['holdout_window'][0]} to "
        f"{metadata['holdout_window'][1]} · high-risk detection ROC-AUC {det['roc_auc']:.3f}, "
        f"precision {det['precision']:.3f}"
    )


def explorer_tab() -> None:
    data = get_data()
    st.subheader("Statistical data explorer")
    st.caption(f"Weekly ward-level observations, {data['date'].min():%Y-%m-%d} to "
               f"{data['date'].max():%Y-%m-%d} · {data['ward_id'].nunique()} wards · "
               f"{data['city'].nunique()} city corporations.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(city_weekly_fig(), use_container_width=True)
    with right:
        st.plotly_chart(seasonality_fig(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(corr_fig(), use_container_width=True)
    with right:
        st.plotly_chart(rain_scatter_fig(), use_container_width=True)
        st.plotly_chart(dist_fig(), use_container_width=True)

    fig = outlier_fig()
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    with st.sidebar:
        st.title("🦟 Dengue Ward Risk")
        st.caption("Ward-level dengue early-warning dashboard")
        _model, metadata, _cutoffs, _global = get_artifacts()
        m = metadata["metrics"]
        st.markdown(
            f"**Model:** {metadata['model']}  \n"
            f"**Trained on:** {metadata['rows']['train']:,} ward-weeks "
            f"({metadata['train_window'][0]} → {metadata['train_window'][1]})  \n"
            f"**Holdout:** {metadata['rows']['holdout']:,} ward-weeks (2024–2025)"
        )
        st.divider()
        st.markdown(
            "**Holdout performance**  \n"
            f"MAE {m['mae_cases']:.2f} · RMSE {m['rmse_cases']:.2f} · "
            f"R² {m['r2_cases']:.2f}  \n"
            f"Tier accuracy {m['tier_accuracy']:.1%}"
        )
        st.divider()
        legend = " · ".join(
            f"<span style='color:{TIER_COLORS[t]};font-weight:700;'>{t}</span>"
            for t in TIER_ORDER
        )
        st.markdown(f"**Risk tiers:**  {legend}", unsafe_allow_html=True)
        st.caption(
            "Tiers use per-ward quantiles of observed weekly cases "
            "(medium ≥ q75, high ≥ q90, critical ≥ q97)."
        )
        data = get_data()
        st.caption(
            f"Coverage: {data['date'].min():%Y-%m-%d} → {data['date'].max():%Y-%m-%d} · "
            f"{data['ward_id'].nunique()} wards · {data['city'].nunique()} cities"
        )


def main() -> None:
    sidebar()
    st.title("Dengue Ward Risk Intelligence")
    data = get_data()

    tab_predict, tab_heat, tab_forecast, tab_valid, tab_explore = st.tabs(
        ["🎯 Predict & Drivers", "🗺️ Risk Heatmap", "📈 Multi-Week Forecast",
         "✅ Model Validation", "📊 Data Explorer"]
    )
    with tab_predict:
        predict_tab(data)
    with tab_heat:
        heatmap_tab()
    with tab_forecast:
        forecast_tab(data)
    with tab_valid:
        validation_tab()
    with tab_explore:
        explorer_tab()


if __name__ == "__main__":
    main()
