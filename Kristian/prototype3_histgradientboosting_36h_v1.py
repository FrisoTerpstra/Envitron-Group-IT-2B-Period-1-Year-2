from pathlib import Path
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE
DATA = ROOT / "data"
OUT = ROOT / "output" / "prototype3_hgb_36h"
MODELS = ROOT / "models" / "prototype3_hgb_36h"

TARGET = "consumption_w"
TIMESTAMP = "timestamp_utc"
BUILDING_ID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"
STEPS = 144
SEED = 42

WEATHER_COLS = [
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "wind_speed_ms",
]

BASE_FEATURES = [
    "origin_consumption",
    "lag_15m",
    "lag_1h",
    "lag_24h",
    "lag_7d",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month",
]

SELECTED_HORIZONS = [1, 24, 48, 96, 144]


def load_data():
    electricity = pd.read_csv(DATA / "electricity.csv")
    weather = pd.read_csv(DATA / "weather.csv")
    buildings = pd.read_csv(DATA / "buildings.csv")

    for df in (electricity, weather):
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True, errors="coerce")

    return electricity, weather, buildings


def prepare_building_frame(building_id: str):
    electricity, weather, buildings = load_data()

    e = (
        electricity.loc[electricity["building_id"] == building_id, ["building_id", TIMESTAMP, TARGET]]
        .dropna(subset=[TIMESTAMP, TARGET])
        .sort_values(TIMESTAMP)
        .drop_duplicates(subset=[TIMESTAMP], keep="last")
        .copy()
    )

    raw_rows = len(e)
    negative_rows = e[e[TARGET] < 0].copy()
    e = e[e[TARGET] >= 0].copy()

    weather_cols_present = [c for c in WEATHER_COLS if c in weather.columns]
    w = (
        weather.loc[weather["building_id"] == building_id, ["building_id", TIMESTAMP] + weather_cols_present]
        .drop_duplicates(subset=["building_id", TIMESTAMP], keep="last")
        .copy()
    )

    x = (
        e.merge(w, on=["building_id", TIMESTAMP], how="left")
        .sort_values(TIMESTAMP)
        .reset_index(drop=True)
    )

    series_map = x.set_index(TIMESTAMP)[TARGET]

    x["origin_consumption"] = x[TARGET]
    lag_map = {
        "lag_15m": pd.Timedelta("15min"),
        "lag_1h": pd.Timedelta("1h"),
        "lag_24h": pd.Timedelta("24h"),
        "lag_7d": pd.Timedelta("7d"),
    }
    for name, delta in lag_map.items():
        x[name] = (x[TIMESTAMP] - delta).map(series_map)

    minutes = x[TIMESTAMP].dt.hour * 60 + x[TIMESTAMP].dt.minute
    day_of_week = x[TIMESTAMP].dt.dayofweek
    x["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    x["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    x["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    x["is_weekend"] = (day_of_week >= 5).astype(int)
    x["month"] = x[TIMESTAMP].dt.month

    for h in range(1, STEPS + 1):
        x[f"y_{h}"] = (x[TIMESTAMP] + pd.Timedelta(minutes=15 * h)).map(series_map)

    description = ""
    building_row = buildings.loc[buildings["building_id"] == building_id]
    if not building_row.empty and "description" in building_row.columns:
        description = building_row["description"].iloc[0]

    quality = pd.DataFrame([
        {
            "building_id": building_id,
            "description": description,
            "raw_rows": raw_rows,
            "negative_rows_excluded": len(negative_rows),
            "negative_pct_raw": (len(negative_rows) / raw_rows * 100) if raw_rows else np.nan,
            "rows_after_exclusion": len(x),
            "start": x[TIMESTAMP].min(),
            "end": x[TIMESTAMP].max(),
        }
    ])

    return x, negative_rows, quality, description, weather_cols_present


def build_model():
    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=SEED,
    )
    return MultiOutputRegressor(base)


def baseline_arrays(df):
    y = df[[f"y_{h}" for h in range(1, STEPS + 1)]].to_numpy(dtype=float)
    persistence = np.repeat(df["origin_consumption"].to_numpy(dtype=float)[:, None], STEPS, axis=1)

    smap = df.set_index(TIMESTAMP)[TARGET]
    copy_yesterday = np.full_like(y, np.nan)
    for h in range(1, STEPS + 1):
        target_ts = df[TIMESTAMP] + pd.Timedelta(minutes=15 * h)
        back_days = 1 if h <= 96 else 2
        copy_yesterday[:, h - 1] = (target_ts - pd.Timedelta(days=back_days)).map(smap).to_numpy(dtype=float)

    return y, persistence, copy_yesterday


def metric_rows(validation_name, model_name, actual, predicted):
    rows = []
    for h in range(STEPS):
        mask = np.isfinite(actual[:, h]) & np.isfinite(predicted[:, h])
        yy = actual[mask, h]
        pp = predicted[mask, h]
        if len(yy) == 0:
            continue
        rows.append(
            {
                "validation": validation_name,
                "model": model_name,
                "horizon_step": h + 1,
                "horizon_hours": (h + 1) / 4,
                "n": len(yy),
                "mae_w": mean_absolute_error(yy, pp),
                "rmse_w": mean_squared_error(yy, pp) ** 0.5,
            }
        )
    return rows


def run_split(df, validation_name, train_end, test_start, test_end, weather_cols, save_model=False):
    train = df[df[TIMESTAMP] <= train_end].copy()
    test = df[(df[TIMESTAMP] >= test_start) & (df[TIMESTAMP] <= test_end)].copy()

    y_train = train[[f"y_{h}" for h in range(1, STEPS + 1)]].to_numpy(dtype=float)
    y_test, persistence, copy_yesterday = baseline_arrays(test)

    valid_train = np.isfinite(y_train).all(axis=1)
    valid_test = np.isfinite(y_test).all(axis=1)

    rows = []
    predictions = {
        "Persistence": persistence,
        "Copy-Yesterday": copy_yesterday,
    }

    for baseline_name, baseline_pred in predictions.items():
        rows += metric_rows(validation_name, baseline_name, y_test[valid_test], baseline_pred[valid_test])

    model_variants = [("HGB", BASE_FEATURES)]
    if weather_cols:
        model_variants.append(("HGB + Weather", BASE_FEATURES + weather_cols))

    for label, features in model_variants:
        model = build_model()
        x_train = train.loc[valid_train, features]
        x_test = test.loc[valid_test, features]

        model.fit(x_train, y_train[valid_train])
        pred = np.full_like(y_test, np.nan)
        pred[valid_test] = model.predict(x_test)
        predictions[label] = pred
        rows += metric_rows(validation_name, label, y_test[valid_test], pred[valid_test])

        if save_model:
            MODELS.mkdir(parents=True, exist_ok=True)
            safe_name = label.lower().replace(" + ", "_").replace(" ", "_")
            joblib.dump(
                {
                    "model": model,
                    "features": features,
                    "building_id": BUILDING_ID,
                    "forecast_horizon": "36 hours / 144 outputs",
                    "validation": validation_name,
                },
                MODELS / f"{safe_name}_{validation_name}.joblib",
            )

    prediction_exports = []
    export_columns = {
        "Persistence": "persistence_w",
        "Copy-Yesterday": "copy_yesterday_w",
        "HGB": "hgb_w",
        "HGB + Weather": "hgb_weather_w",
    }
    for h in SELECTED_HORIZONS:
        idx = h - 1
        frame = pd.DataFrame(
            {
                "validation": validation_name,
                "origin_timestamp": test[TIMESTAMP],
                "target_timestamp": test[TIMESTAMP] + pd.Timedelta(minutes=15 * h),
                "horizon_step": h,
                "horizon_hours": h / 4,
                "actual_w": y_test[:, idx],
            }
        )
        for model_name, col_name in export_columns.items():
            if model_name in predictions:
                frame[col_name] = predictions[model_name][:, idx]
        prediction_exports.append(frame)

    return pd.DataFrame(rows), pd.concat(prediction_exports, ignore_index=True)


def create_summary(metrics):
    holdout = metrics[~metrics["validation"].str.startswith("walk_forward_")]
    summary = holdout.groupby(["validation", "model"], as_index=False).agg(
        mae_w=("mae_w", "mean"),
        rmse_w=("rmse_w", "mean"),
        horizons=("horizon_step", "nunique"),
    )

    wf = metrics[metrics["validation"].str.startswith("walk_forward_")].groupby("model", as_index=False).agg(
        mae_w=("mae_w", "mean"),
        rmse_w=("rmse_w", "mean"),
        horizons=("horizon_step", "nunique"),
    )
    if not wf.empty:
        wf["validation"] = "walk_forward"
        summary = pd.concat([summary, wf], ignore_index=True)

    return summary


def make_charts(summary, metrics, predictions):
    for metric, label in [("mae_w", "MAE"), ("rmse_w", "RMSE")]:
        pivot = summary.pivot(index="model", columns="validation", values=metric)
        ax = pivot.plot(kind="bar", figsize=(10, 6))
        ax.set_title(f"36-hour {label} — Models and Validation")
        ax.set_ylabel(f"{label} (W)")
        ax.set_xlabel("Model")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(OUT / f"model_validation_{label.lower()}.png", dpi=180)
        plt.close()

        subset = metrics[metrics["validation"] == "90_10"]
        if not subset.empty:
            plt.figure(figsize=(11, 6))
            for model_name, group in subset.groupby("model"):
                plt.plot(group["horizon_hours"], group[metric], label=model_name)
            plt.title(f"{label} Across 36-hour Forecast Horizon — 90/10")
            plt.xlabel("Forecast horizon (hours)")
            plt.ylabel(f"{label} (W)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT / f"horizon_{label.lower()}_90_10.png", dpi=180)
            plt.close()

    view = predictions[(predictions["validation"] == "90_10") & (predictions["horizon_step"] == 144)].dropna().tail(7 * 96)
    if not view.empty:
        plt.figure(figsize=(13, 6))
        for col, label in [
            ("actual_w", "Actual"),
            ("persistence_w", "Persistence"),
            ("copy_yesterday_w", "Copy-Yesterday"),
            ("hgb_w", "HGB"),
            ("hgb_weather_w", "HGB + Weather"),
        ]:
            if col in view.columns:
                plt.plot(view["target_timestamp"], view[col], label=label, alpha=0.85)
        plt.title("Actual vs Forecast — 36-hour Horizon (90/10)")
        plt.ylabel("Consumption (W)")
        plt.legend()
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(OUT / "actual_vs_forecast_36h.png", dpi=180)
        plt.close()


def main():
    start = time.perf_counter()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df, negatives, quality, description, weather_cols = prepare_building_frame(BUILDING_ID)
    negatives.to_csv(OUT / "excluded_negative_consumption.csv", index=False)
    quality.to_csv(OUT / "data_quality.csv", index=False)

    origins = df[np.isfinite(df[f"y_{STEPS}"])].copy()
    n = len(origins)
    if n < 500:
        raise ValueError("Not enough complete forecast origins to run a reliable 36-hour experiment.")

    metric_frames = []
    prediction_frames = []

    for frac, name in [(0.20, "80_20"), (0.10, "90_10")]:
        cut = int(n * (1 - frac))
        train_end = origins[TIMESTAMP].iloc[cut - 1]
        test_start = origins[TIMESTAMP].iloc[cut]
        test_end = origins[TIMESTAMP].iloc[-1]
        m, p = run_split(df, name, train_end, test_start, test_end, weather_cols, save_model=True)
        metric_frames.append(m)
        prediction_frames.append(p)

    bounds = np.linspace(int(n * 0.60), n, 4, dtype=int)
    for i in range(3):
        left, right = bounds[i], bounds[i + 1]
        train_end = origins[TIMESTAMP].iloc[left - 1]
        test_start = origins[TIMESTAMP].iloc[left]
        test_end = origins[TIMESTAMP].iloc[right - 1]
        m, _ = run_split(df, f"walk_forward_{i + 1}", train_end, test_start, test_end, weather_cols, save_model=False)
        metric_frames.append(m)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = create_summary(metrics)

    metrics.to_csv(OUT / "metrics_by_horizon.csv", index=False)
    predictions.to_csv(OUT / "predictions_selected_horizons.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)

    make_charts(summary, metrics, predictions)

    run_summary = {
        "experiment": "Prototype 3 - HistGradientBoostingRegressor 36-Hour Forecasting Experiment",
        "building_id": BUILDING_ID,
        "description": description,
        "forecast_horizon": "36 hours",
        "resolution": "15 minutes",
        "steps": STEPS,
        "models": ["Persistence", "Copy-Yesterday", "HGB"] + (["HGB + Weather"] if weather_cols else []),
        "validations": ["80/20 chronological", "90/10 chronological", "3-fold expanding walk-forward"],
        "metrics": ["MAE", "RMSE"],
        "negative_handling": f"{len(negatives)} negative rows excluded from modelling and exported separately; raw source data unchanged.",
        "weather_note": "Historical observed weather is used if weather columns are available. For real deployment, forecast-time weather should be used instead.",
        "runtime_seconds": time.perf_counter() - start,
    }
    (OUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))

    print(summary.to_string(index=False))
    print(f"\nRuntime: {run_summary['runtime_seconds']:.2f} seconds")
    print(f"\nOutput folder: {OUT}")


if __name__ == "__main__":
    main()
