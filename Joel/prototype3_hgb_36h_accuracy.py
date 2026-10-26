from pathlib import Path
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output" / "prototype3_hgb_36h_accuracy"
MODELS = ROOT / "models" / "prototype3_hgb_36h_accuracy"

TARGET = "consumption_w"
TIMESTAMP = "timestamp_utc"
BUILDING_ID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"
BUILDING_TYPE = "Farm"
HISTORY_DAYS = 460
TEST_FRACTION = 0.10
MAPE_MIN_ACTUAL_W = 100.0  # 0.1 kW, matching the Ridge percentage evaluation
STEPS = 144
SEED = 42
MAX_ITER = int(os.environ.get("HGB_MAX_ITER", "200"))

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
    "lag_2h",
    "lag_6h",
    "lag_24h",
    "lag_48h",
    "lag_7d",
    "change_15m",
    "change_1h",
    "consumption_mean_1h",
    "consumption_mean_6h",
    "consumption_mean_24h",
    "consumption_std_1h",
    "consumption_std_24h",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "month",
]

SELECTED_HORIZONS = [1, 24, 48, 96, 144]


def prepare_data():
    electricity = pd.read_csv(DATA / "electricity.csv")
    weather = pd.read_csv(DATA / "weather.csv")
    buildings = pd.read_csv(DATA / "buildings.csv")

    building_rows = buildings.loc[
        buildings["building_id"].astype(str) == BUILDING_ID
    ]
    if len(building_rows) != 1:
        raise ValueError(
            f"Expected one metadata row for building {BUILDING_ID}; "
            f"found {len(building_rows)}."
        )
    building_description = str(building_rows.iloc[0]["description"])
    if BUILDING_TYPE.lower() not in building_description.lower():
        raise ValueError(
            f"Building {BUILDING_ID} is described as {building_description!r}, "
            f"not {BUILDING_TYPE!r}."
        )

    for df in (electricity, weather):
        df[TIMESTAMP] = pd.to_datetime(
            df[TIMESTAMP], utc=True, errors="coerce"
        )

    e = (
        electricity.loc[
            electricity["building_id"] == BUILDING_ID,
            ["building_id", TIMESTAMP, TARGET],
        ]
        .dropna(subset=[TIMESTAMP, TARGET])
        .sort_values(TIMESTAMP)
        .drop_duplicates(subset=[TIMESTAMP], keep="last")
        .copy()
    )

    if e.empty:
        raise ValueError(f"No electricity rows found for building {BUILDING_ID}.")

    # Use the latest available 460-day history for the selected Farm building.
    available_end = e[TIMESTAMP].max()
    available_days = (available_end - e[TIMESTAMP].min()).total_seconds() / 86400
    if available_days < HISTORY_DAYS - 1:
        raise ValueError(
            f"The selected building has only {available_days:.1f} days of data; "
            f"approximately {HISTORY_DAYS} days are required."
        )
    history_window_start = available_end - pd.Timedelta(days=HISTORY_DAYS)
    e = e[e[TIMESTAMP] >= history_window_start].copy()
    raw_rows = len(e)
    negative_rows = e[e[TARGET] < 0].copy()
    e = e[e[TARGET] >= 0].copy()
    if e.empty:
        raise ValueError("No non-negative consumption rows remain in the 460-day window.")
    history_start = e[TIMESTAMP].min()
    history_end = e[TIMESTAMP].max()
    history_days = (history_end - history_start).total_seconds() / 86400

    weather_cols = [c for c in WEATHER_COLS if c in weather.columns]
    w = weather.loc[
        weather["building_id"] == BUILDING_ID,
        ["building_id", TIMESTAMP] + weather_cols,
    ].drop_duplicates(subset=[TIMESTAMP], keep="last")

    frame = (
        e.merge(w, on=["building_id", TIMESTAMP], how="left")
        .sort_values(TIMESTAMP)
        .reset_index(drop=True)
    )

    series = frame.set_index(TIMESTAMP)[TARGET]
    frame["origin_consumption"] = frame[TARGET]

    lag_deltas = {
        "lag_15m": "15min",
        "lag_1h": "1h",
        "lag_2h": "2h",
        "lag_6h": "6h",
        "lag_24h": "24h",
        "lag_48h": "48h",
        "lag_7d": "7D",
    }
    for name, delta in lag_deltas.items():
        frame[name] = (
            frame[TIMESTAMP] - pd.Timedelta(delta)
        ).map(series)

    frame["change_15m"] = frame["origin_consumption"] - frame["lag_15m"]
    frame["change_1h"] = frame["origin_consumption"] - frame["lag_1h"]

    indexed = frame.set_index(TIMESTAMP)[TARGET]
    for window in ["1h", "6h", "24h"]:
        rolling = indexed.rolling(window, min_periods=1)
        frame[f"consumption_mean_{window}"] = rolling.mean().to_numpy()
        if window in ("1h", "24h"):
            frame[f"consumption_std_{window}"] = rolling.std().to_numpy()

    minutes = frame[TIMESTAMP].dt.hour * 60 + frame[TIMESTAMP].dt.minute
    day = frame[TIMESTAMP].dt.dayofweek
    frame["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    frame["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    frame["dow_sin"] = np.sin(2 * np.pi * day / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * day / 7)
    frame["is_weekend"] = (day >= 5).astype(int)
    frame["month"] = frame[TIMESTAMP].dt.month

    # Build all future-label columns in a separate frame to avoid pandas
    # fragmentation from repeatedly inserting 144 columns.
    target_columns = {}
    for step in range(1, STEPS + 1):
        target_time = frame[TIMESTAMP] + pd.Timedelta(minutes=15 * step)
        target_columns[f"y_{step}"] = target_time.map(series).to_numpy(dtype=float)
    frame = pd.concat(
        [frame, pd.DataFrame(target_columns, index=frame.index)], axis=1
    )

    quality = {
        "building_id": BUILDING_ID,
        "building_type": BUILDING_TYPE,
        "building_description": building_description,
        "history_days_requested": HISTORY_DAYS,
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "history_days_used": history_days,
        "rows_before_negative_filter": raw_rows,
        "negative_rows_excluded": len(negative_rows),
        "rows_after_filter": len(frame),
        "weather_features": weather_cols,
    }

    return frame, series, negative_rows, weather_cols, quality


def features_for_horizon(frame, step, weather_cols):
    target_time = frame[TIMESTAMP] + pd.Timedelta(minutes=15 * step)
    minutes = target_time.dt.hour * 60 + target_time.dt.minute
    day = target_time.dt.dayofweek

    features = frame[BASE_FEATURES + weather_cols].copy()
    features["target_tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    features["target_tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    features["target_dow_sin"] = np.sin(2 * np.pi * day / 7)
    features["target_dow_cos"] = np.cos(2 * np.pi * day / 7)
    features["target_month"] = target_time.dt.month

    return features


def make_model():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=MAX_ITER,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=SEED,
    )


def make_baselines(test, series, step):
    actual = test[f"y_{step}"].to_numpy(dtype=float)
    persistence = test["origin_consumption"].to_numpy(dtype=float)

    target_time = test[TIMESTAMP] + pd.Timedelta(minutes=15 * step)
    back_days = 1 if step <= 96 else 2
    copy_time = target_time - pd.Timedelta(days=back_days)
    copy_yesterday = copy_time.map(series).to_numpy(dtype=float)

    return actual, persistence, copy_yesterday


def add_metrics(rows, name, actual, predicted, step):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return

    actual_valid = actual[mask]
    predicted_valid = predicted[mask]
    mae = mean_absolute_error(actual_valid, predicted_valid)
    rmse = mean_squared_error(actual_valid, predicted_valid) ** 0.5
    mean_actual = float(np.mean(actual_valid))

    # Match the team's 90/10 Ridge percentage evaluation: omit actuals below
    # 0.1 kW from MAPE. NMAE and NRMSE are normalized by mean actual demand.
    mape_mask = mask & (np.abs(actual) >= MAPE_MIN_ACTUAL_W)
    if mape_mask.any():
        mape_pct = float(
            np.mean(
                np.abs(
                    (actual[mape_mask] - predicted[mape_mask])
                    / actual[mape_mask]
                )
            )
            * 100
        )
    else:
        mape_pct = np.nan

    nmae_pct = mae / mean_actual * 100 if mean_actual > 0 else np.nan
    nrmse_pct = rmse / mean_actual * 100 if mean_actual > 0 else np.nan

    rows.append(
        {
            "model": name,
            "horizon_step": step,
            "horizon_hours": step / 4,
            "n": int(mask.sum()),
            "n_mape": int(mape_mask.sum()),
            "mean_actual_w": mean_actual,
            "mae_w": mae,
            "rmse_w": rmse,
            "mape_pct": mape_pct,
            "nmae_pct": nmae_pct,
            "nrmse_pct": nrmse_pct,
        }
    )


def add_overall_metrics(rows, name, actual_matrix, predicted_matrix):
    mask = np.isfinite(actual_matrix) & np.isfinite(predicted_matrix)
    if not mask.any():
        return

    actual = actual_matrix[mask]
    predicted = predicted_matrix[mask]
    mae_w = mean_absolute_error(actual, predicted)
    rmse_w = mean_squared_error(actual, predicted) ** 0.5
    mean_actual_w = float(np.mean(actual))

    mape_mask = np.abs(actual) >= MAPE_MIN_ACTUAL_W
    if mape_mask.any():
        mape_pct = float(
            np.mean(
                np.abs(
                    (actual[mape_mask] - predicted[mape_mask])
                    / actual[mape_mask]
                )
            )
            * 100
        )
    else:
        mape_pct = np.nan

    nmae_pct = mae_w / mean_actual_w * 100 if mean_actual_w > 0 else np.nan
    nrmse_pct = rmse_w / mean_actual_w * 100 if mean_actual_w > 0 else np.nan

    rows.append(
        {
            "validation": "90_10",
            "model": name,
            "mae_kw": mae_w / 1000,
            "rmse_kw": rmse_w / 1000,
            "mean_actual_kw": mean_actual_w / 1000,
            "mape_pct": mape_pct,
            "mape_excluded_n": int((~mape_mask).sum()),
            "horizons": int(actual_matrix.shape[1]),
            "n_predictions": int(mask.sum()),
            "nmae_pct": nmae_pct,
            "nrmse_pct": nrmse_pct,
        }
    )


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    frame, series, negative_rows, weather_cols, quality = prepare_data()
    negative_rows.to_csv(OUT / "excluded_negative_consumption.csv", index=False)

    complete = frame[np.isfinite(frame[f"y_{STEPS}"])].copy()
    if len(complete) < 500:
        raise ValueError("Not enough complete forecast origins for a 36-hour run.")

    cut = int(len(complete) * (1.0 - TEST_FRACTION))
    test_start = complete[TIMESTAMP].iloc[cut]
    test = complete[complete[TIMESTAMP] >= test_start].copy()

    # Training labels must end before the test period starts.
    train_before = test_start - pd.Timedelta(hours=36)
    train = complete[complete[TIMESTAMP] < train_before].copy()
    if len(train) < 500:
        raise ValueError("Not enough training rows after the 36-hour gap.")

    models = []
    predictions = np.full((len(test), STEPS), np.nan)
    actuals_all = np.full((len(test), STEPS), np.nan)
    persistence_all = np.full((len(test), STEPS), np.nan)
    yesterday_all = np.full((len(test), STEPS), np.nan)
    metric_rows = []
    prediction_parts = []

    for step in range(1, STEPS + 1):
        y_train = train[f"y_{step}"].to_numpy(dtype=float)
        valid_train = np.isfinite(y_train)

        model = make_model()
        x_train = features_for_horizon(
            train.loc[valid_train], step, weather_cols
        )
        x_test = features_for_horizon(test, step, weather_cols)

        model.fit(x_train, y_train[valid_train])
        predictions[:, step - 1] = model.predict(x_test)
        if step == 1 or step % 12 == 0 or step == STEPS:
            print(f"Completed horizon model {step}/{STEPS}", flush=True)
        models.append(model)

        actual, persistence, yesterday = make_baselines(test, series, step)
        forecast = predictions[:, step - 1]
        actuals_all[:, step - 1] = actual
        persistence_all[:, step - 1] = persistence
        yesterday_all[:, step - 1] = yesterday

        add_metrics(metric_rows, "HGB", actual, forecast, step)
        add_metrics(metric_rows, "Persistence", actual, persistence, step)
        add_metrics(metric_rows, "Copy-Yesterday", actual, yesterday, step)

        if step in SELECTED_HORIZONS:
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "origin_timestamp": test[TIMESTAMP].to_numpy(),
                        "target_timestamp": (
                            test[TIMESTAMP] + pd.Timedelta(minutes=15 * step)
                        ).to_numpy(),
                        "horizon_step": step,
                        "horizon_hours": step / 4,
                        "actual_w": actual,
                        "hgb_w": forecast,
                        "persistence_w": persistence,
                        "copy_yesterday_w": yesterday,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions_out = pd.concat(prediction_parts, ignore_index=True)

    metrics.to_csv(OUT / "metrics_by_horizon.csv", index=False)
    predictions_out.to_csv(
        OUT / "predictions_selected_horizons.csv", index=False
    )

    model_path = MODELS / "horizon_models.joblib"
    joblib.dump(
        {
            "models": models,
            "base_features": BASE_FEATURES,
            "weather_features": weather_cols,
            "building_id": BUILDING_ID,
            "building_type": BUILDING_TYPE,
            "building_description": quality["building_description"],
            "forecast_horizon": "36 hours / 144 steps",
            "resolution": "15 minutes",
        },
        model_path,
        compress=3,
    )

    overall_rows = []
    for name, predicted in [
        ("Copy-Yesterday", yesterday_all),
        ("Persistence", persistence_all),
        ("HGB", predictions),
    ]:
        add_overall_metrics(overall_rows, name, actuals_all, predicted)
    summary = pd.DataFrame(overall_rows)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)

    run_summary = {
        "experiment": "36-hour horizon-specific HGB",
        "building_id": BUILDING_ID,
        "building_type": BUILDING_TYPE,
        "building_description": quality["building_description"],
        "history_days_requested": HISTORY_DAYS,
        "history_start": quality["history_start"],
        "history_end": quality["history_end"],
        "history_days_used": quality["history_days_used"],
        "training_origins": len(train),
        "test_origins": len(test),
        "test_fraction": TEST_FRACTION,
        "test_start": test_start.isoformat(),
        "validation": (
            "chronological 90/10 holdout on complete forecast origins; "
            "purge 36 hours of training origins before the test start"
        ),
        "model_count": STEPS,
        "weather_features": weather_cols,
        "metrics": {
            "MAE": "mean absolute error, reported in kW",
            "RMSE": "root mean squared error, reported in kW",
            "MAPE": (
                "pooled absolute percentage error; excludes actual values "
                "below 100 W (0.1 kW)"
            ),
            "NMAE": "pooled MAE / pooled mean actual consumption * 100",
            "NRMSE": "pooled RMSE / pooled mean actual consumption * 100",
            "summary_scope": "all finite test-origin by forecast-horizon pairs",
        },
        "negative_rows_excluded": len(negative_rows),
        "model_file": str(model_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_quality": quality,
        "weather_note": (
            "Uses weather observed at forecast origin. Future weather forecasts "
            "can be added if available."
        ),
    }
    (OUT / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    print("90/10 chronological validation - controlled results:")
    print(summary.to_string(index=False))
    print(f"\nRuntime: {run_summary['runtime_seconds']:.2f} seconds")
    print(f"Output folder: {OUT}")


if __name__ == "__main__":
    main()
