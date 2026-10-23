#!/usr/bin/env python3
"""Auditable HGB 36-hour experiment with purged chronological evaluation.

Destinations:
    output/hgb_36h_v3/
    models/hgb_36h_v3/

Evaluation:
    Preserve the original chronological 90/10 cutoff calculation.
    Remove training windows whose targets reach the test boundary.
    Use only observations available at each origin for baselines.
    Compare all four models on identical origin-horizon pairs.
    Report native prediction availability separately.

Unchanged:
    Building, features, model parameters, preprocessing, forecast horizons,
    negative-value cleaning, and thread limit.

Existing matching filenames are overwritten atomically where practical.
Unrelated files in the destination directories are not deleted.
"""

import os

THREAD_LIMIT = 2

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, str(THREAD_LIMIT))

from pathlib import Path
from datetime import datetime, timezone
import json
import time
import traceback
import uuid

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Explicit names: never inferred from evaluation options.
EXPERIMENT = "hgb_36h_v3"
OUT = HERE / "output" / EXPERIMENT
MODELS = HERE / "models" / EXPERIMENT

TARGET = "consumption_w"
BID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"

STEPS = 144
INTERVAL_MINUTES = 15
SELECTED = [1, 24, 48, 96, 144]

CHECKPOINT_EVERY = 12
CSV_CHUNK_SIZE = 200_000
MAPE_MIN_ACTUAL_W = 100.0

PURGE_TRAINING_OVERLAP = True
BASELINE_LOOKUP = "full_history"
SAVE_MODELS = True

EVALUATION_PROTOCOL = "purged_90_10_common_pairs_v1"

WEATHER = [
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "wind_speed_ms",
]

BASE = [
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

TARGETS = [f"y_{h}" for h in range(1, STEPS + 1)]

FORECAST_DURATION = pd.Timedelta(
    minutes=INTERVAL_MINUTES * STEPS
)

# Identical to the previous experiment.
PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.1,
    "max_iter": 60,
    "max_leaf_nodes": 15,
    "max_bins": 63,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
}

VARIANTS = [
    ("HGB", BASE, "hgb_w", "hgb"),
    ("HGB + Weather", BASE + WEATHER, "hgb_weather_w", "hgb_weather"),
]

PREDICTION_COLUMNS = {
    "Persistence": "persistence_w",
    "Copy-Yesterday": "copy_yesterday_w",
    "HGB": "hgb_w",
    "HGB + Weather": "hgb_weather_w",
}

if not PURGE_TRAINING_OVERLAP:
    raise ValueError("This script requires purged training windows.")

if BASELINE_LOOKUP != "full_history":
    raise ValueError("This script requires full-history baseline lookup.")

if not all(1 <= h <= STEPS for h in SELECTED):
    raise ValueError("Selected horizons must be between 1 and STEPS.")


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def atomic_csv(table, path):
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(data, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_figure(fig, path):
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(
        temporary,
        format=path.suffix.lstrip("."),
        dpi=120,
    )
    temporary.replace(path)


# ---------------------------------------------------------------------
# Loading and feature engineering
# ---------------------------------------------------------------------

def read_building_csv(path, columns):
    """Read selected columns in chunks for one building."""
    parts = []

    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"building_id": "string"},
        chunksize=CSV_CHUNK_SIZE,
    ):
        selected = chunk.loc[chunk["building_id"] == BID].copy()
        if not selected.empty:
            parts.append(selected)

    if not parts:
        return pd.DataFrame(columns=columns)

    return pd.concat(parts, ignore_index=True)


def load_prepare():
    required = ["electricity.csv", "weather.csv", "buildings.csv"]
    missing = [
        str(DATA / name)
        for name in required
        if not (DATA / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(missing)
        )

    electricity = read_building_csv(
        DATA / "electricity.csv",
        ["building_id", "timestamp_utc", TARGET],
    )
    weather = read_building_csv(
        DATA / "weather.csv",
        ["building_id", "timestamp_utc"] + WEATHER,
    )
    buildings = read_building_csv(
        DATA / "buildings.csv",
        ["building_id", "description"],
    )

    if electricity.empty:
        raise ValueError(
            f"No electricity observations for building {BID}."
        )

    raw_n = len(electricity)

    for frame in (electricity, weather):
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"],
            utc=True,
            errors="coerce",
        )

    electricity[TARGET] = pd.to_numeric(
        electricity[TARGET], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    for column in WEATHER:
        weather[column] = pd.to_numeric(
            weather[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    invalid_n = int(
        (
            electricity["timestamp_utc"].isna()
            | electricity[TARGET].isna()
        ).sum()
    )

    x = electricity.dropna(
        subset=["timestamp_utc", TARGET]
    ).copy()

    x = x.sort_values("timestamp_utc", kind="stable")

    duplicate_n = int(
        x.duplicated("timestamp_utc", keep="last").sum()
    )
    x = x.drop_duplicates("timestamp_utc", keep="last")

    negatives = x.loc[x[TARGET] < 0].copy()
    x = x.loc[x[TARGET] >= 0].copy()

    if x.empty:
        raise ValueError("No valid nonnegative electricity observations.")

    weather_invalid_timestamp_n = int(
        weather["timestamp_utc"].isna().sum()
    )

    wx = weather.dropna(subset=["timestamp_utc"]).copy()
    weather_duplicate_n = int(
        wx.duplicated(
            ["building_id", "timestamp_utc"],
            keep="last",
        ).sum()
    )
    wx = wx.drop_duplicates(
        ["building_id", "timestamp_utc"],
        keep="last",
    )

    x = (
        x.merge(
            wx,
            on=["building_id", "timestamp_utc"],
            how="left",
            validate="one_to_one",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    series = x.set_index("timestamp_utc")[TARGET]
    x["origin_consumption"] = x[TARGET]

    for name, delta in [
        ("lag_15m", "15min"),
        ("lag_1h", "1h"),
        ("lag_24h", "24h"),
        ("lag_7d", "7D"),
    ]:
        x[name] = (
            x["timestamp_utc"] - pd.Timedelta(delta)
        ).map(series)

    # Retain the original UTC forecast-origin calendar features.
    minutes = (
        x["timestamp_utc"].dt.hour * 60
        + x["timestamp_utc"].dt.minute
    )
    dow = x["timestamp_utc"].dt.dayofweek

    x["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    x["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    x["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    x["is_weekend"] = (dow >= 5).astype(int)
    x["month"] = x["timestamp_utc"].dt.month

    # Exact timestamp lookups avoid shifting across missing intervals.
    future = {
        f"y_{h}": (
            x["timestamp_utc"]
            + pd.Timedelta(minutes=INTERVAL_MINUTES * h)
        ).map(series)
        for h in range(1, STEPS + 1)
    }

    x = pd.concat(
        [x, pd.DataFrame(future, index=x.index)],
        axis=1,
    )

    description = (
        str(buildings["description"].iloc[0])
        if not buildings.empty
        else ""
    )

    quality = {
        "building_id": BID,
        "building_description": description,
        "raw_rows": raw_n,
        "invalid_timestamp_or_target_rows": invalid_n,
        "duplicate_timestamps_removed": duplicate_n,
        "negative_rows_excluded": len(negatives),
        "negative_pct_raw": 100 * len(negatives) / raw_n,
        "clean_rows": len(x),
        "start": x["timestamp_utc"].min(),
        "end": x["timestamp_utc"].max(),
        "weather_rows_for_building": len(wx),
        "weather_invalid_timestamp_rows": weather_invalid_timestamp_n,
        "weather_duplicate_rows_removed": weather_duplicate_n,
        "rows_with_any_missing_weather": int(
            x[WEATHER].isna().any(axis=1).sum()
        ),
        "rows_with_all_missing_weather": int(
            x[WEATHER].isna().all(axis=1).sum()
        ),
        "non_15min_successive_gaps": int(
            x["timestamp_utc"]
            .diff()
            .dropna()
            .ne(pd.Timedelta(minutes=INTERVAL_MINUTES))
            .sum()
        ),
    }

    return x, negatives, quality


# ---------------------------------------------------------------------
# Chronological split and baselines
# ---------------------------------------------------------------------

def split_data(d):
    # Preserve the previous cutoff definition for comparability.
    eligible = d.loc[np.isfinite(d[f"y_{STEPS}"])].copy()
    cut = int(len(eligible) * 0.90)

    if cut < 1 or cut >= len(eligible):
        raise ValueError("Not enough eligible origins for a 90/10 split.")

    train_end = eligible["timestamp_utc"].iloc[cut - 1]
    test_start = eligible["timestamp_utc"].iloc[cut]
    test_end = eligible["timestamp_utc"].iloc[-1]

    tr = d.loc[d["timestamp_utc"] <= train_end].copy()
    te = d.loc[
        d["timestamp_utc"].between(test_start, test_end)
    ].copy()

    train_candidate_n = len(tr)
    test_candidate_n = len(te)

    # All models use the same complete-target test-origin cohort.
    tr = tr.loc[
        np.isfinite(tr[TARGETS].to_numpy(dtype=float)).all(axis=1)
    ].copy()

    te = te.loc[
        np.isfinite(te[TARGETS].to_numpy(dtype=float)).all(axis=1)
    ].copy()

    train_complete_before_purge_n = len(tr)
    test_complete_n = len(te)

    overlap = (
        tr["timestamp_utc"] + FORECAST_DURATION >= test_start
    )
    purged = tr.loc[overlap, ["timestamp_utc"]].copy()
    purged = purged.rename(
        columns={"timestamp_utc": "origin_timestamp"}
    )
    purged["latest_target_timestamp"] = (
        purged["origin_timestamp"] + FORECAST_DURATION
    )
    purged["reason"] = "target_window_reaches_test_boundary"

    tr = tr.loc[~overlap].copy()

    if tr.empty or te.empty:
        raise ValueError("No complete training or testing windows remain.")

    overlap_after = int(
        (
            tr["timestamp_utc"] + FORECAST_DURATION >= test_start
        ).sum()
    )

    latest_training_target = (
        tr["timestamp_utc"].max() + FORECAST_DURATION
    )

    if overlap_after != 0 or latest_training_target >= test_start:
        raise AssertionError("Training targets overlap the test boundary.")

    info = {
        "legacy_train_origin_cutoff": train_end,
        "test_origin_cutoff_start": test_start,
        "test_origin_cutoff_end": test_end,
        "actual_train_last_origin": tr["timestamp_utc"].max(),
        "actual_test_first_origin": te["timestamp_utc"].min(),
        "actual_test_last_origin": te["timestamp_utc"].max(),
        "latest_training_target": latest_training_target,
        "eligible_origins_for_cutoff": len(eligible),
        "train_candidate_origins": train_candidate_n,
        "test_candidate_origins": test_candidate_n,
        "incomplete_train_origins_removed": (
            train_candidate_n - train_complete_before_purge_n
        ),
        "incomplete_test_origins_removed": (
            test_candidate_n - test_complete_n
        ),
        "complete_train_origins_before_purge": (
            train_complete_before_purge_n
        ),
        "complete_train_origins": len(tr),
        "complete_test_origins": len(te),
        "overlapping_train_windows_before_purge": len(purged),
        "purged_train_origins": len(purged),
        "training_windows_reaching_test_start": overlap_after,
        "purge_training_overlap": True,
        "baseline_lookup": BASELINE_LOOKUP,
    }

    return (
        tr.reset_index(drop=True),
        te.reset_index(drop=True),
        info,
        purged.reset_index(drop=True),
    )


def baseline_arrays(te, history):
    actual = te[TARGETS].to_numpy(dtype=float)

    persistence = np.repeat(
        te["origin_consumption"].to_numpy(dtype=float)[:, None],
        STEPS,
        axis=1,
    )

    lookup = history.set_index("timestamp_utc")[TARGET]
    yesterday = np.full_like(actual, np.nan)

    for h in range(1, STEPS + 1):
        lead = pd.Timedelta(minutes=INTERVAL_MINUTES * h)
        target_time = te["timestamp_utc"] + lead

        # Latest same-time-of-day observation available at the origin.
        # Beyond 24 hours this uses two days back, not future actuals.
        days_back = max(1, int(np.ceil(lead / pd.Timedelta(days=1))))
        reference_time = target_time - pd.Timedelta(days=days_back)

        if (reference_time > te["timestamp_utc"]).any():
            raise AssertionError("Baseline attempted to use future data.")

        yesterday[:, h - 1] = (
            reference_time.map(lookup).to_numpy(dtype=float)
        )

    return actual, persistence, yesterday


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def horizon_metrics(label, actual, predicted, h, cohort):
    valid = np.isfinite(actual) & np.isfinite(predicted)
    a = actual[valid]
    p = predicted[valid]

    error = p - a
    absolute_error = np.abs(error)
    n = len(a)

    mape_valid = np.abs(a) >= MAPE_MIN_ACTUAL_W
    mape_n = int(mape_valid.sum())
    mape_sum = float(
        np.sum(
            100 * absolute_error[mape_valid]
            / np.abs(a[mape_valid])
        )
    )

    eligible_n = int(np.isfinite(actual).sum())

    return {
        "validation": "90_10",
        "cohort": cohort,
        "model": label,
        "horizon_step": h,
        "horizon_hours": h * INTERVAL_MINUTES / 60,
        "eligible_test_n": eligible_n,
        "n": n,
        "coverage_pct": (
            100 * n / eligible_n if eligible_n else np.nan
        ),
        "mean_actual_w": float(a.mean()) if n else np.nan,
        "mae_w": float(absolute_error.mean()) if n else np.nan,
        "rmse_w": (
            float(np.sqrt(np.mean(error ** 2))) if n else np.nan
        ),
        "bias_w": float(error.mean()) if n else np.nan,
        "absolute_error_sum_w": float(absolute_error.sum()),
        "squared_error_sum_w2": float(np.sum(error ** 2)),
        "error_sum_w": float(error.sum()),
        "actual_sum_w": float(a.sum()),
        "mape_percentage_sum": mape_sum,
        "mape_pct": mape_sum / mape_n if mape_n else np.nan,
        "mape_valid_n": mape_n,
        "mape_excluded_n": n - mape_n,
    }


def make_summary(metrics):
    """Pool origin-horizon pairs; retain mean horizon RMSE separately."""
    records = []

    for (validation, cohort, model), group in metrics.groupby(
        ["validation", "cohort", "model"],
        sort=False,
    ):
        n = int(group["n"].sum())
        eligible_n = int(group["eligible_test_n"].sum())
        mape_n = int(group["mape_valid_n"].sum())

        mean_actual = (
            group["actual_sum_w"].sum() / n if n else np.nan
        )
        mae = (
            group["absolute_error_sum_w"].sum() / n if n else np.nan
        )
        rmse = (
            np.sqrt(group["squared_error_sum_w2"].sum() / n)
            if n else np.nan
        )
        bias = group["error_sum_w"].sum() / n if n else np.nan

        records.append({
            "validation": validation,
            "cohort": cohort,
            "model": model,
            "MAE (kW)": mae / 1000,
            "Pooled RMSE (kW)": rmse / 1000,
            "Mean horizon RMSE (kW)": (
                group["rmse_w"].mean() / 1000
            ),
            "Mean actual (kW)": mean_actual / 1000,
            "MAPE (%)": (
                group["mape_percentage_sum"].sum() / mape_n
                if mape_n else np.nan
            ),
            "Mean bias (kW)": bias / 1000,
            "NMAE (%)": (
                100 * mae / mean_actual
                if mean_actual > 0 else np.nan
            ),
            "Pooled NRMSE (%)": (
                100 * rmse / mean_actual
                if mean_actual > 0 else np.nan
            ),
            "Pair coverage (%)": (
                100 * n / eligible_n if eligible_n else np.nan
            ),
            "prediction_pairs": n,
            "eligible_pairs": eligible_n,
            "mape_excluded_n": int(group["mape_excluded_n"].sum()),
            "horizons": int(group["horizon_step"].nunique()),
            "all_horizons_have_pairs": bool(group["n"].gt(0).all()),
        })

    return pd.DataFrame(records)


def evaluate_predictions(actual, predictions):
    expected_models = set(PREDICTION_COLUMNS)

    if set(predictions) != expected_models:
        raise AssertionError("Not all four model predictions are present.")

    actual_valid = np.isfinite(actual)
    common = actual_valid.copy()

    for label, prediction in predictions.items():
        if prediction.shape != actual.shape:
            raise AssertionError(f"Invalid prediction shape for {label}.")
        common &= np.isfinite(prediction)

    if not common.any():
        raise ValueError("No common prediction pairs across all models.")

    if not common.any(axis=0).all():
        raise ValueError(
            "At least one horizon has no common prediction pairs."
        )

    native_rows = []
    common_rows = []
    coverage_rows = []

    for label, prediction in predictions.items():
        available = actual_valid & np.isfinite(prediction)

        for h in range(1, STEPS + 1):
            j = h - 1

            native_rows.append(
                horizon_metrics(
                    label,
                    actual[:, j],
                    prediction[:, j],
                    h,
                    "native_available_pairs",
                )
            )

            masked_prediction = np.where(
                common[:, j],
                prediction[:, j],
                np.nan,
            )
            common_rows.append(
                horizon_metrics(
                    label,
                    actual[:, j],
                    masked_prediction,
                    h,
                    "common_pairs",
                )
            )

            eligible_n = int(actual_valid[:, j].sum())
            available_n = int(available[:, j].sum())

            coverage_rows.append({
                "validation": "90_10",
                "model": label,
                "horizon_step": h,
                "horizon_hours": h * INTERVAL_MINUTES / 60,
                "eligible_pairs": eligible_n,
                "available_pairs": available_n,
                "missing_prediction_pairs": eligible_n - available_n,
                "native_coverage_pct": (
                    100 * available_n / eligible_n
                    if eligible_n else np.nan
                ),
                "common_pairs": int(common[:, j].sum()),
            })

    native_metrics = pd.DataFrame(native_rows)
    common_metrics = pd.DataFrame(common_rows)
    coverage = pd.DataFrame(coverage_rows)

    equal_counts = (
        common_metrics.groupby("horizon_step")["n"]
        .nunique()
        .eq(1)
        .all()
    )
    if not equal_counts:
        raise AssertionError("Common-pair counts differ between models.")

    coverage_summary = coverage.groupby(
        ["validation", "model"],
        as_index=False,
        sort=False,
    ).agg(
        eligible_pairs=("eligible_pairs", "sum"),
        available_pairs=("available_pairs", "sum"),
        missing_prediction_pairs=("missing_prediction_pairs", "sum"),
        common_pairs=("common_pairs", "sum"),
    )

    coverage_summary["native_coverage_pct"] = (
        100 * coverage_summary["available_pairs"]
        / coverage_summary["eligible_pairs"]
    )
    coverage_summary["common_pair_coverage_pct"] = (
        100 * coverage_summary["common_pairs"]
        / coverage_summary["eligible_pairs"]
    )

    return (
        native_metrics,
        common_metrics,
        coverage,
        coverage_summary,
        common,
    )


# ---------------------------------------------------------------------
# Prediction exports and charts
# ---------------------------------------------------------------------

def selected_exports(te, actual, predictions, run_id, common=None):
    exports = {}

    for h in SELECTED:
        j = h - 1

        table = pd.DataFrame({
            "run_id": run_id,
            "validation": "90_10",
            "origin_timestamp": te["timestamp_utc"],
            "target_timestamp": (
                te["timestamp_utc"]
                + pd.Timedelta(minutes=INTERVAL_MINUTES * h)
            ),
            "horizon_step": h,
            "horizon_hours": h * INTERVAL_MINUTES / 60,
            "actual_w": actual[:, j],
        })

        for label, column in PREDICTION_COLUMNS.items():
            table[column] = (
                predictions[label][:, j]
                if label in predictions
                else np.nan
            )

        if common is not None:
            table["common_pair"] = common[:, j]

        exports[h] = table

    return exports


def save_charts(metrics, exports):
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, group in metrics.groupby("model"):
        group = group.sort_values("horizon_step")
        ax.plot(
            group["horizon_hours"],
            group["mae_w"] / 1000,
            label=label,
        )

    ax.set(
        title="36-hour forecast — purged 90/10, identical prediction pairs",
        xlabel="Forecast horizon (hours)",
        ylabel="MAE (kW)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    atomic_figure(fig, OUT / "horizon_mae_90_10.png")
    plt.close(fig)

    view = exports[STEPS].sort_values("target_timestamp").copy()
    end = view["target_timestamp"].max()
    view = view.loc[
        view["target_timestamp"] >= end - pd.Timedelta(days=7)
    ].copy()

    if view.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for column, label in [
        ("actual_w", "Actual"),
        ("hgb_w", "HGB"),
        ("hgb_weather_w", "HGB + Weather"),
    ]:
        values = view[column].where(view["common_pair"])
        ax.plot(
            view["target_timestamp"],
            values / 1000,
            label=label,
            alpha=0.85,
        )

    ax.set(
        title="Fixed 36h lead — last 7 calendar days, common pairs",
        xlabel="Target timestamp (UTC)",
        ylabel="Consumption (kW)",
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    atomic_figure(fig, OUT / "actual_vs_predicted_90_10.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def main():
    started = time.perf_counter()
    run_id = uuid.uuid4().hex

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    run = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "status": "RUNNING",
        "building_id": BID,
        "forecast_steps": STEPS,
        "interval_minutes": INTERVAL_MINUTES,
        "parameters": PARAMS,
        "thread_limit": THREAD_LIMIT,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "purge_training_overlap": True,
        "baseline_lookup": BASELINE_LOOKUP,
        "output_directory": str(OUT),
        "model_directory": str(MODELS),
        "save_models": SAVE_MODELS,
        "preprocessing": "training-only median imputation + scaling",
        "calendar": "UTC forecast-origin calendar features",
        "weather_note": (
            "Observed weather at forecast origin only. Assumes origin "
            "consumption and weather are available when issuing the "
            "forecast. No future weather inputs."
        ),
        "baseline_note": (
            "Copy-Yesterday uses the latest same-time-of-day historical "
            "observation available at or before each origin: one day back "
            "through 24h lead, two days back beyond 24h."
        ),
        "evaluation_note": (
            "Fixed models evaluated at successive test origins. Primary "
            "errors use identical origin-horizon pairs across all four "
            "models. Native availability is reported separately."
        ),
        "metric_note": (
            "MAE, RMSE, bias and MAPE pool prediction pairs. Mean horizon "
            "RMSE is a separate diagnostic. NMAE and pooled NRMSE use "
            "the mean actual on the evaluated pairs as denominator."
        ),
        "mape_min_actual_w": MAPE_MIN_ACTUAL_W,
        "variants": {},
    }

    def update_run():
        run["runtime_seconds"] = time.perf_counter() - started
        write_json(run, OUT / "run_summary.json")

    update_run()

    try:
        print("Run ID:", run_id, flush=True)
        print("Data directory:", DATA, flush=True)
        print("Output directory:", OUT, flush=True)
        print("Model directory:", MODELS, flush=True)
        print(
            "Existing matching filenames will be overwritten; "
            "unrelated files will remain.",
            flush=True,
        )
        print("Loading and preparing data...", flush=True)

        d, negatives, quality = load_prepare()
        tr, te, split_info, purged = split_data(d)

        run.update(split_info)
        quality.update(split_info)
        quality["run_id"] = run_id

        atomic_csv(
            negatives,
            OUT / "excluded_negative_consumption.csv",
        )
        atomic_csv(
            pd.DataFrame([quality]),
            OUT / "data_quality_90_10.csv",
        )

        purged.insert(0, "run_id", run_id)
        atomic_csv(
            purged,
            OUT / "purged_training_origins.csv",
        )

        origins = pd.concat([
            pd.DataFrame({
                "partition": "train",
                "origin_timestamp": tr["timestamp_utc"],
            }),
            pd.DataFrame({
                "partition": "test",
                "origin_timestamp": te["timestamp_utc"],
            }),
        ], ignore_index=True)
        origins.insert(0, "run_id", run_id)

        atomic_csv(
            origins,
            OUT / "evaluation_origins.csv",
        )

        missing_features = []
        for partition, frame in [("train", tr), ("test", te)]:
            for feature in BASE + WEATHER:
                missing_features.append({
                    "run_id": run_id,
                    "partition": partition,
                    "feature": feature,
                    "missing_n": int(frame[feature].isna().sum()),
                    "missing_pct": float(
                        frame[feature].isna().mean() * 100
                    ),
                })

        atomic_csv(
            pd.DataFrame(missing_features),
            OUT / "feature_missingness.csv",
        )

        print(
            f"Complete origins: train={len(tr):,}, test={len(te):,}",
            flush=True,
        )
        print(
            "Training windows removed by purge:",
            split_info["purged_train_origins"],
            flush=True,
        )
        print(
            "Training windows reaching test boundary:",
            split_info["training_windows_reaching_test_start"],
            flush=True,
        )
        print(
            "Latest training target:",
            split_info["latest_training_target"],
            flush=True,
        )
        print(
            "Test boundary:",
            split_info["test_origin_cutoff_start"],
            flush=True,
        )
        print(
            "Missing weather rows:",
            quality["rows_with_any_missing_weather"],
            flush=True,
        )

        train_y = tr[TARGETS].to_numpy(dtype=float)
        test_y, persistence, yesterday = baseline_arrays(te, d)

        predictions = {
            "Persistence": persistence,
            "Copy-Yesterday": yesterday,
        }

        total_fits = STEPS * len(VARIANTS)
        completed_fits = 0
        fitting_started = time.perf_counter()

        for label, features, column, filename in VARIANTS:
            print(f"\nPreparing {label}...", flush=True)

            all_missing = [
                feature
                for feature in features
                if tr[feature].isna().all()
            ]
            if all_missing:
                print(
                    "WARNING: entirely missing training features:",
                    ", ".join(all_missing),
                    flush=True,
                )

            preprocessing = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])

            tick = time.perf_counter()
            Xtr = preprocessing.fit_transform(tr[features])
            Xte = preprocessing.transform(te[features])

            variant = {
                "status": "TRAINING",
                "completed_horizons": 0,
                "preprocessing_seconds": time.perf_counter() - tick,
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
                "model_saved": False,
            }
            run["variants"][label] = variant
            update_run()

            models = []
            predicted = np.full(test_y.shape, np.nan)
            predictions[label] = predicted
            variant_started = time.perf_counter()

            for h in range(1, STEPS + 1):
                model = HistGradientBoostingRegressor(**PARAMS)

                tick = time.perf_counter()
                model.fit(Xtr, train_y[:, h - 1])
                variant["fit_seconds"] += time.perf_counter() - tick

                tick = time.perf_counter()
                predicted[:, h - 1] = model.predict(Xte)
                variant["predict_seconds"] += time.perf_counter() - tick

                if SAVE_MODELS:
                    models.append(model)

                completed_fits += 1
                variant["completed_horizons"] = h

                if h % CHECKPOINT_EVERY == 0 or h == STEPS:
                    elapsed = time.perf_counter() - fitting_started
                    eta = (
                        elapsed / completed_fits
                        * (total_fits - completed_fits)
                    )

                    partial = selected_exports(
                        te,
                        test_y,
                        predictions,
                        run_id,
                    )
                    atomic_csv(
                        pd.concat(partial.values(), ignore_index=True),
                        OUT / "predictions_checkpoint_90_10.csv",
                    )
                    update_run()

                    print(
                        f"  {label}: {h}/{STEPS} horizons | "
                        f"overall {completed_fits}/{total_fits} fits | "
                        f"elapsed {elapsed / 60:.1f} min | "
                        f"approx. remaining {eta / 60:.1f} min",
                        flush=True,
                    )

            if not np.isfinite(predicted).all():
                raise AssertionError(
                    f"{label} produced missing or infinite predictions."
                )

            variant["fit_and_predict_wall_seconds"] = (
                time.perf_counter() - variant_started
            )
            variant["status"] = "TRAINED"

            if SAVE_MODELS:
                print(f"Saving {label} model bundle...", flush=True)
                tick = time.perf_counter()

                model_path = MODELS / f"{filename}_90_10.joblib"
                temporary = model_path.with_name(
                    model_path.name + ".tmp"
                )

                joblib.dump({
                    "run_id": run_id,
                    "preprocessing": preprocessing,
                    "models": models,
                    "features": features,
                    "building_id": BID,
                    "target": TARGET,
                    "forecast_steps": STEPS,
                    "interval_minutes": INTERVAL_MINUTES,
                    "validation": "90_10",
                    "experiment": EXPERIMENT,
                    "evaluation_protocol": EVALUATION_PROTOCOL,
                    "purge_training_overlap": True,
                    "baseline_lookup": BASELINE_LOOKUP,
                    "parameters": PARAMS,
                    "sklearn_version": sklearn.__version__,
                    "weather_mode": (
                        "observed forecast-origin weather only"
                        if label == "HGB + Weather"
                        else "none"
                    ),
                    "split": split_info,
                }, temporary, compress=3)

                temporary.replace(model_path)

                variant["model_saved"] = True
                variant["model_path"] = str(model_path)
                variant["model_file_bytes"] = model_path.stat().st_size
                variant["model_save_seconds"] = (
                    time.perf_counter() - tick
                )

            variant["status"] = "COMPLETED"
            update_run()
            print(f"Completed {label}.", flush=True)

            del models, Xtr, Xte, preprocessing, model

        (
            native_metrics,
            common_metrics,
            coverage,
            coverage_summary,
            common,
        ) = evaluate_predictions(test_y, predictions)

        common_summary = make_summary(common_metrics)
        native_summary = make_summary(native_metrics)

        if common_summary["prediction_pairs"].nunique() != 1:
            raise AssertionError("Unequal common-pair summary counts.")

        for table in [
            native_metrics,
            common_metrics,
            coverage,
            coverage_summary,
            common_summary,
            native_summary,
        ]:
            table.insert(0, "run_id", run_id)

        exports = selected_exports(
            te,
            test_y,
            predictions,
            run_id,
            common=common,
        )

        output_tables = {
            "metrics_by_horizon_90_10.csv": common_metrics,
            "metrics_90_10.csv": common_summary,
            "metrics_by_horizon_native_90_10.csv": native_metrics,
            "metrics_native_90_10.csv": native_summary,
            "coverage_by_horizon_90_10.csv": coverage,
            "coverage_summary_90_10.csv": coverage_summary,
            "predictions_selected_horizons_90_10.csv": pd.concat(
                exports.values(), ignore_index=True
            ),
        }

        for filename, table in output_tables.items():
            atomic_csv(table, OUT / filename)

        save_charts(common_metrics, exports)

        run["common_prediction_pairs"] = int(common.sum())
        run["eligible_prediction_pairs"] = int(
            np.isfinite(test_y).sum()
        )
        run["common_pair_coverage_pct"] = float(
            100 * common.sum() / np.isfinite(test_y).sum()
        )
        run["equal_common_pair_counts_verified"] = True
        run["zero_training_target_overlap_verified"] = True
        run["primary_metrics_file"] = "metrics_90_10.csv"
        run["coverage_file"] = "coverage_summary_90_10.csv"
        run["interpretation"] = (
            "Purged chronological holdout with identical prediction "
            "pairs across all four models. Single building and split. "
            "This previously inspected holdout is not a fresh independent "
            "test set. Metrics are not a production validation."
        )
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run["status"] = "COMPLETED"
        update_run()

        # Checkpoints are not final comparison outputs.
        checkpoint = OUT / "predictions_checkpoint_90_10.csv"
        if checkpoint.exists():
            try:
                checkpoint.unlink()
            except OSError as exc:
                print(
                    f"Warning: could not remove checkpoint: {exc}",
                    flush=True,
                )

        print(
            "\nPRIMARY RESULTS — IDENTICAL PREDICTION PAIRS",
            flush=True,
        )
        print(
            common_summary.drop(columns="run_id")
            .round(2)
            .to_string(index=False),
            flush=True,
        )

        print("\nNATIVE PREDICTION AVAILABILITY", flush=True)
        print(
            coverage_summary.drop(columns="run_id")
            .round(2)
            .to_string(index=False),
            flush=True,
        )

        print(
            "\nVerified: zero training target overlap and identical "
            "comparison pairs across all four models.",
            flush=True,
        )
        print(
            "Use metrics_90_10.csv for the primary comparison.",
            flush=True,
        )
        print(
            "Use coverage_summary_90_10.csv for prediction availability.",
            flush=True,
        )
        print(
            "Charts are PNG files; older JPG files are not updated.",
            flush=True,
        )
        print(
            f"Total runtime: {run['runtime_seconds']:.1f} seconds",
            flush=True,
        )
        print("Results:", OUT, flush=True)

        if SAVE_MODELS:
            print("Models: ", MODELS, flush=True)

    except KeyboardInterrupt:
        run["status"] = "INTERRUPTED"
        update_run()
        print(
            "\nInterrupted. Checkpoint predictions may be partial. "
            "Existing final files may belong to an earlier run. "
            "Trust final outputs only when run_summary.json is COMPLETED "
            "and their run_id matches. Training does not resume.",
            flush=True,
        )
        raise

    except Exception as exc:
        run["status"] = "FAILED"
        run["error"] = str(exc)
        run["traceback"] = traceback.format_exc()
        update_run()
        print(
            "\nRun failed. Existing final files may belong to an earlier "
            "run. Check run_summary.json and matching run_id values.",
            flush=True,
        )
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=THREAD_LIMIT):
        main()