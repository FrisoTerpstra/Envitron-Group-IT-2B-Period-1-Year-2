#!/usr/bin/env python3
"""Kristian–Joel: direct 36-hour HistGradientBoosting prototype.

Setup:
    One building; 144 forecast steps at 15-minute resolution.
    Chronological 90/10 cutoff with strictly purged training targets.
    Complete target windows and identical comparison pairs.
    Separate HGB and HGB + Weather variants.

Combined features:
    Original consumption lags and UTC origin-calendar features.
    Additional 2h/6h/48h lags and recent consumption changes.
    Causal rolling means and standard deviations.
    Deterministic target-calendar features for each forecast lead.

Outputs:
    output/hgb_36h_joint/charts/       Five PNG charts
    output/hgb_36h_joint/
        metrics_90_10.csv
        run_summary.json
    models/hgb_36h_joint/
        hgb_90_10.joblib
        hgb_weather_90_10.joblib

Existing matching filenames are overwritten.
Data and unrelated existing files are never deleted.
"""

import os

THREAD_LIMIT = 2
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, str(THREAD_LIMIT))

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import platform
import sys
import time
import uuid

import joblib
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

EXPERIMENT = "hgb_36h_joint"
OUT = HERE / "output" / EXPERIMENT
CHARTS = OUT / "charts"
MODELS = HERE / "models" / EXPERIMENT

BID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"
TARGET = "consumption_w"
TS = "timestamp_utc"

STEPS = 144
INTERVAL_MINUTES = 15
CHUNK_SIZE = 50_000
PROGRESS_EVERY = 12
MAPE_MIN_ACTUAL_W = 100.0
DURATION = pd.Timedelta(minutes=STEPS * INTERVAL_MINUTES)

PROTOCOL = "purged_90_10_common_pairs_v1"
FEATURE_VERSION = "kristian_joel_combined_v1"

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

# Retain the established settings to isolate feature changes.
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
    ("HGB", False, "hgb"),
    ("HGB + Weather", True, "hgb_weather"),
]

ORDER = [
    "Persistence",
    "Copy-Yesterday",
    "HGB",
    "HGB + Weather",
]

COLORS = {
    "Actual": "#222222",
    "Persistence": "#d62728",
    "Copy-Yesterday": "#1f77b4",
    "HGB": "#ff7f0e",
    "HGB + Weather": "#2ca02c",
}

PNG_FILES = [
    "percentage_metrics_90_10.png",
    "model_comparison_90_10.png",
    "horizon_mae_90_10.png",
    "horizon_rmse_90_10.png",
    "actual_vs_predicted_90_10.png",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def write_json(data, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def peak_memory_mib():
    """Peak process RSS; not total container memory."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        return float(value / divisor)
    except (ImportError, AttributeError):
        return None


def origins_hash(origins):
    values = pd.DatetimeIndex(origins).asi8
    return hashlib.sha256(values.tobytes()).hexdigest()


def read_building(filename, columns):
    parts = []
    for chunk in pd.read_csv(
        DATA / filename,
        usecols=columns,
        dtype={"building_id": "string"},
        chunksize=CHUNK_SIZE,
    ):
        selected = chunk.loc[chunk["building_id"].eq(BID)].copy()
        if not selected.empty:
            parts.append(selected)

    return (
        pd.concat(parts, ignore_index=True)
        if parts else pd.DataFrame(columns=columns)
    )


# ---------------------------------------------------------------------
# Loading and causal feature engineering
# ---------------------------------------------------------------------

def prepare():
    required = ["buildings.csv", "electricity.csv", "weather.csv"]
    missing = [str(DATA / name) for name in required if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    e = read_building("electricity.csv", ["building_id", TS, TARGET])
    w = read_building("weather.csv", ["building_id", TS] + WEATHER)
    b = read_building("buildings.csv", ["building_id", "description"])

    if e.empty:
        raise ValueError(f"No electricity observations for building {BID}.")
    if len(b) != 1:
        raise ValueError(
            f"Expected one metadata row for {BID}; found {len(b)}."
        )

    raw_rows = len(e)
    for frame, columns in [(e, [TARGET]), (w, WEATHER)]:
        frame[TS] = pd.to_datetime(frame[TS], utc=True, errors="coerce")
        for column in columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)

    invalid_rows = int((e[TS].isna() | e[TARGET].isna()).sum())
    e = e.dropna(subset=[TS, TARGET]).sort_values(TS, kind="stable")

    duplicates = int(e.duplicated(TS, keep="last").sum())
    e = e.drop_duplicates(TS, keep="last")

    negative_rows = int(e[TARGET].lt(0).sum())
    e = e.loc[e[TARGET].ge(0)].copy()
    if e.empty:
        raise ValueError("No valid nonnegative electricity observations remain.")

    w = w.dropna(subset=[TS]).drop_duplicates(
        ["building_id", TS], keep="last"
    )
    d = (
        e.merge(
            w,
            on=["building_id", TS],
            how="left",
            validate="one_to_one",
        )
        .sort_values(TS)
        .reset_index(drop=True)
    )

    history = d.set_index(TS)[TARGET]
    d["origin_consumption"] = d[TARGET]

    for name, delta in [
        ("lag_15m", "15min"),
        ("lag_1h", "1h"),
        ("lag_2h", "2h"),
        ("lag_6h", "6h"),
        ("lag_24h", "24h"),
        ("lag_48h", "48h"),
        ("lag_7d", "7D"),
    ]:
        d[name] = (d[TS] - pd.Timedelta(delta)).map(history)

    d["change_15m"] = d[TARGET] - d["lag_15m"]
    d["change_1h"] = d[TARGET] - d["lag_1h"]

    # Windows are (origin - window, origin].
    # Current consumption is assumed known when issuing the forecast.
    for window in ["1h", "6h", "24h"]:
        rolling = history.rolling(window, min_periods=1, closed="right")
        d[f"consumption_mean_{window}"] = rolling.mean().to_numpy()
        if window in {"1h", "24h"}:
            d[f"consumption_std_{window}"] = rolling.std(ddof=1).to_numpy()

    minutes = d[TS].dt.hour * 60 + d[TS].dt.minute
    dow = d[TS].dt.dayofweek

    d["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    d["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    d["is_weekend"] = (dow >= 5).astype(int)
    d["month"] = d[TS].dt.month

    quality = {
        "description": str(b.iloc[0]["description"]),
        "raw_rows": raw_rows,
        "clean_rows": len(d),
        "invalid_timestamp_or_target_rows": invalid_rows,
        "duplicate_timestamps_removed": duplicates,
        "negative_rows_excluded": negative_rows,
        "start": d[TS].min(),
        "end": d[TS].max(),
        "rows_with_any_missing_weather": int(
            d[WEATHER].isna().any(axis=1).sum()
        ),
        "non_15min_successive_gaps": int(
            d[TS].diff().dropna()
            .ne(pd.Timedelta(minutes=INTERVAL_MINUTES)).sum()
        ),
    }
    return d, history, quality


def targets(history, origins, step):
    times = origins + pd.Timedelta(minutes=INTERVAL_MINUTES * step)
    return history.reindex(pd.DatetimeIndex(times)).to_numpy(float)


def features(frame, step, use_weather):
    columns = BASE + (WEATHER if use_weather else [])
    x = frame[columns].copy()

    # Calendar information is known in advance; no future observations.
    target_time = frame[TS] + pd.Timedelta(minutes=INTERVAL_MINUTES * step)
    minutes = target_time.dt.hour * 60 + target_time.dt.minute
    dow = target_time.dt.dayofweek

    x["target_tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    x["target_tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    x["target_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    x["target_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    x["target_month"] = target_time.dt.month
    return x


# ---------------------------------------------------------------------
# Fixed chronological split and historical baselines
# ---------------------------------------------------------------------

def split(d, history):
    times = d[TS]
    eligible = times.loc[np.isfinite(targets(history, times, STEPS))]
    cut = int(len(eligible) * 0.90)
    if not 0 < cut < len(eligible):
        raise ValueError("Insufficient eligible origins for a 90/10 split.")

    train_end = eligible.iloc[cut - 1]
    test_start = eligible.iloc[cut]
    test_end = eligible.iloc[-1]

    # Avoid retaining a complete training-target matrix in memory.
    complete = np.ones(len(d), dtype=bool)
    for step in range(1, STEPS + 1):
        complete &= np.isfinite(targets(history, times, step))

    train_mask = times.le(train_end).to_numpy() & complete
    test_mask = times.between(test_start, test_end).to_numpy() & complete

    overlap = (times + DURATION >= test_start).to_numpy()
    purged_rows = int((train_mask & overlap).sum())
    train_mask &= ~overlap

    tr = d.loc[train_mask].copy()
    te = d.loc[test_mask].copy()
    if tr.empty or te.empty:
        raise ValueError("No complete training or testing windows remain.")

    latest_target = tr[TS].max() + DURATION
    if latest_target >= test_start:
        raise AssertionError("Training targets reach the test boundary.")

    info = {
        "legacy_train_origin_cutoff": train_end,
        "test_origin_cutoff_start": test_start,
        "test_origin_cutoff_end": test_end,
        "actual_train_last_origin": tr[TS].max(),
        "actual_test_first_origin": te[TS].min(),
        "actual_test_last_origin": te[TS].max(),
        "latest_training_target": latest_target,
        "complete_train_origins": len(tr),
        "complete_test_origins": len(te),
        "purged_train_origins": purged_rows,
        "training_windows_reaching_test_start": 0,
        "incomplete_test_origins_removed": int(
            (
                times.between(test_start, test_end).to_numpy()
                & ~complete
            ).sum()
        ),
        "train_origins_sha256": origins_hash(tr[TS]),
        "test_origins_sha256": origins_hash(te[TS]),
    }
    return tr.reset_index(drop=True), te.reset_index(drop=True), info


def baselines(te, history):
    actual = np.column_stack([
        targets(history, te[TS], step)
        for step in range(1, STEPS + 1)
    ])
    persistence = np.repeat(
        te[TARGET].to_numpy(float)[:, None], STEPS, axis=1
    )
    yesterday = np.full_like(actual, np.nan)

    for step in range(1, STEPS + 1):
        lead = pd.Timedelta(minutes=INTERVAL_MINUTES * step)
        days_back = int(np.ceil(lead / pd.Timedelta(days=1)))
        reference = te[TS] + lead - pd.Timedelta(days=days_back)

        if (reference > te[TS]).any():
            raise AssertionError("Baseline attempted to use future data.")

        yesterday[:, step - 1] = history.reindex(
            pd.DatetimeIndex(reference)
        ).to_numpy(float)

    return actual, {
        "Persistence": persistence,
        "Copy-Yesterday": yesterday,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(actual, predictions):
    if set(predictions) != set(ORDER):
        raise ValueError("Predictions from all four approaches are required.")

    actual_valid = np.isfinite(actual)
    common = actual_valid.copy()

    for name in ORDER:
        if predictions[name].shape != actual.shape:
            raise ValueError(f"Prediction shape mismatch: {name}")
        common &= np.isfinite(predictions[name])

    if not common.any(axis=0).all():
        raise ValueError("At least one horizon has no common comparison pairs.")

    a = actual[common]
    mean_actual = float(a.mean())
    mape_mask = np.abs(a) >= MAPE_MIN_ACTUAL_W
    eligible_pairs = int(actual_valid.sum())
    common_pairs = int(common.sum())

    summary_rows, horizon_rows = [], []

    for name in ORDER:
        p = predictions[name]
        error = p[common] - a
        mae = float(np.abs(error).mean())
        rmse = float(np.sqrt(np.mean(error**2)))
        available = int((actual_valid & np.isfinite(p)).sum())
        horizon_rmse = []

        for j in range(STEPS):
            mask = common[:, j]
            e = p[mask, j] - actual[mask, j]
            r = float(np.sqrt(np.mean(e**2)))
            horizon_rmse.append(r)

            horizon_rows.append({
                "model": name,
                "hours": (j + 1) * INTERVAL_MINUTES / 60,
                "n": int(mask.sum()),
                "mae": float(np.abs(e).mean()) / 1000,
                "rmse": r / 1000,
            })

        summary_rows.append({
            "validation": "90_10",
            "cohort": "common_pairs",
            "model": name,
            "MAE (kW)": mae / 1000,
            "Pooled RMSE (kW)": rmse / 1000,
            "Mean horizon RMSE (kW)": float(np.mean(horizon_rmse)) / 1000,
            "Mean actual (kW)": mean_actual / 1000,
            "NMAE (%)": (
                100 * mae / mean_actual if mean_actual > 0 else np.nan
            ),
            "Pooled NRMSE (%)": (
                100 * rmse / mean_actual if mean_actual > 0 else np.nan
            ),
            "MAPE (%)": (
                float(100 * np.mean(np.abs(error[mape_mask] / a[mape_mask])))
                if mape_mask.any() else np.nan
            ),
            "Mean bias (kW)": float(error.mean()) / 1000,
            "prediction_pairs": common_pairs,
            "eligible_pairs": eligible_pairs,
            "Common pair coverage (%)": 100 * common_pairs / eligible_pairs,
            "Native coverage (%)": 100 * available / eligible_pairs,
            "native_missing_pairs": eligible_pairs - available,
            "mape_excluded_n": int((~mape_mask).sum()),
            "horizons": STEPS,
        })

    horizons = pd.DataFrame(horizon_rows)
    if not horizons.groupby("hours")["n"].nunique().eq(1).all():
        raise AssertionError("Models have unequal common-pair counts.")

    return pd.DataFrame(summary_rows), horizons, common


# ---------------------------------------------------------------------
# Five presentation charts
# ---------------------------------------------------------------------

def save_charts(summary, horizons, te, actual, predictions, common, run_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    def save(fig, filename):
        fig.tight_layout()
        path = CHARTS / filename
        temporary = path.with_name(path.stem + ".tmp.png")
        fig.savefig(
            temporary,
            format="png",
            dpi=150,
            metadata={
                "Description": (
                    f"run_id={run_id}; protocol={PROTOCOL}; "
                    f"features={FEATURE_VERSION}"
                )
            },
        )
        temporary.replace(path)
        plt.close(fig)

    for columns, title, unit, filename in [
        (
            ["NMAE (%)", "Pooled NRMSE (%)", "MAPE (%)"],
            "Percentage errors", "Error (%)", PNG_FILES[0],
        ),
        (
            ["MAE (kW)", "Pooled RMSE (kW)"],
            "Absolute errors", "Error (kW)", PNG_FILES[1],
        ),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        summary.set_index("model")[columns].plot.bar(ax=ax, rot=15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)

        ax.set(
            title=f"{title} — joint HGB, purged 90/10, common pairs",
            ylabel=unit,
            xlabel="",
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)
        save(fig, filename)

    for metric, filename in [
        ("mae", PNG_FILES[2]),
        ("rmse", PNG_FILES[3]),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for name in ORDER:
            group = horizons.loc[horizons["model"].eq(name)]
            ax.plot(
                group["hours"], group[metric],
                label=name, color=COLORS[name],
            )

        ax.set(
            title=f"{metric.upper()} by horizon — purged 90/10, common pairs",
            xlabel="Forecast horizon (hours)",
            ylabel=f"{metric.upper()} (kW)",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, filename)

    times = pd.DatetimeIndex(te[TS] + DURATION)
    end = times.max()
    start = end - pd.Timedelta(days=7)
    view = pd.DataFrame(index=times)

    for name, values in [("Actual", actual)] + list(predictions.items()):
        view[name] = np.where(
            common[:, -1], values[:, -1], np.nan
        ) / 1000

    view = view.loc[view.index >= start]
    grid = pd.date_range(start, end, freq=f"{INTERVAL_MINUTES}min")
    view = view.reindex(view.index.union(grid)).sort_index()

    fig, ax = plt.subplots(figsize=(13, 6))
    for name in ["Actual"] + ORDER:
        baseline = name in ORDER[:2]
        ax.plot(
            view.index,
            view[name],
            label=name,
            color=COLORS[name],
            linestyle="--" if baseline else "-",
            alpha=0.55 if baseline else 0.9,
        )

    ax.set(
        title="Actual vs predicted — fixed 36h lead, last 7 calendar days",
        xlabel="Target timestamp (UTC)",
        ylabel="Consumption (kW)",
        xlim=(start, end),
    )
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d", tz=timezone.utc)
    )
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate(rotation=25)
    save(fig, PNG_FILES[4])


# ---------------------------------------------------------------------
# Model training and experiment
# ---------------------------------------------------------------------

def main():
    started = time.perf_counter()
    run_id = uuid.uuid4().hex

    CHARTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    run = {
        "run_id": run_id,
        "experiment": EXPERIMENT,
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "building_id": BID,
        "forecast_steps": STEPS,
        "interval_minutes": INTERVAL_MINUTES,
        "evaluation_protocol": PROTOCOL,
        "feature_version": FEATURE_VERSION,
        "parameters": PARAMS,
        "thread_limit": THREAD_LIMIT,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "purge_training_overlap": True,
        "baseline_lookup": "full_history",
        "preprocessing": "training-only median imputation + scaling",
        "mape_min_actual_w": MAPE_MIN_ACTUAL_W,
        "weather_note": (
            "Observed forecast-origin weather only. Origin consumption "
            "and weather are assumed available when issuing the forecast."
        ),
        "feature_note": (
            "Causal historical lags/rolling statistics and deterministic "
            "UTC origin/target calendar features."
        ),
        "metric_note": (
            "Primary errors pool identical origin-horizon pairs. "
            "NMAE and pooled NRMSE normalize by their mean actual. "
            "Mean horizon RMSE is a separate diagnostic."
        ),
        "coverage_note": (
            "Native availability is reported separately. Coverage is "
            "conditional on complete 144-target test windows."
        ),
        "variants": {},
    }

    def update():
        run["runtime_seconds"] = time.perf_counter() - started
        run["process_peak_rss_mib"] = peak_memory_mib()
        write_json(run, OUT / "run_summary.json")

    update()

    try:
        print("Run ID:", run_id, flush=True)
        print("Data:", DATA, flush=True)
        print("Charts:", CHARTS, flush=True)
        print("Models:", MODELS, flush=True)

        d, history, quality = prepare()
        tr, te, split_info = split(d, history)

        run["data_quality"] = quality
        run.update(split_info)
        update()

        print(
            f"Complete origins: train={len(tr):,}, test={len(te):,}",
            flush=True,
        )
        print(
            "Training windows purged:",
            split_info["purged_train_origins"],
            flush=True,
        )
        print("Training windows reaching test boundary: 0", flush=True)
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

        actual, predictions = baselines(te, history)
        del d
        gc.collect()

        for name, use_weather, filename in VARIANTS:
            tick = time.perf_counter()
            models = []
            predicted = np.empty_like(actual)
            state = {
                "status": "TRAINING",
                "completed_horizons": 0,
                "model_saved": False,
            }
            run["variants"][name] = state

            all_missing = [
                col for col in BASE + (WEATHER if use_weather else [])
                if tr[col].isna().all()
            ]
            if all_missing:
                print(
                    "WARNING: entirely missing training features:",
                    ", ".join(all_missing),
                    flush=True,
                )

            print(f"\nTraining {name}...", flush=True)

            for step in range(1, STEPS + 1):
                x_train = features(tr, step, use_weather)
                x_test = features(te, step, use_weather)
                y_train = targets(history, tr[TS], step)

                model = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("hgb", HistGradientBoostingRegressor(**PARAMS)),
                ])

                model.fit(x_train, y_train)
                predicted[:, step - 1] = model.predict(x_test)
                models.append(model)
                state["completed_horizons"] = step

                if step % PROGRESS_EVERY == 0 or step == STEPS:
                    print(f"  {name}: {step}/{STEPS} horizons", flush=True)
                    update()

            if not np.isfinite(predicted).all():
                raise AssertionError(f"{name} produced nonfinite predictions.")

            predictions[name] = predicted

            bundle = {
                "run_id": run_id,
                "model_format": "per_horizon_sklearn_pipelines_v1",
                "models": models,
                "feature_names": list(x_train.columns),
                "base_features": BASE,
                "weather_features": WEATHER if use_weather else [],
                "target_calendar_features": [
                    "target_tod_sin", "target_tod_cos",
                    "target_dow_sin", "target_dow_cos", "target_month",
                ],
                "use_weather": use_weather,
                "building_id": BID,
                "target": TARGET,
                "forecast_steps": STEPS,
                "interval_minutes": INTERVAL_MINUTES,
                "experiment": EXPERIMENT,
                "feature_version": FEATURE_VERSION,
                "evaluation_protocol": PROTOCOL,
                "parameters": PARAMS,
                "sklearn_version": sklearn.__version__,
                "split": split_info,
            }

            path = MODELS / f"{filename}_90_10.joblib"
            temporary = path.with_name(path.name + ".tmp")
            joblib.dump(bundle, temporary, compress=3)
            temporary.replace(path)

            state.update(
                status="COMPLETED",
                model_saved=True,
                model_file_bytes=path.stat().st_size,
                wall_seconds=time.perf_counter() - tick,
            )

            del bundle, models, model, x_train, x_test, y_train
            gc.collect()
            update()

        summary, horizons, common = evaluate(actual, predictions)
        summary.insert(0, "run_id", run_id)

        print("\nGenerating five PNG charts...", flush=True)
        save_charts(
            summary, horizons, te, actual,
            predictions, common, run_id,
        )

        temporary = OUT / "metrics_90_10.csv.tmp"
        summary.to_csv(temporary, index=False)
        temporary.replace(OUT / "metrics_90_10.csv")

        run.update(
            status="COMPLETED",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            common_prediction_pairs=int(common.sum()),
            eligible_prediction_pairs=int(np.isfinite(actual).sum()),
            common_pair_coverage_pct=float(
                100 * common.sum() / np.isfinite(actual).sum()
            ),
            common_pairs_sha256=hashlib.sha256(
                common.tobytes()
            ).hexdigest(),
            zero_training_target_overlap_verified=True,
            equal_common_pair_counts_verified=True,
            primary_metrics_file="metrics_90_10.csv",
            chart_files=[f"charts/{name}" for name in PNG_FILES],
            interpretation=(
                "Single building and previously inspected chronological "
                "holdout. Not fresh independent validation, proof of "
                "superiority, or a Raspberry Pi performance benchmark."
            ),
        )
        update()

        display_columns = [
            "model", "MAE (kW)", "Pooled RMSE (kW)",
            "NMAE (%)", "Pooled NRMSE (%)", "MAPE (%)",
            "Mean bias (kW)", "Common pair coverage (%)",
            "Native coverage (%)",
        ]
        print("\nRESULTS — IDENTICAL PREDICTION PAIRS", flush=True)
        print(
            summary[display_columns].round(3).to_string(index=False),
            flush=True,
        )
        print(
            f"\nRuntime: {run['runtime_seconds']:.1f} seconds",
            flush=True,
        )
        print(
            "Peak process RSS (MiB):",
            run["process_peak_rss_mib"],
            flush=True,
        )
        print("Results:", OUT, flush=True)
        print("Models:", MODELS, flush=True)

    except (Exception, KeyboardInterrupt) as exc:
        run["status"] = (
            "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        )
        run["error"] = repr(exc)
        update()
        print(
            "\nRun incomplete. Existing outputs may belong to an older run. "
            "Check run_summary.json before using results.",
            flush=True,
        )
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=THREAD_LIMIT):
        main()