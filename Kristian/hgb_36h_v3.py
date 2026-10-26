#!/usr/bin/env python3
"""Compact HGB 36h experiment: purged 90/10, common-pair evaluation.

Output: five charts, metrics_90_10.csv, run_summary.json.
Models are saved separately. Old output files are archived after success.
Training targets are looked up one horizon at a time to reduce memory.
"""

import os

THREAD_LIMIT = 2
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, str(THREAD_LIMIT))

from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import shutil
import sys
import time
import uuid

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


# Configuration: unchanged experiment and model settings.
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXPERIMENT = "hgb_36h_v3"
OUT = HERE / "output" / EXPERIMENT
CHARTS = OUT / "charts"
MODELS = HERE / "models" / EXPERIMENT

BID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"
TARGET = "consumption_w"
STEPS, INTERVAL_MINUTES = 144, 15
CHUNK_SIZE, PROGRESS_EVERY = 50_000, 12
MAPE_MIN_ACTUAL_W = 100.0
DURATION = pd.Timedelta(minutes=STEPS * INTERVAL_MINUTES)
PROTOCOL = "purged_90_10_common_pairs_v1"

WEATHER = [
    "temperature_c", "relative_humidity_pct", "precipitation_mm",
    "cloud_cover_pct", "shortwave_radiation_wm2", "direct_radiation_wm2",
    "diffuse_radiation_wm2", "wind_speed_ms",
]
BASE = [
    "origin_consumption", "lag_15m", "lag_1h", "lag_24h", "lag_7d",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "month",
]
PARAMS = dict(
    loss="squared_error", learning_rate=0.1, max_iter=60,
    max_leaf_nodes=15, max_bins=63, min_samples_leaf=30,
    l2_regularization=1.0, early_stopping=False, random_state=42,
)
VARIANTS = [
    ("HGB", BASE, "hgb"),
    ("HGB + Weather", BASE + WEATHER, "hgb_weather"),
]
ORDER = ["Persistence", "Copy-Yesterday", "HGB", "HGB + Weather"]
COLORS = {
    "Actual": "#222222", "Persistence": "#d62728",
    "Copy-Yesterday": "#1f77b4", "HGB": "#ff7f0e",
    "HGB + Weather": "#2ca02c",
}
PNG_FILES = [
    "percentage_metrics_90_10.png", "model_comparison_90_10.png",
    "horizon_mae_90_10.png", "horizon_rmse_90_10.png",
    "actual_vs_predicted_90_10.png",
]


def save_json(data):
    tmp = OUT / "run_summary.json.tmp"
    tmp.write_text(
        json.dumps(data, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(OUT / "run_summary.json")


def peak_memory_mib():
    """Process peak RSS; not a complete container-memory measurement."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024**2 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        return None


def read_building(filename, columns):
    parts = []
    for chunk in pd.read_csv(
        DATA / filename, usecols=columns,
        dtype={"building_id": "string"}, chunksize=CHUNK_SIZE,
    ):
        selected = chunk.loc[chunk["building_id"].eq(BID)].copy()
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=columns
    )


def prepare():
    e = read_building(
        "electricity.csv", ["building_id", "timestamp_utc", TARGET]
    )
    w = read_building(
        "weather.csv", ["building_id", "timestamp_utc"] + WEATHER
    )
    b = read_building("buildings.csv", ["building_id", "description"])
    if e.empty:
        raise ValueError(f"No electricity observations for {BID}.")

    raw_n = len(e)
    for frame in (e, w):
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], utc=True, errors="coerce"
        )
    for frame, columns in ((e, [TARGET]), (w, WEATHER)):
        for col in columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    invalid = int((e["timestamp_utc"].isna() | e[TARGET].isna()).sum())
    e = e.dropna(subset=["timestamp_utc", TARGET]).sort_values(
        "timestamp_utc", kind="stable"
    )
    duplicates = int(e.duplicated("timestamp_utc", keep="last").sum())
    e = e.drop_duplicates("timestamp_utc", keep="last")
    negatives = int(e[TARGET].lt(0).sum())
    e = e.loc[e[TARGET].ge(0)].copy()
    if e.empty:
        raise ValueError("No valid nonnegative electricity observations.")

    w = w.dropna(subset=["timestamp_utc"]).drop_duplicates(
        ["building_id", "timestamp_utc"], keep="last"
    )
    d = e.merge(
        w, on=["building_id", "timestamp_utc"],
        how="left", validate="one_to_one",
    ).sort_values("timestamp_utc").reset_index(drop=True)

    history = d.set_index("timestamp_utc")[TARGET]
    d["origin_consumption"] = d[TARGET]
    for name, delta in [
        ("lag_15m", "15min"), ("lag_1h", "1h"),
        ("lag_24h", "24h"), ("lag_7d", "7D"),
    ]:
        d[name] = (d["timestamp_utc"] - pd.Timedelta(delta)).map(history)

    minutes = d["timestamp_utc"].dt.hour * 60 + d["timestamp_utc"].dt.minute
    dow = d["timestamp_utc"].dt.dayofweek
    d["tod_sin"], d["tod_cos"] = (
        np.sin(2 * np.pi * minutes / 1440),
        np.cos(2 * np.pi * minutes / 1440),
    )
    d["dow_sin"], d["dow_cos"] = (
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    )
    d["is_weekend"], d["month"] = (dow >= 5).astype(int), d["timestamp_utc"].dt.month

    quality = dict(
        raw_rows=raw_n, clean_rows=len(d),
        invalid_timestamp_or_target_rows=invalid,
        duplicate_timestamps_removed=duplicates,
        negative_rows_excluded=negatives,
        building_description=str(b["description"].iloc[0]) if not b.empty else "",
        rows_with_any_missing_weather=int(d[WEATHER].isna().any(axis=1).sum()),
        non_15min_successive_gaps=int(
            d["timestamp_utc"].diff().dropna()
            .ne(pd.Timedelta(minutes=INTERVAL_MINUTES)).sum()
        ),
    )
    return d, history, quality


def targets(history, origins, h):
    times = origins + pd.Timedelta(minutes=INTERVAL_MINUTES * h)
    return history.reindex(pd.DatetimeIndex(times)).to_numpy(dtype=float)


def split(d, history):
    times = d["timestamp_utc"]
    eligible = times.loc[np.isfinite(targets(history, times, STEPS))]
    cut = int(len(eligible) * 0.90)
    if not 0 < cut < len(eligible):
        raise ValueError("Not enough eligible origins for a 90/10 split.")

    train_end, test_start, test_end = (
        eligible.iloc[cut - 1], eligible.iloc[cut], eligible.iloc[-1]
    )

    # Retain only a Boolean mask, not a full training-target matrix.
    complete = np.ones(len(d), dtype=bool)
    for h in range(1, STEPS + 1):
        complete &= np.isfinite(targets(history, times, h))

    train_mask = times.le(train_end).to_numpy() & complete
    test_mask = times.between(test_start, test_end).to_numpy() & complete
    overlap = (times + DURATION >= test_start).to_numpy()
    purged_n = int((train_mask & overlap).sum())
    train_mask &= ~overlap

    tr, te = d.loc[train_mask].copy(), d.loc[test_mask].copy()
    if tr.empty or te.empty:
        raise ValueError("No complete training or testing windows remain.")
    latest = tr["timestamp_utc"].max() + DURATION
    if latest >= test_start:
        raise AssertionError("Training targets reach the test boundary.")

    info = dict(
        legacy_train_origin_cutoff=train_end,
        test_origin_cutoff_start=test_start,
        test_origin_cutoff_end=test_end,
        latest_training_target=latest,
        actual_train_last_origin=tr["timestamp_utc"].max(),
        actual_test_first_origin=te["timestamp_utc"].min(),
        actual_test_last_origin=te["timestamp_utc"].max(),
        complete_train_origins=len(tr), complete_test_origins=len(te),
        purged_train_origins=purged_n,
        incomplete_test_origins_removed=int(
            (times.between(test_start, test_end).to_numpy() & ~complete).sum()
        ),
        training_windows_reaching_test_start=0,
    )
    return tr.reset_index(drop=True), te.reset_index(drop=True), info


def baselines(te, history):
    times = te["timestamp_utc"]
    actual = np.column_stack([
        targets(history, times, h) for h in range(1, STEPS + 1)
    ])
    persistence = np.repeat(
        te["origin_consumption"].to_numpy(float)[:, None], STEPS, axis=1
    )
    seasonal = np.full_like(actual, np.nan)
    for h in range(1, STEPS + 1):
        lead = pd.Timedelta(minutes=INTERVAL_MINUTES * h)
        back = int(np.ceil(lead / pd.Timedelta(days=1)))
        reference = times + lead - pd.Timedelta(days=back)
        if (reference > times).any():
            raise AssertionError("Baseline uses future observations.")
        seasonal[:, h - 1] = history.reindex(
            pd.DatetimeIndex(reference)
        ).to_numpy(float)
    return actual, {"Persistence": persistence, "Copy-Yesterday": seasonal}


def evaluate(actual, predictions):
    common = np.isfinite(actual)
    for name in ORDER:
        if predictions[name].shape != actual.shape:
            raise AssertionError(f"Prediction shape mismatch: {name}")
        common &= np.isfinite(predictions[name])
    if not common.any(axis=0).all():
        raise ValueError("A horizon has no common prediction pairs.")

    a = actual[common]
    mean_actual = float(a.mean())
    mape_mask = np.abs(a) >= MAPE_MIN_ACTUAL_W
    summary, horizon_rows = [], []

    for name in ORDER:
        p = predictions[name]
        error = p[common] - a
        mae, rmse = np.abs(error).mean(), np.sqrt(np.mean(error**2))
        available = int((np.isfinite(actual) & np.isfinite(p)).sum())
        h_rmse = []
        for j in range(STEPS):
            mask = common[:, j]
            e = p[mask, j] - actual[mask, j]
            r = float(np.sqrt(np.mean(e**2)))
            h_rmse.append(r)
            horizon_rows.append({
                "model": name, "hours": (j + 1) * INTERVAL_MINUTES / 60,
                "mae": float(np.abs(e).mean()) / 1000, "rmse": r / 1000,
            })
        summary.append({
            "model": name, "MAE (kW)": float(mae / 1000),
            "Pooled RMSE (kW)": float(rmse / 1000),
            "Mean horizon RMSE (kW)": float(np.mean(h_rmse) / 1000),
            "NMAE (%)": float(100 * mae / mean_actual) if mean_actual > 0 else np.nan,
            "Pooled NRMSE (%)": float(100 * rmse / mean_actual) if mean_actual > 0 else np.nan,
            "MAPE (%)": float(
                100 * np.mean(np.abs(error[mape_mask] / a[mape_mask]))
            ) if mape_mask.any() else np.nan,
            "Mean actual (kW)": mean_actual / 1000,
            "Mean bias (kW)": float(error.mean() / 1000),
            "prediction_pairs": int(common.sum()),
            "eligible_pairs": actual.size,
            "Common pair coverage (%)": 100 * common.sum() / actual.size,
            "Native coverage (%)": 100 * available / actual.size,
            "native_missing_pairs": actual.size - available,
            "mape_excluded_n": int((~mape_mask).sum()),
        })
    return pd.DataFrame(summary), pd.DataFrame(horizon_rows), common


def charts(summary, horizons, te, actual, predictions, common, run_id):
    def save(fig, filename):
        fig.tight_layout()
        path = CHARTS / filename
        tmp = path.with_name(path.stem + ".tmp.png")
        fig.savefig(
            tmp, format="png", dpi=150,
            metadata={"Description": f"run_id={run_id}; protocol={PROTOCOL}"},
        )
        tmp.replace(path)
        plt.close(fig)

    for columns, title, unit, filename in [
        (["NMAE (%)", "Pooled NRMSE (%)", "MAPE (%)"],
         "Percentage errors", "Error (%)", PNG_FILES[0]),
        (["MAE (kW)", "Pooled RMSE (kW)"],
         "Absolute errors", "Error (kW)", PNG_FILES[1]),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        summary.set_index("model")[columns].plot.bar(ax=ax, rot=15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)
        ax.set(title=f"{title} — purged 90/10, common pairs", ylabel=unit, xlabel="")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.margins(y=0.15)
        save(fig, filename)

    for metric, filename in [("mae", PNG_FILES[2]), ("rmse", PNG_FILES[3])]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for name in ORDER:
            g = horizons.loc[horizons["model"].eq(name)]
            ax.plot(g["hours"], g[metric], label=name, color=COLORS[name])
        ax.set(
            title=f"{metric.upper()} by horizon — purged 90/10, common pairs",
            xlabel="Forecast horizon (hours)", ylabel=f"{metric.upper()} (kW)",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, filename)

    times = pd.DatetimeIndex(te["timestamp_utc"] + DURATION)
    end, start = times.max(), times.max() - pd.Timedelta(days=7)
    view = pd.DataFrame(index=times)
    for name, values in [("Actual", actual)] + list(predictions.items()):
        view[name] = np.where(common[:, -1], values[:, -1], np.nan) / 1000
    view = view.loc[view.index >= start]
    grid = pd.date_range(start, end, freq=f"{INTERVAL_MINUTES}min")
    view = view.reindex(view.index.union(grid)).sort_index()

    fig, ax = plt.subplots(figsize=(13, 6))
    for name in ["Actual"] + ORDER:
        baseline = name in ORDER[:2]
        ax.plot(
            view.index, view[name], label=name, color=COLORS[name],
            linestyle="--" if baseline else "-",
            alpha=0.55 if baseline else 0.9,
        )
    ax.set(
        title="Actual vs predicted — fixed 36h lead, last 7 calendar days, common pairs",
        xlabel="Target timestamp (UTC)", ylabel="Consumption (kW)",
        xlim=(start, end),
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=timezone.utc))
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate(rotation=25)
    save(fig, PNG_FILES[4])


def archive_old_outputs(run_id):
    """Move obsolete output items outside OUT; never touch DATA or MODELS."""
    archive = OUT.parent / f"{EXPERIMENT}_archive" / run_id
    moved = []

    for folder, allowed in [
        (OUT, {"charts", "metrics_90_10.csv", "run_summary.json"}),
        (CHARTS, set(PNG_FILES)),
    ]:
        for item in list(folder.iterdir()):
            if item.name not in allowed:
                destination = archive / item.relative_to(OUT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(destination))
                moved.append(str(item.relative_to(OUT)))
    return str(archive) if moved else None, moved


def main():
    started, run_id = time.perf_counter(), uuid.uuid4().hex
    CHARTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    run = dict(
        run_id=run_id, experiment=EXPERIMENT, status="RUNNING",
        evaluation_protocol=PROTOCOL, building_id=BID,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        parameters=PARAMS, forecast_steps=STEPS, interval_minutes=INTERVAL_MINUTES,
        thread_limit=THREAD_LIMIT, purge_training_overlap=True,
        baseline_lookup="full_history", sklearn_version=sklearn.__version__,
        numpy_version=np.__version__, pandas_version=pd.__version__,
        preprocessing="training-only median imputation + scaling",
        weather_note="Observed forecast-origin weather only; assumed available at issue time.",
        metric_note="Pooled common-pair errors; normalization by common-pair mean actual.",
        coverage_note="Native availability is separate; coverage uses complete test windows.",
        mape_min_actual_w=MAPE_MIN_ACTUAL_W, variants={},
    )

    def update():
        run["runtime_seconds"] = time.perf_counter() - started
        run["process_peak_rss_mib"] = peak_memory_mib()
        save_json(run)

    update()
    try:
        print("Output:", OUT, "\nModels:", MODELS, flush=True)
        d, history, quality = prepare()
        tr, te, split_info = split(d, history)
        run.update(split_info)
        run["data_quality"] = quality
        actual, predictions = baselines(te, history)
        del d
        gc.collect()
        print(
            f"Origins: train={len(tr):,}, test={len(te):,}; "
            f"purged={split_info['purged_train_origins']}; overlap=0",
            flush=True,
        )

        for name, features, filename in VARIANTS:
            tick = time.perf_counter()
            pre = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            Xtr, Xte = pre.fit_transform(tr[features]), pre.transform(te[features])
            models, predicted = [], np.empty_like(actual)
            state = {"status": "TRAINING", "completed_horizons": 0}
            run["variants"][name] = state
            for h in range(1, STEPS + 1):
                y = targets(history, tr["timestamp_utc"], h)
                model = HistGradientBoostingRegressor(**PARAMS).fit(Xtr, y)
                predicted[:, h - 1] = model.predict(Xte)
                models.append(model)
                state["completed_horizons"] = h
                if h % PROGRESS_EVERY == 0:
                    print(f"{name}: {h}/{STEPS}", flush=True)
                    update()
            if not np.isfinite(predicted).all():
                raise ValueError(f"{name} produced nonfinite predictions.")
            predictions[name] = predicted
            path = MODELS / f"{filename}_90_10.joblib"
            tmp = path.with_name(path.name + ".tmp")
            joblib.dump(dict(
                run_id=run_id, preprocessing=pre, models=models,
                features=features, building_id=BID, target=TARGET,
                forecast_steps=STEPS, interval_minutes=INTERVAL_MINUTES,
                validation="90_10", experiment=EXPERIMENT,
                evaluation_protocol=PROTOCOL, purge_training_overlap=True,
                baseline_lookup="full_history", parameters=PARAMS,
                sklearn_version=sklearn.__version__, split=split_info,
                weather_mode="observed forecast-origin weather only"
                if name == "HGB + Weather" else "none",
            ), tmp, compress=3)
            tmp.replace(path)
            state.update(
                status="COMPLETED", model_saved=True,
                model_file_bytes=path.stat().st_size,
                wall_seconds=time.perf_counter() - tick,
            )
            del models, model, pre, Xtr, Xte, y
            gc.collect()
            update()

        summary, horizons, common = evaluate(actual, predictions)
        charts(summary, horizons, te, actual, predictions, common, run_id)
        summary.insert(0, "run_id", run_id)
        tmp = OUT / "metrics_90_10.csv.tmp"
        summary.to_csv(tmp, index=False)
        tmp.replace(OUT / "metrics_90_10.csv")

        archive, moved = archive_old_outputs(run_id)
        run.update(
            status="COMPLETED", common_prediction_pairs=int(common.sum()),
            eligible_prediction_pairs=actual.size,
            common_pair_coverage_pct=float(100 * common.sum() / actual.size),
            zero_training_target_overlap_verified=True,
            equal_common_pair_counts_verified=True,
            chart_files=[f"charts/{p}" for p in PNG_FILES],
            archived_old_outputs=archive, archived_items=moved,
            interpretation="Single building and previously inspected holdout; not production validation.",
        )
        update()
        print(summary.drop(columns="run_id").round(2).to_string(index=False))
        print(f"\nRuntime: {run['runtime_seconds']:.1f} seconds")
        print("Process peak RSS (MiB):", run["process_peak_rss_mib"])
        if archive:
            print("Old output files moved to:", archive)
        print("Charts:", CHARTS)

    except (Exception, KeyboardInterrupt) as exc:
        run.update(
            status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
            error=repr(exc),
        )
        update()
        print("Run incomplete. Existing results may belong to an older run.", flush=True)
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=THREAD_LIMIT):
        main()