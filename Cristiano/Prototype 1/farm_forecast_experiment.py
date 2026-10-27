

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BUILDING_ID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"
STEPS_PER_HOUR = 4
HORIZON = 36 * STEPS_PER_HOUR            # 144 steps
DAY, WEEK = 96, 672                      # steps
TRAIN_FRAC = 0.90
LAGS = [1, 2, 4, 8, 16, 48, DAY, 2 * DAY, WEEK]
WEATHER_COLS = ["temperature_c", "relative_humidity_pct", "precipitation_mm",
                "cloud_cover_pct", "shortwave_radiation_wm2", "wind_speed_ms"]
TRAIN_ORIGIN_STRIDE = 8                  # one training origin every 2 hours
TEST_ORIGIN_STRIDE = 4                   # one test forecast every hour
SEED = 42
LOCAL_TZ = "Europe/Amsterdam"            # farm routines follow local clock time


# ------------------------------------------------------------------ data
def load(elec_path, weather_path):
    e = pd.read_csv(elec_path)
    w = pd.read_csv(weather_path)
    e = e[e.building_id == BUILDING_ID]
    w = w[w.building_id == BUILDING_ID]
    for d in (e, w):
        d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    df = (e.set_index("timestamp_utc")[["consumption_w"]]
          .join(w.set_index("timestamp_utc")[WEATHER_COLS], how="left")
          .sort_index())
    df = df[~df.index.duplicated()].asfreq("15min")
    df["load_kw"] = df.pop("consumption_w") / 1000.0

    # Cleaning: a handful of readings (0.13%) are negative, which is physically
    # impossible for consumption -> metering glitches. Set to NaN and interpolate.
    n_bad = int((df["load_kw"] < 0).sum())
    df.loc[df["load_kw"] < 0, "load_kw"] = np.nan
    n_missing = int(df["load_kw"].isna().sum())
    df["load_kw"] = df["load_kw"].interpolate(limit_direction="both")
    df[WEATHER_COLS] = df[WEATHER_COLS].interpolate(limit_direction="both")
    return df, {"negative_readings_replaced": n_bad, "total_interpolated": n_missing}


# ------------------------------------------------------------------ features
def build_rows(df, origin_pos):
    """One row per (forecast origin, horizon step). Origin = last known value.
    All load features use data at or before the origin only. Weather features
    use the value at the target time (stands in for a weather forecast)."""
    y = df["load_kw"].values
    W = df[WEATHER_COLS].values
    idx = df.index
    p = np.asarray(origin_pos)
    n = len(y)

    cs = np.concatenate([[0], np.cumsum(y)])
    cs2 = np.concatenate([[0], np.cumsum(y ** 2)])
    def roll_mean(k): return (cs[p + 1] - cs[p + 1 - k]) / k
    def roll_std(k):
        m = roll_mean(k)
        return np.sqrt(np.maximum((cs2[p + 1] - cs2[p + 1 - k]) / k - m ** 2, 0))

    base = {f"lag_{L}": y[p - L + 1] for L in LAGS}
    base.update({"mean_4h": roll_mean(16), "mean_24h": roll_mean(DAY),
                 "std_24h": roll_std(DAY), "mean_7d": roll_mean(WEEK),
                 "temp_origin": W[p, 0]})

    frames = []
    for h in range(1, HORIZON + 1):
        t = p + h
        valid = t < n
        tt = np.where(valid, t, n - 1)
        local = idx[tt].tz_convert(LOCAL_TZ)
        tod = local.hour + local.minute / 60
        r = {"h": np.full(len(p), h)}
        r.update(base)
        # same time-of-day yesterday / last week, relative to the TARGET,
        # always taken from before the origin (no leakage)
        r["same_time_yday"] = y[tt - DAY * int(np.ceil(h / DAY))]
        r["same_time_lastweek"] = y[tt - WEEK]
        r["tod_sin"] = np.sin(2 * np.pi * tod / 24)
        r["tod_cos"] = np.cos(2 * np.pi * tod / 24)
        r["tod"] = tod
        r["dow"] = local.dayofweek
        r["weekend"] = (local.dayofweek >= 5).astype(int)
        r["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
        r["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
        for j, c in enumerate(WEATHER_COLS):
            r[c] = W[tt, j]
        r["_origin"] = p
        r["_target"] = tt
        r["_valid"] = valid
        r["y"] = y[tt]
        frames.append(pd.DataFrame(r))
    out = pd.concat(frames, ignore_index=True)
    return out[out["_valid"]].drop(columns="_valid").reset_index(drop=True)


def feats(frame):
    return [c for c in frame.columns if not c.startswith("_") and c != "y"]


def ridge_X(X, cols=None):
    Xd = pd.get_dummies(X.assign(tod=X["tod"].astype(int)), columns=["dow", "tod"])
    return Xd if cols is None else Xd.reindex(columns=cols, fill_value=0)


# ------------------------------------------------------------------ metrics
def metrics(y, yhat):
    e = yhat - y
    mae, rmse = np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))
    ybar = np.mean(y)
    return {"MAE_kW": mae, "RMSE_kW": rmse,
            "MAPE_%": np.mean(np.abs(e) / np.abs(y)) * 100,
            "NMAE_%": mae / ybar * 100,
            "NRMSE_%": rmse / ybar * 100}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elec", default="electricity.csv")
    ap.add_argument("--weather", default="weather.csv")
    args = ap.parse_args()

    df, cleaning = load(args.elec, args.weather)
    n = len(df)
    split = int(n * TRAIN_FRAC)          # first index of the test period
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    print(f"Building {BUILDING_ID} (farm)")
    print(f"Data: {n:,} x 15-min, {df.index[0]} -> {df.index[-1]} ({days:.0f} days)")
    print(f"Cleaning: {cleaning}")
    print(f"Train: {df.index[0]} -> {df.index[split - 1]}  ({split:,} steps)")
    print(f"Test : {df.index[split]} -> {df.index[-1]}  ({n - split:,} steps)")

    # Train origins: need a week of history; their targets must stay inside train.
    train_o = np.arange(WEEK, split - HORIZON, TRAIN_ORIGIN_STRIDE)
    # Test origins: forecast made from inside the test period, full 36h ahead.
    # (History before the origin may come from train - that is normal operation.)
    test_o = np.arange(split - 1, n - HORIZON, TEST_ORIGIN_STRIDE)

    train = build_rows(df, train_o)
    test = build_rows(df, test_o)
    assert train["_target"].max() < split, "leakage: train target in test period"
    assert test["_target"].min() >= split
    F = feats(train)
    print(f"Training rows: {len(train):,} | test forecasts: {len(test_o):,} x {HORIZON} steps "
          f"= {len(test):,} predictions")

    # ---- models
    gbm_kw = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                  min_samples_leaf=100, l2_regularization=1.0, random_state=SEED)
    gbm = HistGradientBoostingRegressor(**gbm_kw).fit(train[F], train["y"])
    q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **gbm_kw).fit(train[F], train["y"])
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **gbm_kw).fit(train[F], train["y"])
    Xr = ridge_X(train[F])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xr, train["y"])

    test["gbm"] = gbm.predict(test[F])
    test["p10"] = q10.predict(test[F])
    test["p90"] = q90.predict(test[F])
    test["ridge"] = ridge.predict(ridge_X(test[F], Xr.columns))
    test["naive_week"] = test["same_time_lastweek"]
    test["naive_day"] = test["same_time_yday"]

    names = {"gbm": "Gradient boosting (proposed)", "ridge": "Ridge regression",
             "naive_week": "Seasonal naive (1 week)", "naive_day": "Daily naive (1 day)"}
    res = pd.DataFrame({names[m]: metrics(test["y"].values, test[m].values) for m in names}).T
    cov = ((test.y >= test.p10) & (test.y <= test.p90)).mean() * 100
    print("\nTest-set results (all origins, all 144 horizon steps):")
    print(res.round(3).to_string())
    print(f"GBM 80% prediction interval coverage: {cov:.1f}%")
    res.round(4).to_csv("results_metrics.csv")

    # error per horizon (in hours)
    test["h_hours"] = np.ceil(test["h"] / STEPS_PER_HOUR).astype(int)
    by_h = pd.DataFrame({names[m]: test.assign(ae=(test[m] - test.y).abs())
                         .groupby("h_hours")["ae"].mean() for m in names})
    by_h.round(4).to_csv("results_mae_by_horizon_hour.csv")

    # feature importance (permutation, on a sample of test rows)
    from sklearn.inspection import permutation_importance
    samp = test.sample(20000, random_state=SEED)
    pi = permutation_importance(gbm, samp[F], samp["y"], n_repeats=3,
                                random_state=SEED, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=F).sort_values(ascending=False)
    imp.round(4).to_csv("results_feature_importance.csv", header=["MAE_increase_kW"])

    with open("experiment_setup.json", "w") as f:
        json.dump({
            "building_id": BUILDING_ID, "building_type": "farm",
            "data_start": str(df.index[0]), "data_end": str(df.index[-1]),
            "days": round(days, 1), "resolution": "15 min",
            "horizon_steps": HORIZON, "horizon_hours": 36,
            "split": "90/10 chronological", "train_end": str(df.index[split - 1]),
            "test_start": str(df.index[split]), "test_origins": int(len(test_o)),
            "test_origin_every": "1 hour", "train_origin_every": "2 hours",
            "cleaning": cleaning, "interval_coverage_pct": round(cov, 2),
            "normalisation": "NMAE and NRMSE divided by mean actual load in the test set",
            "weather_note": "Observed weather at target time used as a perfect weather forecast",
        }, f, indent=2)

    # ---- plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    for ax, frac in zip(axes[:2], (0.3, 0.75)):
        o = test_o[int(len(test_o) * frac)]
        ex = test[test._origin == o]
        hist = df["load_kw"].iloc[o - 2 * DAY:o + 1]
        tx = df.index[ex._target].tz_convert(LOCAL_TZ)
        ax.plot(hist.index.tz_convert(LOCAL_TZ), hist.values, color="#444", lw=1, label="History")
        ax.plot(tx, ex.y, color="#444", lw=1, ls=":", label="Actual")
        ax.fill_between(tx, ex.p10, ex.p90, color="#1f6feb", alpha=0.18, label="GBM 80% interval")
        ax.plot(tx, ex.gbm, color="#1f6feb", lw=2, label="Gradient boosting")
        ax.plot(tx, ex.ridge, color="#d97706", lw=1.4, label="Ridge")
        ax.axvline(df.index[o].tz_convert(LOCAL_TZ), color="grey", ls="--", lw=1)
        ax.set_title(f"36-hour forecast made at {df.index[o].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M} (local)")
        ax.set_ylabel("Consumption (kW)"); ax.legend(loc="upper left", fontsize=8)
    ax = axes[2]
    styles = {"Gradient boosting (proposed)": ("#1f6feb", "-", 2),
              "Ridge regression": ("#d97706", "-", 1.5),
              "Seasonal naive (1 week)": ("#888", "--", 1.3),
              "Daily naive (1 day)": ("#888", ":", 1.3)}
    for c in by_h:
        col, ls, lw = styles[c]
        ax.plot(by_h.index, by_h[c], color=col, ls=ls, lw=lw, label=c)
    ax.set_title("Test MAE by forecast horizon"); ax.set_xlabel("Hours ahead")
    ax.set_ylabel("MAE (kW)"); ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    for a in axes: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("results_report.png", dpi=120)
    print("\nSaved: results_metrics.csv, results_mae_by_horizon_hour.csv, "
          "results_feature_importance.csv, experiment_setup.json, results_report.png")


if __name__ == "__main__":
    main()
