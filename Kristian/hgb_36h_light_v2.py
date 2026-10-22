#!/usr/bin/env python3
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent

DATA = HERE / 'data'
OUT = HERE / 'output' / 'hgb_36h_light'

for filename in ['electricity.csv', 'weather.csv', 'buildings.csv']:
    filepath = DATA / filename

    if not filepath.is_file():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

print(f"Datasets found in: {DATA}", flush=True)
print(f"Results will be saved to: {OUT}", flush=True)

TARGET = 'consumption_w'
BID = '6a1bf87e-d798-4e93-a648-a8e948fc24c2'
STEPS = 144
VAL_NAME = '90_10'

WEATHER = [
    'temperature_c','relative_humidity_pct','precipitation_mm','cloud_cover_pct',
    'shortwave_radiation_wm2','direct_radiation_wm2','diffuse_radiation_wm2','wind_speed_ms'
]
BASE = [
    'origin_consumption','lag_15m','lag_1h','lag_24h','lag_7d',
    'tod_sin','tod_cos','dow_sin','dow_cos','is_weekend','month'
]

USE_WEATHER = True
EXPORT_SELECTED_HORIZONS_ONLY = [1, 24, 48, 96, 144]
MAX_ITER = 120
MAX_DEPTH = 6
LEARNING_RATE = 0.08
MIN_SAMPLES_LEAF = 30
L2 = 0.0
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
RANDOM_STATE = 42


def load_prepare():
    e = pd.read_csv(DATA / 'electricity.csv', usecols=['building_id', 'timestamp_utc', TARGET])
    w_cols = ['building_id', 'timestamp_utc'] + WEATHER
    w = pd.read_csv(DATA / 'weather.csv', usecols=w_cols)
    b = pd.read_csv(DATA / 'buildings.csv')

    for z in (e, w):
        z['timestamp_utc'] = pd.to_datetime(z['timestamp_utc'], utc=True, errors='coerce')

    x = (
        e[e.building_id == BID][['building_id', 'timestamp_utc', TARGET]]
        .dropna()
        .sort_values('timestamp_utc')
        .drop_duplicates('timestamp_utc', keep='last')
        .copy()
    )
    neg = x[x[TARGET] < 0].copy()
    x = x[x[TARGET] >= 0].copy()

    wx = (
        w[w.building_id == BID][w_cols]
        .drop_duplicates(['building_id', 'timestamp_utc'], keep='last')
        .copy()
    )

    x = x.merge(wx, on=['building_id', 'timestamp_utc'], how='left').sort_values('timestamp_utc').reset_index(drop=True)
    s = x.set_index('timestamp_utc')[TARGET]
    x['origin_consumption'] = x[TARGET]

    for name, delta in [('lag_15m', '15min'), ('lag_1h', '1h'), ('lag_24h', '24h'), ('lag_7d', '7d')]:
        x[name] = (x['timestamp_utc'] - pd.Timedelta(delta)).map(s)

    mins = x['timestamp_utc'].dt.hour * 60 + x['timestamp_utc'].dt.minute
    dow = x['timestamp_utc'].dt.dayofweek
    x['tod_sin'] = np.sin(2 * np.pi * mins / 1440)
    x['tod_cos'] = np.cos(2 * np.pi * mins / 1440)
    x['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    x['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    x['is_weekend'] = (dow >= 5).astype(int)
    x['month'] = x['timestamp_utc'].dt.month

    for h in range(1, STEPS + 1):
        x[f'y_{h}'] = (x['timestamp_utc'] + pd.Timedelta(minutes=15 * h)).map(s)

    desc = b.loc[b.building_id == BID, 'description'].iloc[0] if BID in set(b.building_id) else ''
    return x, neg, desc, len(e[e.building_id == BID])


def build_model():
    features = BASE + WEATHER if USE_WEATHER else BASE
    model = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('hgb', MultiOutputRegressor(HistGradientBoostingRegressor(
            loss='squared_error',
            learning_rate=LEARNING_RATE,
            max_iter=MAX_ITER,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            l2_regularization=L2,
            early_stopping=EARLY_STOPPING,
            validation_fraction=VALIDATION_FRACTION,
            random_state=RANDOM_STATE
        ), n_jobs=-1))
    ])
    return model, features


def baseline_arrays(d):
    y = d[[f'y_{h}' for h in range(1, STEPS + 1)]].to_numpy(float)
    persistence = np.repeat(d['origin_consumption'].to_numpy(float)[:, None], STEPS, axis=1)
    smap = d.set_index('timestamp_utc')[TARGET]
    cy = np.full_like(y, np.nan)
    for h in range(1, STEPS + 1):
        target = d['timestamp_utc'] + pd.Timedelta(minutes=15 * h)
        back = 1 if h <= 96 else 2
        cy[:, h - 1] = (target - pd.Timedelta(days=back)).map(smap).to_numpy(float)
    return y, persistence, cy


def metrics_by_horizon(validation, model, y, p):
    rows = []
    for h in range(STEPS):
        m = np.isfinite(y[:, h]) & np.isfinite(p[:, h])
        yy, pp = y[m, h], p[m, h]
        if len(yy) == 0:
            continue
        mae = mean_absolute_error(yy, pp)
        rmse = mean_squared_error(yy, pp) ** 0.5
        mean_actual = np.mean(np.abs(yy)) if np.mean(np.abs(yy)) != 0 else np.nan
        nmae = (mae / mean_actual * 100) if np.isfinite(mean_actual) else np.nan
        nrmse = (rmse / mean_actual * 100) if np.isfinite(mean_actual) else np.nan
        mape_mask = yy != 0
        mape = np.mean(np.abs((yy[mape_mask] - pp[mape_mask]) / yy[mape_mask])) * 100 if np.any(mape_mask) else np.nan
        rows.append({
            'validation': validation,
            'model': model,
            'horizon_step': h + 1,
            'horizon_hours': (h + 1) / 4,
            'n': len(yy),
            'mae_w': mae,
            'rmse_w': rmse,
            'nmae_pct': nmae,
            'nrmse_pct': nrmse,
            'mape_pct': mape,
        })
    return rows


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    d, neg, desc, raw_n = load_prepare()
    neg.to_csv(OUT / 'excluded_negative_consumption.csv', index=False)
    pd.DataFrame([{
        'building_id': BID,
        'description': desc,
        'raw_rows': raw_n,
        'negative_rows_excluded': len(neg),
        'negative_pct_raw': len(neg) / raw_n * 100 if raw_n else np.nan,
        'rows_after_exclusion': len(d),
        'start': d['timestamp_utc'].min(),
        'end': d['timestamp_utc'].max(),
    }]).to_csv(OUT / 'data_quality.csv', index=False)

    origins = d[np.isfinite(d['y_144'])].copy()
    n = len(origins)
    cut = int(n * 0.90)
    train_end = origins['timestamp_utc'].iloc[cut - 1]
    test_start = origins['timestamp_utc'].iloc[cut]
    test_end = origins['timestamp_utc'].iloc[-1]

    tr = d[d['timestamp_utc'] <= train_end].copy()
    te = d[(d['timestamp_utc'] >= test_start) & (d['timestamp_utc'] <= test_end)].copy()

    Ytr = tr[[f'y_{h}' for h in range(1, STEPS + 1)]].to_numpy(float)
    Yte, pers, cy = baseline_arrays(te)
    oktr = np.isfinite(Ytr).all(axis=1)
    okte = np.isfinite(Yte).all(axis=1)

    model, features = build_model()
    model.fit(tr.loc[oktr, features], Ytr[oktr])
    pred = np.full_like(Yte, np.nan)
    pred[okte] = model.predict(te.loc[okte, features])

    rows = []
    rows += metrics_by_horizon(VAL_NAME, 'Persistence', Yte[okte], pers[okte])
    rows += metrics_by_horizon(VAL_NAME, 'Copy-Yesterday', Yte[okte], cy[okte])
    rows += metrics_by_horizon(VAL_NAME, 'HistGradientBoosting' + (' + Weather' if USE_WEATHER else ''), Yte[okte], pred[okte])
    met = pd.DataFrame(rows)
    met.to_csv(OUT / 'metrics_by_horizon.csv', index=False)

    summary = met.groupby(['validation', 'model'], as_index=False).agg(
        mae_w=('mae_w', 'mean'),
        rmse_w=('rmse_w', 'mean'),
        nmae_pct=('nmae_pct', 'mean'),
        nrmse_pct=('nrmse_pct', 'mean'),
        mape_pct=('mape_pct', 'mean'),
        horizons=('horizon_step', 'nunique')
    )
    summary.to_csv(OUT / 'metrics_summary.csv', index=False)

    pr = []
    model_name = 'HistGradientBoosting' + (' + Weather' if USE_WEATHER else '')
    for h in EXPORT_SELECTED_HORIZONS_ONLY:
        idx = h - 1
        pr.append(pd.DataFrame({
            'validation': VAL_NAME,
            'origin_timestamp': te['timestamp_utc'],
            'target_timestamp': te['timestamp_utc'] + pd.Timedelta(minutes=15 * h),
            'horizon_step': h,
            'horizon_hours': h / 4,
            'actual_w': Yte[:, idx],
            'persistence_w': pers[:, idx],
            'copy_yesterday_w': cy[:, idx],
            'hgb_w': pred[:, idx],
            'model_name': model_name,
        }))
    preds = pd.concat(pr, ignore_index=True)
    preds.to_csv(OUT / 'predictions_selected_horizons.csv', index=False)

    pv = summary.pivot(index='model', columns='validation', values='nmae_pct')
    ax = pv.plot(kind='bar', figsize=(8, 5), legend=False, color=['#1E4FA3'])
    ax.set_title('36-hour NMAE (%) — 90/10')
    ax.set_ylabel('NMAE (%)')
    ax.set_xlabel('Model')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / 'model_validation_nmae_pct.png', dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    for model_name, g in met.groupby('model'):
        plt.plot(g['horizon_hours'], g['nmae_pct'], label=model_name)
    plt.title('NMAE (%) Across 36-hour Forecast Horizon — 90/10')
    plt.xlabel('Forecast horizon (hours)')
    plt.ylabel('NMAE (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'horizon_nmae_pct_90_10.png', dpi=180)
    plt.close()

    v = preds[preds['horizon_step'] == 144].dropna().tail(7 * 96)
    if len(v):
        plt.figure(figsize=(12, 5.5))
        plt.plot(v['target_timestamp'], v['actual_w'], label='Actual', alpha=0.9)
        plt.plot(v['target_timestamp'], v['persistence_w'], label='Persistence', alpha=0.8)
        plt.plot(v['target_timestamp'], v['copy_yesterday_w'], label='Copy-Yesterday', alpha=0.8)
        plt.plot(v['target_timestamp'], v['hgb_w'], label=model_name, alpha=0.85)
        plt.title('Actual vs Forecast — 36-hour Horizon (90/10)')
        plt.ylabel('Consumption (W)')
        plt.xticks(rotation=25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / 'actual_vs_forecast_36h.png', dpi=180)
        plt.close()

    run = {
        'experiment': 'HistGradientBoosting 36-Hour Forecasting Experiment (Lightweight)',
        'building_id': BID,
        'description': desc,
        'forecast_horizon': '36 hours',
        'resolution': '15 minutes',
        'steps': 144,
        'validation': '90/10 chronological',
        'models': ['Persistence', 'Copy-Yesterday', model_name],
        'metrics': ['MAE', 'RMSE', 'NMAE', 'NRMSE', 'MAPE'],
        'negative_handling': f'{len(neg)} negative rows excluded from modelling and exported separately; raw data unchanged.',
        'weather_mode': 'included' if USE_WEATHER else 'not included',
        'speed_optimizations': [
            '90/10 validation only',
            'single HistGradientBoosting variant',
            'selected-horizon export only',
            'lightweight chart set only',
            'early stopping enabled',
            'bounded tree depth and iterations'
        ],
        'runtime_seconds': time.perf_counter() - t0,
    }
    (OUT / 'run_summary.json').write_text(json.dumps(run, indent=2, default=str))
    print(summary.to_string(index=False))
    print('\nRuntime seconds:', round(run['runtime_seconds'], 2))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
