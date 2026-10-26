#!/usr/bin/env python3
"""Benchmark Envitron Ridge V3. Run from the experiment root.
This benchmark does NOT modify the forecasting experiment or its metrics.
"""
from pathlib import Path
import sys, time, csv, os, platform, resource
import numpy as np
import joblib

ROOT = Path.cwd()
SRC = ROOT / 'src'
if not (SRC / 'ridge_36h_90_10.py').exists():
    sys.exit('Run this from Ridge_36h_90_10_percentage_evaluation/ (folder with src/ and data/).')
sys.path.insert(0, str(SRC))
import ridge_36h_90_10 as exp

print('Loading and preparing building data...')
t0 = time.perf_counter()
d, _, _, _ = exp.load_prepare()
prep_seconds = time.perf_counter() - t0
origins = d[np.isfinite(d['y_144'])].copy()
cut = int(len(origins) * 0.90)
train_end = origins.timestamp_utc.iloc[cut - 1]
test_start = origins.timestamp_utc.iloc[cut]
tr = d[d.timestamp_utc <= train_end]
te = d[d.timestamp_utc >= test_start]
ytr = tr[[f'y_{h}' for h in range(1, exp.STEPS + 1)]].to_numpy(float)
yte = te[[f'y_{h}' for h in range(1, exp.STEPS + 1)]].to_numpy(float)
oktr = np.isfinite(ytr).all(axis=1)
okte = np.isfinite(yte).all(axis=1)
if not oktr.any() or not okte.any():
    sys.exit('No eligible train/test origins found.')

rows = []
for name, features, model_file in [
    ('Ridge', exp.BASE, 'ridge_90_10.joblib'),
    ('Ridge + Weather', exp.BASE + exp.WEATHER, 'ridge_weather_90_10.joblib'),
]:
    model_path = ROOT / 'models' / model_file
    if not model_path.exists():
        sys.exit(f'Missing model: {model_path}. First run python3 src/ridge_36h_90_10.py')
    saved = joblib.load(model_path)
    model = saved['model']
    xtrain = tr.loc[oktr, features]
    xone = te.loc[okte, features].iloc[[0]]
    # Warm up to exclude one-time library initialization from inference timing.
    model.predict(xone)
    samples = []
    for _ in range(100):
        t = time.perf_counter()
        result = model.predict(xone)
        samples.append((time.perf_counter() - t) * 1000)
    assert result.shape == (1, 144), result.shape
    fresh = exp.pipe()
    t = time.perf_counter()
    fresh.fit(xtrain, ytr[oktr])
    fit_seconds = time.perf_counter() - t
    ridge = model.named_steps['ridge']
    # 144 target horizons; coefficients are for scaled features, not raw kW per unit.
    coefficients = ridge.coef_
    print(f'\n{name}:')
    print(f'  Training samples: {len(xtrain):,} | Features: {len(features)} | Outputs: 144')
    print(f'  Training time: {fit_seconds:.3f} seconds')
    print(f'  One 36-hour forecast: median {np.median(samples):.3f} ms, p95 {np.percentile(samples,95):.3f} ms')
    print(f'  Saved model: {model_path.stat().st_size/1024:.2f} KB')
    print(f'  Coefficient matrix: {coefficients.shape} (144 forecast steps x {len(features)} features)')
    # Example explanation: largest standardized coefficients at horizon 36h.
    ranks = np.argsort(np.abs(coefficients[-1]))[::-1][:5]
    print('  Largest absolute standardized coefficients at 36h (not causal effects):')
    for i in ranks:
        print(f'    {features[i]}: {coefficients[-1, i]:.3f}')
    rows.append({
        'model': name, 'training_samples': len(xtrain), 'features': len(features),
        'forecast_outputs': 144, 'training_seconds': round(fit_seconds, 5),
        'one_forecast_median_ms': round(float(np.median(samples)), 5),
        'one_forecast_p95_ms': round(float(np.percentile(samples,95)), 5),
        'saved_model_kb': round(model_path.stat().st_size / 1024, 3),
    })

# ru_maxrss is process-wide peak since start, including Python and data preparation.
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024
print(f'\nFull benchmark process peak RAM: {peak_mb:.1f} MB (includes CSV loading and preprocessing)')
print(f'Data loading + preparation: {prep_seconds:.2f} seconds')
print('Note: This benchmark is run on your current computer, not a Raspberry Pi.')
print('Note: Training/test boundary overlap in the original experiment is NOT corrected by this benchmark.')
out = ROOT / 'output' / 'ridge_performance_benchmark.csv'
out.parent.mkdir(exist_ok=True)
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ['process_peak_ram_mb', 'data_preparation_seconds'])
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, 'process_peak_ram_mb': round(peak_mb, 2), 'data_preparation_seconds': round(prep_seconds, 2)})
print(f'Saved: {out}')
