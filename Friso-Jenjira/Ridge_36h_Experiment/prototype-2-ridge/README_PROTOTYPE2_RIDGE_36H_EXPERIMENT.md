# Project 2.1 Envitron
# Team IT2B IT Program NHL Stenden
# Prototype 2 Extension — 36-Hour Ridge Experiment

This experiment extends the existing Prototype 2 after Envitron's 29 September 2026 feedback. 
The original 15-minute prototype and its outputs are preserved.

## Experiment scope

- Forecast horizon: 36 hours (144 x 15-minute outputs)
- Building: `6a1bf87e-d798-4e93-a648-a8e948fc24c2` (new building with fewest nagative consumption values)
- Negative consumption: 59 rows excluded from modelling and exported separately; raw CSV remains unchanged
- Validation: 80/20 chronological, 90/10 chronological, and 3-fold expanding walk-forward
- Models: Persistence, Copy-Yesterday, Ridge, Ridge + Weather
- Metrics: MAE and RMSE

## Leakage prevention

All 144 Ridge outputs are predicted directly from information available at the forecast origin. 
Future actual consumption is never used as a model feature.

The supplied weather file contains historical observed weather, not archived weather-forecast snapshots.
Therefore Ridge + Weather uses weather observed at the forecast origin only. 
It does **not** use future observed weather. 
A later experiment can use Envitron's forecast-weather API data when suitable forecast snapshots are available.

For Copy-Yesterday, targets up to 24 hours ahead use the previous day's same time slot. 
Targets beyond 24 hours cannot use `target - 24h` because that would occur after the forecast origin. 
They therefore use the most recent fully known corresponding daily profile (48 hours back).

## Main measured results

Average error across the 144 forecast horizons:

| Validation | Model | MAE | RMSE |
|---|---|---:|---:|
| 80/20 | Copy-Yesterday | 3.60 kW | 5.12 kW |
| 80/20 | Persistence | 4.93 kW | 6.39 kW |
| 80/20 | Ridge | 3.21 kW | **4.14 kW** |
| 80/20 | Ridge + Weather | **3.00 kW** | 4.24 kW |

| 90/10 | Copy-Yesterday | 3.51 kW | 5.06 kW |
| 90/10 | Persistence | 4.99 kW | 6.49 kW |
| 90/10 | Ridge | 3.18 kW | 4.10 kW |
| 90/10 | Ridge + Weather | **2.87 kW** | **3.96 kW** |

| Walk-forward | Copy-Yesterday | 4.54 kW | 6.17 kW |
| Walk-forward | Persistence | 5.34 kW | 6.87 kW |
| Walk-forward | Ridge | 4.02 kW | **4.95 kW** |
| Walk-forward | Ridge + Weather | **3.93 kW** | 4.97 kW |

## Initial interpretation

- Ridge + Weather has the lowest average MAE in all 3 validation approaches.
- Ridge + Weather also has the lowest RMSE for 90/10; Ridge is marginally lower for 80/20 and walk-forward.
- Walk-forward errors are higher than the single 80/20 and 90/10 holdouts, 
showing that performance changes across time periods. 
This supports Envitron's concern that one chronological split can give an incomplete picture.
- Persistence is much weaker at a 36-hour horizon than it was in the original 15-minute experiment. 
This is expected because one current value becomes less informative farther into the future.
- These results are experimental evidence for this building only; they do not select a final model.

## New outputs

`output/experiment_36h/`
- `metrics_summary.csv`
- `metrics_by_horizon.csv`
- `predictions_selected_horizons.csv`
- `data_quality.csv`
- `excluded_negative_consumption.csv`
- `model_validation_mae.png`
- `model_validation_rmse.png`
- `horizon_mae_90_10.png`
- `horizon_rmse_90_10.png`
- `actual_vs_forecast_36h.png`
- `run_summary.json`

`models/experiment_36h/`
- saved 36-hour Ridge and Ridge + Weather models for the 80/20 and 90/10 holdouts
