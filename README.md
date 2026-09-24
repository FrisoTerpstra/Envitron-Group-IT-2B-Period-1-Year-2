# Envitron-Group-IT-2B-Period-1-Year-2
# Power Time-Series Forecasting Prototype

This local Python prototype tests whether a machine-learning forecast can predict power output more accurately than a simple copy-yesterday baseline.

The first machine-learning model is `RandomForestRegressor` from scikit-learn. It is not an LLM, does not use Ollama, does not call cloud APIs, and does not need API keys.

## Data Used

The project currently contains:

- `electricity.csv`: timestamped electricity measurements per `building_id`
- `weather.csv`: timestamped weather measurements per `building_id`
- `buildings.csv`: static building metadata such as PV and battery size

The script looks for these files first in `data/` and then in the project root. The CSV files stay separate from the Python code.

## Inspected Columns

`electricity.csv` contains:

- `building_id`
- `timestamp_utc`
- `consumption_w`
- `production_w`
- `delivery_w`
- `return_delivery_w`
- `charge_w`
- `discharge_w`

`weather.csv` contains:

- `building_id`
- `timestamp_utc`
- `temperature_c`
- `relative_humidity_pct`
- `precipitation_mm`
- `cloud_cover_pct`
- `shortwave_radiation_wm2`
- `direct_radiation_wm2`
- `diffuse_radiation_wm2`
- `wind_speed_ms`

`buildings.csv` contains:

- `building_id`
- `description`
- `pv_kwp`
- `battery_kwh`
- `battery_kw`
- `production_available`
- `first_timestamp_utc`
- `last_timestamp_utc`

## Prediction Target

The prototype predicts:

```text
production_w
```

This was selected because the project goal is power output forecasting, and `production_w` is the direct production measurement in the electricity data.

## Input Features

The Random Forest uses:

- Time features from `timestamp_utc`: hour, minute, day of week, day of year, month, weekend flag, and cyclical sine/cosine encodings
- Weather features: temperature, humidity, precipitation, cloud cover, radiation, and wind speed
- Building metadata: PV size, battery capacity, battery power, and production availability
- Building identity using one-hot encoded `building_id`
- Lagged production values:
  - previous measured production
  - production one hour earlier
  - production at the same time yesterday

Lag features are created only from earlier timestamps, so the model is not trained using future power observations.

Weather is treated as an external input. For real future forecasting, these weather columns would need to come from a weather forecast rather than measured future weather.

## RandomForestRegressor

`RandomForestRegressor` is a supervised machine-learning model that combines many decision trees. Each tree learns patterns between input features and the target value. The forest averages the trees' predictions, which often makes it more stable than a single decision tree.

This prototype uses:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

## Training and Testing

This is a time-series problem, so the data is not randomly shuffled.

The script sorts observations chronologically and uses an approximate chronological split:

- first 80% of unique timestamps for training
- last 20% of unique timestamps for testing

This matters because a forecasting model must not learn from observations that happen after the period it is being tested on.

## Copy-Yesterday Baseline

The copy-yesterday baseline predicts:

```text
power at a timestamp = actual power for the same building at the same time one day earlier
```

The data is mostly measured every 15 minutes, but the implementation does not blindly shift by 96 rows. It matches timestamps using `timestamp_utc - 1 day`, which handles small gaps more correctly.

## Metrics

The script evaluates both copy-yesterday and Random Forest on the same test rows.

MAE means Mean Absolute Error. It is the average absolute difference between actual power and predicted power.

RMSE means Root Mean Squared Error. It also measures prediction error, but larger mistakes are penalized more heavily.

For both metrics, lower is better.

## Install Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

On this Windows machine, Python is available through the launcher, so this may be:

```bash
py -3 -m pip install -r requirements.txt
```

## Run

Run the prototype with:

```bash
python forecast.py
```

or:

```bash
py -3 forecast.py
```

## Outputs

The script creates:

- `output/predictions.csv`
- `output/forecast_chart.png`
- `output/forecast_chart_full_sampled.png`
- `models/random_forest_model.joblib`

`output/predictions.csv` contains:

- `timestamp_utc`
- `building_id`
- `actual_power`
- `copy_yesterday_prediction`
- `random_forest_prediction`
- `copy_yesterday_error`
- `random_forest_error`

The model file stores the trained Random Forest plus the feature columns and metadata needed to understand how it was trained.
