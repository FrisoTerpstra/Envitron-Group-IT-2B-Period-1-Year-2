# Setup and Run Guide: Prototype 3 HistGradientBoostingRegressor 36h

## 1. Put the files in the right folders

Inside your repository, make sure your personal folder looks like this:

```text
Envitron-Group-IT-2B-Period-1-Year-2/
└── Kristian/
    ├── data/
    │   ├── electricity.csv
    │   ├── weather.csv
    │   └── buildings.csv
    ├── models/
    ├── output/
    ├── prototype3_histgradientboosting_36h.py
    └── prototype3_histgradientboosting_36h_setup.md
```

Copy both generated files into your `Kristian` folder, because this prototype is configured to use `Kristian/data`, `Kristian/output`, and `Kristian/models`.

## 2. Open the project in PyCharm

1. Open PyCharm.
2. Open your repository folder.
3. Open the `Kristian` folder or keep it visible inside the repo tree.
4. Make sure PyCharm uses a Python interpreter that supports scikit-learn, pandas, matplotlib, numpy, and joblib.

A Python 3.11 virtual environment is a safe choice for this prototype.

## 3. Install the required packages

Open the PyCharm terminal and run:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

If your system uses `python3` and `pip3`, then use:

```bash
python3 -m pip install pandas numpy matplotlib scikit-learn joblib
```

## 4. Check the building ID inside the script

Open the script and look for this line:

```python
BUILDING_ID = "6a1bf87e-d798-4e93-a648-a8e948fc24c2"
```

You can keep that building first, because your team already used it in the Ridge 36h experiment. If you want to test another building, replace the ID with one from `Kristian/data/buildings.csv`.

## 5. Run the script

You can run it in two ways.

### Option A: Run button in PyCharm

1. Right-click the file `prototype3_histgradientboosting_36h.py`.
2. Click **Run 'prototype3_histgradientboosting_36h'**.

### Option B: Terminal

Go into the `Kristian` folder and run:

```bash
python prototype3_histgradientboosting_36h.py
```

Or if needed:

```bash
python3 prototype3_histgradientboosting_36h.py
```

## 6. What the script does

The script will:

1. Load `electricity.csv`, `weather.csv`, and `buildings.csv` from `Kristian/data/`.
2. Select one building.
3. Remove negative `consumption_w` rows from modelling, while exporting them separately.
4. Build lag features and time features.
5. Create 144 future targets for a 36-hour forecast.
6. Compare these models:
   - Persistence
   - Copy-Yesterday
   - HGB
   - HGB + Weather (if weather columns are available)
7. Run:
   - 80/20 validation
   - 90/10 validation
   - 3-fold expanding walk-forward validation
8. Save metrics, predictions, charts, and trained models.

## 7. Where to find the outputs

After the run, look in:

```text
Kristian/output/prototype3_hgb_36h/
```

You should see files like:

- `data_quality.csv`
- `excluded_negative_consumption.csv`
- `metrics_by_horizon.csv`
- `metrics_summary.csv`
- `predictions_selected_horizons.csv`
- `model_validation_mae.png`
- `model_validation_rmse.png`
- `horizon_mae_90_10.png`
- `horizon_rmse_90_10.png`
- `actual_vs_forecast_36h.png`
- `run_summary.json`

And trained model files in:

```text
Kristian/models/prototype3_hgb_36h/
```

## 8. How to read the results

### `metrics_summary.csv`
This gives average MAE and RMSE across the full 36-hour horizon for each validation and each model.

### `metrics_by_horizon.csv`
This shows how error changes over the forecast horizon. Use this to see whether the model performs well at short horizons but weakens later.

### `predictions_selected_horizons.csv`
This exports actual and predicted values for selected forecast steps so you can inspect raw forecast behaviour.

### Charts
- `model_validation_mae.png` and `model_validation_rmse.png` compare models overall.
- `horizon_mae_90_10.png` and `horizon_rmse_90_10.png` show how errors grow with forecast distance.
- `actual_vs_forecast_36h.png` gives a visual comparison at the full 36-hour horizon.

## 9. If something goes wrong

### Error: file not found
Make sure the CSV files are in `Kristian/data/` and the names match exactly:
- `electricity.csv`
- `weather.csv`
- `buildings.csv`

### Error: module not found
Install the missing package with pip.

### Error: not enough complete forecast origins
This means the selected building does not have enough full 36-hour windows after preprocessing. Try another building from `Kristian/data/buildings.csv`.

### The script runs but results look strange
Check:
- whether the chosen building has enough history,
- whether negative values were heavily excluded,
- whether weather fields merged correctly,
- and whether the building is operationally unusual.

## 10. Good next tests

After your first successful run, test these variations:

1. Run the same script with 2 or 3 different building IDs.
2. Compare HGB vs HGB + Weather.
3. Compare your summary metrics against the Ridge 36h metrics from your teammates.
4. Look at the horizon charts to see if HGB performs better at longer horizons.

## 11. What to say in Jira / weekly report

A good short description of your work would be:

> Developed a third forecasting prototype using HistGradientBoostingRegressor for 36-hour building consumption forecasting. Implemented leakage-aware baselines, time-series validation, data-quality exports, model saving, and output visualizations for comparison with existing Ridge-based experiments.

