
# Project 2.1 Envitron
# Team IT2B IT Program NHL Stenden

# Envitron Prototype 2 — Ridge Regression Energy Forecasting

Prototype 2 investigates a **lightweight and explainable forecasting approach** 
for predicting short-term building energy consumption for Envitron's Energy Management System (EMS).

The prototype uses **Ridge Regression** and compares it with simple forecasting baselines.

The current experiment predicts:

- **Target:** `consumption_w`
- **Forecast horizon:** 15 minutes ahead
- **Building:** One office building
- **Data resolution:** 15 minutes
- **Evaluation:** Chronological train/test split
- **Main metrics:** MAE and RMSE

> **Important:** The 15-minute forecast horizon is currently an experimental assumption and has not yet been confirmed as Envitron's operational requirement.

---

## 1. Purpose

The purpose of Prototype 2 is to investigate whether a relatively simple forecasting model 
can provide useful prediction accuracy while remaining:

- lightweight;
- explainable;
- computationally efficient; and
- suitable for possible later Docker deployment.

Prototype 2 does **not** attempt to prove that Ridge Regression is better than Random Forest.

Instead, it investigates the trade-off between forecasting accuracy, explainability and computational requirements.

---

## 2. Forecasting Approaches

Prototype 2 evaluates four forecasting approaches.

### Persistence

Persistence uses the most recent consumption measurement to predict the next 15-minute interval.

Conceptually:

`Forecast(t + 15 min) = Consumption(t)`

This provides a simple short-term baseline.

### Copy-Yesterday

Copy-Yesterday uses consumption from the same time on the previous day.

Conceptually:

`Forecast(t) = Consumption(t - 24 hours)`

This provides a daily-pattern baseline.

### Ridge Regression

Ridge Regression is a regularised linear regression model.

It uses historical consumption and time-related features to predict future building consumption.

Ridge was selected because it is:

- relatively simple;
- computationally lightweight;
- regularised to reduce unstable coefficients caused by correlated features; and
- more explainable than many more complex machine-learning models.

### Ridge Regression + Weather

This model uses the same Ridge Regression approach but adds weather information.

The purpose is to investigate whether weather variables provide additional predictive information.

---

## 3. Project Structure

```text
prototype-2/
│
├── data/
│   ├── electricity.csv
│   ├── weather.csv
│   └── buildings.csv
│
├── prototype2-ridge.py
├── requirements.txt
├── README.md
│
├── output/
│   ├── predictions.csv
│   ├── metrics.csv
│   ├── coefficients.csv
│   ├── data_quality.csv
│   ├── forecast_comparison.png
│   ├── error_comparison.png
│   ├── model_mae.png
│   └── ridge_coefficients.png
│
└── models/
    ├── ridge.joblib
    └── ridge_weather.joblib
```

---

## 4. Dataset

The prototype uses anonymised data provided by Envitron.

The main datasets are:

### `electricity.csv`

Contains building energy measurements at approximately 15-minute intervals.

The prediction target used by Prototype 2 is:

`consumption_w`

### `weather.csv`

Contains historical weather information used in the Ridge + Weather experiment.

### `buildings.csv`

Contains information describing the buildings in the dataset.

The current experiment uses **one office building**.

---

## 5. Data Quality

Before model training, the prototype performs data-quality checks.

These checks help identify potential problems that may influence forecasting performance.

One important finding was:

- **736 negative `consumption_w` observations**
- approximately **1.78%** of the observations examined

These values were preserved because their meaning has not yet been confirmed.

### Question for Envitron

**What do negative `consumption_w` values represent in the provided dataset?**

They should not automatically be treated as errors until their meaning is understood.

---

## 6. Feature Engineering

Prototype 2 creates historical consumption features using lagged observations.

The Ridge model includes:

- `lag_15m`
- `lag_30m`
- `lag_1h`
- `lag_24h`
- `lag_7d`

These features allow the model to use recent, daily and weekly historical consumption patterns.

Time-related features are also included to represent recurring patterns such as:

- time of day;
- day of week; and
- weekend behaviour.

Cyclical time encoding is used where appropriate so that recurring time patterns are represented continuously.

The Ridge + Weather configuration additionally includes weather variables.

---

## 7. Chronological Train/Test Split

Because this is a **time-series forecasting problem**, the data is not randomly shuffled before evaluation.

Instead, Prototype 2 uses a chronological 80/20 split.

```text
PAST                                          FUTURE
───────────────────────────────────────────────────>

Training data                              Test data
31,896 observations                        7,974 observations
```

Earlier observations are used for training and later observations are used for testing.

This better represents the real forecasting situation, where a model learns from historical information and predicts future observations.

---

## 8. Leakage Prevention

Prototype 2 uses a scikit-learn `Pipeline` for preprocessing and Ridge Regression.

The pipeline includes preprocessing such as `StandardScaler`.

The pipeline is fitted using the **training data only**.

This ensures that scaling parameters are learned from the training period rather than from the complete dataset.

This helps prevent information from the test or future period from leaking into the model-training process.

Conceptually, feature scaling brings numerical features to comparable scales so that Ridge Regression can apply regularisation appropriately.

---

## 9. Evaluation Metrics

The primary evaluation metrics are:

### Mean Absolute Error — MAE

MAE measures the average absolute difference between the actual and predicted consumption.

Lower MAE indicates lower average forecasting error.

### Root Mean Square Error — RMSE

RMSE gives greater influence to larger forecasting errors because the errors are squared before averaging.

Lower RMSE indicates better forecasting performance.

Both metrics are reported because they describe different aspects of forecast error.

---

## 10. Prototype 2 Results

The measured results for the selected office building and chronological test period are:

| Method | MAE | RMSE |
|---|---:|---:|
| Persistence | **1.76 kW** | 4.23 kW |
| Copy-Yesterday | 4.21 kW | 9.18 kW |
| Ridge Regression | 1.82 kW | 4.00 kW |
| Ridge + Weather | 1.79 kW | **3.98 kW** |

### Initial Findings

**Persistence achieved the lowest MAE.**

**Ridge + Weather achieved the lowest RMSE.**

Therefore, no single method performed best according to every evaluation metric.

The strong Persistence result suggests that the most recent consumption measurement contains useful predictive information for this short 15-minute forecasting horizon.

---

## 11. Weather Contribution

Ridge Regression without weather achieved approximately:

- MAE: **1.82 kW**
- RMSE: **4.00 kW**

Ridge Regression with Weather achieved approximately:

- MAE: **1.79 kW**
- RMSE: **3.98 kW**

Adding weather reduced Ridge MAE by approximately **1.58%**.

The initial experiment therefore suggests that weather provides some additional predictive information, but the improvement is modest for this building and evaluation period.

### Important Limitation

The current experiment uses **historical observed weather**.

In a real forecasting system, actual future weather would not yet be known.

An operational EMS would therefore need to use weather forecasts that are available at prediction time.

The current weather experiment should be interpreted as an investigation of the predictive value of weather variables rather than a complete operational weather-forecasting test.

---

## 12. Model Explainability

One advantage of Ridge Regression is that its coefficients can be inspected.

In the current experiment, `lag_15m` has the largest absolute standardised Ridge coefficient.

This supports the observation that recent consumption contains strong predictive information for the short 15-minute forecast horizon.

However, Ridge coefficients represent relationships learned by the model.

They should **not** be interpreted as evidence that a feature causes a change in energy consumption.

---

## 13. Controlled Random Forest Comparison

The original Prototype 1 and Prototype 2 cannot be compared directly because they predict different targets.


Original Prototype 1:

`Random Forest → production_w`


Prototype 2:

`Ridge Regression → consumption_w`

Therefore, their original MAE and RMSE values should not be directly compared.

A separate **controlled comparison** was created.

The Random Forest methodology was adapted to use:

- the same target: `consumption_w`;
- the same selected building;
- an aligned chronological evaluation period; and
- the same MAE and RMSE evaluation metrics.

The original Prototype 1 was **not modified**.

The adapted Random Forest produced:

| Method | MAE | RMSE |
|---|---:|---:|
| Adapted Random Forest | 2.10 kW | 4.83 kW |

Under these controlled experimental conditions, the adapted Random Forest did not outperform the simpler approaches.

This does **not** demonstrate that Random Forest is generally worse than Ridge Regression.

It only shows that the additional model complexity did not provide lower forecasting error under the conditions of this initial experiment.

---

## 14. Computational Requirements

The project also investigates whether forecasting approaches are lightweight enough for practical deployment.

Measured results from the current test environment include:

| Model | Training Time | Saved Model Size |
|---|---:|---:|
| Ridge + Weather | ~0.065 s | ~4.1 KB |
| Adapted Random Forest | ~21 s | ~581 MB |

Ridge with Weather required substantially less training time and storage in the current experiment.

However, these measurements are from the **current development/test environment**.

They are **not Raspberry Pi benchmarks**.

Hardware-specific benchmarking should be performed later if Raspberry Pi deployment remains relevant.

---

## 15. Output Files

After execution, Prototype 2 generates several output files.

### `predictions.csv`

Contains actual and predicted values for the evaluated forecasting approaches.

This can be used for:

- forecast inspection;
- error analysis;
- plotting; and
- comparison between methods.

### `metrics.csv`

Contains measured forecasting metrics such as MAE and RMSE.

### `coefficients.csv`

Contains Ridge Regression coefficients for model interpretation.

### `data_quality.csv`

Contains results from the data-quality analysis.

### `forecast_comparison.png`

Visual comparison between actual consumption and forecast values.

### `error_comparison.png`

Visualises forecasting errors.

### `model_mae.png`

Compares model MAE values.

### `ridge_coefficients.png`

Visualises Ridge Regression coefficients.

---

## 16. Saved Models

The trained Ridge models are stored in the `models/` directory.

### `ridge.joblib`

Saved Ridge Regression pipeline without weather features.

### `ridge_weather.joblib`

Saved Ridge Regression pipeline including weather features.

Saving the complete pipeline allows the preprocessing and trained Ridge model to remain together.

---

## 17. Installation

Python 3 is required.

Install the required packages using:

```bash
pip install -r requirements.txt
```

The required Python packages are defined in `requirements.txt`.

---

## 18. Running Prototype 2

Place the Envitron dataset files in the `data/` directory:

```text
data/
├── electricity.csv
├── weather.csv
└── buildings.csv
```

Then run:

```bash
python prototype2-ridge.py
```

After successful execution, the generated results are stored in:

```text
output/
```

and the trained Ridge models are stored in:

```text
models/
```

---

## 19. Current Limitations

The current Prototype 2 experiment has several important limitations.

### One Building

The measured results currently represent only one office building.

The findings should not yet be generalised to other buildings.

### One Chronological Holdout

The current evaluation uses one chronological holdout period.

Additional time periods should be tested.

### 15-Minute Forecast Horizon

The current experiment assumes a 15-minute-ahead forecast.

This has not yet been confirmed as Envitron's operational forecasting requirement.

### Observed Weather

The current weather experiment uses historical observed weather rather than operational weather forecasts.

### Negative Consumption

The meaning of negative `consumption_w` values has not yet been confirmed.

### No Final Model Selection

The current experiment provides initial evidence only.

**No final forecasting model has been selected.**

---

## 20. Planned Next Steps

Future research will focus on:

1. testing additional buildings;
2. testing additional chronological periods;
3. analysing periods with large forecasting errors;
4. comparing regular and irregular consumption patterns;
5. investigating how much historical data is required;
6. confirming the operational forecast horizon with Envitron;
7. evaluating realistic forecast-weather inputs if weather remains useful; and
8. later testing Docker deployment and Raspberry Pi performance.

These items are **planned future work and are not findings from the current experiment**.

---

## 21. Current Research Position

The current experiment indicates that simple forecasting approaches deserve further investigation.

For the selected building and evaluation period:

- Persistence achieved the lowest MAE.
- Ridge + Weather achieved the lowest RMSE.
- Weather produced a modest improvement to Ridge.
- The adapted Random Forest did not outperform the simpler approaches in the controlled experiment.
- Ridge remained substantially lighter than the adapted Random Forest in the current test environment.

However, the evidence is currently limited to one building and one chronological holdout period.

The next stage should determine whether these findings remain consistent across additional buildings, time periods and operationally relevant forecast horizons before a final forecasting approach is selected.

---

## Project Context

This prototype was developed as part of Team IT2B - NHL Stenden Information Technology student project:

**Predicting Energy Demand — One Building at a Time**

Client: **Envitron**

The prototype is intended for **research and experimental evaluation** 
and is not currently a production forecasting system.