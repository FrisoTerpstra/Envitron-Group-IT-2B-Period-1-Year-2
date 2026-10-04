# Electricity Forecast

This is a small Python-only script for predicting hourly electricity consumption up to 36 hours ahead.

It trains on the first 3 calendar months of data for one building, predicts the next 36 hourly values, compares them with the actual data, and writes:

- `results/forecast_results.csv`
- `results/forecast_chart.svg`

The input CSV files are read from the project `data` folder:

- `data/electricity.csv`
- `data/weather.csv`
- `data/buildings.csv`

The script creates the `data` and `results` folders automatically when it runs.
If the CSV files are missing, it will tell you which files to place in `data`.

## Run

Open this folder in PyCharm and run:

```powershell
py forecast_consumption.py
```

To see the available building IDs:

```powershell
py forecast_consumption.py --list-buildings
```

To run a specific building:

```powershell
py forecast_consumption.py --building-id 305a2fe5-a67a-4268-9b6a-e360b36f7961
```

## Method

The model is ordinary least squares regression. It uses:

- hour-of-day pattern
- day-of-week pattern
- weather values
- same hour yesterday
- same hour last week

No UI and no external Python packages are required.
