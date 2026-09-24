import os
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = PROJECT_DIR / "models"
MPLCONFIG_DIR = OUTPUT_DIR / "matplotlib_cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt

ELECTRICITY_FILE = "electricity.csv"
WEATHER_FILE = "weather.csv"
BUILDINGS_FILE = "buildings.csv"

TIMESTAMP_COLUMN = "timestamp_utc"
ID_COLUMN = "building_id"
TARGET_COLUMN = "production_w"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.80


def find_csv(filename: str) -> Path:
    """Find CSVs either in data/ or in the project root."""
    candidates = [DATA_DIR / filename, PROJECT_DIR / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Put it in {DATA_DIR} or next to forecast.py."
    )


def print_dataset_report(electricity: pd.DataFrame, weather: pd.DataFrame, buildings: pd.DataFrame) -> None:
    print("\nDATASET INSPECTION")
    print("==================")
    for name, df in [
        ("electricity.csv", electricity),
        ("weather.csv", weather),
        ("buildings.csv", buildings),
    ]:
        print(f"\n{name}")
        print(f"Rows: {len(df):,}")
        print("Columns:")
        for column in df.columns:
            print(f"  - {column}: {df[column].dtype}, missing={df[column].isna().sum():,}")

    datetime_columns = [
        column
        for column in set(electricity.columns).union(weather.columns).union(buildings.columns)
        if "timestamp" in column.lower() or "date" in column.lower() or "time" in column.lower()
    ]
    power_candidates = [
        column
        for column in electricity.columns
        if any(term in column.lower() for term in ["power", "_w", "production", "consumption", "delivery"])
    ]

    print("\nLikely date/time columns:", ", ".join(sorted(datetime_columns)))
    print("Possible power/target columns:", ", ".join(power_candidates))
    print(f"Selected timestamp column: {TIMESTAMP_COLUMN}")
    print(f"Selected prediction target: {TARGET_COLUMN}")


def load_data() -> pd.DataFrame:
    electricity_path = find_csv(ELECTRICITY_FILE)
    weather_path = find_csv(WEATHER_FILE)
    buildings_path = find_csv(BUILDINGS_FILE)

    electricity = pd.read_csv(electricity_path)
    weather = pd.read_csv(weather_path)
    buildings = pd.read_csv(buildings_path)

    if TIMESTAMP_COLUMN not in electricity.columns:
        raise ValueError(f"Expected {TIMESTAMP_COLUMN!r} in {electricity_path}")
    if TARGET_COLUMN not in electricity.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} in {electricity_path}")

    electricity[TIMESTAMP_COLUMN] = pd.to_datetime(electricity[TIMESTAMP_COLUMN], utc=True)
    weather[TIMESTAMP_COLUMN] = pd.to_datetime(weather[TIMESTAMP_COLUMN], utc=True)
    buildings["first_timestamp_utc"] = pd.to_datetime(buildings["first_timestamp_utc"], utc=True)
    buildings["last_timestamp_utc"] = pd.to_datetime(buildings["last_timestamp_utc"], utc=True)

    print_dataset_report(electricity, weather, buildings)

    data = electricity.merge(
        weather,
        on=[ID_COLUMN, TIMESTAMP_COLUMN],
        how="left",
        validate="one_to_one",
    )
    data = data.merge(
        buildings,
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )
    return data


def infer_frequency(data: pd.DataFrame) -> pd.Timedelta:
    diffs = (
        data[[ID_COLUMN, TIMESTAMP_COLUMN]]
        .drop_duplicates()
        .sort_values([ID_COLUMN, TIMESTAMP_COLUMN])
        .groupby(ID_COLUMN)[TIMESTAMP_COLUMN]
        .diff()
        .dropna()
    )
    if diffs.empty:
        raise ValueError("Could not infer measurement frequency because no timestamp differences exist.")

    mode_frequency = diffs.value_counts().idxmax()
    print(f"\nApparent measurement frequency: {mode_frequency}")
    print("Most common timestamp gaps:")
    for gap, count in diffs.value_counts().head(8).items():
        print(f"  {gap}: {count:,} occurrences")
    return mode_frequency


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    ts = data[TIMESTAMP_COLUMN]
    data["hour"] = ts.dt.hour
    data["minute"] = ts.dt.minute
    data["day_of_week"] = ts.dt.dayofweek
    data["day_of_year"] = ts.dt.dayofyear
    data["month"] = ts.dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * (data["hour"] + data["minute"] / 60) / 24)
    data["hour_cos"] = np.cos(2 * np.pi * (data["hour"] + data["minute"] / 60) / 24)
    data["day_of_year_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 366)
    data["day_of_year_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 366)
    return data


def add_timestamp_lag(data: pd.DataFrame, lag: pd.Timedelta, output_column: str) -> pd.DataFrame:
    lag_lookup = data[[ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN]].rename(
        columns={
            TIMESTAMP_COLUMN: "lag_timestamp",
            TARGET_COLUMN: output_column,
        }
    )
    data["lag_timestamp"] = data[TIMESTAMP_COLUMN] - lag
    data = data.merge(
        lag_lookup,
        on=[ID_COLUMN, "lag_timestamp"],
        how="left",
        validate="many_to_one",
    )
    return data.drop(columns=["lag_timestamp"])


def engineer_features(data: pd.DataFrame, frequency: pd.Timedelta) -> tuple[pd.DataFrame, list[str]]:
    data = data.sort_values([ID_COLUMN, TIMESTAMP_COLUMN]).copy()
    data = add_time_features(data)

    data["production_previous_measurement_w"] = data.groupby(ID_COLUMN)[TARGET_COLUMN].shift(1)
    data = add_timestamp_lag(data, pd.Timedelta(hours=1), "production_lag_1h_w")
    data = add_timestamp_lag(data, pd.Timedelta(days=1), "production_lag_24h_w")

    weather_features = [
        "temperature_c",
        "relative_humidity_pct",
        "precipitation_mm",
        "cloud_cover_pct",
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
        "wind_speed_ms",
    ]
    building_features = ["pv_kwp", "battery_kwh", "battery_kw", "production_available"]
    time_features = [
        "hour",
        "minute",
        "day_of_week",
        "day_of_year",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    lag_features = [
        "production_previous_measurement_w",
        "production_lag_1h_w",
        "production_lag_24h_w",
    ]

    available_features = [
        column
        for column in weather_features + building_features + time_features + lag_features
        if column in data.columns
    ]

    data["production_available"] = data["production_available"].astype(int)
    data = pd.get_dummies(data, columns=[ID_COLUMN], prefix="building", dtype=int)
    building_id_features = [column for column in data.columns if column.startswith("building_")]

    feature_columns = available_features + building_id_features
    print("\nSelected model features:")
    for column in feature_columns:
        print(f"  - {column}")
    print(
        "\nLag features are based only on earlier timestamps. "
        f"The dominant interval is {frequency}, and copy-yesterday uses timestamp minus one day."
    )
    return data, feature_columns


def chronological_split(data: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series, pd.Series]:
    unique_timestamps = np.array(sorted(data[TIMESTAMP_COLUMN].dropna().unique()))
    split_index = int(len(unique_timestamps) * TRAIN_FRACTION)
    if split_index <= 0 or split_index >= len(unique_timestamps):
        raise ValueError("Not enough timestamps for an 80/20 chronological train/test split.")

    split_timestamp = pd.Timestamp(unique_timestamps[split_index])
    train_mask = data[TIMESTAMP_COLUMN] < split_timestamp
    test_mask = data[TIMESTAMP_COLUMN] >= split_timestamp

    print(f"\nChronological split timestamp: {split_timestamp}")
    print(f"Training rows before dropping missing model fields: {train_mask.sum():,}")
    print(f"Testing rows before dropping missing model fields: {test_mask.sum():,}")
    return split_timestamp, train_mask, test_mask


def evaluate(y_true: pd.Series, predictions: pd.Series) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return mae, rmse


def save_charts(results: pd.DataFrame, target_unit: str = "W") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    chart_data = results.copy()
    best_building = chart_data[ID_COLUMN].value_counts().idxmax()
    chart_data = chart_data[chart_data[ID_COLUMN] == best_building].sort_values(TIMESTAMP_COLUMN)

    if chart_data.empty:
        print("No chart created because there are no comparable predictions.")
        return

    start = chart_data[TIMESTAMP_COLUMN].min()
    one_week = start + pd.Timedelta(days=7)
    representative = chart_data[chart_data[TIMESTAMP_COLUMN] < one_week]
    if len(representative) < 10:
        representative = chart_data.head(500)

    plot_forecast(
        representative,
        OUTPUT_DIR / "forecast_chart.png",
        f"Actual vs Forecasted Production ({best_building}, representative period)",
        target_unit,
    )

    full = chart_data
    if len(full) > 2000:
        full = full.iloc[np.linspace(0, len(full) - 1, 2000).astype(int)]
    plot_forecast(
        full,
        OUTPUT_DIR / "forecast_chart_full_sampled.png",
        f"Actual vs Forecasted Production ({best_building}, sampled full test period)",
        target_unit,
    )


def plot_forecast(data: pd.DataFrame, output_path: Path, title: str, target_unit: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data[TIMESTAMP_COLUMN], data["actual_power"], label="Actual power", linewidth=1.5)
    ax.plot(
        data[TIMESTAMP_COLUMN],
        data["copy_yesterday_prediction"],
        label="Copy-yesterday prediction",
        linewidth=1.2,
    )
    ax.plot(
        data[TIMESTAMP_COLUMN],
        data["random_forest_prediction"],
        label="Random Forest prediction",
        linewidth=1.2,
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Power ({target_unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    data = load_data()
    frequency = infer_frequency(data)
    data, feature_columns = engineer_features(data, frequency)
    split_timestamp, train_mask, test_mask = chronological_split(data)

    required_columns = feature_columns + [TARGET_COLUMN, "production_lag_24h_w"]
    train = data.loc[train_mask].dropna(subset=required_columns).copy()
    test = data.loc[test_mask].dropna(subset=required_columns).copy()

    X_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    X_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]

    print(f"Training rows used: {len(train):,}")
    print(f"Testing rows used before fair baseline comparison: {len(test):,}")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    try:
        model.fit(X_train, y_train)
    except PermissionError:
        print(
            "Parallel training with n_jobs=-1 was blocked by the local environment. "
            "Retrying with n_jobs=1."
        )
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
    rf_predictions = model.predict(X_test)

    results = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: test[TIMESTAMP_COLUMN],
            ID_COLUMN: test.filter(like="building_").idxmax(axis=1).str.replace("building_", "", regex=False),
            "actual_power": y_test,
            "copy_yesterday_prediction": test["production_lag_24h_w"],
            "random_forest_prediction": rf_predictions,
        }
    )
    results["copy_yesterday_error"] = (
        results["actual_power"] - results["copy_yesterday_prediction"]
    ).abs()
    results["random_forest_error"] = (
        results["actual_power"] - results["random_forest_prediction"]
    ).abs()
    results = results.dropna().sort_values([TIMESTAMP_COLUMN, ID_COLUMN])

    copy_mae, copy_rmse = evaluate(results["actual_power"], results["copy_yesterday_prediction"])
    rf_mae, rf_rmse = evaluate(results["actual_power"], results["random_forest_prediction"])

    print("\nEVALUATION ON SAME TEST TIMESTAMPS")
    print("==================================")
    print(f"Comparable test rows: {len(results):,}")
    print("Copy-yesterday")
    print(f"  MAE:  {copy_mae:,.2f} W")
    print(f"  RMSE: {copy_rmse:,.2f} W")
    print("Random Forest")
    print(f"  MAE:  {rf_mae:,.2f} W")
    print(f"  RMSE: {rf_rmse:,.2f} W")
    if rf_mae < copy_mae and rf_rmse < copy_rmse:
        print("Result: Random Forest performed better than copy-yesterday on both metrics.")
    else:
        print("Result: Random Forest did not beat copy-yesterday on both metrics.")

    predictions_path = OUTPUT_DIR / "predictions.csv"
    results.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions: {predictions_path}")

    model_path = MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "timestamp_column": TIMESTAMP_COLUMN,
            "split_timestamp": split_timestamp,
            "measurement_frequency": frequency,
        },
        model_path,
    )
    print(f"Saved model: {model_path}")

    save_charts(results, target_unit="W")


if __name__ == "__main__":
    main()
