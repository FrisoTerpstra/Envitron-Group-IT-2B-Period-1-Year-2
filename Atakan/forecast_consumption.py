"""Electricity forecast evaluation using only Python's standard library.

Run: python consumption_forecast.py --building-id YOUR_BUILDING_ID
Place electricity.csv, weather.csv and buildings.csv in data/ beside this script.
Uses all available building history; missing target slots are not scored.
Retains the supplied linear regression (tiny numerical regularization).
No test-set hyperparameter tuning. Negative consumption values are retained.
Outputs all 144-step windows, one prediction per test timestamp, metrics and SVG.
"""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_ELECTRICITY = DEFAULT_DATA_DIR / "electricity.csv"
DEFAULT_WEATHER = DEFAULT_DATA_DIR / "weather.csv"
DEFAULT_BUILDINGS = DEFAULT_DATA_DIR / "buildings.csv"

WEATHER_COLUMNS = [
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "wind_speed_ms",
]


STEP = timedelta(minutes=15)
HORIZON_STEPS = 144


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def quarter_bucket(value):
    timestamp = parse_time(value)
    return timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)


def hour_bucket(value):
    return parse_time(value).replace(minute=0, second=0, microsecond=0)


def safe_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def prepare_folders_and_check_files(electricity_path, weather_path, buildings_path, output_dir):
    for folder in {
        electricity_path.parent,
        weather_path.parent,
        buildings_path.parent,
        output_dir,
    }:
        folder.mkdir(parents=True, exist_ok=True)

    missing = [
        path
        for path in [electricity_path, weather_path, buildings_path]
        if not path.exists()
    ]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Created the folders, but these CSV files are missing:\n"
            f"{missing_list}\n\n"
            "Put the CSV files in the data folder and run the script again."
        )


def choose_building(buildings_path, requested_building_id):
    buildings = []
    with buildings_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            buildings.append(row)

    if requested_building_id:
        for row in buildings:
            if row["building_id"] == requested_building_id:
                return row
        raise ValueError(f"Building id not found in buildings.csv: {requested_building_id}")

    if not buildings:
        raise ValueError("No buildings found.")
    return buildings[0]


def list_buildings(buildings_path):
    with buildings_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        print("Available buildings:")
        for row in reader:
            print(f"{row['building_id']}  {row.get('description', '')}")


def load_quarter_hour_target(path, building_id, target_column):
    totals = defaultdict(float)
    counts = defaultdict(int)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["building_id"] != building_id:
                continue
            value = safe_float(row.get(target_column))
            if value is None:
                continue
            hour = quarter_bucket(row["timestamp_utc"])
            totals[hour] += value
            counts[hour] += 1

    return {hour: totals[hour] / counts[hour] for hour in totals}


def load_hourly_weather(path, building_id):
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["building_id"] != building_id:
                continue
            hour = hour_bucket(row["timestamp_utc"])
            values = {column: safe_float(row.get(column)) for column in WEATHER_COLUMNS}
            if any(value is None for value in values.values()):
                continue
            for column, value in values.items():
                totals[hour][column] += value
            counts[hour] += 1

    weather = {}
    for hour, count in counts.items():
        weather[hour] = {
            column: totals[hour][column] / count for column in WEATHER_COLUMNS
        }
    return weather


def lag_or_typical(history, timestamp, hours_back, typical_by_hour):
    value = history.get(timestamp - hours(hours_back))
    if value is not None:
        return value
    return typical_by_hour.get(timestamp.hour * 4 + timestamp.minute // 15, 0.0)


def hours(count):
    from datetime import timedelta

    return timedelta(hours=count)


def raw_features(timestamp, weather_row, lag_24, lag_168):
    hour_of_day = timestamp.hour + timestamp.minute / 60
    hour_of_week = timestamp.weekday() * 24 + hour_of_day
    return [
        1.0,
        math.sin(2 * math.pi * hour_of_day / 24),
        math.cos(2 * math.pi * hour_of_day / 24),
        math.sin(2 * math.pi * hour_of_week / (7 * 24)),
        math.cos(2 * math.pi * hour_of_week / (7 * 24)),
        weather_row["temperature_c"],
        weather_row["relative_humidity_pct"],
        weather_row["precipitation_mm"],
        weather_row["cloud_cover_pct"],
        weather_row["shortwave_radiation_wm2"],
        weather_row["wind_speed_ms"],
        lag_24,
        lag_168,
    ]


def normalize_rows(rows, means=None, scales=None):
    if means is None:
        means = [0.0] * len(rows[0])
        scales = [1.0] * len(rows[0])
        for col in range(1, len(rows[0])):
            values = [row[col] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[col] = mean
            scales[col] = math.sqrt(variance) or 1.0

    normalized = []
    for row in rows:
        normalized.append(
            [
                row[0],
                *[
                    (row[col] - means[col]) / scales[col]
                    for col in range(1, len(row))
                ],
            ]
        )
    return normalized, means, scales


def solve_linear_system(matrix, values):
    n = len(values)
    a = [matrix[row][:] + [values[row]] for row in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("Cannot solve regression because the feature matrix is singular.")
        a[col], a[pivot] = a[pivot], a[col]

        pivot_value = a[col][col]
        for item in range(col, n + 1):
            a[col][item] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for item in range(col, n + 1):
                a[row][item] -= factor * a[col][item]

    return [a[row][n] for row in range(n)]


def fit_ordinary_least_squares(rows, targets):
    normalized, means, scales = normalize_rows(rows)
    column_count = len(normalized[0])
    xtx = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
    xty = [0.0 for _ in range(column_count)]

    for row, target in zip(normalized, targets):
        for i in range(column_count):
            xty[i] += row[i] * target
            for j in range(column_count):
                xtx[i][j] += row[i] * row[j]

    for i in range(1, column_count):
        xtx[i][i] += 1e-8

    coefficients = solve_linear_system(xtx, xty)
    return coefficients, means, scales


def predict(raw_row, coefficients, means, scales):
    normalized, _, _ = normalize_rows([raw_row], means, scales)
    return sum(value * coefficient for value, coefficient in zip(normalized[0], coefficients))


def make_training_data(target, weather, split_time):
    timestamps = sorted(t for t in target if t < split_time and t in weather)
    if not timestamps:
        raise ValueError("No joined target/weather rows found before the split date.")

    typical_by_hour = defaultdict(list)
    for timestamp in timestamps:
        typical_by_hour[timestamp.hour * 4 + timestamp.minute // 15].append(target[timestamp])
    typical_by_hour = {
        hour: sum(values) / len(values) for hour, values in typical_by_hour.items()
    }

    rows = []
    y = []
    for timestamp in timestamps:
        lag_24 = target.get(timestamp - hours(24))
        lag_168 = target.get(timestamp - hours(168))
        if lag_24 is None or lag_168 is None:
            continue
        rows.append(raw_features(timestamp, weather[timestamp], lag_24, lag_168))
        y.append(target[timestamp])

    if len(rows) < 50:
        raise ValueError("Not enough training rows after building lag features.")
    return rows, y, typical_by_hour


def select_window(target):
    """Keep all observations and a regular grid without inventing target values."""
    if not target:
        raise ValueError("No valid consumption readings for this building.")
    start, end = min(target), max(target)
    times = [start + i * STEP for i in range((end - start) // STEP + 1)]
    return dict(target), times


def forecast_window(target, weather, origin, coefficients, means, scales,
                    typical_by_hour, weather_mode):
    # Only observations available before this origin enter the forecast history.
    history = {time: value for time, value in target.items() if time < origin}
    known_weather_times = [time for time in weather if time + hours(1) <= origin]
    if not known_weather_times:
        raise ValueError("No completed weather hour available before forecast origin.")
    last_weather = weather[max(known_weather_times)]
    rows = []
    for step in range(HORIZON_STEPS):
        timestamp = origin + step * STEP
        if weather_mode == "observed":
            key = timestamp.replace(minute=0, second=0, microsecond=0)
            if key not in weather:
                raise ValueError(f"Missing observed weather for {key}.")
            weather_row = weather[key]
        else:
            # No weather forecast archive was supplied: persist last known weather.
            weather_row = last_weather
        lag_24 = lag_or_typical(history, timestamp, 24, typical_by_hour)
        lag_168 = lag_or_typical(history, timestamp, 168, typical_by_hour)
        predicted = predict(raw_features(timestamp, weather_row, lag_24, lag_168),
                            coefficients, means, scales)
        history[timestamp] = predicted  # Recursive lag after the first 24 hours.
        actual = target.get(timestamp)
        rows.append({
            "forecast_origin_utc": origin.isoformat(),
            "lead_step": step + 1,
            "timestamp_utc": timestamp.isoformat(),
            "predicted_w": predicted,
            "actual_w": actual,
            "error_w": predicted - actual if actual is not None else None,
            "abs_error_w": abs(predicted - actual) if actual is not None else None,
        })
    return rows


def metrics(rows, normalization_range=None):
    """MAPE excludes exact zero actuals; normalized errors use test actual range."""
    rows = [row for row in rows if row["actual_w"] is not None]
    if not rows:
        raise ValueError("Cannot calculate metrics without observed test targets.")
    mae = sum(abs(row["error_w"]) for row in rows) / len(rows)
    rmse = math.sqrt(sum(row["error_w"] ** 2 for row in rows) / len(rows))
    percentages = [abs(row["error_w"] / row["actual_w"]) * 100
                   for row in rows if row["actual_w"] != 0]
    if normalization_range is None:
        actuals = [row["actual_w"] for row in rows]
        normalization_range = max(actuals) - min(actuals)
    return {
        "MAE_W": mae,
        "RMSE_W": rmse,
        "MAPE_pct": sum(percentages) / len(percentages) if percentages else None,
        "NMAE_pct": 100 * mae / normalization_range if normalization_range else None,
        "NRMSE_pct": 100 * rmse / normalization_range if normalization_range else None,
        "MAPE_excluded_zero_actuals": len(rows) - len(percentages),
    }


def write_results_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scale_points(values, point_count, left, top, width, height):
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        minimum -= 1
        maximum += 1

    def scale(index, value):
        x = left + index * width / max(1, point_count - 1)
        y = top + height - (value - minimum) * height / (maximum - minimum)
        return x, y

    return scale, minimum, maximum


def polyline(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_svg_chart(path, rows, title):
    width = 1200
    height = 650
    left = 90
    top = 70
    plot_width = 1030
    plot_height = 450
    actual = [row["actual_w"] for row in rows]
    predicted = [row["predicted_w"] for row in rows]
    all_values = [value for value in actual if value is not None] + predicted
    scale, minimum, maximum = scale_points(all_values, len(rows), left, top, plot_width, plot_height)
    actual_segments = [[]]
    for index, value in enumerate(actual):
        if value is None:
            if actual_segments[-1]:
                actual_segments.append([])
        else:
            actual_segments[-1].append(scale(index, value))
    predicted_points = [scale(index, value) for index, value in enumerate(predicted)]

    y_ticks = []
    for step in range(6):
        value = minimum + (maximum - minimum) * step / 5
        y = top + plot_height - step * plot_height / 5
        y_ticks.append((value, y))

    x_ticks = []
    for index in range(0, len(rows), max(1, len(rows) // 6)):
        x, _ = scale(index, predicted[index])
        label = rows[index]["timestamp_utc"][5:16].replace("T", " ")
        x_ticks.append((x, label))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="35" font-family="Arial" font-size="24" font-weight="700" fill="#202124">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#3c4043" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#3c4043" stroke-width="1"/>',
    ]

    for value, y in y_ticks:
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e8eaed" stroke-width="1"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-family="Arial" font-size="12" text-anchor="end" fill="#5f6368">{value:,.0f}</text>')

    for x, label in x_ticks:
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#3c4043" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" font-family="Arial" font-size="12" text-anchor="middle" fill="#5f6368">{label}</text>')

    svg.extend(
        [
            *[f'<polyline points="{polyline(segment)}" fill="none" stroke="#1a73e8" stroke-width="3"/>' for segment in actual_segments if segment],
            f'<polyline points="{polyline(predicted_points)}" fill="none" stroke="#d93025" stroke-width="3"/>',
            f'<line x1="{left}" y1="{height - 75}" x2="{left + 40}" y2="{height - 75}" stroke="#1a73e8" stroke-width="4"/>',
            f'<text x="{left + 52}" y="{height - 70}" font-family="Arial" font-size="14" fill="#202124">Actual</text>',
            f'<line x1="{left + 145}" y1="{height - 75}" x2="{left + 185}" y2="{height - 75}" stroke="#d93025" stroke-width="4"/>',
            f'<text x="{left + 197}" y="{height - 70}" font-family="Arial" font-size="14" fill="#202124">Predicted</text>',
            f'<text x="{left}" y="{height - 35}" font-family="Arial" font-size="13" fill="#5f6368">15-minute average watts. Training: first 90% of available readings. Gaps in actuals are not scored.</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=(
        "All available history, chronological 90/10 split, 36-hour (144-step) forecasts. "
        "Input consumption must be power in watts, not interval energy."))
    parser.add_argument("--electricity", type=Path, default=DEFAULT_ELECTRICITY)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_BUILDINGS)
    parser.add_argument("--building-id", default=None)
    parser.add_argument("--target", default="consumption_w")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--list-buildings", action="store_true")
    parser.add_argument("--weather-mode", choices=["persistence", "observed"],
                        default="persistence", help=(
                            "Default: last completed weather hour held constant. "
                            "Observed: oracle experiment using future actual weather; "
                            "NOT a leakage-free operational forecast."))
    args = parser.parse_args()
    prepare_folders_and_check_files(args.electricity, args.weather, args.buildings, args.output_dir)
    if args.list_buildings:
        list_buildings(args.buildings)
        return
    building = choose_building(args.buildings, args.building_id)
    building_id = building["building_id"]
    target = load_quarter_hour_target(args.electricity, building_id, args.target)
    if not target:
        raise ValueError(f"No electricity rows found for building {building_id}.")
    target, times = select_window(target)
    observed_times = sorted(target)
    train_count = len(observed_times) * 9 // 10
    if train_count == 0:
        raise ValueError("Not enough readings for a chronological train/test split.")
    split_time = observed_times[train_count]
    split_index = (split_time - times[0]) // STEP
    if len(times) - split_index < HORIZON_STEPS:
        raise ValueError("The final 10% must span at least 36 hours for a full forecast.")
    weather = load_hourly_weather(args.weather, building_id)
    # Hourly historical weather features are shared across their four target slots.
    training_weather = {
        timestamp: weather[timestamp.replace(minute=0, second=0, microsecond=0)]
        for timestamp in observed_times[:train_count]
        if timestamp.replace(minute=0, second=0, microsecond=0) in weather
    }
    missing_training_weather = train_count - len(training_weather)
    if missing_training_weather:
        print(f"Skipping {missing_training_weather} training readings with missing weather.")
    training_rows, training_targets, typical = make_training_data(target, training_weather, split_time)
    coefficients, means, scales = fit_ordinary_least_squares(training_rows, training_targets)

    # Fixed model; observed history updates only between forecast origins.
    # Final full 36h window overlaps the previous window to cover the test tail.
    origins = list(range(split_index, len(times) - HORIZON_STEPS + 1, HORIZON_STEPS))
    last_origin = len(times) - HORIZON_STEPS
    if origins[-1] != last_origin:
        origins.append(last_origin)
    all_rows = []
    for index in origins:
        all_rows.extend(forecast_window(target, weather, times[index], coefficients,
                                        means, scales, typical, args.weather_mode))
    # Aggregate scores weight each test timestamp once (latest origin wins in overlap).
    unique_rows = {row["timestamp_utc"]: row for row in all_rows}
    scored_rows = sorted((row for row in unique_rows.values() if row["actual_w"] is not None),
                         key=lambda row: row["timestamp_utc"])
    test_values = [target[timestamp] for timestamp in observed_times[train_count:]]
    test_range = max(test_values) - min(test_values)
    scores = metrics(scored_rows, test_range)
    write_results_csv(args.output_dir / "forecast_results.csv", all_rows)
    write_results_csv(args.output_dir / "evaluation_results.csv", scored_rows)
    write_results_csv(args.output_dir / "metrics.csv", [scores])
    write_svg_chart(args.output_dir / "forecast_chart.svg", all_rows[:HORIZON_STEPS],
                    "First 36-hour test forecast")
    print(f"Building: {building_id}")
    print(f"History: {times[0].isoformat()} to {(times[-1] + STEP).isoformat()} (end exclusive)")
    print(f"Available span: {len(times) / 96:.2f} days; {len(target)} observed quarter-hour readings")
    print(f"Missing intervals: {len(times) - len(target)} (not imputed or scored)")
    print(f"Chronological 90/10 split: {train_count} training readings; {len(target)-train_count} test readings")
    print(f"Split: {split_time.isoformat()}; no shuffling or test-set fitting")
    print(f"Training rows after lag/weather availability checks: {len(training_rows)}")
    print(f"Forecasts: {len(origins)} windows, each 36 hours / 144 steps")
    print(f"Scored: {len(scored_rows)} unique test timestamps; latest origin used in tail overlap")
    print(f"Weather mode: {args.weather_mode}")
    if args.weather_mode == "observed":
        print("WARNING: Oracle weather experiment uses future observed weather.")
    print("NMAE/NRMSE denominator: range of actual consumption over the entire test set.")
    print("MAPE excludes zero actuals; near-zero actuals may produce very large percentages.")
    for name, value in scores.items():
        print(f"{name}: {value:.4f}" if value is not None else f"{name}: undefined")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
