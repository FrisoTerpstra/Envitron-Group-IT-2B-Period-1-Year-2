import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
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


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def hour_bucket(value):
    return parse_time(value).replace(minute=0, second=0, microsecond=0)


def add_months(dt, months):
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][month - 1]
    return dt.replace(year=year, month=month, day=min(dt.day, days_in_month))


def safe_float(value):
    try:
        return float(value)
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


def load_hourly_target(path, building_id, target_column):
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
            hour = hour_bucket(row["timestamp_utc"])
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
            ok = True
            for column in WEATHER_COLUMNS:
                value = safe_float(row.get(column))
                if value is None:
                    ok = False
                    break
                totals[hour][column] += value
            if ok:
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
    return typical_by_hour.get(timestamp.hour, 0.0)


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
        typical_by_hour[timestamp.hour].append(target[timestamp])
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


def forecast_next_hours(target, weather, split_time, horizon, coefficients, means, scales, typical_by_hour):
    eval_times = [time for time in sorted(target) if time >= split_time and time in weather][:horizon]
    if len(eval_times) < horizon:
        raise ValueError(f"Only found {len(eval_times)} comparable future hours.")

    history = {time: value for time, value in target.items() if time < split_time}
    rows = []

    for timestamp in eval_times:
        lag_24 = lag_or_typical(history, timestamp, 24, typical_by_hour)
        lag_168 = lag_or_typical(history, timestamp, 168, typical_by_hour)
        raw_row = raw_features(timestamp, weather[timestamp], lag_24, lag_168)
        predicted = predict(raw_row, coefficients, means, scales)
        history[timestamp] = predicted
        actual = target[timestamp]
        rows.append(
            {
                "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "predicted_w": predicted,
                "actual_w": actual,
                "error_w": predicted - actual,
                "abs_error_w": abs(predicted - actual),
            }
        )

    return rows


def metrics(rows):
    errors = [row["error_w"] for row in rows]
    abs_errors = [abs(value) for value in errors]
    squared = [value * value for value in errors]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(squared) / len(squared))
    actual_range = max(row["actual_w"] for row in rows) - min(row["actual_w"] for row in rows)
    nmae = mae / actual_range * 100 if actual_range else 0.0
    return mae, rmse, nmae


def write_results_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp_utc",
                "predicted_w",
                "actual_w",
                "error_w",
                "abs_error_w",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "predicted_w": f"{row['predicted_w']:.2f}",
                    "actual_w": f"{row['actual_w']:.2f}",
                    "error_w": f"{row['error_w']:.2f}",
                    "abs_error_w": f"{row['abs_error_w']:.2f}",
                }
            )


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
    all_values = actual + predicted
    scale, minimum, maximum = scale_points(all_values, len(rows), left, top, plot_width, plot_height)
    actual_points = [scale(index, value) for index, value in enumerate(actual)]
    predicted_points = [scale(index, value) for index, value in enumerate(predicted)]

    y_ticks = []
    for step in range(6):
        value = minimum + (maximum - minimum) * step / 5
        y = top + plot_height - step * plot_height / 5
        y_ticks.append((value, y))

    x_ticks = []
    for index in range(0, len(rows), max(1, len(rows) // 6)):
        x, _ = scale(index, actual[index])
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
            f'<polyline points="{polyline(actual_points)}" fill="none" stroke="#1a73e8" stroke-width="3"/>',
            f'<polyline points="{polyline(predicted_points)}" fill="none" stroke="#d93025" stroke-width="3"/>',
            f'<line x1="{left}" y1="{height - 75}" x2="{left + 40}" y2="{height - 75}" stroke="#1a73e8" stroke-width="4"/>',
            f'<text x="{left + 52}" y="{height - 70}" font-family="Arial" font-size="14" fill="#202124">Actual</text>',
            f'<line x1="{left + 145}" y1="{height - 75}" x2="{left + 185}" y2="{height - 75}" stroke="#d93025" stroke-width="4"/>',
            f'<text x="{left + 197}" y="{height - 70}" font-family="Arial" font-size="14" fill="#202124">Predicted</text>',
            f'<text x="{left}" y="{height - 35}" font-family="Arial" font-size="13" fill="#5f6368">Hourly average watts. Model trained on the first 3 calendar months only.</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Forecast electricity consumption up to 36 hours ahead.")
    parser.add_argument("--electricity", type=Path, default=DEFAULT_ELECTRICITY)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_BUILDINGS)
    parser.add_argument("--building-id", default=None)
    parser.add_argument("--target", default="consumption_w")
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--list-buildings", action="store_true")
    args = parser.parse_args()

    if args.list_buildings:
        prepare_folders_and_check_files(args.electricity, args.weather, args.buildings, args.output_dir)
        list_buildings(args.buildings)
        return

    prepare_folders_and_check_files(args.electricity, args.weather, args.buildings, args.output_dir)
    building = choose_building(args.buildings, args.building_id)
    building_id = building["building_id"]

    target = load_hourly_target(args.electricity, building_id, args.target)
    weather = load_hourly_weather(args.weather, building_id)
    if not target:
        raise ValueError(f"No electricity rows found for building {building_id}.")

    first_time = min(target)
    split_time = add_months(first_time, args.train_months)
    training_rows, training_targets, typical_by_hour = make_training_data(target, weather, split_time)
    coefficients, means, scales = fit_ordinary_least_squares(training_rows, training_targets)
    result_rows = forecast_next_hours(
        target,
        weather,
        split_time,
        args.horizon,
        coefficients,
        means,
        scales,
        typical_by_hour,
    )

    csv_path = args.output_dir / "forecast_results.csv"
    chart_path = args.output_dir / "forecast_chart.svg"
    write_results_csv(csv_path, result_rows)
    write_svg_chart(
        chart_path,
        result_rows,
        f"36 hour forecast for {building_id}",
    )

    mae, rmse, nmae = metrics(result_rows)
    print(f"Building: {building_id}")
    print(f"Description: {building.get('description', '')}")
    print(f"Training window: {first_time.isoformat()} to {split_time.isoformat()}")
    print(f"Training rows used: {len(training_rows)} hourly rows")
    print(f"Forecast compared with actuals: {len(result_rows)} hours")
    print(f"MAE: {mae:,.2f} W")
    print(f"RMSE: {rmse:,.2f} W")
    print(f"Normalized MAE: {nmae:.2f}% of actual range")
    print(f"Results CSV: {csv_path.resolve()}")
    print(f"Chart SVG: {chart_path.resolve()}")


if __name__ == "__main__":
    main()
