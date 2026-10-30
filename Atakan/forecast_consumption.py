"""Electricity consumption forecast using Python's standard library.

Reads electricity, weather, and building data from CSV files.
Trains linear regression models with and without weather using the first 90%
of observations, then predicts 36 hours in 15-minute steps over the test period.
Saves accuracy metrics and SVG graphs.

Place electricity.csv, weather.csv, and buildings.csv in the data folder
next to this script. Example command:
    python consumption_forecast.py --building-id YOUR_ID

Negative electricity values are kept. NMAE uses average actual test consumption.
"""
import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The data folder sits next to this Python file.
DATA = Path(__file__).resolve().parent / "data"
WEATHER = [
    "temperature_c", "relative_humidity_pct", "precipitation_mm",
    "cloud_cover_pct", "shortwave_radiation_wm2", "direct_radiation_wm2",
    "diffuse_radiation_wm2", "wind_speed_ms",
]
STEP = timedelta(minutes=15)
STEPS = 144  # 144 * 15 minutes = 36 hours


def read_csv(path):
    """Read a CSV file and return all its rows.

    INPUT: The location of a CSV file.
    DOES: Reads each row. Column headers become dictionary keys.
    RETURNS: A list of row dictionaries.
    WHY: Our models need electricity, weather and building information.
    """
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def as_time(text):
    """Read a timestamp and round it down to a 15-minute slot.

    INPUT: A date and time written as text in the CSV.
    DOES: Converts it to UTC and removes seconds and extra minutes.
    RETURNS: A Python datetime value.
    WHY: Electricity and weather must match at the same time.
    """
    # Some CSV timestamps end with Z. Here Z means the UTC time zone.
    time = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    time = time.astimezone(timezone.utc)
    # Round down to the start of the current 15-minute interval.
    rounded_minute = (time.minute // 15) * 15
    time = time.replace(minute=rounded_minute, second=0, microsecond=0)
    return time


def number(text):
    """Safely turn text from a CSV cell into a number.

    INPUT: Text, such as "24.5", or a missing value.
    DOES: Tries float(); rejects invalid, infinite or NaN values.
    RETURNS: A decimal number, or None if the value is unusable.
    WHY: We should not train on broken measurements.
    """
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None  # Text such as "hello" cannot become a number.

    if math.isfinite(value):
        return value
    else:
        return None  # Infinity and NaN are not usable readings.


def load_readings(path, building_id, columns):
    """Load valid readings for one building.

    INPUT: CSV path, building ID, and the columns we want.
    DOES: Skips other buildings and broken readings; averages duplicates.
    RETURNS: A dictionary: timestamp -> dictionary of values.
    WHY: Gives us clean, matching 15-minute measurements.
    Negative numbers and zero are kept because they are valid readings.
    """
    # totals stores sums for each time and column.
    # counts stores how many valid readings exist for each time.
    # Example: totals[time]["temperature_c"] is the sum at that time.
    # defaultdict creates an empty value automatically for a new key.
    # This avoids checking whether a time or column already exists.
    totals = defaultdict(lambda: defaultdict(float))
    # Example: counts[time] tells us how many valid CSV rows we saw.
    counts = defaultdict(int)

    for row in read_csv(path):
        # Only include data for the selected building.
        if row["building_id"] != building_id:
            continue

        values = {}
        for column in columns:
            values[column] = number(row.get(column))

        # If ANY required value is missing, skip the whole CSV row.
        incomplete = False
        for value in values.values():
            if value is None:
                incomplete = True
        if incomplete:
            continue

        time = as_time(row["timestamp_utc"])
        for column in columns:
            totals[time][column] += values[column]
        counts[time] += 1

    # If several valid readings share a 15-minute slot, average them.
    readings = {}
    for time, values in totals.items():
        readings[time] = {}
        for column, total in values.items():
            readings[time][column] = total / counts[time]
    return readings


def features(time, weather, yesterday, last_week, use_weather):
    """Build the list of information the model sees for one time.

    INPUT: Time, weather, yesterday's power, last week's power, flag.
    DOES: Adds time patterns, optional weather, and older power readings.
    RETURNS: A list of numbers called features.
    WHY: Linear regression needs numerical inputs to make predictions.
    """
    # 14:30 becomes 14.5, so the model can use parts of an hour.
    hour = time.hour + time.minute / 60
    week_hour = time.weekday() * 24 + hour
    # A day and week repeat in circles. sin/cos represent that pattern.
    # We use BOTH sin and cos so that the end of the day/week connects
    # smoothly to the start instead of creating a sudden jump.
    day_angle = 2 * math.pi * hour / 24
    week_angle = 2 * math.pi * week_hour / 168
    values = [1.0]  # Intercept: a constant starting feature.
    values.append(math.sin(day_angle))
    values.append(math.cos(day_angle))
    values.append(math.sin(week_angle))
    values.append(math.cos(week_angle))

    if use_weather:
        for column in WEATHER:
            values.append(weather[column])

    # Electricity usage from a day ago and a week ago.
    values.append(yesterday)
    values.append(last_week)
    return values


# ===== PART 1: PREPARE MODEL INPUT NUMBERS =====
def normalize_rows(rows, means=None, scales=None):
    """Make feature values easier for regression to work with.

    INPUT: Feature rows, and optionally training means and scales.
    DOES: Subtracts each feature's average and divides by its spread.
    RETURNS: New rows, means, and scales.
    WHY: Temperature and power have very different number sizes.
    The first number, 1.0, stays unchanged as the intercept.
    """
    if means is None:
        means = [0.0] * len(rows[0])
        scales = [1.0] * len(rows[0])
        for col in range(1, len(rows[0])):
            values = [row[col] for row in rows]
            mean = sum(values) / len(values)
            # Calculate how far the values spread around their average.
            squared_differences = []
            for value in values:
                difference = value - mean
                squared_differences.append(difference * difference)
            variance = sum(squared_differences) / len(squared_differences)
            means[col] = mean
            standard_deviation = math.sqrt(variance)
            if standard_deviation == 0:
                standard_deviation = 1.0  # Avoid division by zero.
            scales[col] = standard_deviation
    normalized = []
    for row in rows:
        new_row = [row[0]]  # Never change the intercept value.
        for col in range(1, len(row)):
            new_value = (row[col] - means[col]) / scales[col]
            new_row.append(new_value)
        normalized.append(new_row)
    return normalized, means, scales


# ===== PART 2: LEARN THE REGRESSION WEIGHTS =====
def solve_linear_system(matrix, values):
    """Solve a set of equations needed by linear regression.

    INPUT: A matrix of numbers and the values on the right side.
    DOES: Uses Gauss-Jordan elimination: rearrange rows and cancel terms.
    RETURNS: The solution numbers (one per equation).
    WHY: Regression uses these equations to find the model's weights.
    This is the most mathematical part; no library is used for it.
    """
    # Think of this as solving several equations at once.
    # These rows are numbers in the equations, NOT electricity CSV rows.
    n = len(values)
    # Make a working copy so the original matrix is not changed.
    a = []
    for row in range(n):
        a.append(matrix[row][:] + [values[row]])
    for col in range(n):
        # Find the row with the biggest usable number in this column.
        # Swapping it to the top makes the calculations more stable.
        pivot = col
        for candidate in range(col + 1, n):
            if abs(a[candidate][col]) > abs(a[pivot][col]):
                pivot = candidate
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError('Cannot solve regression because the feature matrix is singular.')
        # Put the best pivot row at the current position.
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
    solution = []
    for row in range(n):
        solution.append(a[row][n])
    return solution

def fit_ordinary_least_squares(rows, targets):
    """Train the linear regression model using historical examples.

    INPUT: Feature rows and their true electricity consumption values.
    DOES: Normalizes features and solves the least-squares equations.
    RETURNS: Coefficients (weights), means, and scales.
    WHY: The model needs to learn how inputs relate to consumption.
    """
    normalized, means, scales = normalize_rows(rows)
    column_count = len(normalized[0])
    # xtx is a square table of zeros, used by the regression equations.
    xtx = []
    for i in range(column_count):
        zero_row = [0.0] * column_count
        xtx.append(zero_row)
    xty = [0.0 for _ in range(column_count)]
    # Each example adds information to the equations used for learning.
    # These calculations find weights that minimize squared training errors.
    # Loop over training examples, one at a time.
    for row_index in range(len(normalized)):
        row = normalized[row_index]
        target = targets[row_index]
        for i in range(column_count):
            xty[i] += row[i] * target
            for j in range(column_count):
                xtx[i][j] += row[i] * row[j]
    # Tiny stabilizer: helps when inputs are almost duplicates.
    # The first (intercept) feature is left unchanged.
    for i in range(1, column_count):
        xtx[i][i] += 1e-08
    coefficients = solve_linear_system(xtx, xty)
    return coefficients, means, scales

def predict(raw_row, coefficients, means, scales):
    """Calculate one predicted electricity consumption value.

    INPUT: Features and the values learned during training.
    DOES: Normalizes the features and adds feature * weight.
    RETURNS: The predicted power in watts.
    WHY: This is how trained linear regression makes a prediction.
    """
    normalized, _, _ = normalize_rows([raw_row], means, scales)
    # A regression prediction is a sum of: input * learned weight.
    prediction = 0.0
    for i in range(len(coefficients)):
        value = normalized[0][i]
        coefficient = coefficients[i]
        prediction += value * coefficient
    return prediction




# ===== PART 3: MAKE TRAINING EXAMPLES =====
def training_data(power, weather, split):
    """Create the training examples from historical readings only.

    INPUT: Electricity, weather, and where the test period starts.
    DOES: Builds input lists and the correct answers for each time.
    RETURNS: Inputs, answers, and typical power by time of day.
    WHY: The model must learn from the past, not future test answers.
    """
    times = []
    for time in power:
        if time < split and time in weather:
            times.append(time)
    times.sort()
    if not times:
        raise ValueError("No matching electricity and weather training data.")
    # Group consumption values by the same 15-minute time of day.
    # Example: all values recorded at 09:15 go into one group.
    daily = defaultdict(list)
    for t in times:
        daily[t.hour * 4 + t.minute // 15].append(power[t])
    typical = {}
    for slot, values in daily.items():
        typical[slot] = sum(values) / len(values)

    inputs = []  # What the model uses to make predictions.
    answers = []  # Correct electricity values to learn from.
    for t in times:
        yesterday = power.get(t - timedelta(hours=24))
        last_week = power.get(t - timedelta(hours=168))
        if yesterday is None or last_week is None:
            continue
        inputs.append(features(t, weather[t], yesterday, last_week, True))
        answers.append(power[t])
    if len(inputs) < 50:
        raise ValueError("Too few training examples with 1-day and 1-week history.")
    return inputs, answers, typical



# ===== PART 4: MAKE 36-HOUR FORECASTS =====
def forecast(power, weather, start, model, typical, use_weather, weather_mode):
    """Predict electricity consumption for one 36-hour window.

    INPUT: Past readings, weather, start time, model, and settings.
    DOES: Repeats 144 steps of 15 minutes and calculates predictions.
    RETURNS: A list of result dictionaries for each prediction.
    WHY: We want 36-hour forecasts, not just a single prediction.
    When necessary, later steps use earlier predictions as history.
    """
    coefficients, means, scales = model
    history = {}
    for time, value in power.items():
        if time < start:
            history[time] = value  # Never include future real readings.
    last_weather = {}
    if use_weather:
        # Find weather measurements available BEFORE forecast start.
        known = []
        for t in weather:
            if t + STEP <= start:
                known.append(t)
        if not known:
            raise ValueError("No past weather available for the forecast.")
        last_weather = weather[max(known)]

    result = []
    for step in range(STEPS):
        time = start + step * STEP
        if use_weather and weather_mode == "observed":
            if time not in weather:
                raise ValueError(f"Missing observed weather at {time}")
            weather_now = weather[time]
        else:
            weather_now = last_weather

        # Look up consumption from yesterday and one week earlier.
        # If a reading is missing, use typical power for this time of day.
        slot = time.hour * 4 + time.minute // 15
        typical_power = typical.get(slot, 0.0)
        yesterday = history.get(time - timedelta(hours=24))
        last_week = history.get(time - timedelta(hours=168))
        if yesterday is None:
            yesterday = typical_power
        if last_week is None:
            last_week = typical_power

        model_inputs = features(time, weather_now, yesterday, last_week, use_weather)
        prediction = predict(model_inputs, coefficients, means, scales)
        # Store the prediction as history for later steps.
        # This matters once we forecast more than 24 hours ahead.
        history[time] = prediction
        actual = power.get(time)
        if actual is None:
            error = None
            absolute_error = None
        else:
            error = prediction - actual
            absolute_error = abs(error)
        # Keep all information about this one prediction together.
        result.append({
            "forecast_origin_utc": start.isoformat(), "lead_step": step + 1,
            "timestamp_utc": time.isoformat(), "predicted_w": prediction,
            "actual_w": actual, "error_w": error,
            "abs_error_w": absolute_error,
        })
    return result



# ===== PART 5: CHECK HOW GOOD THE PREDICTIONS WERE =====
def metrics(rows, test_range, test_average):
    """Measure how far predictions are from actual electricity usage.

    INPUT: Forecast results, test range, and average test consumption.
    DOES: Calculates MAE, RMSE, MAPE, NMAE, and NRMSE.
    RETURNS: A dictionary containing the scores.
    WHY: We use these scores to compare the two models fairly.
    NMAE uses average actual consumption, NOT max-minus-min.
    NRMSE still uses the test range, as in the earlier version.
    """
    # Only use predictions where the real electricity reading is known.
    errors = []
    percentages = []
    for row in rows:
        actual = row["actual_w"]
        if actual is None:
            continue
        error = row["error_w"]
        errors.append(error)
        if actual != 0:
            percentages.append(abs(error / actual) * 100)

    if not errors:
        raise ValueError("No actual test readings to score.")

    # MAE: average size of the error, ignoring + and - signs.
    mae = sum(abs(error) for error in errors) / len(errors)

    # RMSE: like MAE, but large mistakes count more heavily.
    squared_errors = []
    for error in errors:
        squared_errors.append(error ** 2)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    # MAPE is undefined for zero actual consumption, so skip those.
    if percentages:
        mape = sum(percentages) / len(percentages)
    else:
        mape = None

    # NMAE divides the average absolute error by average actual consumption.
    if test_average > 0:
        nmae = 100 * mae / test_average
    else:
        nmae = None

    # NRMSE still uses max(real usage) - min(real usage).
    if test_range != 0:
        nrmse = 100 * rmse / test_range
    else:
        nrmse = None

    return {
        "MAE_W": mae,
        "RMSE_W": rmse,
        "MAPE_pct": mape,
        "NMAE_pct": nmae,
        "NRMSE_pct": nrmse,
        "MAPE_excluded_zero_actuals": len(errors) - len(percentages),
    }



# ===== PART 6: SAVE RESULT FILES =====
def save_csv(path, rows):
    """Save rows of results into a CSV file.

    INPUT: A destination path and a list of dictionaries.
    DOES: Writes the column headers and the values below them.
    RETURNS: Nothing; it creates a file on disk.
    WHY: We can open predictions and results in Excel or Sheets.
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_svg_chart(path, rows, title):
    """Make a basic graph of predicted and real electricity usage.

    INPUT: An output path, forecast results, and a graph title.
    DOES: Turns power measurements into line positions in an SVG.
    RETURNS: Nothing; it writes an SVG image file.
    WHY: It is easier to see model errors on a graph.
    """
    width, height = 1000, 500
    actual = []
    predicted = []
    values = []
    for row in rows:
        actual.append(row["actual_w"])
        predicted.append(row["predicted_w"])
        if row["actual_w"] is not None:
            values.append(row["actual_w"])
        values.append(row["predicted_w"])
    low, high = min(values), max(values)
    if low == high:
        low, high = low - 1, high + 1

    def point(index, value):
        """Turn one time index and one power value into graph x,y text."""
        # Convert each time and watt reading into a point on the picture.
        x = 70 + index * 880 / max(1, len(rows) - 1)
        y = 430 - (value - low) * 360 / (high - low)
        return f"{x:.1f},{y:.1f}"

    # Missing actual readings leave breaks in the blue line.
    actual_lines, segment = [], []
    for i, value in enumerate(actual):
        if value is None:
            if segment:
                actual_lines.append(segment)
                segment = []
        else:
            segment.append(point(i, value))
    if segment:
        actual_lines.append(segment)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="70" y="35" font-size="22" font-family="Arial">{title}</text>',
        '<line x1="70" y1="70" x2="70" y2="430" stroke="black"/>',
        '<line x1="70" y1="430" x2="950" y2="430" stroke="black"/>',
        f'<text x="5" y="77" font-size="13">{high:,.0f} W</text>',
        f'<text x="5" y="430" font-size="13">{low:,.0f} W</text>',
    ]
    for line in actual_lines:
        svg.append(f'<polyline points="{" ".join(line)}" fill="none" stroke="#1a73e8" stroke-width="2"/>')
    prediction_points = []
    for i in range(len(predicted)):
        prediction_points.append(point(i, predicted[i]))
    points = " ".join(prediction_points)
    svg.append(f'<polyline points="{points}" fill="none" stroke="#d93025" stroke-width="2"/>')
    svg += [
        '<text x="70" y="465" fill="#1a73e8" font-size="14">Blue: actual</text>',
        '<text x="230" y="465" fill="#d93025" font-size="14">Red: prediction</text>',
        '</svg>',
    ]
    path.write_text("\n".join(svg), encoding="utf-8")



# ===== PART 7: RUN ALL STEPS IN THE CORRECT ORDER =====
def main():
    """Run the whole project from beginning to end.

    INPUT: Command-line options such as --building-id.
    DOES: Load files -> split data -> train -> forecast -> score -> save.
    RETURNS: Nothing; it prints a summary and saves result files.
    WHY: It keeps the main steps of the project in one place.
    """
    parser = argparse.ArgumentParser(description="36-hour electricity forecasting")
    parser.add_argument("--electricity", type=Path, default=DATA / "electricity.csv")
    parser.add_argument("--weather", type=Path, default=DATA / "weather.csv")
    parser.add_argument("--buildings", type=Path, default=DATA / "buildings.csv")
    parser.add_argument("--building-id")
    parser.add_argument("--target", default="consumption_w")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--list-buildings", action="store_true")
    parser.add_argument("--weather-mode", choices=["persistence", "observed"],
                        default="persistence")
    args = parser.parse_args()
    for path in (args.electricity, args.weather, args.buildings):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for path in (args.electricity, args.weather, args.buildings):
        if not path.exists():
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {missing}")

    buildings = read_csv(args.buildings)
    if args.list_buildings:
        for b in buildings:
            print(b["building_id"], b.get("description", ""))
        return
    # Find the requested building; otherwise use the first available one.
    building = None
    if args.building_id:
        for item in buildings:
            if item["building_id"] == args.building_id:
                building = item
                break
    else:
        if buildings:
            building = buildings[0]
    if building is None:
        raise ValueError("Building not found in buildings.csv")
    building_id = building["building_id"]

    electricity = load_readings(args.electricity, building_id, [args.target])
    power = {}
    for time, row in electricity.items():
        power[time] = row[args.target]
    if not power:
        raise ValueError("No valid electricity readings.")
    weather = load_readings(args.weather, building_id, WEATHER)
    if not weather:
        raise ValueError("No complete weather readings.")

    # STEP 1: Split in time order. We never shuffle future into the past.
    # First 90% trains the model; last 10% tests it.
    observed = sorted(power)
    train_count = len(observed) * 9 // 10
    if train_count == 0 or train_count == len(observed):
        raise ValueError("Not enough readings for training and testing.")
    split = observed[train_count]
    times = []
    total_slots = (observed[-1] - observed[0]) // STEP + 1
    for i in range(total_slots):
        times.append(observed[0] + i * STEP)
    split_index = (split - times[0]) // STEP
    if len(times) - split_index < STEPS:
        raise ValueError("Test period must span at least 36 hours.")
    train_weather = {}
    for time in observed[:train_count]:
        if time in weather:
            train_weather[time] = weather[time]
    # STEP 2: Build examples from TRAINING data only.
    inputs, answers, typical = training_data(power, train_weather, split)
    test_actuals = []
    for time in observed[train_count:]:
        test_actuals.append(power[time])
    test_range = max(test_actuals) - min(test_actuals)
    test_average = sum(test_actuals) / len(test_actuals)

    # STEP 3: Choose starts for 36-hour forecast windows.
    # Forecast all of the test period in 36-hour windows.
    origins = []
    for start_index in range(split_index, len(times) - STEPS + 1, STEPS):
        origins.append(start_index)
    if origins[-1] != len(times) - STEPS:
        origins.append(len(times) - STEPS)
    comparison, scored_times = [], None
    # STEP 4: Run the experiment twice for a fair comparison.
    for name, with_weather in (("without_weather", False), ("with_weather", True)):
        if with_weather:
            model_inputs = inputs
        else:
            # The first five features are time features.
            # The last two are yesterday's and last week's power values.
            # Remove the eight weather features for the baseline model.
            model_inputs = []
            for row in inputs:
                model_inputs.append(row[:5] + row[-2:])
        # Learn the weights from the chosen input columns.
        model = fit_ordinary_least_squares(model_inputs, answers)
        forecasts = []
        for index in origins:
            forecasts.extend(forecast(power, weather, times[index], model,
                                      typical, with_weather, args.weather_mode))
        # If windows overlap, score the latest prediction for a timestamp once.
        unique = {}
        for row in forecasts:
            unique[row["timestamp_utc"]] = row
        scored = []
        for row in unique.values():
            if row["actual_w"] is not None:
                scored.append(row)
        scored.sort(key=lambda row: row["timestamp_utc"])
        current_times = []
        for row in scored:
            current_times.append(row["timestamp_utc"])
        if scored_times is not None and current_times != scored_times:
            raise ValueError("Models were scored on different timestamps.")
        scored_times = current_times
        # STEP 5: Calculate MAE, RMSE, MAPE, NMAE, and NRMSE.
        score = metrics(scored, test_range, test_average)
        comparison_row = {"model": name}
        comparison_row.update(score)
        comparison.append(comparison_row)
        # STEP 6: Save the accuracy metrics and one graph for this model.
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        save_csv(folder / "metrics.csv", [score])
        write_svg_chart(folder / "forecast_chart.svg", forecasts[:STEPS],
                        f"First 36-hour forecast: {name.replace('_', ' ')}")

    save_csv(args.output_dir / "metrics_comparison.csv", comparison)
    print(f"Building: {building_id}")
    print(f"90/10 split: {train_count} train; {len(observed) - train_count} test")
    print(f"Forecast windows: {len(origins)}; 144 steps (36 hours) each")
    print(f"Test mean consumption: {test_average:.4f} W (NMAE denominator)")
    print(f"Test range: {test_range:.4f} W (NRMSE denominator)")
    if args.weather_mode == "observed":
        print("WARNING: observed mode uses actual future weather (oracle test).")
    for row in comparison:
        print("Model:", row["model"])
        for metric_name, metric_value in row.items():
            if metric_name == "model":
                continue
            if isinstance(metric_value, float):
                print("  ", metric_name, ":", round(metric_value, 4))
            else:
                print("  ", metric_name, ":", metric_value)
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
