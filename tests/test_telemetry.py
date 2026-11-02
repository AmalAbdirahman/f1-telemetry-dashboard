import numpy as np
import pandas as pd

from f1dash.models import telemetry


def lap_frame(times, speed_fn, length=None):
    """Build a telemetry frame from a function of elapsed time."""
    times = np.asarray(times, dtype=float)
    speed = speed_fn(times)
    distance = np.concatenate([[0], np.cumsum(np.diff(times) * (speed[1:] + speed[:-1]) / 2 / 3.6)])
    if length:
        distance = distance * length / distance[-1]
    return pd.DataFrame({"Time": times, "Distance": distance, "Speed": speed})


def test_regression_speed_delta_compares_equal_distance_not_equal_index():
    # The same constant-speed-by-position lap, sampled at different rates.
    # Index-by-index subtraction reports a phantom delta here.
    distance_a = np.linspace(0, 5000, 300)
    distance_b = np.linspace(0, 5000, 400) ** 1.3 / 5000**0.3
    speed = lambda d: 200 + 80 * np.sin(d / 400)
    lap_a = pd.DataFrame({"Distance": distance_a, "Speed": speed(distance_a)})
    lap_b = pd.DataFrame({"Distance": distance_b, "Speed": speed(distance_b)})
    assert abs(telemetry.speed_delta(lap_a, lap_b)) < 0.1


def test_speed_delta_sign_is_driver_one_minus_driver_two():
    lap_a = pd.DataFrame({"Distance": np.linspace(0, 1000, 50), "Speed": np.full(50, 210.0)})
    lap_b = pd.DataFrame({"Distance": np.linspace(0, 1000, 80), "Speed": np.full(80, 200.0)})
    assert telemetry.speed_delta(lap_a, lap_b) == 10


def test_regression_sector_delta_negative_when_driver_one_faster():
    lap_1 = pd.Series({"Sector1": 20.0, "Sector2": 30.0, "Sector3": 25.0, "LapTimeSec": 75.0})
    lap_2 = pd.Series({"Sector1": 20.5, "Sector2": 29.5, "Sector3": 25.0, "LapTimeSec": 75.0})
    deltas = telemetry.sector_deltas(lap_1, lap_2).set_index("Sector")["Delta"]
    assert deltas["S1"] == -0.5
    assert deltas["S2"] == 0.5
    assert deltas["S3"] == 0


def test_regression_each_sector_is_checked_independently():
    lap_1 = pd.Series({"Sector1": None, "Sector2": 30.0, "Sector3": None, "LapTimeSec": 75.0})
    lap_2 = pd.Series({"Sector1": 20.5, "Sector2": 29.5, "Sector3": 25.0, "LapTimeSec": 75.5})
    deltas = telemetry.sector_deltas(lap_1, lap_2).set_index("Sector")["Delta"]
    assert pd.isna(deltas["S1"]) and pd.isna(deltas["S3"])
    assert deltas["S2"] == 0.5
    assert deltas["Lap"] == -0.5


def test_common_grid_stops_at_the_shorter_lap():
    grid = telemetry.common_grid(np.array([0, 4990.0]), np.array([0, 5010.0]))
    assert len(grid) == 500 and grid[0] == 0 and grid[-1] == 4990


def test_time_delta_is_negative_where_driver_one_gains():
    quick = lap_frame(np.linspace(0, 80, 321), lambda t: np.full_like(t, 220.0))
    slow = lap_frame(np.linspace(0, 81, 325), lambda t: np.full_like(t, 220.0 * 80 / 81))
    grid, delta = telemetry.time_delta(quick, slow)
    assert delta[-1] < 0 and len(grid) == len(delta)
    assert (np.diff(delta) <= 1e-9).all()


def test_time_delta_ends_on_the_true_lap_time_gap():
    # Integrated distances differ slightly, as they do with real telemetry.
    # The trace must still finish on the official gap between the two laps.
    lap_1 = lap_frame(np.linspace(0, 79.3, 300), lambda t: 200 + 70 * np.sin(t / 5), length=4150)
    lap_2 = lap_frame(np.linspace(0, 79.9, 300), lambda t: 200 + 70 * np.sin(t / 5), length=4163)
    _, delta = telemetry.time_delta(lap_1, lap_2)
    assert abs(delta[-1] - (79.3 - 79.9)) < 1e-6


def test_prepare_lap_rejects_too_few_samples():
    start = pd.Timestamp("2024-01-01T12:00:00Z")
    raw = pd.DataFrame(
        {"date": ["2024-01-01T12:00:01+00:00"], "speed": [100], "throttle": [50], "brake": [0], "n_gear": [4]}
    )
    assert telemetry.prepare_lap(raw, start, start + pd.Timedelta(seconds=80)).empty
    assert telemetry.prepare_lap(pd.DataFrame(), start, start).empty
