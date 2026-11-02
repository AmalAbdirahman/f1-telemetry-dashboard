"""Lap telemetry maths: distance alignment, speed delta, time delta, sector deltas.

Pure functions only. Sign convention throughout: Driver 1 minus Driver 2, so a
negative time delta means Driver 1 was quicker.
"""

import numpy as np
import pandas as pd

GRID_POINTS = 500
MIN_SAMPLES = 10


def prepare_lap(raw: pd.DataFrame, lap_start, lap_end) -> pd.DataFrame:
    """Clip car data to one lap and add elapsed time and integrated distance.

    Assumption: OpenF1 gives no distance channel, so distance is the integral of
    speed over time (about 4 samples a second). Two laps therefore differ by a
    few metres, which the common grid tolerates.
    """
    columns = ["Time", "Distance", "Speed", "Throttle", "Brake", "Gear"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    dates = pd.to_datetime(raw["date"], format="ISO8601", utc=True)
    lap = pd.DataFrame(
        {
            "Time": (dates - lap_start).dt.total_seconds(),
            "Speed": pd.to_numeric(raw["speed"], errors="coerce"),
            "Throttle": pd.to_numeric(raw["throttle"], errors="coerce").clip(0, 100),
            "Brake": pd.to_numeric(raw["brake"], errors="coerce"),
            "Gear": pd.to_numeric(raw["n_gear"], errors="coerce"),
        }
    )
    lap = lap.dropna(subset=["Speed"]).sort_values("Time").drop_duplicates("Time")
    duration = (lap_end - lap_start).total_seconds()
    inside = lap[(lap["Time"] > 0) & (lap["Time"] < duration)]
    if len(inside) < MIN_SAMPLES:
        return pd.DataFrame(columns=columns)
    # Samples rarely land on the timing line, so add both ends by interpolation.
    # Time then runs exactly from 0 to the official lap time.
    times = np.concatenate([[0.0], inside["Time"].to_numpy(float), [duration]])
    filled = lap.set_index("Time").interpolate(method="index", limit_direction="both")
    lap = pd.DataFrame(
        {"Time": times}
        | {c: np.interp(times, filled.index.to_numpy(float), filled[c].to_numpy(float)) for c in filled}
    )
    speed_ms = lap["Speed"].to_numpy(dtype=float) / 3.6
    time = lap["Time"].to_numpy(dtype=float)
    steps = np.diff(time) * (speed_ms[1:] + speed_ms[:-1]) / 2
    lap["Distance"] = np.concatenate([[0.0], np.cumsum(steps)])
    return lap[columns].reset_index(drop=True)



def common_grid(distance_1, distance_2) -> np.ndarray:
    """Return the shared distance grid for two laps."""
    return np.linspace(0, min(np.max(distance_1), np.max(distance_2)), GRID_POINTS)


def align(lap: pd.DataFrame, grid: np.ndarray, channel: str) -> np.ndarray:
    """Interpolate a channel onto the grid, stretching the lap to the grid's length.

    Both laps cover the same track, so any difference in integrated distance is
    measurement drift. Stretching keeps corners lined up and makes a time-delta
    trace finish exactly on the difference between the two lap times.
    """
    distance = lap["Distance"].to_numpy(float)
    scaled = distance * grid[-1] / distance[-1]
    return np.interp(grid, scaled, lap[channel].to_numpy(float))


def speed_delta(lap_1: pd.DataFrame, lap_2: pd.DataFrame) -> float:
    """Mean speed difference (Driver 1 − Driver 2), compared at equal distance."""
    grid = common_grid(lap_1["Distance"], lap_2["Distance"])
    return float(np.mean(align(lap_1, grid, "Speed") - align(lap_2, grid, "Speed")))


def time_delta(lap_1: pd.DataFrame, lap_2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid, t1 − t2) along the lap. Negative means Driver 1 is ahead."""
    grid = common_grid(lap_1["Distance"], lap_2["Distance"])
    return grid, align(lap_1, grid, "Time") - align(lap_2, grid, "Time")


def sector_deltas(lap_1: pd.Series, lap_2: pd.Series) -> pd.DataFrame:
    """Return S1, S2, S3 and whole-lap deltas as lap 1 − lap 2.

    Negative means Driver 1 was faster. Each row is checked on its own, and a
    missing time gives None so the UI can show "n/a".
    """
    rows = []
    for label, column in (("S1", "Sector1"), ("S2", "Sector2"), ("S3", "Sector3"), ("Lap", "LapTimeSec")):
        t1, t2 = lap_1.get(column), lap_2.get(column)
        delta = None if pd.isna(t1) or pd.isna(t2) else float(t1) - float(t2)
        rows.append({"Sector": label, "Delta": delta})
    return pd.DataFrame(rows, dtype=object)
