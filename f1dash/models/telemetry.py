"""Lap telemetry maths before the overhaul fixes: ported as a starting point."""

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
