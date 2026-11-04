"""The race as a strategist sees it at the end of lap N, using nothing later.

Order and gaps come from the moment each car crossed the line to finish lap N
(``LapEnd``). That crossing order is the race order, and the difference between
two crossings is the timing-screen gap.

A car is lapped when the leader finished another lap before it finished lap N.
Checking that exactly needs the leader's lap N + 1, which is after N, so it is
estimated as floor(gap ÷ the leader's lap N time). The same goes for the car
ahead. Under a safety car every lap is slow, so the estimate stays consistent.

Every input is first cut to laps 1..N, including the clean-lap filter, whose
107% threshold is relative to a stint median. Nothing after lap N can leak in.

Recent pace is taken from the current set of tyres, because a fresh set can be
a second a lap different. Until the set has MIN_RECENT_LAPS clean laps, the
last RECENT_LAPS clean laps of any stint are used instead (``PaceFromStint``
says which).

Assumptions in ``project_clock``:
- Each car keeps its recent mean pace plus its current tyre trend, as in the
  overtake simulation. Pit stops, safety cars and traffic in the projected
  laps are not known at lap N and are not modelled.
- A car with no clean laps yet runs at the field's median pace.
"""

import numpy as np
import pandas as pd

from f1dash import config
from f1dash.models import degradation
from f1dash.models.filters import clean_laps

COLUMNS = [
    "Driver", "Position", "GapToLeader", "LapsDown", "Interval", "IntervalLapsDown",
    "Compound", "TyreLife", "Stint", "Stops", "InPitLane",
    "RecentPace", "RecentSD", "RecentLaps", "PaceFromStint", "Slope", "SlopeLaps",
]

# Track status codes from the timing feed, most serious first.
NEUTRALISED = (("5", "Red flag"), ("4", "SC"), ("6", "VSC"))
MAJORITY = 0.5  # Share of cars whose lap must overlap a status for the lap to count.


def laps_up_to(laps: pd.DataFrame, lap_n: int) -> pd.DataFrame:
    """Only the laps a strategist has seen by the end of lap N."""
    return laps[laps["LapNumber"] <= lap_n]


def last_timed_lap(laps: pd.DataFrame) -> int:
    """The last lap that at least two cars have a finishing time for.

    OpenF1 often leaves the final lap's duration blank, which leaves no line
    crossing to order the cars by.
    """
    timed = laps[laps["LapEnd"].notna()].groupby("LapNumber")["Driver"].nunique()
    timed = timed[timed >= 2]
    return int(timed.index.max()) if not timed.empty else 0


def recent_laps(clean: pd.DataFrame, driver: str, stint=None, window: int = config.RECENT_LAPS) -> tuple[np.ndarray, bool]:
    """A driver's last ``window`` clean lap times, oldest first, and whether they are all from ``stint``.

    The current stint is used once it has MIN_RECENT_LAPS clean laps; before
    that, recent laps from any stint stand in.
    """
    own = clean[clean["Driver"] == driver].sort_values("LapNumber")
    if stint is not None and not pd.isna(stint):
        on_set = own[own["Stint"] == stint]
        if len(on_set) >= config.MIN_RECENT_LAPS:
            return on_set["LapTimeSec"].to_numpy(float)[-window:], True
    return own["LapTimeSec"].to_numpy(float)[-window:], False


def neutralised_at(laps: pd.DataFrame, lap_n: int) -> str | None:
    """'Red flag', 'SC' or 'VSC' if most cars' lap N overlapped it, else None."""
    codes = laps.loc[laps["LapNumber"] == lap_n, "TrackStatus"]
    if codes.empty:
        return None
    for code, label in NEUTRALISED:
        if codes.str.contains(code, regex=False).mean() > MAJORITY:
            return label
    return None


def laps_down(gap: float, reference_lap: float) -> int:
    """Whole laps a car is behind, estimated from the gap and the reference car's lap time."""
    if not np.isfinite(gap) or not np.isfinite(reference_lap) or reference_lap <= 0:
        return 0
    return int(max(gap, 0.0) // reference_lap)


def _current_slope(clean: pd.DataFrame, driver: str, stint) -> tuple[float, int]:
    own = clean[(clean["Driver"] == driver) & (clean["Stint"] == stint) & clean["TyreLife"].notna()]
    slope = degradation.stint_slope(own["TyreLife"], degradation.fuel_corrected(own))
    return (np.nan if slope is None else slope), len(own)


def race_state(laps: pd.DataFrame, lap_n: int) -> pd.DataFrame:
    """One row per car that finished lap N, in race order.

    Gaps are in seconds at the line. ``LapsDown`` counts laps behind the
    leader and ``IntervalLapsDown`` laps behind the car ahead. Pace, spread
    and slope are NaN when there are too few clean laps by lap N.
    """
    seen = laps_up_to(laps, lap_n)
    at_n = seen[(seen["LapNumber"] == lap_n) & seen["LapEnd"].notna()].sort_values("LapEnd")
    if at_n.empty:
        return pd.DataFrame(columns=COLUMNS)
    clean = clean_laps(seen)
    stops = seen.groupby("Driver")["PitInLap"].sum()
    lap_seconds = (at_n["LapEnd"] - at_n["LapStart"]).dt.total_seconds().to_numpy()
    leader = at_n.iloc[0]

    rows = []
    previous = None
    for index, lap in enumerate(at_n.itertuples(index=False)):
        gap = (lap.LapEnd - leader.LapEnd).total_seconds()
        interval = np.nan if previous is None else (lap.LapEnd - previous.LapEnd).total_seconds()
        recent, from_stint = recent_laps(clean, lap.Driver, lap.Stint)
        slope, slope_laps = _current_slope(clean, lap.Driver, lap.Stint)
        rows.append(
            {
                "Driver": lap.Driver,
                "Position": index + 1,
                "GapToLeader": gap,
                "LapsDown": laps_down(gap, lap_seconds[0]),
                "Interval": interval,
                "IntervalLapsDown": 0 if previous is None else laps_down(interval, lap_seconds[index - 1]),
                "Compound": lap.Compound,
                "TyreLife": lap.TyreLife,
                "Stint": lap.Stint,
                "Stops": int(stops[lap.Driver]),
                "InPitLane": bool(lap.PitInLap or lap.PitOutLap),
                "RecentPace": float(recent.mean()) if len(recent) else np.nan,
                "RecentSD": float(recent.std(ddof=1)) if len(recent) > 1 else np.nan,
                "RecentLaps": len(recent),
                "PaceFromStint": from_stint,
                "Slope": slope,
                "SlopeLaps": slope_laps,
            }
        )
        previous = lap
    return pd.DataFrame(rows, columns=COLUMNS)


def project_clock(state: pd.DataFrame, laps_ahead: int) -> pd.Series:
    """Seconds after the leader finished lap N at which each car finishes lap N + k.

    Lap j ahead takes RecentPace + Slope × j, so wear builds up as the stint
    goes on. Missing pace takes the field median and a missing slope is zero.
    """
    pace = state["RecentPace"].fillna(state["RecentPace"].median())
    pace = pace.fillna(0.0)  # Nobody has a clean lap yet: keep gaps as they are.
    slope = state["Slope"].fillna(0.0)
    k = laps_ahead
    clock = state["GapToLeader"] + k * pace + slope * k * (k + 1) / 2
    return pd.Series(clock.to_numpy(float), index=state["Driver"].to_numpy())


def recent_laps_at(laps: pd.DataFrame, lap_n: int, driver: str, window: int = config.RECENT_LAPS) -> np.ndarray:
    """A driver's recent clean lap times by the end of lap N, from the tyres they had on at lap N."""
    seen = laps_up_to(laps, lap_n)
    at_n = seen[(seen["Driver"] == driver) & (seen["LapNumber"] == lap_n)]
    stint = at_n["Stint"].iloc[0] if not at_n.empty else None
    return recent_laps(clean_laps(seen), driver, stint, window)[0]
