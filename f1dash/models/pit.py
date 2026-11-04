"""Pit loss estimated from the session, and the chance of a safe rejoin.

Loss for one stop = (in-lap + out-lap) − 2 × the driver's median clean lap. Only
green-flag stops count: a safety car makes a stop look cheap, and a red-flag
"stop" lasts many minutes.

Pit loss L ~ Normal(μ, σ²), and P_safe = P(gap − L > clearance), where the
clearance approximates the margin needed to stay out of dirty air.

The spread across stops already contains the entry, stationary and launch
phases and the data cannot separate them. So σ² is either that observed spread
plus rival pace variance, or a manual sum of assumed phase variances. Adding the
phases to the observed spread would count them twice.

Rejoin projector: if a car stops on lap L, its time at the line plus the pit
loss is re-ranked against everyone else's time at the end of lap L. Laps after
N are projected from lap N (``race_state.project_clock``), so stops other cars
make later are unknown, as they would be on the pit wall.

Clean air is judged on the road, not in race order. A lapped car can be just
in front on track while behind in the race, so the time gap to every car is
wrapped into half a lap either side: a car is (gap mod lap time) ahead.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from f1dash.models import race_state
from f1dash.models.filters import clean_laps

SD_RANGE = 4
WINDOW_LAPS = 10  # The pit window covers lap N to N + 10.


@dataclass(frozen=True)
class PitLoss:
    mean: float
    sd: float | None  # None with a single stop.
    n: int


def estimate_pit_loss(laps: pd.DataFrame, clean_laps: pd.DataFrame) -> PitLoss | None:
    """Pool the loss over all green-flag stops, or None if there are none."""
    medians = clean_laps.groupby("Driver")["LapTimeSec"].median().rename("Median")
    green = laps[laps["TrackStatus"] == "1"]
    in_laps = green[green["PitInLap"]][["Driver", "LapNumber", "LapTimeSec"]]
    out_laps = green[green["PitOutLap"]][["Driver", "LapNumber", "LapTimeSec"]]
    out_laps = out_laps.assign(LapNumber=out_laps["LapNumber"] - 1)
    stops = in_laps.merge(out_laps, on=["Driver", "LapNumber"], suffixes=("In", "Out")).join(medians, on="Driver")
    losses = (stops["LapTimeSecIn"] + stops["LapTimeSecOut"] - 2 * stops["Median"]).dropna()
    if losses.empty:
        return None
    return PitLoss(float(losses.mean()), float(losses.std(ddof=1)) if len(losses) > 1 else None, len(losses))


def loss_known_at(laps: pd.DataFrame, lap_n: int) -> tuple[PitLoss | None, bool]:
    """Pit loss from stops completed by lap N, or else from the whole race.

    Returns (estimate, True) when stops before lap N were enough, and
    (estimate, False) for the whole-race fallback, which uses later laps.
    """
    seen = race_state.laps_up_to(laps, lap_n)
    early = estimate_pit_loss(seen, clean_laps(seen))
    if early is not None:
        return early, True
    return estimate_pit_loss(laps, clean_laps(laps)), False


@dataclass(frozen=True)
class Rejoin:
    position: int
    ahead: str | None  # Car ahead in race order, and the gap to it (s).
    gap_ahead: float
    behind: str | None
    gap_behind: float
    road_ahead: str | None  # Nearest car in front on track, lapped or not.
    road_gap_ahead: float
    clean_air: bool


def _road_gaps(clock: pd.Series, moment: float, lap_time: float) -> pd.Series:
    """How far each car is in front on track (s), wrapped to within half a lap."""
    delta = moment - clock
    if not np.isfinite(lap_time) or lap_time <= 0:
        return delta
    return (delta + lap_time / 2) % lap_time - lap_time / 2


def rejoin(clock: pd.Series, driver: str, loss: float, lap_time: float, clearance: float) -> Rejoin:
    """Where ``driver`` comes out if ``loss`` seconds are added to their time at the line.

    ``clock`` holds every car's time at the line for the same lap.
    """
    moment = float(clock[driver]) + loss
    others = clock.drop(driver)
    in_front, behind = others[others < moment], others[others >= moment]
    road = _road_gaps(others, moment, lap_time)
    road_in_front = road[road > 0]
    nearest = road_in_front.idxmin() if not road_in_front.empty else None
    road_gap = float(road_in_front.min()) if nearest is not None else np.nan
    return Rejoin(
        position=len(in_front) + 1,
        ahead=in_front.idxmax() if not in_front.empty else None,
        gap_ahead=float(moment - in_front.max()) if not in_front.empty else np.nan,
        behind=behind.idxmin() if not behind.empty else None,
        gap_behind=float(behind.min() - moment) if not behind.empty else np.nan,
        road_ahead=nearest,
        road_gap_ahead=road_gap,
        clean_air=nearest is None or road_gap > clearance,
    )


def pit_window(
    state: pd.DataFrame, driver: str, loss: float, clearance: float, lap_n: int, last_lap: int
) -> pd.DataFrame:
    """The rejoin for a stop on each lap from N to N + 10, never on the final lap."""
    lap_time = float(state["RecentPace"].median())
    rows = []
    for k in range(0, WINDOW_LAPS + 1):
        if lap_n + k >= last_lap:
            break
        result = rejoin(race_state.project_clock(state, k), driver, loss, lap_time, clearance)
        rows.append({"PitLap": lap_n + k, **result.__dict__})
    columns = ["PitLap", "Position", "Ahead", "GapAhead", "Behind", "GapBehind", "RoadAhead", "RoadGapAhead", "CleanAir"]
    names = dict(zip(["PitLap", *Rejoin.__dataclass_fields__], columns))
    return pd.DataFrame(rows).rename(columns=names).reindex(columns=columns)


def total_sigma(*sigmas: float) -> float:
    """Combine independent sources of spread: variances add."""
    return float(np.sqrt(sum(s**2 for s in sigmas)))


def p_safe(gap: float, loss: float, sigma: float, clearance: float = 0.0) -> float:
    """Probability of rejoining with more than ``clearance`` seconds in hand."""
    margin = gap - loss - clearance
    if sigma <= 0:
        return float(margin > 0)
    return float(stats.norm.cdf(margin / sigma))


def rejoin_curve(gap: float, loss: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Density of the gap after the stop, for plotting."""
    mean = gap - loss
    spread = max(sigma, 0.01)
    x = np.linspace(mean - SD_RANGE * spread, mean + SD_RANGE * spread, 400)
    return x, stats.norm.pdf(x, mean, spread)
