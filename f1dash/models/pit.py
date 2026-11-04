"""Pit loss estimated from the session, and the chance of a safe rejoin.

Loss for one green-flag stop = (in-lap + out-lap) − 2 × the driver's median
clean lap. Safety-car stops are measured separately because they really are
cheaper: the field is slow while the pit lane is not. Against the driver's
normal lap they would look dearer, since both laps are slow anyway, so a
safety-car stop is measured against the median of the other cars on the same
two laps. Red-flag "stops" last many minutes and are never used.

Pit loss L ~ Normal(μ, σ²), and P_safe = P(gap − L > clearance), where the
clearance approximates the margin needed to stay out of dirty air.

The spread across stops already contains the entry, stationary and launch
phases and the data cannot separate them. So σ² is either that observed spread
plus rival pace variance, or a manual sum of assumed phase variances. Adding the
phases to the observed spread would count them twice.

Rejoin projector: at the end of lap N, the earliest possible in-lap is N + 1.
For an in-lap L, the loss (which covers in- and out-lap) is added to the car's
time at the line after its out-lap, L + 1, and the field is re-ranked there.
Laps after N are projected from lap N (``race_state.project_clock``), so stops
other cars make later are unknown, as they would be on the pit wall.

The rejoin is judged on the road, not in race order. A lapped car can be just
in front on track while behind in the race, so the time gap to every car is
wrapped into half a lap either side: a car is (gap mod lap time) ahead. It is
clear when the nearest car in front is more than the clearance away (dirty
air), and so is the nearest car behind, which would otherwise have a good
chance of passing on the out-lap while the new tyres are still cold.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from f1dash.models import race_state
from f1dash.models.filters import clean_laps

SD_RANGE = 4
WINDOW_LAPS = 11  # In-laps N + 1 to N + 11.
SAFETY_CAR, VSC, RED = "4", "6", "5"  # Track status codes, as in data.py.


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


def _neutralised(status: pd.Series) -> pd.Series:
    """Laps under a safety car or VSC, but never a red flag."""
    under = status.str.contains(SAFETY_CAR, regex=False) | status.str.contains(VSC, regex=False)
    return under & ~status.str.contains(RED, regex=False)


def estimate_neutralised_pit_loss(laps: pd.DataFrame) -> PitLoss | None:
    """Pool the loss over stops whose in- and out-lap were both under a SC or VSC.

    Loss = (in-lap + out-lap) − the median of the other cars' laps on those two
    lap numbers, so the slow safety-car laps cancel and the pit lane is left.
    """
    racing = laps[~laps["PitInLap"] & ~laps["PitOutLap"] & laps["LapTimeSec"].notna()]
    field = racing.groupby("LapNumber")["LapTimeSec"].median()
    under = laps[_neutralised(laps["TrackStatus"])]
    in_laps = under[under["PitInLap"]][["Driver", "LapNumber", "LapTimeSec"]]
    out_laps = under[under["PitOutLap"]][["Driver", "LapNumber", "LapTimeSec"]]
    out_laps = out_laps.assign(LapNumber=out_laps["LapNumber"] - 1)
    stops = in_laps.merge(out_laps, on=["Driver", "LapNumber"], suffixes=("In", "Out"))
    reference = stops["LapNumber"].map(field) + (stops["LapNumber"] + 1).map(field)
    losses = (stops["LapTimeSecIn"] + stops["LapTimeSecOut"] - reference).dropna()
    if losses.empty:
        return None
    return PitLoss(float(losses.mean()), float(losses.std(ddof=1)) if len(losses) > 1 else None, len(losses))


def loss_known_at(laps: pd.DataFrame, lap_n: int, neutralised: bool = False) -> tuple[PitLoss | None, bool]:
    """Pit loss from stops completed by lap N, or else from the whole race.

    ``neutralised`` asks for safety-car stops instead of green-flag ones.
    Returns (estimate, True) when stops before lap N were enough, and
    (estimate, False) for the whole-race fallback, which uses later laps.
    """

    def measure(frame):
        if neutralised:
            return estimate_neutralised_pit_loss(frame)
        return estimate_pit_loss(frame, clean_laps(frame))

    early = measure(race_state.laps_up_to(laps, lap_n))
    if early is not None:
        return early, True
    return measure(laps), False


@dataclass(frozen=True)
class Rejoin:
    position: int
    ahead: str | None  # Car ahead in race order, and the gap to it (s).
    gap_ahead: float
    behind: str | None
    gap_behind: float
    road_ahead: str | None  # Nearest car in front on track, lapped or not.
    road_gap_ahead: float
    road_behind: str | None  # Nearest car behind on track.
    road_gap_behind: float
    clean_air: bool  # Nothing within the clearance in front.
    under_pressure: bool  # A car within the clearance behind, while the new tyres are cold.
    clear: bool  # Both of the above are fine.


def _road_gaps(clock: pd.Series, moment: float, lap_time: float) -> pd.Series:
    """How far each car is in front on track (s), wrapped to within half a lap. Negative is behind."""
    delta = moment - clock
    if not np.isfinite(lap_time) or lap_time <= 0:
        return delta
    return (delta + lap_time / 2) % lap_time - lap_time / 2


def _nearest(gaps: pd.Series) -> tuple[str | None, float]:
    if gaps.empty:
        return None, np.nan
    return gaps.idxmin(), float(gaps.min())


def rejoin(clock: pd.Series, driver: str, loss: float, lap_time: float, clearance: float) -> Rejoin:
    """Where ``driver`` comes out if ``loss`` seconds are added to their time at the line.

    ``clock`` holds every car's time at the line for the same lap.
    """
    moment = float(clock[driver]) + loss
    others = clock.drop(driver)
    in_front, behind = others[others < moment], others[others >= moment]
    road = _road_gaps(others, moment, lap_time)
    road_ahead, road_gap_ahead = _nearest(road[road > 0])
    road_behind, road_gap_behind = _nearest(-road[road <= 0])
    clean_air = road_ahead is None or road_gap_ahead > clearance
    under_pressure = road_behind is not None and road_gap_behind <= clearance
    return Rejoin(
        position=len(in_front) + 1,
        ahead=in_front.idxmax() if not in_front.empty else None,
        gap_ahead=float(moment - in_front.max()) if not in_front.empty else np.nan,
        behind=behind.idxmin() if not behind.empty else None,
        gap_behind=float(behind.min() - moment) if not behind.empty else np.nan,
        road_ahead=road_ahead,
        road_gap_ahead=road_gap_ahead,
        road_behind=road_behind,
        road_gap_behind=road_gap_behind,
        clean_air=clean_air,
        under_pressure=under_pressure,
        clear=clean_air and not under_pressure,
    )


WINDOW_COLUMNS = [
    "PitLap", "Loss", "Position", "Ahead", "GapAhead", "Behind", "GapBehind", "RoadAhead", "RoadGapAhead",
    "RoadBehind", "RoadGapBehind", "CleanAir", "UnderPressure", "Clear",
]


def pit_window(
    state: pd.DataFrame,
    driver: str,
    loss: float,
    clearance: float,
    lap_n: int,
    last_lap: int,
    later_loss: float | None = None,
) -> pd.DataFrame:
    """The rejoin for an in-lap on each of laps N + 1 to N + 11, judged after the out-lap.

    ``later_loss`` is for a cheap safety-car stop: ``loss`` then applies to lap
    N + 1 only, because nobody knows at lap N how long the safety car stays
    out, and every later lap costs ``later_loss``.
    """
    lap_time = float(state["RecentPace"].median())
    rows = []
    for pit_lap in range(lap_n + 1, lap_n + WINDOW_LAPS + 1):
        if pit_lap + 1 > last_lap:
            break
        this_loss = loss if later_loss is None or pit_lap == lap_n + 1 else later_loss
        clock = race_state.project_clock(state, pit_lap + 1 - lap_n)
        result = rejoin(clock, driver, this_loss, lap_time, clearance)
        rows.append([pit_lap, this_loss, *result.__dict__.values()])
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


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
