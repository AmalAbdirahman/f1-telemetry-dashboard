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
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SD_RANGE = 4


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
