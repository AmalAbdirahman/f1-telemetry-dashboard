"""Monte Carlo simulation of the gap between a chasing and a leading car.

gap_n = gap_0 + Σ (chaser_lap_i − leader_lap_i), so the chaser is ahead once the
gap is negative.

Assumptions:
- Each driver's future laps are their recent mean plus their tyre trend.
- Deviations follow an AR(1) process, so good and bad laps come in runs, while
  each individual lap keeps the driver's recent spread.
- Being ahead on the clock is not the same as passing. Dirty air, defending and
  the overtaking aid of the era (DRS before 2026, the new overtake mode from
  2026) are not modelled, and neither are pit stops or safety cars.
"""

from dataclasses import dataclass

import numpy as np

from f1dash import config

BAND_COVERAGE = 90  # The fan chart plots the 5th to 95th percentiles.
MIN_LAPS_FOR_RHO = 4


@dataclass(frozen=True)
class RecentForm:
    mean: float
    sd: float
    n: int


@dataclass(frozen=True)
class GapSummary:
    p_any: float  # P(chaser ahead at any point).
    p_flag: float  # P(chaser ahead at the final lap).
    median_catch_lap: int | None  # Median first lap ahead, among runs that got ahead.
    band_low: np.ndarray
    median_gap: np.ndarray
    band_high: np.ndarray


def recent_form(lap_times, window: int = config.RECENT_LAPS) -> RecentForm:
    """Mean and SD of up to the last ``window`` laps, oldest first."""
    recent = np.asarray(lap_times, dtype=float)[-window:]
    return RecentForm(float(recent.mean()), float(recent.std(ddof=1)) if len(recent) > 1 else 0.0, len(recent))


def estimate_rho(lap_times) -> float:
    """Lag-1 autocorrelation of detrended laps, clipped to [0, RHO_MAX]."""
    times = np.asarray(lap_times, dtype=float)
    if len(times) < MIN_LAPS_FOR_RHO:
        return 0.0
    index = np.arange(len(times))
    residuals = times - np.polyval(np.polyfit(index, times, 1), index)
    if np.allclose(residuals, 0):
        return 0.0
    rho = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    return float(np.clip(np.nan_to_num(rho), 0, config.RHO_MAX))


def _deviations(sd: float, n_sims: int, n_laps: int, rho: float, rng) -> np.ndarray:
    """AR(1) deviations whose stationary SD is ``sd``."""
    shocks = rng.standard_normal((n_sims, n_laps)) * sd
    deviations = np.empty((n_sims, n_laps))
    deviations[:, 0] = shocks[:, 0]
    innovation = np.sqrt(1 - rho**2)
    for k in range(1, n_laps):
        deviations[:, k] = rho * deviations[:, k - 1] + innovation * shocks[:, k]
    return deviations


def simulate_gaps(
    gap0: float,
    mu_chaser: float,
    sd_chaser: float,
    mu_leader: float,
    sd_leader: float,
    n_laps: int,
    slope_chaser: float = 0.0,
    slope_leader: float = 0.0,
    rho: float = 0.0,
    n_sims: int = config.N_SIMULATIONS,
    seed: int = config.SEED,
) -> np.ndarray:
    """Return simulated gaps with shape (n_sims, n_laps)."""
    rng = np.random.default_rng(seed)
    future = np.arange(1, n_laps + 1)
    chaser = mu_chaser + slope_chaser * future + _deviations(sd_chaser, n_sims, n_laps, rho, rng)
    leader = mu_leader + slope_leader * future + _deviations(sd_leader, n_sims, n_laps, rho, rng)
    return gap0 + np.cumsum(chaser - leader, axis=1)


def summarise(gaps: np.ndarray) -> GapSummary:
    """Turn simulated gaps into probabilities and fan-chart percentiles."""
    ahead = gaps < 0
    caught = ahead.any(axis=1)
    first_lap = ahead.argmax(axis=1)[caught] + 1
    low, median, high = np.percentile(gaps, [5, 50, 95], axis=0)
    return GapSummary(
        p_any=float(caught.mean()),
        p_flag=float(ahead[:, -1].mean()),
        median_catch_lap=int(np.median(first_lap)) if caught.any() else None,
        band_low=low,
        median_gap=median,
        band_high=high,
    )
