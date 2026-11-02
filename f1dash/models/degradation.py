"""Tyre degradation: fuel correction and a robust fit per driver and stint.

Assumptions:
- Pace falls linearly with tyre age over a stint. A cliff is not modelled.
- Fuel burn is a constant FUEL_EFFECT seconds per lap. Lap number and tyre age
  rise together within a stint, so this constant adds directly to every slope.
  A wrong fuel figure shifts all slopes by the same amount.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from f1dash import config

HUBER_K = 1.345  # Standard tuning constant for 95% efficiency on normal data.
IRLS_ITERATIONS = 8


@dataclass(frozen=True)
class StintFit:
    slope: float  # Seconds lost per lap of tyre age.
    intercept: float
    r2: float
    rmse: float
    n: int
    ci_low: float  # 90% bootstrap interval for the slope.
    ci_high: float


def fuel_corrected(laps: pd.DataFrame) -> pd.Series:
    """Lap time with the fuel effect removed.

    Cars speed up as fuel burns, so adding FUEL_EFFECT × lap number cancels that
    speed-up and leaves the tyre's contribution.
    """
    return laps["LapTimeSec"] + config.FUEL_EFFECT * laps["LapNumber"]


def _huber_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Huber slope by iteratively reweighted least squares.

    A fast stand-in for statsmodels' RLM inside the bootstrap, where hundreds of
    fits per stint would otherwise take seconds.
    """
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(IRLS_ITERATIONS):
        residuals = y - design @ beta
        scale = np.median(np.abs(residuals)) / 0.6745
        if scale <= 0:
            break
        size = np.abs(residuals) / scale
        weights = np.where(size <= HUBER_K, 1.0, HUBER_K / np.maximum(size, 1e-12))
        weighted = design * weights[:, None]
        beta = np.linalg.solve(design.T @ weighted, weighted.T @ y)
    return float(beta[1])


def _bootstrap_slope_ci(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(config.BOOTSTRAP_RESAMPLES):
        picks = rng.integers(0, len(x), len(x))
        if np.unique(x[picks]).size > 1:
            slopes.append(_huber_slope(x[picks], y[picks]))
    low, high = np.percentile(slopes, [5, 95])
    return float(low), float(high)


def fit_stint(tyre_life, pace, seed: int = config.SEED) -> StintFit | None:
    """Fit pace against tyre age. Returns None when the stint is too short."""
    x = np.asarray(tyre_life, dtype=float)
    y = np.asarray(pace, dtype=float)
    if len(x) < config.MIN_STINT_LAPS or np.unique(x).size < 2:
        return None
    result = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    intercept, slope = float(result.params[0]), float(result.params[1])
    residuals = y - (intercept + slope * x)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((residuals**2).sum()) / total if total > 0 else 0.0
    ci_low, ci_high = _bootstrap_slope_ci(x, y, seed)
    return StintFit(slope, intercept, r2, float(np.sqrt((residuals**2).mean())), len(x), ci_low, ci_high)


def fit_all_stints(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """Fit every driver's every stint. Stints that are too short are left out."""
    columns = ["Driver", "Stint", "Compound", "Slope", "CILow", "CIHigh", "R2", "RMSE", "Laps", "Intercept"]
    rows = []
    for (driver, stint), group in clean_laps.groupby(["Driver", "Stint"]):
        fit = fit_stint(group["TyreLife"], fuel_corrected(group))
        if fit is None:
            continue
        rows.append(
            {
                "Driver": driver,
                "Stint": stint,
                "Compound": group["Compound"].mode().iloc[0],
                "Slope": fit.slope,
                "CILow": fit.ci_low,
                "CIHigh": fit.ci_high,
                "R2": fit.r2,
                "RMSE": fit.rmse,
                "Laps": fit.n,
                "Intercept": fit.intercept,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def laps_to_lose(slope: float, seconds: float) -> float | None:
    """Linear heuristic: laps until wear costs ``seconds``. Not a tyre cliff."""
    return seconds / slope if slope > 0 else None
