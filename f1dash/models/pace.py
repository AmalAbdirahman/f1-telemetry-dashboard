"""Empirical-Bayes estimate of a driver's true mean pace.

Prior: the spread of per-driver mean laps across the field in this session.
Likelihood: the driver's laps so far, with σ the pooled within-driver SD.
Posterior: the Normal-Normal conjugate update for the driver's mean.

All three distributions describe the *mean* lap time, which is why the
likelihood's SD is σ/√n. Laps are treated as independent given the mean, and
the field prior includes the driver being estimated. With around 20 drivers
that makes little difference.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from f1dash.models.degradation import fuel_corrected

MIN_DRIVERS = 3
MIN_LAPS_PER_DRIVER = 3
SD_RANGE = 4  # Plot ±4 SD around the outermost curves.
GRID_POINTS = 600


@dataclass(frozen=True)
class Normal:
    mean: float
    sd: float


@dataclass(frozen=True)
class Curves:
    x: np.ndarray
    prior: Normal
    likelihood: Normal
    posterior: Normal
    prior_pdf: np.ndarray
    likelihood_pdf: np.ndarray
    posterior_pdf: np.ndarray


def _by_driver(clean_laps: pd.DataFrame):
    pace = fuel_corrected(clean_laps)
    enough = pace.groupby(clean_laps["Driver"]).transform("size") >= MIN_LAPS_PER_DRIVER
    return pace[enough].groupby(clean_laps["Driver"])


def field_prior(clean_laps: pd.DataFrame) -> Normal | None:
    """Distribution of per-driver mean pace, or None if the field is too small."""
    means = _by_driver(clean_laps).mean()
    if len(means) < MIN_DRIVERS or means.std(ddof=1) <= 0:
        return None
    return Normal(float(means.mean()), float(means.std(ddof=1)))


def pooled_sd(clean_laps: pd.DataFrame) -> float:
    """Lap-to-lap SD within a driver, pooled across the field."""
    grouped = _by_driver(clean_laps)
    variances, counts = grouped.var(ddof=1), grouped.size()
    return float(np.sqrt(((counts - 1) * variances).sum() / (counts - 1).sum()))


def driver_observations(clean_laps: pd.DataFrame, driver: str) -> pd.Series:
    """One driver's fuel-corrected laps in race order."""
    own = clean_laps[clean_laps["Driver"] == driver].sort_values("LapNumber")
    return fuel_corrected(own)


def posterior(prior: Normal, sample_mean: float, n: int, sigma: float) -> Normal:
    """Normal-Normal update of the mean after ``n`` laps."""
    prior_precision = 1 / prior.sd**2
    data_precision = n / sigma**2
    precision = prior_precision + data_precision
    mean = (prior_precision * prior.mean + data_precision * sample_mean) / precision
    return Normal(float(mean), float(np.sqrt(1 / precision)))


def curves(prior: Normal, sample_mean: float, n: int, sigma: float) -> Curves:
    """Densities for the prior, likelihood and posterior of the mean."""
    likelihood = Normal(float(sample_mean), float(sigma / np.sqrt(n)))
    updated = posterior(prior, sample_mean, n, sigma)
    trio = (prior, likelihood, updated)
    low = min(d.mean - SD_RANGE * d.sd for d in trio)
    high = max(d.mean + SD_RANGE * d.sd for d in trio)
    x = np.linspace(low, high, GRID_POINTS)
    pdfs = [stats.norm.pdf(x, d.mean, d.sd) for d in trio]
    return Curves(x, prior, likelihood, updated, *pdfs)
