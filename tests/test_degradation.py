import numpy as np
import pandas as pd
import pytest

from f1dash import config
from f1dash.models import degradation


def stint(driver, number, compound, n, slope, base=80.0, first_lap=2, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    life = np.arange(1, n + 1, dtype=float)
    laps = first_lap + life - 1
    # Build raw lap times so that fuel-corrected pace has exactly the slope wanted.
    times = base + slope * life - config.FUEL_EFFECT * laps + rng.normal(0, noise, n)
    return pd.DataFrame(
        {
            "Driver": driver,
            "Stint": float(number),
            "Compound": compound,
            "TyreLife": life,
            "LapNumber": laps,
            "LapTimeSec": times,
        }
    )


def test_fuel_correction_adds_time_for_later_laps():
    laps = pd.DataFrame({"LapTimeSec": [80.0, 80.0], "LapNumber": [1.0, 11.0]})
    corrected = degradation.fuel_corrected(laps)
    assert corrected.iloc[1] - corrected.iloc[0] == pytest.approx(10 * config.FUEL_EFFECT)


def test_fit_recovers_a_known_slope():
    laps = stint("VER", 1, "SOFT", 25, 0.08)
    fit = degradation.fit_stint(laps["TyreLife"], degradation.fuel_corrected(laps))
    assert abs(fit.slope - 0.08) < 0.01
    assert fit.ci_low < 0.08 < fit.ci_high
    assert fit.n == 25


def test_fit_resists_an_outlier():
    laps = stint("VER", 1, "SOFT", 25, 0.05)
    laps.loc[20, "LapTimeSec"] += 6
    fit = degradation.fit_stint(laps["TyreLife"], degradation.fuel_corrected(laps))
    assert abs(fit.slope - 0.05) < 0.015


def test_regression_r2_is_the_fits_r2_not_a_correlation():
    laps = stint("VER", 1, "SOFT", 25, 0.05)
    laps.loc[24, "LapTimeSec"] += 8
    x, y = laps["TyreLife"], degradation.fuel_corrected(laps)
    fit = degradation.fit_stint(x, y)
    residuals = y - (fit.intercept + fit.slope * x)
    assert abs(fit.r2 - (1 - (residuals**2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-9
    assert abs(fit.rmse - np.sqrt((residuals**2).mean())) < 1e-9
    assert abs(fit.r2 - np.corrcoef(x, y)[0, 1] ** 2) > 1e-4


def test_bootstrap_is_seeded():
    laps = stint("VER", 1, "SOFT", 15, 0.05, noise=0.2)
    first = degradation.fit_stint(laps["TyreLife"], degradation.fuel_corrected(laps))
    second = degradation.fit_stint(laps["TyreLife"], degradation.fuel_corrected(laps))
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_too_few_laps_or_one_tyre_age_gives_no_fit():
    laps = stint("VER", 1, "SOFT", 5, 0.05)
    assert degradation.fit_stint(laps["TyreLife"], laps["LapTimeSec"]) is None
    assert degradation.fit_stint(pd.Series([3.0] * 8), pd.Series(np.arange(8.0))) is None


def test_regression_each_driver_and_stint_gets_its_own_fit():
    laps = pd.concat(
        [
            stint("VER", 1, "MEDIUM", 20, 0.02, first_lap=2),
            stint("VER", 2, "HARD", 20, 0.10, first_lap=23),
            stint("LEC", 1, "MEDIUM", 20, 0.06, first_lap=2),
        ],
        ignore_index=True,
    )
    fits = degradation.fit_all_stints(laps).set_index(["Driver", "Stint"])
    assert len(fits) == 3
    assert abs(fits.loc[("VER", 1.0), "Slope"] - 0.02) < 0.01
    assert abs(fits.loc[("VER", 2.0), "Slope"] - 0.10) < 0.01
    assert abs(fits.loc[("LEC", 1.0), "Slope"] - 0.06) < 0.01
    assert fits.loc[("VER", 2.0), "Compound"] == "HARD"


def test_laps_to_lose_is_undefined_without_wear():
    assert degradation.laps_to_lose(0.1, 1.5) == 15
    assert degradation.laps_to_lose(-0.1, 1.5) is None
    assert degradation.laps_to_lose(0.0, 1.5) is None
