import numpy as np
import pandas as pd
import pytest

from f1dash import config
from f1dash.models import pace


def field_laps():
    """Five drivers with a known fuel-corrected mean pace and lap-to-lap spread."""
    rng = np.random.default_rng(2)
    rows = []
    for i, mean in enumerate([80.0, 80.5, 81.0, 81.5, 82.0]):
        noise = rng.normal(0, 0.4, 30)
        noise = noise - noise.mean()
        for lap, value in enumerate(noise, start=2):
            raw_time = mean + value - config.FUEL_EFFECT * lap
            rows.append({"Driver": f"D{i}", "LapNumber": float(lap), "LapTimeSec": raw_time})
    return pd.DataFrame(rows)


def test_regression_prior_comes_from_the_field_not_early_laps():
    prior = pace.field_prior(field_laps())
    assert abs(prior.mean - 81.0) < 1e-6
    assert abs(prior.sd - np.std([80, 80.5, 81, 81.5, 82], ddof=1)) < 1e-6
    assert prior.sd > 0.5  # A 3-lap prior could be near zero and swamp the data.


def test_pooled_sd_is_the_within_driver_spread():
    laps = field_laps()
    assert abs(pace.pooled_sd(laps) - 0.4) < 0.05
    # The 2-second spread between drivers must not leak into it.
    assert pace.pooled_sd(laps) < laps["LapTimeSec"].std() / 1.5


def test_posterior_matches_the_normal_normal_formula():
    prior = pace.Normal(81.0, 0.8)
    post = pace.posterior(prior, sample_mean=80.0, n=4, sigma=0.4)
    precision = 1 / 0.8**2 + 4 / 0.4**2
    assert post.sd == pytest.approx(np.sqrt(1 / precision))
    assert post.mean == pytest.approx((81.0 / 0.8**2 + 80.0 * 4 / 0.4**2) / precision)


def test_posterior_tends_to_the_data_as_laps_accumulate():
    prior = pace.Normal(81.0, 0.8)
    few = pace.posterior(prior, 80.0, 1, 0.4)
    many = pace.posterior(prior, 80.0, 40, 0.4)
    assert abs(many.mean - 80.0) < abs(few.mean - 80.0)
    assert many.sd < few.sd


def test_regression_all_three_curves_describe_the_mean():
    curves = pace.curves(pace.Normal(81.0, 0.8), sample_mean=80.0, n=16, sigma=0.4)
    # Likelihood of the mean has SD σ/√n, not the lap-to-lap σ.
    assert curves.likelihood.sd == pytest.approx(0.4 / 4)
    assert curves.posterior.sd < min(curves.prior.sd, curves.likelihood.sd)


def test_regression_x_range_comes_from_the_sds():
    narrow = pace.curves(pace.Normal(81.0, 0.05), 81.0, 100, 0.1)
    wide = pace.curves(pace.Normal(81.0, 5.0), 81.0, 1, 3.0)
    assert narrow.x[-1] - narrow.x[0] < 2
    assert wide.x[-1] - wide.x[0] > 20
    assert len(narrow.prior_pdf) == len(narrow.x)


def test_too_small_a_field_gives_no_prior():
    laps = field_laps()
    assert pace.field_prior(laps[laps["Driver"] == "D0"]) is None
