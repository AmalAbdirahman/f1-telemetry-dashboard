import numpy as np

from f1dash import config
from f1dash.models import overtake


def probability(chaser_pace, leader_pace=90.0, gap=5.0, laps=20, sigma=0.3, **kwargs):
    gaps = overtake.simulate_gaps(gap, chaser_pace, sigma, leader_pace, sigma, laps, **kwargs)
    return overtake.summarise(gaps)


def test_regression_faster_chaser_catches_and_slower_does_not():
    faster = probability(89.5)
    slower = probability(90.5)
    assert faster.p_any > 0.95
    assert slower.p_any < 0.05
    assert faster.p_any > slower.p_any


def test_regression_same_seed_gives_the_same_answer():
    assert probability(89.8, seed=7).p_any == probability(89.8, seed=7).p_any
    assert probability(89.8, seed=7).p_any != probability(89.8, seed=8).p_any


def test_regression_band_is_labelled_with_its_true_coverage():
    gaps = overtake.simulate_gaps(5, 90, 0.3, 90, 0.3, 10)
    summary = overtake.summarise(gaps)
    assert overtake.BAND_COVERAGE == 90
    assert ((gaps >= summary.band_low) & (gaps <= summary.band_high)).mean() > 0.88


def test_gap_mean_follows_the_pace_difference():
    gaps = overtake.simulate_gaps(5, 89.5, 0.3, 90, 0.3, 20, n_sims=20000)
    assert abs(gaps[:, -1].mean() - (5 - 0.5 * 20)) < 0.1


def test_chaser_ahead_means_negative_gap_and_probabilities_are_ordered():
    summary = probability(89.75)
    assert summary.p_any >= summary.p_flag
    assert summary.median_catch_lap is not None
    assert 1 <= summary.median_catch_lap <= 20


def test_no_catch_gives_no_median_lap():
    assert probability(95.0).median_catch_lap is None


def test_degradation_trend_is_added_to_future_laps():
    flat = overtake.simulate_gaps(5, 90, 0.3, 90, 0.3, 20, n_sims=20000)
    worn_leader = overtake.simulate_gaps(5, 90, 0.3, 90, 0.3, 20, slope_leader=0.05, n_sims=20000)
    # The leader loses 0.05 × k on each future lap k = 1..20.
    expected = -0.05 * sum(range(1, 21))
    assert abs(worn_leader[:, -1].mean() - flat[:, -1].mean() - expected) < 0.15


def test_correlated_laps_spread_the_outcome_but_keep_each_lap_variance():
    independent = overtake.simulate_gaps(5, 90, 0.3, 90, 0.3, 20, rho=0.0, n_sims=20000)
    correlated = overtake.simulate_gaps(5, 90, 0.3, 90, 0.3, 20, rho=0.7, n_sims=20000)
    assert correlated[:, -1].std() > 1.5 * independent[:, -1].std()
    single_lap = np.diff(correlated, axis=1)[:, -1]
    assert abs(single_lap.std() - np.sqrt(2) * 0.3) < 0.02


def test_rho_is_estimated_from_detrended_laps_and_clipped():
    rng = np.random.default_rng(0)
    trend = 0.5 * np.arange(10)
    # A trend alone looks strongly correlated unless it is removed first.
    trended = 90 + trend + rng.normal(0, 0.01, 10)
    assert np.corrcoef(trended[:-1], trended[1:])[0, 1] > 0.99
    assert overtake.estimate_rho(trended) < 0.5
    assert overtake.estimate_rho(90 + trend) == 0
    alternating = 90 + np.array([1, -1] * 5, dtype=float)
    assert overtake.estimate_rho(alternating) == 0
    drifting = 90 + np.array([0, 1, 2, 3, 3, 2, 1, 0, 1, 2], dtype=float) * 2
    assert 0 < overtake.estimate_rho(drifting) <= config.RHO_MAX
    assert overtake.estimate_rho(np.array([90.0, 90.1])) == 0


def test_recent_form_uses_at_most_the_last_ten_laps():
    times = np.array([100.0] * 5 + [90.0, 90.2] * 5)
    form = overtake.recent_form(times)
    assert form.n == config.RECENT_LAPS
    assert abs(form.mean - 90.1) < 1e-9
    assert form.sd > 0


def test_short_history_reports_how_many_laps_were_used():
    assert overtake.recent_form(np.array([90.0, 90.1, 90.2])).n == 3
