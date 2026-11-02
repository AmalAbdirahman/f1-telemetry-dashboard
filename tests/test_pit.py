import numpy as np
import pandas as pd
import pytest

from f1dash import data
from f1dash.models import pit


def laps_with_stop(driver, stop_lap, in_time, out_time, status="1", median=80.0):
    rows = []
    for lap in range(2, 31):
        rows.append(
            {
                "Driver": driver,
                "Stint": 1.0 if lap <= stop_lap else 2.0,
                "LapNumber": float(lap),
                "LapTimeSec": median,
                "TrackStatus": "1",
                "PitInLap": lap == stop_lap,
                "PitOutLap": lap == stop_lap + 1,
                "IsAccurate": True,
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[frame["LapNumber"] == stop_lap, ["LapTimeSec", "TrackStatus"]] = [in_time, status]
    frame.loc[frame["LapNumber"] == stop_lap + 1, ["LapTimeSec", "TrackStatus"]] = [out_time, status]
    return frame


def estimate(frame):
    return pit.estimate_pit_loss(frame, data.clean_laps(frame))


def test_pit_loss_is_in_plus_out_minus_two_median_laps():
    result = estimate(laps_with_stop("VER", 10, 84.0, 98.0))
    assert result.mean == 22
    assert result.n == 1
    assert result.sd is None


def test_loss_is_pooled_over_all_stops():
    frame = pd.concat([laps_with_stop("VER", 10, 84, 98), laps_with_stop("LEC", 12, 85, 99)], ignore_index=True)
    result = estimate(frame)
    assert result.mean == 23
    assert result.n == 2
    assert result.sd == pytest.approx(np.std([22, 24], ddof=1))


def test_stops_under_safety_car_or_red_flag_are_excluded():
    # A red-flag "stop" lasting 2,300 s must not touch the estimate.
    frame = pd.concat(
        [
            laps_with_stop("VER", 10, 84, 98),
            laps_with_stop("LEC", 12, 139, 2433, status="245"),
            laps_with_stop("NOR", 14, 110, 120, status="4"),
        ],
        ignore_index=True,
    )
    result = estimate(frame)
    assert result.n == 1 and result.mean == 22


def test_no_usable_stops_gives_no_estimate():
    frame = laps_with_stop("VER", 10, 84, 98, status="4")
    assert estimate(frame) is None


def test_regression_total_sigma_adds_the_phases_as_variances():
    assert pit.total_sigma(0.3, 0.4) == pytest.approx(0.5)
    assert pit.total_sigma(0.2, 0.3, 0.4, 0.5) == pytest.approx(np.sqrt(0.04 + 0.09 + 0.16 + 0.25))


def test_p_safe_is_half_when_the_mean_rejoin_is_on_the_clearance():
    assert pit.p_safe(gap=22, loss=20, sigma=0.5, clearance=2) == pytest.approx(0.5)


def test_clearance_makes_a_rejoin_less_safe():
    assert pit.p_safe(22, 20, 0.5, clearance=1.5) < pit.p_safe(22, 20, 0.5, clearance=0)


def test_larger_gap_is_safer():
    assert pit.p_safe(25, 20, 0.5) > pit.p_safe(20.5, 20, 0.5)


def test_zero_sigma_is_a_clean_yes_or_no():
    assert pit.p_safe(25, 20, 0) == 1
    assert pit.p_safe(15, 20, 0) == 0
