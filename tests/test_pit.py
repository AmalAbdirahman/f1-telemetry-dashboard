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


# ── Known at lap N, and the rejoin projector ─────────────────────────────

def test_pit_loss_uses_only_stops_finished_by_lap_n(race):
    # Fake stops: VER in-lap 10, PER 11, LEC 12... Both laps of a stop must be by lap N.
    early, before_n = pit.loss_known_at(race.laps, 12)
    assert before_n and early.n == 2
    whole = pit.estimate_pit_loss(race.laps, data.clean_laps(race.laps))
    assert whole.n > early.n


def test_pit_loss_falls_back_to_the_whole_race_when_nothing_is_known_yet(race):
    estimate, before_n = pit.loss_known_at(race.laps, 10)
    assert not before_n
    assert estimate.n == pit.estimate_pit_loss(race.laps, data.clean_laps(race.laps)).n


def test_rejoin_reranks_with_the_pit_loss():
    clock = pd.Series({"A": 0.0, "X": 5.0, "B": 10.0, "C": 40.0})
    result = pit.rejoin(clock, "X", loss=20.0, lap_time=80.0, clearance=1.0)
    assert result.position == 3
    assert (result.ahead, result.gap_ahead) == ("B", pytest.approx(15.0))
    assert (result.behind, result.gap_behind) == ("C", pytest.approx(15.0))
    assert result.clean_air


def test_rejoining_close_behind_a_car_is_not_clean_air():
    clock = pd.Series({"A": 0.0, "X": 2.0, "B": 21.5})
    result = pit.rejoin(clock, "X", loss=20.0, lap_time=80.0, clearance=1.0)
    assert result.ahead == "B" and result.gap_ahead == pytest.approx(0.5)
    assert not result.clean_air


def test_a_lapped_car_just_ahead_on_the_road_spoils_clean_air():
    # X rejoins 30 s after the leader. LAP, a lap down, finishes the same lap
    # 79.5 s after X: one 80 s lap later, less half a second. So on the road
    # LAP is 0.5 s ahead of X even though X is ahead of it in the race.
    clock = pd.Series({"A": 0.0, "X": 10.0, "LAP": 109.5})
    result = pit.rejoin(clock, "X", loss=20.0, lap_time=80.0, clearance=1.0)
    assert result.position == 2 and result.behind == "LAP"
    assert result.road_ahead == "LAP" and result.road_gap_ahead == pytest.approx(0.5)
    assert not result.clean_air


def test_last_car_rejoins_last_with_nobody_behind():
    clock = pd.Series({"A": 0.0, "X": 5.0})
    result = pit.rejoin(clock, "X", loss=20.0, lap_time=80.0, clearance=1.0)
    assert result.position == 2 and result.behind is None and np.isnan(result.gap_behind)


def test_pit_window_projects_from_lap_n_only():
    state = pd.DataFrame(
        {
            "Driver": ["A", "X", "B"],
            "GapToLeader": [0.0, 10.0, 25.0],
            "RecentPace": [80.0, 80.0, 81.0],
            "Slope": [0.0, 0.0, 0.0],
        }
    )
    window = pit.pit_window(state, "X", loss=20.0, clearance=1.0, lap_n=20, last_lap=27)
    # No stop on the final lap.
    assert window["PitLap"].tolist() == list(range(20, 27))
    # After the stop X is 5 s behind B, and B is 1 s a lap slower.
    assert window["Position"].tolist() == [3, 3, 3, 3, 3, 2, 2]
    assert window["GapAhead"].iloc[0] == pytest.approx(5.0)
    last = window.iloc[-1]
    assert last["Behind"] == "B" and last["GapBehind"] == pytest.approx(1.0)


def test_pit_window_is_at_most_ten_laps():
    state = pd.DataFrame({"Driver": ["A", "X"], "GapToLeader": [0.0, 3.0], "RecentPace": [80.0, 80.0], "Slope": [0.0, 0.0]})
    window = pit.pit_window(state, "X", loss=20.0, clearance=1.0, lap_n=5, last_lap=60)
    assert window["PitLap"].tolist() == list(range(5, 16))
