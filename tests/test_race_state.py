import numpy as np
import pandas as pd
import pytest

from f1dash import config
from f1dash.models import race_state

START = pd.Timestamp("2024-05-26 13:00", tz="UTC")


def make_race(paces: dict[str, float], n_laps: int = 20, stop: dict[str, int] | None = None, pit_loss: float = 20.0):
    """A race where every lap of a driver takes their pace, plus a pit loss at their stop."""
    stop = stop or {}
    rows = []
    for driver, pace in paces.items():
        clock = START
        for lap in range(1, n_laps + 1):
            seconds = pace
            if lap == stop.get(driver):
                seconds += pit_loss / 2
            if lap == stop.get(driver, -9) + 1:
                seconds += pit_loss / 2
            stint = 1 if driver not in stop or lap <= stop[driver] else 2
            first = 1 if stint == 1 else stop[driver] + 1
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapStart": clock,
                    "LapEnd": clock + pd.Timedelta(seconds=seconds),
                    "LapTimeSec": seconds,
                    "Stint": float(stint),
                    "Compound": "MEDIUM" if stint == 1 else "HARD",
                    "TyreLife": float(lap - first + 1),
                    "TrackStatus": "1",
                    "PitInLap": lap == stop.get(driver),
                    "PitOutLap": lap == 1 or lap == stop.get(driver, -9) + 1,
                    "IsAccurate": lap > 1,
                }
            )
            clock += pd.Timedelta(seconds=seconds)
    return pd.DataFrame(rows)


def test_order_and_gaps_come_from_line_crossings():
    laps = make_race({"VER": 80.0, "LEC": 80.5, "NOR": 81.0})
    state = race_state.race_state(laps, 10).set_index("Driver")
    assert state["Position"].to_dict() == {"VER": 1, "LEC": 2, "NOR": 3}
    assert state.loc["LEC", "GapToLeader"] == pytest.approx(5.0)
    assert state.loc["NOR", "GapToLeader"] == pytest.approx(10.0)
    assert state.loc["NOR", "Interval"] == pytest.approx(5.0)
    assert np.isnan(state.loc["VER", "Interval"])


def test_a_lapped_car_is_counted_in_laps_not_seconds():
    # 10 s a lap slower: by lap 10 the leader has finished lap 11 before SLO finishes lap 10.
    laps = make_race({"VER": 80.0, "LEC": 80.5, "SLO": 90.0})
    state = race_state.race_state(laps, 10).set_index("Driver")
    assert state.loc["SLO", "Position"] == 3
    assert state.loc["SLO", "LapsDown"] == 1
    assert state.loc["SLO", "IntervalLapsDown"] == 1
    assert state.loc["LEC", "LapsDown"] == 0


def test_tyres_stops_pace_and_slope_at_lap_n():
    laps = make_race({"VER": 80.0, "LEC": 80.5}, stop={"VER": 8})
    state = race_state.race_state(laps, 14).set_index("Driver")
    assert state.loc["VER", "Compound"] == "HARD" and state.loc["VER", "TyreLife"] == 6
    assert state.loc["VER", "Stops"] == 1 and state.loc["LEC", "Stops"] == 0
    assert state.loc["LEC", "RecentPace"] == pytest.approx(80.5)
    assert state.loc["LEC", "RecentLaps"] == config.RECENT_LAPS
    # Constant raw laps are fuel-corrected into a slope of exactly FUEL_EFFECT.
    assert state.loc["LEC", "Slope"] == pytest.approx(config.FUEL_EFFECT, abs=1e-6)
    # Lap 9 is an out-lap, so by lap 14 VER has 5 clean laps on the new set: too few for a fit.
    assert np.isnan(state.loc["VER", "Slope"])


def test_retired_drivers_are_not_running():
    laps = make_race({"VER": 80.0, "LEC": 80.5, "RUS": 81.0})
    laps = laps[~((laps["Driver"] == "RUS") & (laps["LapNumber"] > 6))]
    assert "RUS" not in set(race_state.race_state(laps, 10)["Driver"])
    assert "RUS" in set(race_state.race_state(laps, 6)["Driver"])


def test_regression_nothing_after_lap_n_is_used(race):
    lap_n = 12
    before = race_state.race_state(race.laps, lap_n)
    changed = race.laps.copy()
    later = changed["LapNumber"] > lap_n
    changed.loc[later, "LapTimeSec"] *= 1.5
    changed.loc[later, "LapEnd"] += pd.Timedelta(seconds=300)
    changed.loc[later, "LapStart"] += pd.Timedelta(seconds=200)
    changed.loc[later, ["TrackStatus", "Compound"]] = ["4", "WET"]
    changed.loc[later, "PitInLap"] = ~changed.loc[later, "PitInLap"]
    changed.loc[later, "TyreLife"] += 30
    changed.loc[later, "Position"] = 20
    pd.testing.assert_frame_equal(before, race_state.race_state(changed, lap_n))


def test_last_timed_lap_skips_a_final_lap_with_no_times():
    laps = make_race({"VER": 80.0, "LEC": 80.5}, n_laps=12)
    laps.loc[laps["LapNumber"] == 12, "LapEnd"] = pd.NaT
    assert race_state.last_timed_lap(laps) == 11


def test_projection_adds_pace_and_wear():
    state = pd.DataFrame(
        {"Driver": ["A", "B"], "GapToLeader": [0.0, 2.0], "RecentPace": [80.0, 79.0], "Slope": [0.0, 0.1]}
    )
    clock = race_state.project_clock(state, 3)
    assert clock["A"] == pytest.approx(240.0)
    # 2 + 3 × 79 + 0.1 × (1 + 2 + 3)
    assert clock["B"] == pytest.approx(2 + 237 + 0.6)
    assert race_state.project_clock(state, 0)["B"] == pytest.approx(2.0)


def test_projection_fills_missing_pace_with_the_field():
    state = pd.DataFrame(
        {"Driver": ["A", "B", "C"], "GapToLeader": [0.0, 1.0, 2.0], "RecentPace": [80.0, np.nan, 82.0],
         "Slope": [np.nan, np.nan, np.nan]}
    )
    clock = race_state.project_clock(state, 1)
    assert clock["B"] == pytest.approx(1 + 81.0)


def test_recent_laps_at_lap_n_ignore_later_laps():
    laps = make_race({"VER": 80.0, "LEC": 80.5}, n_laps=20)
    laps.loc[laps["LapNumber"] > 12, "LapTimeSec"] = 79.0
    recent = race_state.recent_laps_at(laps, 12, "VER")
    assert len(recent) == config.RECENT_LAPS and (recent == 80.0).all()
