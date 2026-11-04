import pandas as pd
import pytest

import fake_openf1
from f1dash.models import race_overview
from test_race_state import make_race


def test_trace_is_time_behind_the_reference_drivers_average_pace():
    laps = make_race({"VER": 80.0, "LEC": 80.5}, n_laps=10)
    trace = race_overview.race_trace(laps, "VER").set_index(["Driver", "LapNumber"])["Behind"]
    assert trace[("VER", 10)] == pytest.approx(0.0)
    assert trace[("VER", 4)] == pytest.approx(0.0)
    assert trace[("LEC", 4)] == pytest.approx(2.0)
    assert trace[("LEC", 10)] == pytest.approx(5.0)


def test_trace_reference_ignores_a_final_lap_without_a_time():
    laps = make_race({"VER": 80.0, "LEC": 80.5}, n_laps=10)
    laps.loc[laps["LapNumber"] == 10, "LapEnd"] = pd.NaT
    trace = race_overview.race_trace(laps, "VER")
    assert trace["LapNumber"].max() == 9
    assert trace.loc[trace["Driver"] == "VER", "Behind"].abs().max() == pytest.approx(0.0)


def test_a_pit_stop_shows_as_lost_time_on_the_trace():
    laps = make_race({"VER": 80.0, "LEC": 80.0}, n_laps=10, stop={"LEC": 5}, pit_loss=20.0)
    trace = race_overview.race_trace(laps, "VER").set_index(["Driver", "LapNumber"])
    assert trace.loc[("LEC", 4), "Behind"] == pytest.approx(0.0)
    assert trace.loc[("LEC", 6), "Behind"] == pytest.approx(20.0)
    assert bool(trace.loc[("LEC", 5), "PitInLap"])


def test_reference_is_the_winner_or_else_the_leader():
    laps = make_race({"VER": 80.0, "LEC": 79.0}, n_laps=5)
    drivers = pd.DataFrame({"Driver": ["VER", "LEC"], "Finish": [1.0, 2.0]})
    assert race_overview.reference_driver(drivers, laps) == "VER"
    drivers["Finish"] = float("nan")
    assert race_overview.reference_driver(drivers, laps) == "LEC"


def test_neutralised_spans_group_consecutive_laps():
    laps = make_race({"A": 80.0, "B": 80.5, "C": 81.0}, n_laps=12)
    laps.loc[laps["LapNumber"].isin([3, 4]), "TrackStatus"] = "4"
    laps.loc[laps["LapNumber"] == 7, "TrackStatus"] = "6"
    laps.loc[laps["LapNumber"] == 10, "TrackStatus"] = "45"
    # One driver's lap overlapping a safety car is not enough to shade the lap.
    laps.loc[(laps["LapNumber"] == 5) & (laps["Driver"] == "C"), "TrackStatus"] = "4"
    spans = race_overview.neutralised_spans(laps)
    assert spans.to_dict("records") == [
        {"First": 3, "Last": 4, "Label": "SC"},
        {"First": 7, "Last": 7, "Label": "VSC"},
        {"First": 10, "Last": 10, "Label": "Red flag"},
    ]


def test_fake_race_safety_car_is_found(race):
    spans = race_overview.neutralised_spans(race.laps)
    sc = spans[spans["Label"] == "SC"]
    assert len(sc) == 1
    assert sc.iloc[0]["First"] <= fake_openf1.SC_LAPS[0] and sc.iloc[0]["Last"] >= fake_openf1.SC_LAPS[1]


def test_stint_bars_cover_each_drivers_laps(race):
    bars = race_overview.stint_bars(race.laps)
    ver = bars[bars["Driver"] == "VER"].to_dict("records")
    pit = fake_openf1.PIT_LAP[1]
    assert ver == [
        {"Driver": "VER", "Stint": 1, "Compound": "MEDIUM", "First": 1, "Last": pit},
        {"Driver": "VER", "Stint": 2, "Compound": "HARD", "First": pit + 1, "Last": fake_openf1.RACE_LAPS},
    ]
    rus = bars[bars["Driver"] == "RUS"]
    assert rus["Last"].max() == fake_openf1.RETIRE_LAP
