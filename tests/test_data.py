"""The rebuilt lap table: tyre age, position, track status and lap filters."""

from datetime import datetime, timezone

import fake_openf1
from f1dash import data


def driver_lap(session, driver, lap):
    laps = session.laps
    return laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap)].iloc[0]


def test_tyre_life_counts_from_stint_start_and_used_tyre_age(race):
    pit = fake_openf1.PIT_LAP[1]
    assert driver_lap(race, "VER", 1)["TyreLife"] == 1
    assert driver_lap(race, "VER", pit)["TyreLife"] == pit
    # A stint starting on 2-lap-old tyres shows 3 on its first lap.
    assert driver_lap(race, "VER", pit + 1)["TyreLife"] == 3
    assert driver_lap(race, "VER", pit + 1)["Compound"] == "HARD"


def test_position_is_taken_at_the_end_of_each_lap(race):
    assert driver_lap(race, "VER", 4)["Position"] == 1
    assert driver_lap(race, "VER", 5)["Position"] == 2
    assert driver_lap(race, "PER", 5)["Position"] == 1


def test_lap_one_grid_record_is_not_a_pit_stop(race):
    assert not race.laps[race.laps["LapNumber"] == 1]["PitInLap"].any()
    assert driver_lap(race, "VER", fake_openf1.PIT_LAP[1])["PitInLap"]


def test_safety_car_laps_are_not_green(race):
    assert driver_lap(race, "VER", 14)["TrackStatus"] == "4"
    assert driver_lap(race, "VER", 13)["TrackStatus"] == "1"


def test_red_flag_shown_only_as_session_aborted_is_detected():
    base = {"flag": None, "scope": None, "sector": None}
    events = [
        {**base, "date": "2024-01-01T12:00:00+00:00", "category": "SessionStatus", "message": "SESSION ABORTED"},
        {**base, "date": "2024-01-01T12:30:00+00:00", "category": "SessionStatus", "message": "SESSION STARTED"},
    ]
    [(start, end, code)] = data.track_status_intervals(events)
    assert code == data.RED
    assert (end - start).total_seconds() == 1800


def test_sector_yellow_lasts_until_every_sector_clears():
    def flag(minute, name, sector):
        return {
            "date": f"2024-01-01T12:{minute:02d}:00+00:00",
            "category": "Flag",
            "flag": name,
            "scope": "Sector",
            "sector": sector,
            "message": "",
        }

    events = [flag(0, "YELLOW", 1), flag(1, "DOUBLE YELLOW", 2), flag(2, "CLEAR", 1), flag(5, "CLEAR", 2)]
    [(start, end, code)] = data.track_status_intervals(events)
    assert code == data.YELLOW
    assert (end - start).total_seconds() == 300


def test_clean_laps_drop_lap_one_pit_laps_and_safety_car(race):
    clean = data.clean_laps(race.laps)
    assert (clean["LapNumber"] > 1).all()
    assert clean["TrackStatus"].eq("1").all()
    assert not clean["PitInLap"].any() and not clean["PitOutLap"].any()
    assert not clean["LapNumber"].isin(fake_openf1.SC_LAPS).any()


def test_laps_with_a_missing_sector_are_not_accurate():
    api = fake_openf1.build_api()
    raw = api[("laps", fake_openf1.RACE_KEY)]
    target = next(r for r in raw if r["driver_number"] == 1 and r["lap_number"] == 5)
    target["duration_sector_2"] = None
    drivers = data.build_drivers(api[("drivers", fake_openf1.RACE_KEY)], [])
    laps = data.build_laps(raw, [], [], [], [], drivers, True)
    row = laps[(laps["DriverNumber"] == 1) & (laps["LapNumber"] == 5)].iloc[0]
    assert not row["IsAccurate"]


def test_drivers_use_finish_order_and_flag_retirements(race):
    assert race.drivers.iloc[0]["Driver"] == "VER"
    assert race.drivers.iloc[-1]["Driver"] == "RUS"
    assert bool(race.drivers.iloc[-1]["Retired"])


def test_missing_team_colour_falls_back(race):
    assert race.drivers.set_index("Driver").loc["HAM", "TeamColor"].startswith("#")


def test_qualifying_has_no_race_positions(qualifying):
    assert not qualifying.is_race
    assert qualifying.laps["Position"].isna().all()


def test_a_session_with_no_laps_loads_empty(fake_api):
    session = data.load_session(fake_openf1.EMPTY_KEY)
    assert session.laps.empty and session.drivers.empty


def test_available_sessions_hide_testing_cancelled_and_future():
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    meetings = [
        {"meeting_key": 1, "meeting_name": "Pre-Season Testing"},
        {"meeting_key": 2, "meeting_name": "Mexico City Grand Prix"},
    ]

    def session(key, meeting, end, cancelled=False):
        return {
            "session_key": key,
            "meeting_key": meeting,
            "session_name": "Race",
            "session_type": "Race",
            "date_start": end,
            "date_end": end,
            "is_cancelled": cancelled,
        }

    sessions = [
        session(1, 1, "2026-02-01T10:00:00+00:00"),
        session(2, 2, "2026-11-01T22:00:00+00:00"),
        session(3, 2, "2026-11-01T21:00:00+00:00", cancelled=True),
        session(4, 2, "2026-11-08T22:00:00+00:00"),
    ]
    assert data.available_sessions(meetings, sessions, now)["session_key"].tolist() == [2]


def test_telemetry_gets_distance_and_elapsed_time(race):
    lap = driver_lap(race, "VER", 3)
    tel = data.load_telemetry(race.key, 1, lap["LapStart"], lap["LapEnd"])
    assert tel["Distance"].iloc[0] == 0 and tel["Distance"].is_monotonic_increasing
    assert 0 <= tel["Time"].min() and tel["Time"].max() <= lap["LapTimeSec"]


def test_final_lap_with_blank_duration_still_gets_a_position():
    api = fake_openf1.build_api()
    key = fake_openf1.RACE_KEY
    raw = api[("laps", key)]
    target = next(r for r in raw if r["driver_number"] == 1 and r["lap_number"] == fake_openf1.RACE_LAPS)
    target["lap_duration"] = None
    drivers = data.build_drivers(api[("drivers", key)], [])
    laps = data.build_laps(raw, [], [], api[("position", key)], [], drivers, True)
    last = laps[(laps["DriverNumber"] == 1) & (laps["LapNumber"] == fake_openf1.RACE_LAPS)].iloc[0]
    assert last["Position"] == 2


def test_in_laps_are_inferred_when_the_pit_feed_is_empty():
    api = fake_openf1.build_api()
    key = fake_openf1.RACE_KEY
    drivers = data.build_drivers(api[("drivers", key)], [])
    laps = data.build_laps(api[("laps", key)], [], [], [], [], drivers, True)
    pit = fake_openf1.PIT_LAP[1]
    in_laps = laps[(laps["DriverNumber"] == 1) & laps["PitInLap"]]
    assert in_laps["LapNumber"].tolist() == [pit]


def test_clean_laps_drop_a_slow_cool_down_lap(race):
    laps = race.laps.copy()
    index = laps[(laps["Driver"] == "VER") & (laps["LapNumber"] == 5)].index[0]
    laps.loc[index, "LapTimeSec"] = 120.0
    clean = data.clean_laps(laps)
    assert not ((clean["Driver"] == "VER") & (clean["LapNumber"] == 5)).any()


def test_telemetry_runs_exactly_from_zero_to_lap_time(race):
    lap = driver_lap(race, "VER", 3)
    tel = data.load_telemetry(race.key, 1, lap["LapStart"], lap["LapEnd"])
    assert tel["Time"].iloc[0] == 0
    assert abs(tel["Time"].iloc[-1] - lap["LapTimeSec"]) < 1e-6
