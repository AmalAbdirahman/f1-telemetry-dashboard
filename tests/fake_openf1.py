"""A deterministic fake of the OpenF1 API, so tests never touch the network."""

from datetime import datetime, timedelta, timezone

import numpy as np

START = datetime(2024, 5, 26, 13, 0, tzinfo=timezone.utc)
RACE_KEY, QUALI_KEY, EMPTY_KEY = 9001, 9002, 9003
DRIVERS = [
    (1, "VER", "Max VERSTAPPEN", "Red Bull Racing", "3671C6"),
    (11, "PER", "Sergio PEREZ", "Red Bull Racing", "3671C6"),
    (16, "LEC", "Charles LECLERC", "Ferrari", "E8002D"),
    (55, "SAI", "Carlos SAINZ", "Ferrari", "E8002D"),
    (4, "NOR", "Lando NORRIS", "McLaren", "FF8000"),
    (81, "PIA", "Oscar PIASTRI", "McLaren", "FF8000"),
    (44, "HAM", "Lewis HAMILTON", "Mercedes", None),
    (63, "RUS", "George RUSSELL", "Mercedes", None),
]
RACE_LAPS = 30
SC_LAPS = (14, 15)
PIT_LAP = {n: 10 + i for i, (n, *_) in enumerate(DRIVERS)}
RETIRED = 63
RETIRE_LAP = 8


def iso(moment: datetime) -> str:
    return moment.isoformat()


def _lap_rows(key, number, pace, n_laps, rng, start):
    rows = []
    clock = start
    is_race = key == RACE_KEY
    for lap in range(1, n_laps + 1):
        duration = pace + 0.05 * lap + rng.normal(0, 0.15)
        if is_race and lap in SC_LAPS:
            duration += 25
        if is_race and lap == PIT_LAP.get(number):
            duration += 8
        if is_race and lap == PIT_LAP.get(number, 0) + 1:
            duration += 14
        s1, s2 = duration * 0.3, duration * 0.4
        rows.append(
            {
                "session_key": key,
                "driver_number": number,
                "lap_number": lap,
                "date_start": iso(clock),
                "lap_duration": round(duration, 3),
                "duration_sector_1": round(s1, 3),
                "duration_sector_2": round(s2, 3),
                "duration_sector_3": round(duration - s1 - s2, 3),
                "is_pit_out_lap": lap == 1 or (is_race and lap == PIT_LAP.get(number, 0) + 1),
            }
        )
        clock += timedelta(seconds=duration)
    return rows


def _stint(number, stint, first, last, compound, age):
    return {
        "driver_number": number,
        "stint_number": stint,
        "lap_start": first,
        "lap_end": last,
        "compound": compound,
        "tyre_age_at_start": age,
    }


def _race_control(date, category, message, flag=None, scope=None, sector=None):
    return {
        "date": iso(date),
        "category": category,
        "flag": flag,
        "scope": scope,
        "sector": sector,
        "message": message,
    }


def build_api() -> dict:
    """Return {(endpoint, session_key): records} for a race and a qualifying."""
    rng = np.random.default_rng(1)
    api = {}
    drivers = [
        {"driver_number": n, "name_acronym": a, "full_name": f, "team_name": t, "team_colour": c}
        for n, a, f, t, c in DRIVERS
    ]

    laps, stints, pits, positions, results = [], [], [], [], []
    for i, (number, *_) in enumerate(DRIVERS):
        n_laps = RETIRE_LAP if number == RETIRED else RACE_LAPS
        laps += _lap_rows(RACE_KEY, number, 80 + 0.3 * i, n_laps, rng, START)
        pit = PIT_LAP[number]
        pits.append({"driver_number": number, "lap_number": 1, "pit_duration": None})
        if number == RETIRED:
            stints.append(_stint(number, 1, 1, n_laps, "SOFT", 0))
        else:
            pits.append({"driver_number": number, "lap_number": pit, "pit_duration": 23.0})
            stints.append(_stint(number, 1, 1, pit, "MEDIUM", 0))
            stints.append(_stint(number, 2, pit + 1, RACE_LAPS, "HARD", 2))
        positions.append({"driver_number": number, "date": iso(START - timedelta(minutes=5)), "position": i + 1})
        results.append(
            {
                "driver_number": number,
                "position": None if number == RETIRED else i + 1,
                "dnf": number == RETIRED,
                "dns": False,
                "dsq": False,
            }
        )

    # The top two swap places partway through lap 5.
    swap = START + timedelta(seconds=80 * 4 + 40)
    positions.append({"driver_number": 1, "date": iso(swap), "position": 2})
    positions.append({"driver_number": 11, "date": iso(swap), "position": 1})

    lead = [r for r in laps if r["driver_number"] == 1]
    sc_start = datetime.fromisoformat(lead[SC_LAPS[0] - 1]["date_start"]) + timedelta(seconds=5)
    sc_end = datetime.fromisoformat(lead[SC_LAPS[1]]["date_start"]) - timedelta(seconds=5)
    race_control = [
        _race_control(START, "SessionStatus", "SESSION STARTED"),
        _race_control(sc_start, "SafetyCar", "SAFETY CAR DEPLOYED"),
        _race_control(sc_end, "Flag", "TRACK CLEAR", flag="CLEAR", scope="Track"),
    ]
    api.update(
        {
            ("drivers", RACE_KEY): drivers,
            ("laps", RACE_KEY): laps,
            ("stints", RACE_KEY): stints,
            ("pit", RACE_KEY): pits,
            ("position", RACE_KEY): positions,
            ("race_control", RACE_KEY): race_control,
            ("session_result", RACE_KEY): results,
        }
    )

    q_laps, q_stints = [], []
    for i, (number, *_) in enumerate(DRIVERS):
        q_laps += _lap_rows(QUALI_KEY, number, 72 + 0.2 * i, 4, rng, START)
        q_stints.append(_stint(number, 1, 1, 4, "SOFT", 0))
    api.update(
        {
            ("drivers", QUALI_KEY): drivers,
            ("laps", QUALI_KEY): q_laps,
            ("stints", QUALI_KEY): q_stints,
        }
    )
    api[("drivers", EMPTY_KEY)] = drivers
    return api


def _session(key, name, kind, start, hours=2):
    return {
        "session_key": key,
        "meeting_key": 1,
        "session_name": name,
        "session_type": kind,
        "year": 2024,
        "date_start": iso(start),
        "date_end": iso(start + timedelta(hours=hours)),
        "location": "Monte Carlo",
        "is_cancelled": False,
    }


SESSIONS = [
    _session(EMPTY_KEY, "Practice 1", "Practice", START - timedelta(days=2), 1),
    _session(QUALI_KEY, "Qualifying", "Qualifying", START - timedelta(days=1), 1),
    _session(RACE_KEY, "Race", "Race", START),
]
MEETINGS = [{"meeting_key": 1, "meeting_name": "Monaco Grand Prix", "location": "Monte Carlo"}]


def _car_data(params) -> list[dict]:
    first = datetime.fromisoformat(params["date>"]).replace(tzinfo=timezone.utc)
    last = datetime.fromisoformat(params["date<"]).replace(tzinfo=timezone.utc)
    offset = params["driver_number"] / 100
    rows = []
    for t in np.arange(0, (last - first).total_seconds(), 0.25):
        speed = 200 + 90 * np.sin(t / 6 + offset)
        rows.append(
            {
                "date": iso(first + timedelta(seconds=float(t))),
                "speed": float(speed),
                "throttle": 100 if speed > 200 else 20,
                "brake": 0 if speed > 150 else 100,
                "n_gear": 5,
            }
        )
    return rows


def make_fetch(api: dict | None = None):
    """Return a stand-in for ``openf1.fetch`` backed by synthetic data."""
    api = api if api is not None else build_api()

    def fetch(endpoint, **params):
        if endpoint == "meetings":
            return MEETINGS
        if endpoint == "sessions":
            if "session_key" in params:
                return [s for s in SESSIONS if s["session_key"] == params["session_key"]]
            return SESSIONS
        if endpoint == "car_data":
            return _car_data(params)
        return api.get((endpoint, params.get("session_key")), [])

    return fetch
