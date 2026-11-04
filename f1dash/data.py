"""Load sessions from OpenF1 and rebuild the per-lap table the models use.

OpenF1 serves separate tables, so tyre age, race position and track status are
reconstructed here. The ``build_*`` functions are pure so they can be tested
without the network.

Observed OpenF1 quirks handled below:
- A red flag can appear only as a "SESSION ABORTED" status message.
- In a race, lap 1 is marked as a pit-out lap and has an empty pit record.
- A red-flag stop is recorded as a pit stop lasting many minutes.
- The final lap's duration is often blank.
- The 2023 pit feed can be empty, so in-laps are also inferred from out-laps.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from f1dash import config, openf1
from f1dash.models import telemetry as telemetry_model
from f1dash.models.filters import clean_laps  # noqa: F401  (re-exported for the UI)

# Track status codes follow the F1 timing feed: 1 = clear.
YELLOW, SAFETY_CAR, RED, VSC = "2", "4", "5", "6"


@dataclass
class Session:
    """One loaded session. Plain data, so it can be pickled by st.cache_data."""

    key: int
    year: int
    event_name: str
    location: str
    name: str
    is_race: bool
    date_start: pd.Timestamp
    drivers: pd.DataFrame
    laps: pd.DataFrame


def _to_utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), format="ISO8601", utc=True).astype("datetime64[ns, UTC]")


def available_sessions(meetings: list[dict], sessions: list[dict], now: datetime) -> pd.DataFrame:
    """Return finished, non-testing, non-cancelled sessions in calendar order."""
    columns = ["session_key", "gp", "session_name", "session_type", "date_start"]
    names = {m["meeting_key"]: m.get("meeting_name", "") for m in meetings}
    rows = []
    for s in sessions:
        gp = names.get(s.get("meeting_key"), "")
        if not gp or "testing" in gp.lower() or s.get("is_cancelled"):
            continue
        if pd.Timestamp(s["date_end"]) >= pd.Timestamp(now):
            continue
        rows.append({**s, "gp": gp})
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("date_start")[columns].reset_index(drop=True)


def season_sessions(year: int, now: datetime | None = None) -> pd.DataFrame:
    """Fetch the sessions of a season that have already finished."""
    return available_sessions(
        openf1.fetch("meetings", year=year),
        openf1.fetch("sessions", year=year),
        now or datetime.now(timezone.utc),
    )


def track_status_intervals(race_control: list[dict]) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Turn race-control messages into (start, end, status code) intervals."""
    if not race_control:
        return []
    events = sorted(race_control, key=lambda e: pd.Timestamp(e["date"]))
    final = pd.Timestamp(events[-1]["date"])
    open_since: dict[str, pd.Timestamp] = {}
    yellow_sectors: set = set()
    intervals = []

    def start(code, at):
        open_since.setdefault(code, at)

    def stop(code, at):
        if code in open_since:
            intervals.append((open_since.pop(code), at, code))

    for event in events:
        at = pd.Timestamp(event["date"])
        category = event.get("category")
        flag = event.get("flag")
        scope = event.get("scope")
        message = (event.get("message") or "").upper()

        if category == "SafetyCar":
            is_virtual = "VSC" in message or "VIRTUAL" in message
            if "DEPLOYED" in message:
                start(VSC if is_virtual else SAFETY_CAR, at)
            elif is_virtual and "ENDING" in message:
                stop(VSC, at)
        elif category == "SessionStatus":
            if "ABORTED" in message or "SUSPENDED" in message:
                start(RED, at)
            elif "STARTED" in message or "RESUMED" in message:
                stop(RED, at)
            elif "FINISHED" in message or "ENDED" in message:
                for code in list(open_since):
                    stop(code, at)
        elif flag == "RED":
            start(RED, at)
        elif flag in ("YELLOW", "DOUBLE YELLOW"):
            yellow_sectors.add(event.get("sector") if scope == "Sector" else "track")
            start(YELLOW, at)
        elif flag == "CLEAR" and scope == "Sector":
            yellow_sectors.discard(event.get("sector"))
            if not yellow_sectors:
                stop(YELLOW, at)
        elif flag in ("CLEAR", "GREEN") and scope == "Track":
            yellow_sectors.clear()
            stop(YELLOW, at)
            stop(SAFETY_CAR, at)
            if flag == "GREEN":
                stop(RED, at)

    for code in list(open_since):
        stop(code, final)
    return intervals


def _attach_stints(laps: pd.DataFrame, stints: list[dict]) -> pd.DataFrame:
    laps["Stint"] = np.nan
    laps["Compound"] = "UNKNOWN"
    laps["TyreLife"] = np.nan
    for stint in sorted(stints, key=lambda s: s.get("stint_number") or 0):
        lap_start = stint.get("lap_start")
        if lap_start is None:
            continue
        lap_end = stint.get("lap_end")
        mask = (laps["DriverNumber"] == stint["driver_number"]) & (laps["LapNumber"] >= lap_start)
        if lap_end is not None:
            mask &= laps["LapNumber"] <= lap_end
        age = stint.get("tyre_age_at_start") or 0
        laps.loc[mask, "Stint"] = stint.get("stint_number")
        laps.loc[mask, "Compound"] = (stint.get("compound") or "UNKNOWN").upper()
        laps.loc[mask, "TyreLife"] = age + laps.loc[mask, "LapNumber"] - lap_start + 1
    return laps


def _attach_positions(laps: pd.DataFrame, positions: list[dict]) -> pd.DataFrame:
    """Give each lap the driver's position at the moment the lap ended."""
    laps["Position"] = np.nan
    if not positions:
        return laps
    feed = pd.DataFrame(positions)[["driver_number", "date", "position"]]
    feed["date"] = _to_utc(feed["date"])
    feed = feed.rename(columns={"driver_number": "DriverNumber"}).sort_values("date")
    timed = laps[["DriverNumber"]].assign(At=laps["LapEnd"].fillna(laps["LapStart"]))
    merged = pd.merge_asof(
        timed.sort_values("At").reset_index(),
        feed,
        left_on="At",
        right_on="date",
        by="DriverNumber",
    ).set_index("index")
    laps.loc[merged.index, "Position"] = merged["position"]
    return laps


def _attach_track_status(laps: pd.DataFrame, race_control: list[dict]) -> pd.DataFrame:
    """Record every status active during each lap, as in '24'-style timing codes."""
    codes = pd.Series("", index=laps.index, dtype=object)
    for begin, end, code in track_status_intervals(race_control):
        overlaps = (laps["LapStart"] < end) & (laps["LapEnd"] > begin)
        codes[overlaps] += code
    laps["TrackStatus"] = codes.map(lambda c: "".join(sorted(set(c))) or "1")
    return laps


def build_laps(
    laps: list[dict],
    stints: list[dict],
    pits: list[dict],
    positions: list[dict],
    race_control: list[dict],
    drivers: pd.DataFrame,
    is_race: bool,
) -> pd.DataFrame:
    """Combine OpenF1's tables into one row per driver per lap."""
    columns = {
        "driver_number": "DriverNumber",
        "lap_number": "LapNumber",
        "lap_duration": "LapTimeSec",
        "duration_sector_1": "Sector1",
        "duration_sector_2": "Sector2",
        "duration_sector_3": "Sector3",
        "date_start": "LapStart",
        "is_pit_out_lap": "PitOutLap",
    }
    if not laps:
        return pd.DataFrame(
            columns=[
                *columns.values(), "Driver", "Team", "TyreLife", "Compound", "Stint",
                "Position", "PitInLap", "TrackStatus", "IsAccurate", "LapEnd",
            ]
        )

    df = pd.DataFrame(laps).reindex(columns=list(columns)).rename(columns=columns)
    df = df.merge(drivers[["DriverNumber", "Driver", "Team"]], on="DriverNumber", how="inner")
    df = df.dropna(subset=["LapStart"]).sort_values(["DriverNumber", "LapNumber"]).reset_index(drop=True)
    for col in ("LapTimeSec", "Sector1", "Sector2", "Sector3"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["PitOutLap"] = df["PitOutLap"].astype("boolean").fillna(False).astype(bool)

    df["LapStart"] = _to_utc(df["LapStart"])
    next_start = df.groupby("DriverNumber")["LapStart"].shift(-1)
    df["LapEnd"] = (df["LapStart"] + pd.to_timedelta(df["LapTimeSec"], unit="s")).fillna(next_start)

    df = _attach_stints(df, stints)
    if is_race:
        df = _attach_positions(df, positions)
    else:
        df["Position"] = np.nan

    # Lap 1's pit record is the car leaving the grid, not a stop.
    in_laps = {
        (p["driver_number"], p["lap_number"])
        for p in pits
        if p.get("lap_number") is not None and (p["lap_number"] > 1 or p.get("pit_duration") is not None)
    }
    df["PitInLap"] = [key in in_laps for key in zip(df["DriverNumber"], df["LapNumber"])]
    next_is_out_lap = df.groupby("DriverNumber")["PitOutLap"].shift(-1, fill_value=False).astype(bool)
    next_lap = df.groupby("DriverNumber")["LapNumber"].shift(-1)
    df["PitInLap"] |= next_is_out_lap & (next_lap == df["LapNumber"] + 1)

    df = _attach_track_status(df, race_control)

    sector_sum = df[["Sector1", "Sector2", "Sector3"]].sum(axis=1, skipna=False)
    df["IsAccurate"] = (
        df["LapTimeSec"].notna()
        & ((sector_sum - df["LapTimeSec"]).abs() <= config.SECTOR_SUM_TOLERANCE_S)
        & (df["LapNumber"] > 1)
    )
    return df


def build_drivers(drivers: list[dict], results: list[dict]) -> pd.DataFrame:
    """Return one row per driver, in finishing order when a result exists."""
    finishing = {r["driver_number"]: r for r in results}
    rows = {}
    for d in drivers:
        number = d["driver_number"]
        colour = d.get("team_colour")
        result = finishing.get(number, {})
        rows[number] = {
            "DriverNumber": number,
            "Driver": d.get("name_acronym") or str(number),
            "FullName": d.get("full_name") or d.get("broadcast_name") or str(number),
            "Team": d.get("team_name") or "Unknown",
            "TeamColor": f"#{colour}" if colour else config.FALLBACK_TEAM_COLOR,
            "Finish": result.get("position"),
            "Retired": bool(result.get("dnf") or result.get("dns") or result.get("dsq")),
        }
    columns = ["DriverNumber", "Driver", "FullName", "Team", "TeamColor", "Finish", "Retired"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(list(rows.values()), columns=columns)
    df["Finish"] = pd.to_numeric(df["Finish"], errors="coerce")
    return df.sort_values(["Finish", "DriverNumber"], na_position="last").reset_index(drop=True)


def load_session(session_key: int) -> Session:
    """Fetch and assemble a session. Raises OpenF1Error with a readable message."""
    found = openf1.fetch("sessions", session_key=session_key)
    if not found:
        raise openf1.OpenF1Error("OpenF1 has no record of this session.")
    info = found[0]
    meetings = openf1.fetch("meetings", meeting_key=info["meeting_key"])
    location = info.get("location", "")
    is_race = info.get("session_type") == "Race"

    drivers = build_drivers(
        openf1.fetch("drivers", session_key=session_key),
        openf1.fetch("session_result", session_key=session_key),
    )
    laps = build_laps(
        openf1.fetch("laps", session_key=session_key),
        openf1.fetch("stints", session_key=session_key),
        openf1.fetch("pit", session_key=session_key),
        openf1.fetch("position", session_key=session_key) if is_race else [],
        openf1.fetch("race_control", session_key=session_key),
        drivers,
        is_race,
    )
    if not drivers.empty:
        drivers = drivers[drivers["Driver"].isin(laps["Driver"])].reset_index(drop=True)
    return Session(
        key=session_key,
        year=int(info["year"]),
        event_name=meetings[0].get("meeting_name", location) if meetings else location,
        location=location,
        name=info.get("session_name", ""),
        is_race=is_race,
        date_start=pd.Timestamp(info["date_start"]),
        drivers=drivers,
        laps=laps,
    )


def fastest_lap(laps: pd.DataFrame) -> pd.Series | None:
    """Return a driver's fastest timed lap, or None if they set no time."""
    timed = laps[laps["LapTimeSec"].notna() & laps["LapEnd"].notna()]
    return None if timed.empty else timed.loc[timed["LapTimeSec"].idxmin()]


def load_telemetry(session_key: int, driver_number: int, lap_start, lap_end) -> pd.DataFrame:
    """Fetch one lap's car data, with elapsed time and distance added.

    Returns an empty frame when OpenF1 has no samples for the lap.
    """
    margin = timedelta(seconds=1)
    records = openf1.fetch(
        "car_data",
        session_key=session_key,
        driver_number=driver_number,
        **{
            "date>": (lap_start - margin).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "date<": (lap_end + margin).strftime("%Y-%m-%dT%H:%M:%S.%f"),
        },
    )
    return telemetry_model.prepare_lap(pd.DataFrame(records), lap_start, lap_end)
