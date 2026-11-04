"""Whole-race views for the Race tab: the race trace, neutralised laps and tyre stints.

These look back over the full race, unlike ``race_state``. The charts mark
lap N so the user can see where the strategy tabs are standing.
"""

import pandas as pd

from f1dash.models import race_state

# Track status codes from the timing feed, most serious first.
NEUTRALISED = (("5", "Red flag"), ("4", "SC"), ("6", "VSC"))
MAJORITY = 0.5  # Share of cars whose lap must overlap a status for the lap to be shaded.


def reference_driver(drivers: pd.DataFrame, laps: pd.DataFrame) -> str | None:
    """The winner, or the leader at the last timed lap when there is no result."""
    winners = drivers.loc[drivers["Finish"] == 1, "Driver"]
    if not winners.empty:
        return str(winners.iloc[0])
    last = race_state.last_timed_lap(laps)
    state = race_state.race_state(laps, last) if last else pd.DataFrame()
    return None if state.empty else str(state.iloc[0]["Driver"])


def race_trace(laps: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Each car's time behind a car lapping at the reference driver's average pace.

    Behind(lap n) = (time to finish lap n) − n × (reference's race time ÷ its laps).
    The reference ends on zero, positive values are behind it, and a pit stop
    shows as a step up of about the pit loss.
    """
    columns = ["Driver", "LapNumber", "Behind", "PitInLap"]
    timed = laps[laps["LapEnd"].notna()]
    if timed.empty or reference not in set(timed["Driver"]):
        return pd.DataFrame(columns=columns)
    start = laps["LapStart"].min()
    elapsed = (timed["LapEnd"] - start).dt.total_seconds()
    own = timed["Driver"] == reference
    last = timed.loc[own, "LapNumber"].idxmax()
    average = elapsed[last] / timed.loc[last, "LapNumber"]
    trace = timed.assign(Behind=elapsed - timed["LapNumber"] * average)
    return trace[columns].sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def neutralised_spans(laps: pd.DataFrame) -> pd.DataFrame:
    """Runs of laps under a red flag, safety car or virtual safety car.

    A lap counts when most cars' laps overlap the status, so a car that crossed
    the line just after a deployment does not shade an extra lap.
    """
    status = laps.groupby("LapNumber")["TrackStatus"]
    labels = pd.Series(None, index=sorted(laps["LapNumber"].unique()), dtype=object)
    for code, label in reversed(NEUTRALISED):
        share = status.apply(lambda codes, c=code: codes.str.contains(c, regex=False).mean())
        labels[share[share > MAJORITY].index] = label

    spans = []
    for lap, label in labels.items():
        if pd.isna(label):
            continue
        if spans and spans[-1]["Label"] == label and spans[-1]["Last"] == lap - 1:
            spans[-1]["Last"] = int(lap)
        else:
            spans.append({"First": int(lap), "Last": int(lap), "Label": label})
    return pd.DataFrame(spans, columns=["First", "Last", "Label"])


def stint_bars(laps: pd.DataFrame) -> pd.DataFrame:
    """First and last lap of each driver's stints, with the compound."""
    known = laps.assign(Stint=laps["Stint"].fillna(0))
    grouped = known.groupby(["Driver", "Stint"])
    bars = grouped.agg(First=("LapNumber", "min"), Last=("LapNumber", "max")).reset_index()
    bars["Compound"] = grouped["Compound"].agg(lambda c: c.mode().iloc[0]).to_numpy()
    bars = bars.astype({"Stint": int, "First": int, "Last": int})
    return bars[["Driver", "Stint", "Compound", "First", "Last"]].sort_values(["Driver", "First"]).reset_index(drop=True)


def teammates_dashed(drivers: pd.DataFrame) -> set[str]:
    """Drivers drawn dashed: everyone after the first of their team in the table's order."""
    return set(drivers.loc[drivers.duplicated("Team"), "Driver"])

