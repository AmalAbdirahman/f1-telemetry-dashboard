"""Which laps count as representative. Pure, so the models can share it."""

import pandas as pd

from f1dash import config


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep representative laps: green-flag, non-pit, accurately timed.

    Laps over 107% of the median for that driver's stint are also dropped. This
    removes cool-down laps in practice and qualifying, and is relative to the
    stint so that a wet stint keeps its slower laps.
    """
    if laps.empty:
        return laps
    mask = (laps["TrackStatus"] == "1") & ~laps["PitInLap"] & ~laps["PitOutLap"] & laps["IsAccurate"]
    clean = laps[mask]
    typical = clean.groupby(["Driver", "Stint"], dropna=False)["LapTimeSec"].transform("median")
    return clean[clean["LapTimeSec"] <= config.QUICK_LAP_RATIO * typical]
