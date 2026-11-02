"""Turn numbers into the text shown in the UI."""

import pandas as pd


def lap_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - 60 * minutes:06.3f}"


def signed_seconds(value) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.3f} s"


def who_gained(delta, driver_1: str, driver_2: str) -> str:
    """Name the faster driver for a Driver 1 − Driver 2 time delta."""
    if pd.isna(delta):
        return "no time set"
    if delta == 0:
        return "level"
    return f"{driver_1 if delta < 0 else driver_2} faster"


def percent(probability: float) -> str:
    return f"{probability * 100:.1f}%"
