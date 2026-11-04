"""Turn numbers into the text shown in the UI."""

import pandas as pd

from f1dash import config


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


def gap(seconds, laps_down: int = 0) -> str:
    """Timing-screen gap: seconds on the same lap, whole laps once lapped."""
    if pd.isna(seconds):
        return "—"
    if laps_down >= 1:
        return f"+{laps_down} lap" + ("s" if laps_down > 1 else "")
    return f"+{seconds:.3f}"


def tyre(compound: str, age) -> str:
    """Compound initial and tyre age, e.g. 'M 12'."""
    letter = config.COMPOUND_LETTERS.get(compound, "?")
    return letter if pd.isna(age) else f"{letter} {int(age)}"


def slope(value) -> str:
    """Tyre trend in s/lap, or a dash when the stint is too short to fit."""
    return "—" if pd.isna(value) else f"{value:+.3f}"


def position(value: float) -> str:
    """A whole race position: P5, never P5.3."""
    return f"P{int(round(value))}"


def percent(probability: float) -> str:
    return f"{probability * 100:.1f}%"
