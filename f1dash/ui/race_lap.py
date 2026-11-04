"""The one "Race lap" slider that every race tab reads."""

import streamlit as st

from f1dash import data
from f1dash.models import race_state


def slider_key(session: data.Session) -> str:
    # Keyed by session, so choosing another session starts from a fresh default.
    return f"race_lap_{session.key}"


def render(session: data.Session) -> int | None:
    """Draw the slider and return lap N, or None when there is no race to step through."""
    if not session.is_race:
        return None
    last = race_state.last_timed_lap(session.laps)
    if last < 2:
        return None
    lap_n = st.slider(
        "Race lap",
        1,
        last,
        max(1, last // 2),
        key=slider_key(session),
        help="The race tabs only use what was known at the end of this lap, as a strategist would.",
    )
    st.caption(f"Strategy below is worked out at the end of lap {lap_n} of {last}, from nothing that happened later.")
    return lap_n
