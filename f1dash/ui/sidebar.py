"""Season → Grand Prix → session selection."""

from datetime import datetime, timezone

import streamlit as st

from f1dash import config
from f1dash.openf1 import OpenF1Error
from f1dash.ui import state


def render() -> int | None:
    """Draw the sidebar and return the chosen session key, if any."""
    st.sidebar.header("Session")
    this_year = datetime.now(timezone.utc).year
    years = list(range(this_year, config.FIRST_SEASON - 1, -1))
    year = st.sidebar.selectbox("Season", years, key="season")

    try:
        sessions = state.season_sessions(year)
    except OpenF1Error as error:
        st.sidebar.error(str(error))
        return None
    if sessions.empty:
        st.sidebar.info(f"No {year} sessions have finished yet. Try an earlier season.")
        return None

    grands_prix = list(dict.fromkeys(sessions["gp"].iloc[::-1]))
    state.keep_valid("gp", grands_prix)
    gp = st.sidebar.selectbox("Grand Prix", grands_prix, key="gp")

    weekend = sessions[sessions["gp"] == gp]
    names = weekend["session_name"].tolist()
    state.keep_valid("session_name", names)
    default = names.index("Race") if "Race" in names else len(names) - 1
    name = st.sidebar.selectbox("Session", names, index=default, key="session_name")

    st.sidebar.caption(f"Data from OpenF1, which covers {config.FIRST_SEASON} onwards.")
    return int(weekend.loc[weekend["session_name"] == name, "session_key"].iloc[0])
