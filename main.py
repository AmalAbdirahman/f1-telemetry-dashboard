"""Streamlit entry point: page setup, layout and the calls into f1dash.ui."""

import logging

import streamlit as st

from f1dash.openf1 import OpenF1Error
from f1dash.ui import faceoff, race_lap, sidebar, state, theme
from f1dash.ui.tabs import bayes, forecast, pit_safety, predictor, race, tyres

st.set_page_config(page_title="F1 Telemetry & Strategy", page_icon="🏁", layout="wide")
theme.inject_css()
theme.show_header()


def load_with_lights(session_key: int):
    """Load a session, showing the start lights the first time it is chosen."""
    first_view = st.session_state.get("shown_session") != session_key
    placeholder = st.empty()
    if first_view:
        with placeholder:
            st.html(theme.lights_out_html("Loading session data"))
    try:
        return state.load_session(session_key)
    finally:
        placeholder.empty()
        st.session_state["shown_session"] = session_key


def run_tab(render, *args) -> None:
    """Keep one tab's unexpected failure from taking down the whole page."""
    try:
        render(*args)
    except Exception:
        logging.exception("Tab failed")
        st.warning("This tab hit a problem with the data for this session. The other tabs still work.")


def main() -> None:
    session_key = sidebar.render()
    if session_key is None:
        st.info("Choose a season with finished sessions in the sidebar.")
        return

    try:
        session = load_with_lights(session_key)
    except OpenF1Error as error:
        st.error(str(error))
        return

    if session.laps.empty or len(session.drivers) < 2:
        st.info(
            f"OpenF1 has no lap data for {session.event_name} {session.name} yet. "
            "Recent sessions can take a little while to appear, so try another session or check back soon."
        )
        return

    last_lap = int(session.laps["LapNumber"].max())
    detail = f"{session.location} · {len(session.drivers)} drivers · {last_lap} laps"
    theme.show_event(session.event_name, session.name, session.year, detail)

    faceoff.render(session)
    driver_1 = st.session_state[f"driver_1_{session.key}"]
    driver_2 = st.session_state[f"driver_2_{session.key}"]
    lap_n = race_lap.render(session)

    tab_race, tab_tyres, tab_predictor, tab_forecast, tab_bayes, tab_pit = st.tabs(
        ["Race", "Tyres", "Position Predictor", "Overtake Forecast", "Bayesian Pace", "Pit Safety"]
    )
    with tab_race:
        run_tab(race.render, session, driver_1, driver_2, lap_n)
    with tab_tyres:
        run_tab(tyres.render, session, driver_1, driver_2)
    with tab_predictor:
        run_tab(predictor.render, session)
    with tab_forecast:
        run_tab(forecast.render, session, driver_1, driver_2, lap_n)
    with tab_bayes:
        run_tab(bayes.render, session)
    with tab_pit:
        run_tab(pit_safety.render, session, driver_1, lap_n)

    theme.show_footer()


main()
