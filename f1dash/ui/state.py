"""Cached access to data and slow models, plus session-state helpers.

Cache entries are capped because Streamlit Cloud has little memory. Slow model
functions take plain DataFrames, so the cache key is the data itself and
switching race can never serve a stale result.
"""

import pandas as pd
import streamlit as st

from f1dash import data
from f1dash.models import degradation, position
from f1dash.models import race_state as race_state_model

HOUR = 3600


@st.cache_data(ttl=HOUR, max_entries=8, show_spinner=False)
def season_sessions(year: int) -> pd.DataFrame:
    return data.season_sessions(year)


@st.cache_data(ttl=HOUR, max_entries=6, show_spinner=False)
def load_session(session_key: int) -> data.Session:
    return data.load_session(session_key)


@st.cache_data(ttl=HOUR, max_entries=48, show_spinner=False)
def load_telemetry(session_key: int, driver_number: int, lap_start, lap_end) -> pd.DataFrame:
    return data.load_telemetry(session_key, driver_number, lap_start, lap_end)


@st.cache_data(ttl=HOUR, max_entries=32, show_spinner=False)
def stint_fits(clean_laps: pd.DataFrame) -> pd.DataFrame:
    return degradation.fit_all_stints(clean_laps)


@st.cache_data(ttl=HOUR, max_entries=64, show_spinner=False)
def race_state(laps: pd.DataFrame, lap_n: int) -> pd.DataFrame:
    return race_state_model.race_state(laps, lap_n)


@st.cache_data(ttl=HOUR, max_entries=6, show_spinner=False)
def position_model(laps: pd.DataFrame) -> position.Evaluation | None:
    return position.train_and_evaluate(position.build_dataset(laps))


def keep_valid(key: str, options: list) -> None:
    """Forget a stored choice that is no longer on offer."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
