"""Run the whole app against the fake API, for race-like and qualifying-like data."""

from datetime import datetime
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import fake_openf1
from f1dash import openf1

MAIN = Path(__file__).resolve().parents[1] / "main.py"
PROBLEM = "hit a problem"


@pytest.fixture(autouse=True)
def app_environment(monkeypatch):
    monkeypatch.setattr(openf1, "fetch", fake_openf1.make_fetch())
    st.cache_data.clear()


def run_app(session_name="Race"):
    app = AppTest.from_file(str(MAIN), default_timeout=60)
    app.session_state["season"] = 2024
    app.session_state["session_name"] = session_name
    app.run()
    return app


def assert_healthy(app):
    assert not app.exception, [e.value for e in app.exception]
    assert not [w for w in app.warning if PROBLEM in w.value]
    assert not app.error


def all_text(app):
    return " ".join(m.value for m in [*app.markdown, *app.info, *app.warning, *app.caption])


def test_race_renders_every_tab():
    app = run_app()
    assert_healthy(app)
    assert len(app.tabs) == 6
    assert len(app.dataframe) >= 2  # Order at lap N and the degradation table.
    labels = [m.label for m in app.metric]
    assert "Model error (MAE)" in labels and "Baseline error (MAE)" in labels
    assert "Ahead at any point" in labels and "Safe rejoin" in labels and "Posterior mean" in labels


def test_qualifying_explains_race_only_tabs():
    app = run_app("Qualifying")
    assert_healthy(app)
    text = all_text(app)
    assert "race or sprint" in text


def test_session_without_laps_is_explained():
    app = run_app("Practice 1")
    assert_healthy(app)
    assert "no lap data" in all_text(app)


def test_year_list_reaches_the_current_season():
    app = run_app()
    assert str(datetime.now().year) in app.sidebar.selectbox(key="season").options


def test_changing_drivers_and_lap_mode_keeps_working():
    app = run_app()
    key = fake_openf1.RACE_KEY
    app.selectbox(key=f"driver_1_{key}").set_value("LEC").run()
    assert_healthy(app)
    app.radio(key=f"lap_mode_{key}").set_value("Specific lap").run()
    assert_healthy(app)
    app.slider(key=f"lap_number_{key}").set_value(25).run()
    # RUS retired on lap 8. Choosing him must explain that, not crash.
    app.selectbox(key=f"driver_2_{key}").set_value("RUS").run()
    assert_healthy(app)
    assert "did not complete lap 25" in all_text(app)


def test_teammates_and_manual_pit_mode_work():
    app = run_app()
    key = fake_openf1.RACE_KEY
    app.selectbox(key=f"driver_1_{key}").set_value("VER")
    app.selectbox(key=f"driver_2_{key}").set_value("PER").run()
    assert_healthy(app)
    app.radio(key=f"pit_mode_{key}").set_value("Manual phases").run()
    assert_healthy(app)


def test_api_failure_is_a_friendly_message(monkeypatch):
    def fail(*args, **kwargs):
        raise openf1.OpenF1Error("OpenF1's rate limit was reached. Wait a minute and try again.")

    monkeypatch.setattr(openf1, "fetch", fail)
    app = AppTest.from_file(str(MAIN), default_timeout=60)
    app.run()
    assert not app.exception
    assert "rate limit" in app.sidebar.error[0].value


def test_a_season_with_nothing_finished_is_explained(monkeypatch):
    real = fake_openf1.make_fetch()

    def fetch(endpoint, **params):
        return [] if endpoint == "sessions" and "year" in params else real(endpoint, **params)

    monkeypatch.setattr(openf1, "fetch", fetch)
    app = AppTest.from_file(str(MAIN), default_timeout=60)
    app.run()
    assert not app.exception
    assert "No" in app.sidebar.info[0].value


def test_race_lap_slider_is_shared_and_resets_with_the_session():
    app = run_app()
    key = f"race_lap_{fake_openf1.RACE_KEY}"
    slider = app.slider(key=key)
    assert slider.max == fake_openf1.RACE_LAPS and slider.value == fake_openf1.RACE_LAPS // 2
    slider.set_value(25).run()
    assert_healthy(app)
    app.sidebar.selectbox(key="session_name").set_value("Qualifying").run()
    assert_healthy(app)
    assert not [s for s in app.slider if s.key == key]
    app.sidebar.selectbox(key="session_name").set_value("Race").run()
    assert app.slider(key=key).value == fake_openf1.RACE_LAPS // 2
