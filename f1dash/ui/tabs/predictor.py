"""Race position predictor tab."""

import streamlit as st

from f1dash import config, data
from f1dash.models import position
from f1dash.ui import state


def render(session: data.Session) -> None:
    st.markdown(f"Predicts a driver's position {config.PREDICTION_HORIZON} laps ahead, scored against simply assuming nothing changes.")
    if not session.is_race:
        st.info("Positions only mean something in a race or sprint. Pick one of those to use this tab.")
        return

    columns = ["Driver", "LapNumber", "TyreLife", "Position", "Compound"]
    evaluation = state.position_model(session.laps[columns])
    if evaluation is None:
        st.info("There are too few laps with position and tyre data to train and test a model here.")
        return

    model, baseline, verdict = st.columns(3)
    model.metric("Model error (MAE)", f"{evaluation.mae:.2f} places")
    baseline.metric("Baseline error (MAE)", f"{evaluation.baseline_mae:.2f} places")
    verdict.metric("Better on later laps", "Model" if evaluation.beats_baseline else "Baseline")
    st.caption(
        f"Trained on {evaluation.n_train} laps up to lap {evaluation.cut_lap}, "
        f"tested on {evaluation.n_test} laps after it."
    )
    if not evaluation.beats_baseline:
        st.warning(
            "The model loses to the baseline in this race, so trust \"position stays the same\" over the "
            "number below. Positions rarely change over a few laps, which makes that baseline hard to beat."
        )

    last_lap = int(session.laps["LapNumber"].max())
    field = len(session.drivers)
    compounds = sorted(set(session.laps["Compound"]) & set(config.COMPOUND_CODES))
    key = session.key
    first, second = st.columns(2)
    lap = first.slider("Current lap", 1, last_lap, min(20, last_lap), key=f"pred_lap_{key}")
    tyre_life = second.slider("Tyre age (laps)", 1, 60, 15, key=f"pred_age_{key}")
    third, fourth = st.columns(2)
    current = third.slider("Current position", 1, field, min(5, field), key=f"pred_pos_{key}")
    compound = fourth.selectbox("Compound", compounds, key=f"pred_compound_{key}")

    predicted = position.predict(evaluation, lap, tyre_life, current, compound, field)
    st.metric(f"Predicted position in {config.PREDICTION_HORIZON} laps", f"P{predicted:.1f}", f"Now P{current}", delta_color="off", delta_arrow="off")

    with st.expander("Method"):
        st.markdown(
            "A gradient boosting regressor uses lap number, tyre age, current position and compound, all known "
            f"at prediction time. The target is the same driver's position {config.PREDICTION_HORIZON} laps "
            "later, and laps with no known future are dropped. The split is by lap: the last "
            f"{config.TEST_FRACTION:.0%} of the race is held out, and training rows whose target lands in that "
            "period are removed. The baseline predicts the current position. The model sees one race only, and "
            "tree models cannot extrapolate, so lap number adds little on laps it never trained on."
        )
