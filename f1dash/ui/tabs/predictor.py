"""Race position predictor tab."""

import streamlit as st

from f1dash import config, data
from f1dash.models import position
from f1dash.ui import formatting, state


def render(session: data.Session, driver_1: str, lap_n: int | None) -> None:
    st.markdown(f"Predicts a driver's position {config.PREDICTION_HORIZON} laps ahead, scored against simply assuming nothing changes.")
    if not session.is_race or lap_n is None:
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

    snapshot = state.race_state(session.laps, lap_n)
    known = snapshot[snapshot["Compound"].isin(config.COMPOUND_CODES) & snapshot["TyreLife"].notna()]
    if known.empty:
        st.info(f"No car has tyre data at lap {lap_n}.")
        return
    names = known["Driver"].tolist()
    key = session.key
    driver = st.selectbox("Driver", names, index=names.index(driver_1) if driver_1 in names else 0, key=f"pred_driver_{key}")
    now = known.set_index("Driver").loc[driver]
    field = len(session.drivers)
    compounds = sorted(set(session.laps["Compound"]) & set(config.COMPOUND_CODES))

    # Start from the driver's real state at lap N; keys carry the context so a what-if resets with it.
    context = f"{key}_{lap_n}_{driver}"
    with st.expander(f"Inputs, from {driver} at the end of lap {lap_n}"):
        first, second = st.columns(2)
        tyre_life = first.slider("Tyre age (laps)", 1, 80, int(now["TyreLife"]), key=f"pred_age_{context}")
        current = second.slider("Current position", 1, field, int(now["Position"]), key=f"pred_pos_{context}")
        compound = st.selectbox("Compound", compounds, index=compounds.index(now["Compound"]), key=f"pred_compound_{context}")

    predicted = position.predict(evaluation, lap_n, tyre_life, current, compound, field)
    st.metric(
        f"Predicted position after lap {lap_n + config.PREDICTION_HORIZON}",
        formatting.position(predicted),
        f"Now P{current} · raw model output {predicted:.1f}",
        delta_color="off",
        delta_arrow="off",
    )

    with st.expander("Method"):
        st.markdown(
            "A gradient boosting regressor uses lap number, tyre age, current position and compound, all known "
            f"at prediction time. The target is the same driver's position {config.PREDICTION_HORIZON} laps "
            "later, and laps with no known future are dropped. The split is by lap: the last "
            f"{config.TEST_FRACTION:.0%} of the race is held out, and training rows whose target lands in that "
            "period are removed. The baseline predicts the current position. The model sees one race only, and "
            "tree models cannot extrapolate, so lap number adds little on laps it never trained on."
        )
