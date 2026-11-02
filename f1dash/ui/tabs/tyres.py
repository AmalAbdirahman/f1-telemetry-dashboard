"""Tyre degradation tab."""

import streamlit as st

from f1dash import config, data
from f1dash.models import degradation
from f1dash.ui import charts, state

LOSS_TARGET_S = 1.0


def render(session: data.Session, driver_1: str, driver_2: str) -> None:
    st.markdown("How much time each driver lost per lap of tyre age, stint by stint.")
    drivers = [driver_1, driver_2]
    clean = data.clean_laps(session.laps)
    clean = clean[clean["Driver"].isin(drivers) & clean["TyreLife"].notna()]
    if clean.empty:
        st.info("Neither driver has clean green-flag laps with tyre data in this session.")
        return

    columns = ["Driver", "Stint", "Compound", "LapNumber", "LapTimeSec", "TyreLife"]
    fits = state.stint_fits(clean[columns])
    st.plotly_chart(charts.tyre_figure(clean, fits, drivers), width="stretch", key=f"tyres_{session.key}")

    if fits.empty:
        st.info(f"No stint has the {config.MIN_STINT_LAPS} clean laps needed for a fit, which is usual outside races.")
        return

    table = fits.assign(
        Stint=fits["Stint"].astype(int),
        Compound=fits["Compound"].str.title(),
        Interval=[f"{low:+.3f} to {high:+.3f}" for low, high in zip(fits["CILow"], fits["CIHigh"])],
        LapsToLose=[degradation.laps_to_lose(slope, LOSS_TARGET_S) for slope in fits["Slope"]],
    )
    st.dataframe(
        table[["Driver", "Stint", "Compound", "Slope", "Interval", "R2", "RMSE", "Laps", "LapsToLose"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Slope": st.column_config.NumberColumn("Slope (s/lap)", format="%+.3f"),
            "Interval": st.column_config.TextColumn("90% interval"),
            "R2": st.column_config.NumberColumn("R²", format="%.2f"),
            "RMSE": st.column_config.NumberColumn("RMSE (s)", format="%.3f"),
            "LapsToLose": st.column_config.NumberColumn(f"Laps to lose {LOSS_TARGET_S:.0f} s", format="%.0f"),
        },
    )
    st.caption(
        "A negative slope means the car got faster on older tyres, usually a drying or improving track. "
        "Laps to lose is a linear heuristic, not a prediction of a tyre cliff."
    )

    with st.expander("Method"):
        st.markdown(
            f"Pace is lap time + {config.FUEL_EFFECT} × lap number, which cancels the speed gained as fuel "
            "burns. Each driver's stint gets a Huber robust line against tyre age, using green-flag, non-pit, "
            f"accurately timed laps, and needs at least {config.MIN_STINT_LAPS} of them. R² and RMSE come from "
            f"that line's residuals. The interval is a seeded bootstrap with {config.BOOTSTRAP_RESAMPLES} "
            "resamples.\n\n"
            f"**Limitation:** tyre age and lap number rise together within a stint, so the {config.FUEL_EFFECT} "
            "s/lap fuel figure adds directly to every slope. If the true fuel effect is larger, every slope here "
            "is too low by the difference. Compare stints with each other rather than reading a slope as an "
            "absolute figure."
        )
