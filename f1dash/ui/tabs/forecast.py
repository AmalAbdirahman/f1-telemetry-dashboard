"""Overtake forecast tab: Monte Carlo simulation of the gap."""

import numpy as np
import streamlit as st

from f1dash import config, data
from f1dash.models import overtake
from f1dash.ui import charts, formatting, state


def _latest_slope(fits, driver: str) -> float:
    own = fits[fits["Driver"] == driver]
    return float(own["Slope"].iloc[-1]) if not own.empty else 0.0


def render(session: data.Session) -> None:
    st.markdown("Simulates the laps ahead many times to estimate whether a chasing car gets ahead on the clock.")
    if not session.is_race:
        st.info("This needs a race or sprint, where two cars run the same laps against each other.")
        return

    clean = data.clean_laps(session.laps).sort_values("LapNumber")
    counts = clean["Driver"].value_counts()
    names = [d for d in session.drivers["Driver"] if counts.get(d, 0) >= config.MIN_RECENT_LAPS]
    if len(names) < 2:
        st.info(f"This needs two drivers with at least {config.MIN_RECENT_LAPS} clean green-flag laps.")
        return

    key = session.key
    left, right = st.columns(2)
    chaser = left.selectbox("Chasing driver", names, index=1, key=f"chaser_{key}")
    leaders = [name for name in names if name != chaser]
    leader = right.selectbox("Leading driver", leaders, key=f"leader_{key}")
    gap_col, laps_col, rho_col = st.columns(3)
    gap = gap_col.number_input("Current gap (s)", 0.0, 60.0, 3.0, 0.1, key=f"gap_{key}")
    n_laps = laps_col.slider("Laps remaining", 1, 40, 15, key=f"laps_left_{key}")

    chaser_times = clean.loc[clean["Driver"] == chaser, "LapTimeSec"].to_numpy()
    leader_times = clean.loc[clean["Driver"] == leader, "LapTimeSec"].to_numpy()
    form_chaser, form_leader = overtake.recent_form(chaser_times), overtake.recent_form(leader_times)
    estimated = float(
        np.mean(
            [
                overtake.estimate_rho(chaser_times[-config.RECENT_LAPS :]),
                overtake.estimate_rho(leader_times[-config.RECENT_LAPS :]),
            ]
        )
    )
    rho = rho_col.slider(
        "Lap correlation ρ",
        0.0,
        config.RHO_MAX,
        round(estimated, 2),
        0.05,
        key=f"rho_{key}",
        help=f"Estimated from recent laps as {estimated:.2f}. Higher means good and bad laps come in runs.",
    )

    subset = clean[clean["Driver"].isin([chaser, leader]) & clean["TyreLife"].notna()]
    fits = state.stint_fits(subset[["Driver", "Stint", "Compound", "LapNumber", "LapTimeSec", "TyreLife"]])
    slope_chaser, slope_leader = _latest_slope(fits, chaser), _latest_slope(fits, leader)

    gaps = overtake.simulate_gaps(
        gap,
        form_chaser.mean,
        form_chaser.sd,
        form_leader.mean,
        form_leader.sd,
        n_laps,
        slope_chaser=slope_chaser,
        slope_leader=slope_leader,
        rho=rho,
    )
    summary = overtake.summarise(gaps)

    any_lap, at_flag, catch = st.columns(3)
    any_lap.metric("Ahead at any point", formatting.percent(summary.p_any))
    at_flag.metric("Ahead at the flag", formatting.percent(summary.p_flag))
    catch.metric("Median lap to get ahead", "No catch" if summary.median_catch_lap is None else f"Lap +{summary.median_catch_lap}")
    st.plotly_chart(charts.gap_fan_figure(summary, chaser, leader), width="stretch", key=f"fan_{key}")
    st.caption(
        f"Based on recent clean laps: {chaser} {formatting.lap_time(form_chaser.mean)} (n = {form_chaser.n}, "
        f"σ = {form_chaser.sd:.2f} s, trend {slope_chaser:+.3f} s/lap); {leader} "
        f"{formatting.lap_time(form_leader.mean)} (n = {form_leader.n}, σ = {form_leader.sd:.2f} s, "
        f"trend {slope_leader:+.3f} s/lap). Uses each driver's end-of-race form, whichever gap you enter."
    )

    with st.expander("Method"):
        st.markdown(
            f"Each driver's future lap k is their recent mean (last {config.RECENT_LAPS} clean laps at most) "
            "plus their tyre trend × k, plus an AR(1) deviation so that good and bad laps come in runs. "
            "The gap after n laps is gap₀ + Σ(chaser lap − leader lap), and the chaser is ahead once it is "
            f"negative. There are {config.N_SIMULATIONS:,} seeded simulations, and the shaded band holds the "
            f"middle {overtake.BAND_COVERAGE}%.\n\n"
            "**Limitation:** a negative gap only means quicker on the clock. Passing also needs a way past: "
            "dirty air and defending are not modelled, and neither is the overtaking aid of the era (DRS before "
            "2026, the new overtake mode from 2026). Pit stops and safety cars are left out too."
        )
