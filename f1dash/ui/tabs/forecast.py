"""Overtake forecast tab: Monte Carlo simulation of the gap, starting from lap N."""

import numpy as np
import pandas as pd
import streamlit as st

from f1dash import config, data
from f1dash.models import overtake, race_state
from f1dash.ui import charts, formatting, layout, state


def _pick_pair(session: data.Session, names: list[str], driver_1: str, driver_2: str) -> tuple[str, str]:
    key = session.key
    left, right = st.columns(2)
    first = left.selectbox("Car", names, index=names.index(driver_1) if driver_1 in names else 0, key=f"ot_a_{key}")
    rivals = [n for n in names if n != first]
    default = rivals.index(driver_2) if driver_2 in rivals else 0
    second = right.selectbox("Against", rivals, index=default, key=f"ot_b_{key}")
    return first, second


def render(session: data.Session, driver_1: str, driver_2: str, lap_n: int | None) -> None:
    st.markdown("From the chosen lap, simulates the rest of the race many times: does the car behind get ahead on the clock?")
    if not session.is_race or lap_n is None:
        st.info("This needs a race or sprint, where two cars run the same laps against each other.")
        return

    snapshot = state.race_state(session.laps, lap_n)
    names = [d for d in session.drivers["Driver"] if d in set(snapshot["Driver"])]
    if len(names) < 2:
        st.info(f"Fewer than two cars finished lap {lap_n}.")
        return
    first, second = _pick_pair(session, names, driver_1, driver_2)
    rows = snapshot.set_index("Driver")
    chaser, leader = sorted((first, second), key=lambda d: rows.loc[d, "Position"], reverse=True)
    chasing, leading = rows.loc[chaser], rows.loc[leader]

    chaser_times = race_state.recent_laps_at(session.laps, lap_n, chaser)
    leader_times = race_state.recent_laps_at(session.laps, lap_n, leader)
    short = [d for d, t in ((chaser, chaser_times), (leader, leader_times)) if len(t) < config.MIN_RECENT_LAPS]
    if short:
        st.info(
            f"By lap {lap_n}, {' and '.join(short)} had fewer than {config.MIN_RECENT_LAPS} clean green-flag laps, "
            "which is too few to judge pace. Move the Race lap slider later."
        )
        return
    if chasing["LapsDown"] != leading["LapsDown"]:
        st.info(f"At lap {lap_n}, {chaser} is a lap or more down on {leader}, so they are not racing for position.")
        return

    remaining = int(session.laps["LapNumber"].max()) - lap_n
    if remaining < 1:
        st.info("There are no laps left after this one. Move the Race lap slider earlier.")
        return

    form_chaser, form_leader = overtake.recent_form(chaser_times), overtake.recent_form(leader_times)
    measured = {
        "gap": float(chasing["GapToLeader"] - leading["GapToLeader"]),
        "laps": remaining,
        "mu_chaser": form_chaser.mean,
        "mu_leader": form_leader.mean,
        "slope_chaser": 0.0 if pd.isna(chasing["Slope"]) else float(chasing["Slope"]),
        "slope_leader": 0.0 if pd.isna(leading["Slope"]) else float(leading["Slope"]),
        "rho": float(np.mean([overtake.estimate_rho(chaser_times), overtake.estimate_rho(leader_times)])),
    }

    st.caption(
        f"At the end of lap {lap_n}: **{chaser}** is {measured['gap']:.3f} s behind **{leader}** "
        f"with {remaining} laps to go."
    )
    # Keys carry the lap and pair, so the overrides reset to the measured values when either changes.
    context = f"{session.key}_{lap_n}_{chaser}_{leader}"
    with st.expander("Override the measured numbers"):
        one, two = st.columns(2)
        gap = one.number_input("Gap (s)", 0.0, 120.0, round(measured["gap"], 3), 0.1, key=f"ot_gap_{context}")
        n_laps = two.number_input("Laps remaining", 1, 80, measured["laps"], 1, key=f"ot_laps_{context}")
        three, four = st.columns(2)
        mu_chaser = three.number_input(f"{chaser} pace (s)", 40.0, 200.0, round(measured["mu_chaser"], 3), 0.05, key=f"ot_mu_c_{context}")
        mu_leader = four.number_input(f"{leader} pace (s)", 40.0, 200.0, round(measured["mu_leader"], 3), 0.05, key=f"ot_mu_l_{context}")
        five, six = st.columns(2)
        slope_chaser = five.number_input(f"{chaser} trend (s/lap)", -1.0, 1.0, round(measured["slope_chaser"], 3), 0.005, format="%.3f", key=f"ot_sl_c_{context}")
        slope_leader = six.number_input(f"{leader} trend (s/lap)", -1.0, 1.0, round(measured["slope_leader"], 3), 0.005, format="%.3f", key=f"ot_sl_l_{context}")
        rho = st.slider(
            "Lap correlation ρ", 0.0, config.RHO_MAX, round(min(measured["rho"], config.RHO_MAX), 2), 0.05,
            key=f"rho_{context}", help="Higher means good and bad laps come in runs.",
        )

    gaps = overtake.simulate_gaps(
        gap, mu_chaser, form_chaser.sd, mu_leader, form_leader.sd, int(n_laps),
        slope_chaser=slope_chaser, slope_leader=slope_leader, rho=rho,
    )
    summary = overtake.summarise(gaps)

    any_lap, at_flag, catch = st.columns(3)
    any_lap.metric("Ahead at any point", formatting.percent(summary.p_any))
    at_flag.metric("Ahead at the flag", formatting.percent(summary.p_flag))
    catch.metric(
        "If it happens, median lap",
        "No catch" if summary.median_catch_lap is None else f"Lap {lap_n + summary.median_catch_lap}",
    )
    layout.chart(charts.gap_fan_figure(summary, chaser, leader, first_lap=lap_n), key=f"fan_{session.key}")

    def describe(driver, form, slope):
        trend = "no trend yet" if pd.isna(rows.loc[driver, "Slope"]) else f"trend {slope:+.3f} s/lap"
        return f"{driver} {formatting.lap_time(form.mean)} (n = {form.n}, σ = {form.sd:.2f} s, {trend})"

    st.caption(
        f"Pace from the last clean laps on each car's current tyres, up to lap {lap_n}: {describe(chaser, form_chaser, measured['slope_chaser'])}; "
        f"{describe(leader, form_leader, measured['slope_leader'])}. ρ estimated as {measured['rho']:.2f}."
    )

    with st.expander("Method"):
        st.markdown(
            f"Everything starts from the end of lap {lap_n} and uses nothing later. The gap is the difference "
            "between the two cars' line crossings on that lap. Each driver's future lap k is their recent mean "
            f"(last {config.RECENT_LAPS} clean laps at most, on the current tyres once they have "
            f"{config.MIN_RECENT_LAPS}) plus their current stint's tyre trend × k, plus an "
            "AR(1) deviation so that good and bad laps come in runs. The trend needs "
            f"{config.MIN_STINT_LAPS} clean laps on the current tyres, otherwise it is zero. "
            "The gap after n laps is gap₀ + Σ(chaser lap − leader lap), and the chaser is ahead once it is "
            f"negative. There are {config.N_SIMULATIONS:,} seeded simulations, and the shaded band holds the "
            f"middle {overtake.BAND_COVERAGE}%.\n\n"
            "**Limitation:** a negative gap only means quicker on the clock. Passing also needs a way past: "
            "dirty air and defending are not modelled, and neither is the overtaking aid of the era (DRS before "
            "2026, the new overtake mode from 2026). Pit stops and safety cars are left out too."
        )
