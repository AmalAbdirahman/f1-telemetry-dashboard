"""Pit safety tab: the chance of rejoining clear of a rival."""

import streamlit as st

from f1dash import data
from f1dash.models import overtake, pit
from f1dash.ui import charts, formatting

DEFAULT_LOSS, DEFAULT_SIGMA = 22.0, 1.0
SAFE, RISKY = 0.9, 0.5


def render(session: data.Session) -> None:
    st.markdown("Estimates what a stop costs in this race, then the chance of rejoining clear of a rival.")
    if not session.is_race:
        st.info("Pit loss can only be measured in a race or sprint.")
        return

    clean = data.clean_laps(session.laps)
    estimate = pit.estimate_pit_loss(session.laps, clean) if not clean.empty else None
    key = session.key

    if estimate is None:
        st.info("No green-flag stops to measure here, so the sliders start from typical values.")
        loss_default, sigma_default = DEFAULT_LOSS, DEFAULT_SIGMA
    else:
        spread = "one stop, so no spread" if estimate.sd is None else f"spread {estimate.sd:.1f} s"
        st.caption(f"Measured from this session: {estimate.mean:.1f} s across {estimate.n} green-flag stops ({spread}).")
        loss_default = estimate.mean
        sigma_default = DEFAULT_SIGMA if estimate.sd is None else estimate.sd

    names = session.drivers["Driver"].tolist()
    left, right = st.columns(2)
    rival = left.selectbox("Rival on track", names, key=f"rival_{key}")
    gap = right.number_input("Gap to rival now (s)", 0.0, 90.0, 26.0, 0.5, key=f"pit_gap_{key}")

    first, second = st.columns(2)
    loss = first.slider("Pit loss (s)", 10.0, 45.0, round(min(max(loss_default, 10.0), 45.0), 1), 0.1, key=f"pit_loss_{key}")
    clearance = second.slider("Clearance needed (s)", 0.0, 3.0, 1.0, 0.1, key=f"clearance_{key}", help="A rough allowance for dirty air.")

    rival_laps = clean[clean["Driver"] == rival].sort_values("LapNumber")["LapTimeSec"] if not clean.empty else []
    rival_sd = overtake.recent_form(rival_laps).sd if len(rival_laps) >= 2 else 0.0

    mode = st.radio("Stop uncertainty", ["Measured", "Manual phases"], horizontal=True, key=f"pit_mode_{key}")
    if mode == "Measured":
        stops = st.slider("Spread across stops σ (s)", 0.1, 20.0, round(min(max(sigma_default, 0.1), 20.0), 1), 0.1, key=f"pit_sigma_{key}")
        sigma = pit.total_sigma(stops, rival_sd)
    else:
        entry_col, stationary_col, launch_col = st.columns(3)
        entry = entry_col.slider("Entry σ (s)", 0.0, 2.0, 0.3, 0.05, key=f"sigma_entry_{key}")
        stationary = stationary_col.slider("Stationary σ (s)", 0.0, 3.0, 0.5, 0.05, key=f"sigma_stationary_{key}")
        launch = launch_col.slider("Launch σ (s)", 0.0, 2.0, 0.2, 0.05, key=f"sigma_launch_{key}")
        sigma = pit.total_sigma(entry, stationary, launch, rival_sd)
        st.caption("These three values are assumptions: timing data cannot separate the phases of a stop.")

    probability = pit.p_safe(gap, loss, sigma, clearance)
    verdict = "Safe" if probability >= SAFE else "Risky" if probability >= RISKY else "Unsafe"
    chance, margin, total = st.columns(3)
    chance.metric("Safe rejoin", formatting.percent(probability), verdict, delta_color="off", delta_arrow="off")
    margin.metric("Expected gap after the stop", f"{gap - loss:+.1f} s")
    total.metric("Total σ", f"{sigma:.2f} s", f"includes {rival} pace σ {rival_sd:.2f} s", delta_color="off", delta_arrow="off")

    x, y = pit.rejoin_curve(gap, loss, sigma)
    st.plotly_chart(charts.rejoin_figure(x, y, clearance, rival), width="stretch", key=f"rejoin_{key}")

    with st.expander("Method"):
        st.markdown(
            "One stop's loss is (in-lap + out-lap) − 2 × the driver's median clean lap, pooled over all stops "
            "where both laps were green. Safety-car stops look cheap and red-flag stops last many minutes, so "
            "both are left out.\n\n"
            "P_safe = P(gap − L > clearance), with L ~ Normal(μ, σ²). Independent variances add. In Measured "
            "mode, σ² is the spread across stops plus the rival's lap variance. The spread already contains the "
            "entry, stationary and launch phases, so adding them too would count them twice. Manual mode replaces "
            "the spread with three assumed phase variances.\n\n"
            "**Limitation:** the rival's movement during the stop is covered only by their lap variance, and a "
            "slow stop or a changing track widens the measured spread."
        )
