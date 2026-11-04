"""Pit safety tab: the chance of rejoining clear of a rival, and the pit window from lap N."""

import pandas as pd
import streamlit as st

from f1dash import data
from f1dash.models import overtake, pit, race_state
from f1dash.ui import charts, formatting, layout, state, theme

DEFAULT_LOSS, DEFAULT_SIGMA = 22.0, 1.0
SAFE, RISKY = 0.9, 0.5


def _verdict(probability: float) -> str:
    return "Safe" if probability >= SAFE else "Risky" if probability >= RISKY else "Unsafe"


def _car(name, gap) -> str:
    return "—" if pd.isna(name) else f"{name} {gap:.1f} s"


def _clean_air(row) -> str:
    return "✓ Clean air" if row["CleanAir"] else f"✗ {row['RoadAhead']} {row['RoadGapAhead']:.1f} s ahead"


def window_table(window: pd.DataFrame, phone: bool) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Pit on lap": window["PitLap"],
            "Rejoin": [formatting.position(p) for p in window["Position"]],
            "Car ahead": [_car(n, g) for n, g in zip(window["Ahead"], window["GapAhead"])],
            "Car behind": [_car(n, g) for n, g in zip(window["Behind"], window["GapBehind"])],
            "On the road": [_clean_air(row) for _, row in window.iterrows()],
        }
    )
    return table[["Pit on lap", "Rejoin", "On the road"]] if phone else table


def render(session: data.Session, driver_1: str, lap_n: int | None) -> None:
    st.markdown("What a stop costs in this race, whether it comes out clear of a rival, and where it rejoins.")
    if not session.is_race or lap_n is None:
        st.info("Pit loss can only be measured in a race or sprint.")
        return

    key = session.key
    snapshot = state.race_state(session.laps, lap_n)
    names = snapshot["Driver"].tolist()
    if len(names) < 2:
        st.info(f"Fewer than two cars finished lap {lap_n}.")
        return
    rows = snapshot.set_index("Driver")

    left, right = st.columns(2)
    driver = left.selectbox("Pitting driver", names, index=names.index(driver_1) if driver_1 in names else 0, key=f"pit_driver_{key}")
    rivals = [n for n in names if n != driver]
    position = int(rows.loc[driver, "Position"])
    # The natural rival is the car right behind, or the one ahead for the last car.
    default_rival = names[position] if position < len(names) else names[position - 2]
    rival = right.selectbox("Rival on track", rivals, index=rivals.index(default_rival), key=f"rival_{key}_{driver}")

    estimate, before_n = pit.loss_known_at(session.laps, lap_n)
    if estimate is None:
        st.info("No green-flag stops to measure in this race, so the loss starts from a typical value.")
        loss_default, sigma_default = DEFAULT_LOSS, DEFAULT_SIGMA
    else:
        spread = "one stop, so no spread" if estimate.sd is None else f"spread {estimate.sd:.1f} s"
        source = f"from stops completed by lap {lap_n}" if before_n else (
            f"no green-flag stop was complete by lap {lap_n}, so this uses the whole race's stops"
        )
        st.caption(f"Pit loss {estimate.mean:.1f} s across {estimate.n} green-flag stops ({spread}), {source}.")
        loss_default = estimate.mean
        sigma_default = DEFAULT_SIGMA if estimate.sd is None else estimate.sd

    measured_gap = float(rows.loc[rival, "GapToLeader"] - rows.loc[driver, "GapToLeader"])
    rival_laps = race_state.recent_laps_at(session.laps, lap_n, rival)
    rival_sd = overtake.recent_form(rival_laps).sd if len(rival_laps) >= 2 else 0.0
    clearance = st.slider("Clearance needed (s)", 0.0, 3.0, 1.0, 0.1, key=f"clearance_{key}", help="A rough allowance for dirty air.")

    context = f"{key}_{lap_n}_{driver}_{rival}"
    with st.expander("Override the measured numbers"):
        one, two = st.columns(2)
        gap = one.number_input(
            f"Gap from {driver} back to {rival} (s)", -150.0, 150.0, round(measured_gap, 3), 0.5, key=f"pit_gap_{context}",
            help="Negative when the rival is ahead.",
        )
        loss = two.slider("Pit loss (s)", 10.0, 45.0, round(min(max(loss_default, 10.0), 45.0), 1), 0.1, key=f"pit_loss_{key}_{lap_n}")
        mode = st.radio("Stop uncertainty", ["Measured", "Manual phases"], horizontal=True, key=f"pit_mode_{key}")
        if mode == "Measured":
            stops = st.slider("Spread across stops σ (s)", 0.1, 20.0, round(min(max(sigma_default, 0.1), 20.0), 1), 0.1, key=f"pit_sigma_{key}_{lap_n}")
            sigma = pit.total_sigma(stops, rival_sd)
        else:
            entry_col, stationary_col, launch_col = st.columns(3)
            entry = entry_col.slider("Entry σ (s)", 0.0, 2.0, 0.3, 0.05, key=f"sigma_entry_{key}")
            stationary = stationary_col.slider("Stationary σ (s)", 0.0, 3.0, 0.5, 0.05, key=f"sigma_stationary_{key}")
            launch = launch_col.slider("Launch σ (s)", 0.0, 2.0, 0.2, 0.05, key=f"sigma_launch_{key}")
            sigma = pit.total_sigma(entry, stationary, launch, rival_sd)
            st.caption("These three values are assumptions: timing data cannot separate the phases of a stop.")

    where = "behind" if measured_gap >= 0 else "ahead of"
    st.caption(f"At the end of lap {lap_n}, {rival} is {abs(measured_gap):.3f} s {where} {driver}.")

    probability = pit.p_safe(gap, loss, sigma, clearance)
    chance, margin, total = st.columns(3)
    chance.metric(f"Rejoins ahead of {rival}", formatting.percent(probability), _verdict(probability), delta_color="off", delta_arrow="off")
    margin.metric("Expected gap after the stop", f"{gap - loss:+.1f} s")
    total.metric("Total σ", f"{sigma:.2f} s", f"includes {rival} pace σ {rival_sd:.2f} s", delta_color="off", delta_arrow="off")
    x, y = pit.rejoin_curve(gap, loss, sigma)
    layout.chart(charts.rejoin_figure(x, y, clearance, rival), key=f"rejoin_{key}")

    theme.divider()
    st.markdown(f"**If {driver} pits: the window from lap {lap_n}**")
    last_lap = int(session.laps["LapNumber"].max())
    window = pit.pit_window(snapshot, driver, loss, clearance, lap_n, last_lap)
    if window.empty:
        st.info("There are no laps left to stop on.")
    else:
        now = window.iloc[0]
        here, ahead, behind = st.columns(3)
        here.metric(f"Rejoin if pitting on lap {lap_n}", formatting.position(now["Position"]), f"Now P{position}", delta_color="off", delta_arrow="off")
        ahead.metric("Car ahead", _car(now["Ahead"], now["GapAhead"]))
        behind.metric("Car behind", _car(now["Behind"], now["GapBehind"]))
        st.markdown(_clean_air(now) + (f" (more than {clearance:.1f} s to the next car on the road)" if now["CleanAir"] else ""))
        st.dataframe(window_table(window, layout.is_phone()), hide_index=True, width="stretch")
        st.caption(
            f"Laps after {lap_n} are projected from each car's pace and tyre trend at lap {lap_n}. Stops the other "
            "cars make later are not known yet, so the window gets less reliable further out."
        )

    with st.expander("Method"):
        st.markdown(
            "One stop's loss is (in-lap + out-lap) − 2 × the driver's median clean lap, pooled over stops "
            f"where both laps were green and finished by lap {lap_n}. If there are none yet, the whole race's "
            "stops are used and the caption says so. Safety-car stops look cheap and red-flag stops last many "
            "minutes, so both are left out.\n\n"
            "P_safe = P(gap − L > clearance), with L ~ Normal(μ, σ²). Independent variances add. In Measured "
            "mode, σ² is the spread across stops plus the rival's lap variance. The spread already contains the "
            "entry, stationary and launch phases, so adding them too would count them twice. Manual mode replaces "
            "the spread with three assumed phase variances.\n\n"
            "**Pit window.** Pitting on lap L adds the mean loss to the driver's time at the line on lap L and "
            "re-ranks everyone. For L after lap N, each car's time is projected as its recent pace plus its tyre "
            "trend, lap by lap. Clean air is checked on the road: the time gap to every car, lapped or not, is "
            "wrapped into half a lap either side, and the rejoin is clean when the nearest car in front is more "
            "than the clearance away.\n\n"
            "**Limitation:** the rival's movement during the stop is covered only by their lap variance, a slow "
            "stop or a changing track widens the measured spread, and the window cannot know about other cars' "
            "future stops or safety cars."
        )

