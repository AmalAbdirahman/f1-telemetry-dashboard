"""Race tab: race trace, tyre strategy and the order at lap N."""

import pandas as pd
import streamlit as st

from f1dash import config, data
from f1dash.models import race_overview
from f1dash.ui import charts, formatting, layout, state


def order_table(snapshot: pd.DataFrame, phone: bool) -> pd.DataFrame:
    """The timing screen at lap N, with fewer columns on a phone."""
    table = pd.DataFrame(
        {
            "Pos": snapshot["Position"],
            "Driver": snapshot["Driver"],
            "Gap": [formatting.gap(g, d) if p > 1 else "Leader"
                    for g, d, p in zip(snapshot["GapToLeader"], snapshot["LapsDown"], snapshot["Position"])],
            "Interval": [formatting.gap(i, d) for i, d in zip(snapshot["Interval"], snapshot["IntervalLapsDown"])],
            "Tyre": [formatting.tyre(c, a) + (" · pit lap" if pit else "")
                     for c, a, pit in zip(snapshot["Compound"], snapshot["TyreLife"], snapshot["InPitLane"])],
            "Stops": snapshot["Stops"],
            "Pace": [(formatting.lap_time(p) + ("" if own_set else " *")) if pd.notna(p) else "—"
                     for p, own_set in zip(snapshot["RecentPace"], snapshot["PaceFromStint"])],
            "Trend": [formatting.slope(v) for v in snapshot["Slope"]],
        }
    )
    return table[["Pos", "Driver", "Gap", "Tyre"]] if phone else table


def render(session: data.Session, driver_1: str, driver_2: str, lap_n: int | None) -> None:
    st.markdown("How the race unfolded, and where everyone stood at the chosen lap.")
    if not session.is_race or lap_n is None:
        st.info("This tab needs a race or sprint with lap times, where every car runs the same laps.")
        return

    key = session.key
    reference = race_overview.reference_driver(session.drivers, session.laps)
    trace = race_overview.race_trace(session.laps, reference) if reference else pd.DataFrame()
    if not trace.empty:
        st.markdown("**Race trace**")
        spans = race_overview.neutralised_spans(session.laps)
        dashed = race_overview.teammates_dashed(session.drivers)
        fig = charts.race_trace_figure(trace, spans, session.drivers, (driver_1, driver_2), dashed, reference, lap_n)
        layout.chart(fig, key=f"race_trace_{key}")
        st.caption(
            f"Lower is further behind. A flat line matches {reference}'s average pace and a rising one is "
            f"faster than it. Circles are pit stops; shaded laps were under a safety car, virtual safety car "
            f"or red flag. {driver_1} and {driver_2} (the Face-Off drivers) are drawn bold."
        )

    st.markdown("**Tyre strategy**")
    bars = race_overview.stint_bars(session.laps)
    order = [d for d in session.drivers["Driver"] if d in set(bars["Driver"])]
    layout.chart(charts.strategy_figure(bars, order, lap_n), key=f"strategy_{key}")

    st.markdown(f"**Order at the end of lap {lap_n}**")
    snapshot = state.race_state(session.laps, lap_n)
    if snapshot.empty:
        st.info(f"No car has a recorded finish for lap {lap_n}.")
        return
    st.dataframe(
        order_table(snapshot, layout.is_phone()),
        hide_index=True,
        width="stretch",
        column_config={
            "Pace": st.column_config.TextColumn(
                "Recent pace",
                help=f"Mean of the last {config.RECENT_LAPS} clean laps on the current tyres. * means the set has "
                f"fewer than {config.MIN_RECENT_LAPS} clean laps yet, so earlier laps are used.",
            ),
            "Trend": st.column_config.TextColumn("Trend (s/lap)", help="Current stint's fuel-corrected slope."),
        },
    )

    with st.expander("Method"):
        st.markdown(
            "**Race trace.** For each lap n, a car's line is its time to finish lap n minus n × the reference "
            f"driver's average lap ({reference}, the winner when there is a result). The reference finishes on "
            "zero and the axis is flipped so the leaders are on top. Laps are shaded when most cars' laps "
            "overlapped a safety car, virtual safety car or red flag.\n\n"
            f"**Order at lap {lap_n}.** Cars are ordered by when they crossed the line to finish lap {lap_n}, and "
            "gaps are the difference between those crossings. Whether a car is lapped is estimated as "
            "⌊gap ÷ the leader's lap time⌋, because checking it exactly would need the next lap. Pace and trend "
            f"use only clean laps up to lap {lap_n}; the trend needs {config.MIN_STINT_LAPS} clean laps on the "
            "current set of tyres."
        )
