"""Driver Face-Off: two laps compared along the track."""

import streamlit as st

from f1dash import data
from f1dash.models import telemetry
from f1dash.openf1 import OpenF1Error
from f1dash.ui import charts, formatting, layout, state, theme


def _pick_lap(session: data.Session, driver: str, lap_number: int | None):
    laps = session.laps[session.laps["Driver"] == driver]
    if lap_number is None:
        return data.fastest_lap(laps)
    chosen = laps[(laps["LapNumber"] == lap_number) & laps["LapEnd"].notna()]
    return None if chosen.empty else chosen.iloc[0]


def _load(session: data.Session, lap):
    return state.load_telemetry(session.key, int(lap["DriverNumber"]), lap["LapStart"], lap["LapEnd"])


def _show_sectors(lap_1, lap_2, driver_1: str, driver_2: str) -> None:
    st.markdown("**Sector deltas**")
    deltas = telemetry.sector_deltas(lap_1, lap_2)
    for column, row in zip(st.columns(len(deltas)), deltas.itertuples()):
        column.metric(
            row.Sector,
            formatting.signed_seconds(row.Delta),
            formatting.who_gained(row.Delta, driver_1, driver_2),
            delta_color="off",
            delta_arrow="off",
        )
    st.caption(f"Deltas are {driver_1} minus {driver_2}: negative means {driver_1} was faster.")


def render(session: data.Session) -> None:
    st.subheader("Driver Face-Off")
    names = session.drivers["Driver"].tolist()
    left, middle, right = st.columns(3)
    driver_1 = left.selectbox("Driver 1", names, key=f"driver_1_{session.key}")
    rivals = [name for name in names if name != driver_1]
    driver_2 = middle.selectbox("Driver 2", rivals, key=f"driver_2_{session.key}")
    mode = right.radio("Lap", ["Fastest", "Specific lap"], horizontal=True, key=f"lap_mode_{session.key}")

    lap_number = None
    if mode == "Specific lap":
        last_lap = int(session.laps["LapNumber"].max())
        lap_number = st.slider("Lap number", 1, last_lap, min(10, last_lap), key=f"lap_number_{session.key}")

    lap_1, lap_2 = _pick_lap(session, driver_1, lap_number), _pick_lap(session, driver_2, lap_number)
    missing = [d for d, lap in ((driver_1, lap_1), (driver_2, lap_2)) if lap is None]
    if missing:
        where = f"lap {lap_number}" if lap_number else "a timed lap"
        st.info(f"{' and '.join(missing)} did not complete {where}, so there is nothing to compare.")
        return

    try:
        tel_1, tel_2 = _load(session, lap_1), _load(session, lap_2)
    except OpenF1Error as error:
        st.warning(str(error))
        return
    if tel_1.empty or tel_2.empty:
        st.info("OpenF1 has no car data for one of these laps. Lap times are still compared below.")
        _show_sectors(lap_1, lap_2, driver_1, driver_2)
        return

    gap = lap_1["LapTimeSec"] - lap_2["LapTimeSec"]
    speed = telemetry.speed_delta(tel_1, tel_2)
    first, second, third = st.columns(3)
    first.metric(driver_1, formatting.lap_time(lap_1["LapTimeSec"]), f"Lap {int(lap_1['LapNumber'])}", delta_color="off", delta_arrow="off")
    second.metric(driver_2, formatting.lap_time(lap_2["LapTimeSec"]), f"Lap {int(lap_2['LapNumber'])}", delta_color="off", delta_arrow="off")
    third.metric(
        "Mean speed delta",
        f"{speed:+.1f} km/h",
        f"{driver_1 if speed > 0 else driver_2} quicker on average",
        delta_color="off",
        delta_arrow="off",
    )
    st.caption(f"{formatting.who_gained(gap, driver_1, driver_2)} by {abs(gap):.3f} s over the lap.")

    styles = charts.driver_styles(session.drivers, driver_1, driver_2)
    layout.chart(charts.telemetry_figure(tel_1, tel_2, *styles), key=f"telemetry_{session.key}")
    _show_sectors(lap_1, lap_2, driver_1, driver_2)

    with st.expander("Method"):
        st.markdown(
            "OpenF1 has no distance channel, so distance is speed integrated over time from about four "
            "samples a second. Both laps are stretched to the same length and interpolated onto a shared "
            "500-point grid, so traces compare the same point on track. The time delta is "
            f"t({driver_1}) − t({driver_2}) at each point: where the line falls, {driver_1} is gaining. "
            "It finishes exactly on the official lap time gap."
        )
    theme.divider()
