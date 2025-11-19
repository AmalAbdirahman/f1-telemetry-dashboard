import streamlit as st
import fastf1
from fastf1 import plotting as f1plot   # ← this line is critical for team_color()
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ──────────────────────────────────────
# Fix fastf1 cache for Streamlit Cloud
# ──────────────────────────────────────
cache_dir = "/tmp/fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Optional: nicer matplotlib backend
f1plot.setup_mpl(mpl_backend="Agg")

# ──────────────────────────────────────
# App config & title
# ──────────────────────────────────────
st.set_page_config(page_title="F1 Telemetry & Strategy Dashboard", layout="wide")
st.title("Formula 1 Telemetry & Strategy Dashboard")
st.markdown("Interactive telemetry analysis using `fastf1`")

# ──────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────
st.sidebar.header("Race Selection")
year = st.sidebar.selectbox("Year", options=list(range(2018, 2026)), index=6)  # 2024 default
gp = st.sidebar.text_input("Grand Prix", "Monaco")
session_type = st.sidebar.radio("Session", ["R", "Q", "FP1", "FP2", "FP3", "S"], index=0)

if st.sidebar.button("Load Session", type="primary"):
    with st.spinner(f"Loading {year} {gp} {session_type}..."):
        try:
            session = fastf1.get_session(year, gp, session_type)
            session.load(telemetry=True, laps=True, weather=True)
            st.session_state.session = session
            st.success(f"Session loaded: {session.name}")
        except Exception as e:
            st.error(f"Failed to load session: {e}")

# ──────────────────────────────────────
# Main dashboard
# ──────────────────────────────────────
if "session" in st.session_state:
    session = st.session_state.session

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Drivers & Laps")
        drivers = session.results['Abbreviation'].dropna().unique().tolist()
        driver1 = st.selectbox("Driver 1", drivers, index=0)
        driver2 = st.selectbox("Driver 2", drivers, index=1 if len(drivers) > 1 else 0)

        choice = st.radio("Lap", ["Fastest Lap", "Specific Lap Number"])
        if choice == "Specific Lap Number":
            max_lap = int(session.laps['LapNumber'].max())
            lap_num = st.number_input("Lap Number", 1, max_lap, 10)
        else:
            lap_num = None

    with col2:
        # Get laps
        if lap_num:
            lap1 = session.laps.pick_driver(driver1).pick_lap(lap_num)
            lap2 = session.laps.pick_driver(driver2).pick_lap(lap_num)
        else:
            lap1 = session.laps.pick_driver(driver1).pick_fastest()
            lap2 = session.laps.pick_driver(driver2).pick_fastest()

        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()

        # Speed Trace – FIXED team colors
        st.subheader("Speed Trace Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Speed'],
            mode='lines', name=f"{driver1} – {lap1['Team']}",
            line=dict(color=f1plot.get_team_color(lap1['Team']))
        ))
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Speed'],
            mode='lines', name=f"{driver2} – {lap2['Team']}",
            line=dict(color=f1plot.get_team_color(lap2['Team']))
        ))
        fig.update_layout(height=500, xaxis_title="Distance (m)", yaxis_title="Speed (km/h)",
                          template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Throttle & Brake
        st.subheader("Throttle & Brake Application")
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Throttle (%)", "Brake (0/1)"))
        fig2.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Throttle'], name=f"{driver1} Throttle"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Throttle'], name=f"{driver2} Throttle"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Brake'], name=f"{driver1} Brake"), row=2, col=1)
        fig2.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Brake'], name=f"{driver2} Brake"), row=2, col=1)
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        # Tyre Strategy
        st.subheader("Race Tyre Strategy")
        race_laps = session.laps[session.laps['TyreLife'] > 0]
        fig_tyre = px.scatter(race_laps, x="LapNumber", y="Driver",
                              color="Compound", size="TyreLife",
                              hover_data=["Team", "Stint"],
                              title="Tyre Compounds & Stint Lengths")
        st.plotly_chart(fig_tyre, use_container_width=True)

else:
    st.info("Pick a race → click **Load Session** → explore the data!")
    st.balloons()
