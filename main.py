import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ──────────────────────────────────────
# Hardcoded 2024–2025 team colors (never breaks again)
# ──────────────────────────────────────
TEAM_COLORS = {
    "Red Bull": "#1E41FF",        # Red Bull blue
    "Ferrari": "#DC0000",         # Ferrari red
    "McLaren": "#FF9800",         # McLaren papaya
    "Mercedes": "#00D2BE",        # Mercedes turquoise
    "Aston Martin": "#006F62",    # Aston Martin green
    "Alpine": "#0090FF",          # Alpine blue
    "Williams": "#005AFF",        # Williams blue
    "RB": "#0033FF",              # RB / VCARB blue
    "Haas": "#B6BABD",            # Haas grey/white
    "Kick Sauber": "#9B0000",     # Sauber dark red
    "Sauber": "#9B0000",          # fallback
    "Racing Bulls": "#0033FF",    # fallback
    "AlphaTauri": "#0033FF",      # old name fallback
}

def get_color(team):
    if pd.isna(team):
        return "#CCCCCC"
    team_str = str(team).strip()
    return TEAM_COLORS.get(team_str, "#CCCCCC")  # fallback grey

# ──────────────────────────────────────
# Cache for Streamlit Cloud
# ──────────────────────────────────────
cache_dir = "/tmp/fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# ──────────────────────────────────────
# Page config & title
# ──────────────────────────────────────
st.set_page_config(page_title="F1 Telemetry Dashboard", layout="wide")
st.title("Formula 1 Telemetry & Strategy Dashboard")
st.markdown("Real-time telemetry analysis using `fastf1` • Built with love by **you**")

# ──────────────────────────────────────
# Sidebar – Race selection
# ──────────────────────────────────────
st.sidebar.header("Race Selection")
year = st.sidebar.selectbox("Year", options=list(range(2018, 2026)), index=6)  # 2024 default
gp = st.sidebar.text_input("Grand Prix", value="Monaco")
session_type = st.sidebar.radio("Session", ["R", "Q", "FP1", "FP2", "FP3", "S"], index=0)

# ──────────────────────────────────────
# Load Session
# ──────────────────────────────────────
if st.sidebar.button("Load Session", type="primary"):
    with st.spinner(f"Loading {year} {gp} {session_type}… (first time may take 20–40s)"):
        try:
            session = fastf1.get_session(year, gp, session_type)
            session.load(telemetry=True, laps=True, weather=True)
            st.session_state.session = session
            st.sidebar.success(f"Loaded: {session.name}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Stop if no session
if "session" not in st.session_state:
    st.info("↑ Choose a race and click **Load Session** to begin")
    st.stop()

session = st.session_state.session

# ──────────────────────────────────────
# Layout
# ──────────────────────────────────────
col1, col2 = st.columns([1, 3])

# ──────────────────────────────────────
# Left column – Controls
# ──────────────────────────────────────
with col1:
    st.subheader("Driver & Lap Selection")
    drivers = session.results['Abbreviation'].dropna().tolist()
    driver1 = st.selectbox("Driver 1", drivers, index=0)
    driver2 = st.selectbox("Driver 2", drivers, index=1)

    lap_choice = st.radio("Lap", ["Fastest Lap", "Specific Lap Number"])
    if lap_choice == "Specific Lap Number":
        max_lap = int(session.laps['LapNumber'].max())
        lap_num = st.number_input("Lap Number", 1, max_lap, 10)
    else:
        lap_num = None

# ──────────────────────────────────────
# Right column – Plots
# ──────────────────────────────────────
with col2:
    # Get the two laps
    lap1 = session.laps.pick_driver(driver1).pick_fastest() if lap_num is None else session.laps.pick_driver(driver1).pick_lap(lap_num)
    lap2 = session.laps.pick_driver(driver2).pick_fastest() if lap_num is None else session.laps.pick_driver(driver2).pick_lap(lap_num)

    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()

    # 1. Speed Trace – with hardcoded colors (perfect every time)
    st.subheader("Speed Trace Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tel1['Distance'], y=tel1['Speed'],
        mode='lines', name=f"{driver1} – {lap1['Team']}",
        line=dict(color=get_color(lap1['Team']), width=3)
    ))
    fig.add_trace(go.Scatter(
        x=tel2['Distance'], y=tel2['Speed'],
        mode='lines', name=f"{driver2} – {lap2['Team']}",
        line=dict(color=get_color(lap2['Team']), width=3)
    ))
    fig.update_layout(height=520, xaxis_title="Distance (m)", yaxis_title="Speed (km/h)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 2. Throttle & Brake
    st.subheader("Throttle & Brake Application")
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Throttle (%)", "Brake (On/Off)"))
    fig2.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Throttle'], name=f"{driver1} Throttle", line=dict(color=get_color(lap1['Team']))), row=1, col=1)
    fig2.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Throttle'], name=f"{driver2} Throttle", line=dict(color=get_color(lap2['Team']))), row=1, col=1)
    fig2.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Brake'], name=f"{driver1} Brake", line=dict(color=get_color(lap1['Team']))), row=2, col=1)
    fig2.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Brake'], name=f"{driver2} Brake", line=dict(color=get_color(lap2['Team']))), row=2, col=1)
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Tyre Strategy
    st.subheader("Tyre Strategy Overview")
    race_laps = session.laps[session.laps['TyreLife'] > 0]
    if not race_laps.empty:
        fig_tyre = px.scatter(race_laps, x="LapNumber", y="Driver",
                              color="Compound", size="TyreLife",
                              hover_data=["Team", "Stint"],
                              color_discrete_map={"Soft": "#FF0000", "Medium": "#FFFF00", "Hard": "#FFFFFF",
                                                  "Intermediate": "#00FF00", "Wet": "#0000FF"})
        st.plotly_chart(fig_tyre, use_container_width=True)
    else:
        st.info("No tyre data available for this session.")

st.caption("Built with fastf1 • Deployed on Streamlit Cloud • 2025")
st.caption("Built with fastf1 • Deployed on Streamlit Cloud • 2025")
