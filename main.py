import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
import scipy.stats as stats 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ──────────────────────────────────────
# Hardcoded team colors & logos
# ──────────────────────────────────────
TEAM_COLORS = {
    "Red Bull": {"color": "#1E41FF", "logo": "🐂"},
    "Ferrari": {"color": "#DC0000", "logo": "🐎"},
    "McLaren": {"color": "#FF9800", "logo": "🍊"},
    "Mercedes": {"color": "#00D2BE", "logo": "⭐"},
    "Aston Martin": {"color": "#006F62", "logo": "🟢"},
    "Alpine": {"color": "#0090FF", "logo": "🏔️"},
    "Williams": {"color": "#005AFF", "logo": "🦁"},
    "RB": {"color": "#0033FF", "logo": "🟦🐃"},
    "Haas": {"color": "#B6BABD", "logo": "🇺🇸"},
    "Kick Sauber": {"color": "#9B0000", "logo": "🇨🇭"},
    "Sauber": {"color": "#9B0000", "logo": "🇨🇭"},
    "Racing Bulls": {"color": "#0033FF", "logo": "🟦🐃"},
    "AlphaTauri": {"color": "#0033FF", "logo": "🟦🐃"},
}

def get_team_info(team):
    if hasattr(team, 'item'):
        team = team.item()
    elif hasattr(team, "values"):       
        team = team.values[0]
    team_str = str(team).strip()
    if not team_str or pd.isna(team_str):
        return {"color": "#CCCCCC", "logo": "⚪"}
    team_str = str(team).strip()
    return TEAM_COLORS.get(team_str, {"color": "#CCCCCC", "logo": "⚪"})


# ──────────────────────────────────────
# Cache
# ──────────────────────────────────────
cache_dir = "/tmp/fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# ──────────────────────────────────────
#  CSS: For Frontend
# ──────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&display=swap" rel="stylesheet">
<style>
    .main {background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%); color: #e5e7eb !important; font-family: 'Rajdhani', sans-serif;}
    .stApp {background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);}
    .stMarkdown, .stText, .stCaption, .stSelectbox label, .stRadio label, .stNumberInput label, div[data-testid="stMetricValue"] {color: #e5e7eb !important;}
    .sidebar .sidebar-content {background: linear-gradient(#1a1a2e, #16213e); border-right: 2px solid #dc0000;}
    .stPlotlyChart {border-radius: 12px; border: 2px solid #dc0000; box-shadow: 0 4px 12px rgba(220,0,0,0.4);}
    .metric {background: linear-gradient(135deg, #dc0000, #ff4444); color: white !important; border-radius: 10px;}
    .stButton > button {background: linear-gradient(#dc0000, #ff4444); color: white !important; border-radius: 20px;}
    .stButton > button:hover {background: linear-gradient(#ff4444, #dc0000);}
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
 .stSelectbox > label, .stRadio > label, .stTextInput > label, .stNumberInput > label,
.stExpander > label, div[role="button"] label {
    color: #ffffff !important;
    text-shadow: 0 0 4px rgba(255,255,255,0.5) !important;  
    font-weight: 600 !important;
}

.stCaption, .stMarkdown p {
    color: #d1d5db !important;  
}
.stRadio > div[role="radiogroup"] > label,
.stRadio > div > label,
.css-1cpxqw2,
div[data-testid="stWidgetLabel"] > div > div,
div[data-baseweb="radio"] > div > div {
    color: #ffffff !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Header 
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://www.formula1.com/etc/designs/fom-website/images/f1_logo.svg", width=120)
with col2:
    st.markdown("# 🏁 **Formula 1 Telemetry & Strategy Hub**")
    st.markdown("**Powered by fastf1 + ML Insights** | *Analyse laps, predict pits, dominate the grid*")
# ──────────────────────────────────────
# DATA LOADER FUNCTION
# ──────────────────────────────────────
@st.cache_resource
def load_data(year, gp, session_type):
    """
    Wrapper function to cache the specific session object.
    This prevents reloading the data every time you click a button.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=True)
    return session
    
# Session Setup
with st.expander("Session Setup", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1: year = st.selectbox("Year", range(2018, 2026), index=6)
    with c2: gp = st.text_input("Grand Prix", "Monaco")
    with c3: session_type = st.radio("Session", ["R", "Q", "FP1", "FP2", "FP3", "S"], horizontal=True)

    if st.button("🚀 Load Session", type="primary"):
        with st.spinner("🏎️ Fetching telemetry data..."):
            try:
                # CALL THE CACHED FUNCTION HERE
                session = load_data(year, gp, session_type)
                
                st.session_state.session = session
                st.success(f"✅ Loaded: {session.name}")
                st.markdown(
                    f"<h1 style='text-align:center; color:#dc0000; text-shadow: 0 0 20px #ff0000;'>"
                    f"{session.event.year} {session.event['EventName']} {session.name}</h1>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Load failed: {e}. Try 'Bahrain' for quick test.")
                st.stop()

if "session" not in st.session_state:
    st.stop()

session = st.session_state.session

# Driver Selection
st.subheader("Driver Face-Off")
c1, c2, c3 = st.columns(3)
with c1: driver1 = st.selectbox("Driver 1", session.results['Abbreviation'].dropna())
with c2: driver2 = st.selectbox("Driver 2", session.results['Abbreviation'].dropna(), index=1)
with c3:
    lap_type = st.radio("Lap", ["Fastest", "Specific #"], horizontal=True)
    lap_num = st.number_input("Lap #", 1, 100, 10) if lap_type == "Specific #" else None

# Logic to pick the laps
if lap_type == "Fastest":
    lap1 = session.laps.pick_drivers(driver1).pick_fastest()
    lap2 = session.laps.pick_drivers(driver2).pick_fastest()
else:
    # Validation: Check if the driver actually did this lap
    laps_d1 = session.laps.pick_drivers(driver1)
    laps_d2 = session.laps.pick_drivers(driver2)

    # If user asks for Lap 50 but driver stopped at Lap 20, stop gracefully
    if lap_num > laps_d1['LapNumber'].max():
        st.warning(f"⚠️ {driver1} did not reach Lap {lap_num} (DNF or Lapped).")
        st.stop()
    if lap_num > laps_d2['LapNumber'].max():
        st.warning(f"⚠️ {driver2} did not reach Lap {lap_num} (DNF or Lapped).")
        st.stop()

    lap1 = laps_d1.pick_lap(lap_num)
    lap2 = laps_d2.pick_lap(lap_num)

# Validation: Check for empty telemetry (Extra safety on top of previous)
try:
    tel1 = lap1.get_telemetry().add_distance()
    tel2 = lap2.get_telemetry().add_distance()
except Exception as e:
    st.warning(f"Could not retrieve telemetry for this comparison. One driver might have crashed or has no data.")
    st.stop()

team1 = get_team_info(lap1['Team'])
team2 = get_team_info(lap2['Team'])
diff_teams = team1["color"] != team2["color"]

if diff_teams:
    delta = np.mean(tel1['Speed'] - tel2['Speed'])
    st.metric("Avg Speed Delta", f"{delta:+.1f} km/h", f"{driver1} {'faster' if delta > 0 else 'slower'}")

# Telemetry Plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=("Speed Trace", "Throttle & Brake"))
fig.update_layout(height=800, template="plotly_dark")

fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Speed'], name=driver1, line=dict(color=team1['color'], width=4)), row=1, col=1)
fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Speed'], name=driver2, line=dict(color=team2['color'], width=4)), row=1, col=1)
fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Throttle'], name=f"{driver1} Throttle", line=dict(color=team1['color'])), row=2, col=1)
fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Throttle'], name=f"{driver2} Throttle", line=dict(color=team2['color'])), row=2, col=1)
fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Brake']*100, name=f"{driver1} Brake", line=dict(color=team1['color'], dash='dot')), row=2, col=1)
fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Brake']*100, name=f"{driver2} Brake", line=dict(color=team2['color'], dash='dot')), row=2, col=1)

st.plotly_chart(fig, width="stretch")
st.subheader("Sector-by-Sector Time Delta")

# Use telemetry which always has sector times
if 'Sector1Time' in tel1.columns and 'Sector2Time' in tel1.columns and 'Sector3Time' in tel1.columns:
    s1 = (tel2['Sector1Time'].iloc[0] - tel1['Sector1Time'].iloc[0]).total_seconds()
    s2 = (tel2['Sector2Time'].iloc[0] - tel1['Sector2Time'].iloc[0]).total_seconds()
    s3 = (tel2['Sector3Time'].iloc[0] - tel1['Sector3Time'].iloc[0]).total_seconds()
    
    delta_df = pd.DataFrame({
        "Sector": ["S1", "S2", "S3", "Total"],
        "Delta (s)": [s1, s2, s3, s1+s2+s3]
    }).round(3)

    st.dataframe(
        delta_df.style.format("{:+.3f}")
        .applymap(lambda x: f"background-color: {'#004400' if x < 0 else '#440000'}", subset=["Delta (s)"])
        .applymap(lambda x: "color: lime" if x < 0 else "color: red", subset=["Delta (s)"])
    )
else:
    st.info("Sector times not available for this session")

# Strategy Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tyre Degradation", "Pit Predictor (ML)", "Strategy Forecast", "Bayesian Pace Updater", "Pit Safety (Prob)"])

with tab1:
    st.subheader("Tyre Degradation Analysis")

    if session.laps[session.laps['TyreLife'] > 0].empty:
        st.info("No tyre data available in this session")
    else:
        # Grab data for the two selected drivers only
        data = session.laps.pick_drivers([driver1, driver2]).copy()

        # Convert LapTime to seconds (float)
        data['LapTimeSec'] = data['LapTime'].dt.total_seconds()

        # F1 official compound colours
        compound_colors = {
            'SOFT': '#FF0000',
            'MEDIUM': '#FFFF00',
            'HARD': '#A0A0A0',
            'INTERMEDIATE': '#00FF00',
            'WET': '#00FFFF'
        }

        fig = px.scatter(
            data,
            x='TyreLife',
            y='LapTimeSec',
            color='Compound',
            facet_col='Driver',
            size='LapNumber',
            hover_data=['LapNumber', 'Stint'],
            color_discrete_map=compound_colors,
            title="Tyre Degradation – Lap Time vs Tyre Age"
        )

        # Beautiful styling
        fig.update_traces(marker=dict(line=dict(width=1.5, color='white'), opacity=0.9))
        fig.update_layout(
            template="plotly_dark",
            height=600,
            legend_title="Compound",
            xaxis_title="Tyre Age (laps)",
            yaxis_title="Lap Time (seconds)",
            hovermode="x unified"
        )

        # Add trend lines per compound/driver
        for drv in data['Driver'].unique():
            for comp in data['Compound'].unique():
                subset = data[(data['Driver'] == drv) & (data['Compound'] == comp)]
                subset = subset[subset['IsAccurate'] == True]
                if len(subset) > 3:
                    coeffs = np.polyfit(subset['TyreLife'], subset['LapTimeSec'], 1)
                    line = np.poly1d(coeffs)(subset['TyreLife'])
                    fig.add_trace(go.Scatter(
                        x=subset['TyreLife'], y=line,
                        mode='lines',
                        line=dict(dash='dot', color=compound_colors.get(comp, '#888')),
                        name=f"{drv} {comp} trend",
                        showlegend=False
                    ))

        st.plotly_chart(fig, width="stretch")

        # ──────────────────────────────────────────────────────────────
        # Estimated Cliff Point (Fixed Calculation)
        # ──────────────────────────────────────────────────────────────
        
        # 1. Filter for accurate laps only (removes Pit Stops & Safety Car laps)
        clean_data = data[data['IsAccurate'] == True]

        # 2. Calculate degradation only if we have clean data
        if not clean_data.empty:
            avg_degradation = clean_data.groupby(['Driver', 'Compound'])['LapTimeSec'].apply(
                lambda x: np.polyfit(clean_data.loc[x.index, 'TyreLife'], x, 1)[0] * 10
            ).round(2)

            # 3. Display the table
            if not avg_degradation.empty:
                st.markdown("**Estimated 1-second loss every 10 laps**")
                st.dataframe(
                    avg_degradation.rename("sec/10 laps").reset_index(),
                    hide_index=True,
                    column_config={"sec/10 laps": st.column_config.NumberColumn(format="+%.2f s")}
                )
        else:
            st.info("Not enough clean racing laps to calculate degradation.")
with tab2:
    st.subheader("Race Position Predictor (ML)")
    st.markdown("""
    **How it works:** This model uses `GradientBoosting` to look at **Tyre Age** vs **Track Position**.
    It predicts where a driver will be in **3 laps time** based on current tyre life.
    """)

    @st.cache_data
    def train_model(session_name): # Added session_name to force refresh when race changes
        # Prepare data: we need LapNumber, TyreLife, and Position
        df = session.laps[['LapNumber', 'TyreLife', 'Position']].dropna()
        
        # If not enough data (e.g., FP1), don't crash
        if len(df) < 50:
            return None, None
            
        X = df[['LapNumber', 'TyreLife']]
        # Target: The position of the driver 3 laps into the future
        y = df['Position'].shift(-3).fillna(df['Position'].max())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        mae = mean_absolute_error(y_test, model.predict(X_test))
        return model, mae

    # Pass session.name so cache invalidates if you switch races
    model, mae = train_model(session.name)

    if model:
        st.metric("Model Accuracy (MAE)", f"±{mae:.1f} positions", 
                  delta="High Accuracy" if mae < 2.0 else "Normal Accuracy",
                  delta_color="inverse") # inverse makes low MAE (green) good

        col1, col2 = st.columns(2)
        with col1:
            # Set default to current max lap to avoid confusion
            max_lap = int(session.laps['LapNumber'].max())
            lap = st.slider("Current Lap Number", 1, max_lap, min(20, max_lap))
        with col2:
            life = st.slider("Current Tyre Age", 0, 50, 15)

        if st.button("🔮 Predict Future Position", type="primary"):
            # Get prediction
            pred_pos = model.predict([[lap, life]])[0]
            
            # Round it and ensure it's between P1 and P20
            pred_pos = int(round(pred_pos))
            pred_pos = max(1, min(20, pred_pos))

            st.balloons()
            
            #  Display it as "Position"
            st.success(f"**Predicted Position in 3 Laps: P{pred_pos}**")
            
            # Add strategy context
            if pred_pos <= 3:
                st.markdown("🏆 **Podium Contention** - Pace is strong.")
            elif pred_pos <= 10:
                st.markdown("🔵 **Points Finish** - Good midfield pace.")
            else:
                st.markdown("🔻 **Outside Points** - Consider pitting for fresh rubber.")
    else:
        st.warning("⚠️ Not enough data points in this session to train the AI model. Try loading a full Race session.")

with tab3:
    st.subheader("Monte Carlo Catch Predictor")
    
    # ──────────────────────────────────────────────────────────────
    # Statistical Explanation (The "Why")
    # ──────────────────────────────────────────────────────────────
    st.markdown("""
    > **ℹ️ Why use Monte Carlo Simulation?**
    >
    > A simple "Time to Catch" calculation assumes drivers run constant lap times (e.g., *Gap ÷ Delta*). 
    > In reality, lap times are a **Stochastic Process** influenced by traffic, errors, and degradation.
    >
    > This tool models future lap times as random variables $L \sim \mathcal{N}(\mu, \sigma^2)$ and runs **1,000 race simulations** > to calculate the specific probability of an overtake.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        chaser = st.selectbox("Chasing Driver", session.results['Abbreviation'].unique(), index=0)
    with col2:
        leader = st.selectbox("Leading Driver", session.results['Abbreviation'].unique(), index=1)

    # ──────────────────────────────────────────────────────────────
    # Data Preparation
    # ──────────────────────────────────────────────────────────────
    # Get valid racing laps (exclude pit stops & SC)
    laps_chaser = session.laps.pick_drivers(chaser).pick_wo_box().pick_track_status('1').pick_quicklaps()
    laps_leader = session.laps.pick_drivers(leader).pick_wo_box().pick_track_status('1').pick_quicklaps()

    if len(laps_chaser) < 5 or len(laps_leader) < 5:
        st.error("Not enough data points to build a statistical distribution.")
    else:
        # Calculate Mean (μ) and Std Dev (σ) from recent history
        # We take the last 10 laps to capture *current* form/fuel load
        mu_chaser = laps_chaser['LapTime'].dt.total_seconds().iloc[-10:].mean()
        sigma_chaser = laps_chaser['LapTime'].dt.total_seconds().iloc[-10:].std()

        mu_leader = laps_leader['LapTime'].dt.total_seconds().iloc[-10:].mean()
        sigma_leader = laps_leader['LapTime'].dt.total_seconds().iloc[-10:].std()

        # Inputs for simulation
        c1, c2 = st.columns(2)
        with c1:
            current_gap = st.number_input("Current Gap (seconds)", min_value=0.0, value=5.0, step=0.1)
        with c2:
            laps_remaining = st.slider("Laps Remaining", 5, 30, 15)

        # ──────────────────────────────────────────────────────────────
        # The Monte Carlo Engine
        # ──────────────────────────────────────────────────────────────
        n_simulations = 1000
        
        # Generate random future lap times for both drivers
        # Shape: (1000 simulations, laps_remaining)
        future_chaser = np.random.normal(mu_chaser, sigma_chaser, (n_simulations, laps_remaining))
        future_leader = np.random.normal(mu_leader, sigma_leader, (n_simulations, laps_remaining))

        # Calculate cumulative time for both
        cum_chaser = np.cumsum(future_chaser, axis=1)
        cum_leader = np.cumsum(future_leader, axis=1) + current_gap # Leader starts ahead

        # Calculate Gap Trajectory (Leader - Chaser)
        # If Gap < 0, overtake happened
        gap_trajectories = cum_leader - cum_chaser

        # Calculate Probability
        # Check if gap goes below 0 at any point in the remaining laps
        overtake_matrix = gap_trajectories < 0
        prob_overtake = np.mean(np.any(overtake_matrix, axis=1)) * 100

        # ──────────────────────────────────────────────────────────────
        #  Visualisation (Fan Chart / Cone of Uncertainty)
        # ──────────────────────────────────────────────────────────────
        
        # Calculate percentiles for the "Cone"
        median_gap = np.median(gap_trajectories, axis=0)
        p95_gap = np.percentile(gap_trajectories, 95, axis=0)
        p05_gap = np.percentile(gap_trajectories, 5, axis=0)
        x_axis = list(range(1, laps_remaining + 1))

        st.metric("Probability of Overtake", f"{prob_overtake:.1f}%", 
                  delta="Likely" if prob_overtake > 50 else "Unlikely")

        fig = go.Figure()

        # 95% Confidence Interval (The "Cone")
        fig.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1], # forward then backward for shape
            y=list(p95_gap) + list(p05_gap)[::-1],
            fill='toself',
            fillcolor='rgba(0, 200, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))

        # Median Trajectory
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=median_gap,
            line=dict(color='#00D2BE', width=3),
            name='Median Predicted Gap'
        ))

        # Zero line (The Overtake Point)
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Overtake")

        fig.update_layout(
            title=f"Predicted Gap: {chaser} vs {leader}",
            xaxis_title="Laps into Future",
            yaxis_title="Gap (seconds)",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig, width="stretch")
        
        st.caption(f"Based on last 10 laps: {chaser} (σ={sigma_chaser:.2f}s), {leader} (σ={sigma_leader:.2f}s)")


with tab4:
    st.subheader("Bayesian Race Pace Inference")
    st.markdown("""
    > **The Stats Logic:**
    > We start with a **Prior Belief** of a driver's pace based on FP2 (Long Runs).
    > As the race progresses, we observe new data (**Likelihood**).
    > We use a **Normal-Normal Conjugate Prior** update to calculate the **Posterior** (True Pace).
    """)

    # Driver Selection
    b_driver = st.selectbox("Select Driver for Analysis", session.results['Abbreviation'].unique(), index=0)
    
    # Get Prior Data (Simulated from FP2 or early race laps)
    # In a real app, you'd load the 'FP2' session here. To save time, we'll simulate 
    # the Prior using the first 5 laps of the race as "Initial Belief".
    laps_driver = session.laps.pick_drivers(b_driver).pick_quicklaps().pick_wo_box()
    
    if len(laps_driver) < 5:
        st.warning("Not enough laps to perform Bayesian Inference.")
    else:
        # PRIOR: Based on first 3 clean laps (simulating 'Practice Data')
        prior_data = laps_driver['LapTime'].dt.total_seconds().iloc[:3]
        mu_0 = prior_data.mean()
        sigma_0 = prior_data.std() if len(prior_data) > 1 else 0.5
        
        # LIKELIHOOD: The rest of the race
        race_data = laps_driver['LapTime'].dt.total_seconds().iloc[3:]
        if len(race_data) == 0:
            st.info("Waiting for more race laps to update the model...")
        else:
            n = len(race_data)
            mu_likelihood = race_data.mean()
            sigma_likelihood = race_data.std() # Assumed known variance for simplicity
            
            # Calculate Posterior (The Math)
            # Formula for Normal-Normal Conjugate Prior
            # Precision = 1/variance
            tau_0 = 1 / (sigma_0 ** 2)
            tau_likelihood = n / (sigma_likelihood ** 2)
            tau_post = tau_0 + tau_likelihood
            
            sigma_post = np.sqrt(1 / tau_post)
            mu_post = (tau_0 * mu_0 + tau_likelihood * mu_likelihood) / tau_post

            # 3. Visualization (PDFs)
            x_min = min(mu_0, mu_likelihood, mu_post) - 2
            x_max = max(mu_0, mu_likelihood, mu_post) + 2
            x = np.linspace(x_min, x_max, 1000)

            # Generate Normal Curves
            y_prior = stats.norm.pdf(x, mu_0, sigma_0)
            y_like = stats.norm.pdf(x, mu_likelihood, sigma_likelihood)
            y_post = stats.norm.pdf(x, mu_post, sigma_post)

            fig = go.Figure()
            
            # Plot Prior
            fig.add_trace(go.Scatter(x=x, y=y_prior, name='Prior (Initial Belief)', 
                                     line=dict(color='gray', dash='dot')))
            
            # Plot Likelihood
            fig.add_trace(go.Scatter(x=x, y=y_like, name=f'Likelihood (Last {n} Laps)', 
                                     line=dict(color='blue', width=1), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
            
            # Plot Posterior
            fig.add_trace(go.Scatter(x=x, y=y_post, name='Posterior (Updated Pace)', 
                                     line=dict(color='red', width=4)))

            fig.update_layout(
                title=f"Bayesian Pace Update for {b_driver}",
                xaxis_title="Lap Time (s)",
                yaxis_title="Probability Density",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, width="stretch")
            
            st.success(f"**True Pace Estimate:** {mu_post:.3f}s (Confidence: ±{sigma_post*1.96:.2f}s)")
            st.caption("Notice how the Red curve (Posterior) is taller and narrower than the Grey curve? That visualises our Reduced Uncertainty.")

with tab5:
    st.markdown(r"""
### 🛡️ Probabilistic Pit Exit Model (Traffic Avoidance)

**The Strategic Context:**
In Formula 1, the "Undercut" and "Overcut" rely on precise gap calculations. However, determinstic models fail when margins are tight. 
* **Real-world Example:** In the 2025 Mexico GP, Oliver Bearman extended his stint to avoid pitting into a "DRS Train" led by Alonso. Even though fresh tyres were faster, rejoining *inside* traffic would have resulted in a net time loss due to dirty air.

**The Statistical Model:**
Instead of treating a Pit Stop as a fixed constant (e.g., $L = 20.0s$), we model the **Pit Loss Time** as a Stochastic Process following a Normal Distribution:
$$
L \sim \mathcal{N}(\mu_{pit}, \sigma^2_{total})
$$

**How we Calculate Variability ($\sigma_{total}$):**
The total uncertainty is the sum of independent variances from three distinct phases:
$$
\sigma^2_{total} = \sigma^2_{entry} + \sigma^2_{stationary} + \sigma^2_{launch}
$$
1.  **$\sigma^2_{entry}$ (Driver Phase):** Variance in braking for the speed limit line (e.g., locking up vs. optimal deceleration).
2.  **$\sigma^2_{stationary}$ (Mechanic Phase):** Variance in the wheel-gun operation (e.g., 2.2s vs. 3.5s stop).
3.  **$\sigma^2_{launch}$ (Grip Phase):** Variance in traction on the pit exit with cold tyres.

**The Output:**
We calculate the **Probability of Safe Rejoin** ($P_{safe}$) by integrating the Probability Density Function (PDF) where the gap remains positive:
$$
P_{safe} = P(\text{Gap} - L > 0)
$$
""")
    
    col1, col2 = st.columns(2)
    with col1:
        # User selects who they are racing against (e.g., the car behind)
        rival = st.selectbox("Rival Driver (Traffic)", session.results['Abbreviation'].unique(), index=2)
    with col2:
        current_gap = st.number_input("Current Gap to Rival (s)", value=22.0)

    # Pit Stop Parameters (The Uncertainty)
    st.markdown("#### Pit Stop Parameters")
    c1, c2 = st.columns(2)
    with c1:
        avg_pit_loss = st.slider("Avg Pit Loss Time (s)", 18.0, 25.0, 20.0)
    with c2:
        pit_variability = st.slider("Pit Stop Variability (σ)", 0.2, 2.0, 0.5)

    #  Define Distributions
    # Pit Loss ~ N(20s, 0.5s^2)
    # Rival Pace ~ N(CurrentGap, RivalPaceVariance) - simplified for the "moment of impact"
    
    # We want to calculate P(Gap_After_Pit > 0)
    # Gap_After_Pit = Current_Gap - Pit_Loss
    # Since Pit_Loss is a random variable, Gap_After_Pit is also a random variable.
    # Mean_New = Gap - Mean_Pit
    # Std_New = Pit_Std (assuming Gap is fixed measurement for this instant)
    
    mean_gap_after = current_gap - avg_pit_loss
    # Z-score for 0 (The "Clash" point)
    # Z = (X - μ) / σ
    # We want P(x > 0)
    
    z_score = (0 - mean_gap_after) / pit_variability
    prob_safe = 1 - stats.norm.cdf(z_score) # Probability we are ABOVE 0
    
    #  Visualisation
    x = np.linspace(mean_gap_after - 4, mean_gap_after + 4, 500)
    y = stats.norm.pdf(x, mean_gap_after, pit_variability)
    
    fig = go.Figure()
    
    # Plot the Rejoin Distribution
    fig.add_trace(go.Scatter(x=x, y=y, name='Rejoin Distribution', fill='tozeroy',
                             fillcolor='rgba(0, 255, 0, 0.2)' if prob_safe > 0.5 else 'rgba(255, 0, 0, 0.2)',
                             line=dict(color='white')))
    
    # Add the "Danger Zone" (Negative Gap = Behind Rival)
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Traffic Line")
    
    fig.update_layout(
        title=f"Probability of Rejoining Ahead of {rival}",
        xaxis_title="Gap After Pit Stop (seconds)",
        yaxis_title="Probability Density",
        template="plotly_dark"
    )
    
    st.metric("Probability of Safe Rejoin", f"{prob_safe*100:.1f}%", 
              delta="Safe" if prob_safe > 0.9 else "Risky" if prob_safe > 0.5 else "Unsafe")
    st.plotly_chart(fig, width="stretch")

    if prob_safe < 0.5:
        st.error(f"⚠️ UNDERCUT RISK: You will likely emerge {abs(mean_gap_after):.2f}s BEHIND {rival}.")
    else:
        st.success(f"✅ SAFE: Expected to emerge {mean_gap_after:.2f}s AHEAD of {rival}.")
# ──────────────────────────────────────
# Footer
# ──────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown("**Tech Stack:** fastf1 • Plotly • XGBoost • Streamlit")
with col_f2:
    st.markdown("[⭐ GitHub Repo](https://github.com/amalabdirahman/f1-telemetry-dashboard) ")
if st.button("📤 Share Dashboard"):
    st.code(f"https://formula1-telemetry-dashboard.streamlit.app")
st.caption("© 2025 Amal Abdirahman | Fuelled by F1 obsession 🏁 | [LinkedIn](https://www.linkedin.com/in/amalabdirahman)")
