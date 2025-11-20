# 🏎️ Formula 1 Telemetry & Strategy Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://formula1-telemetry-dashboard.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![Stats](https://img.shields.io/badge/Statistics-Bayesian%20%26%20Monte%20Carlo-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Project Overview

This dashboard goes beyond standard descriptive analytics by applying **Inferential Statistics**, **Probabilistic Modeling**, and **Machine Learning** to Formula 1 telemetry data.

While traditional dashboards focus on *historical traces* (what happened), this tool focuses on **stochastic prediction** (what might happen). It leverages **FastF1** for granular data and **SciPy/Scikit-Learn** for statistical computation, offering insights into tyre degradation, overtake probabilities, and pit window safety.

---

## 🚀 Core Dashboard Features

These modules provide the fundamental analysis layer, allowing users to visualise race dynamics and vehicle performance.

### 1. Telemetry Analysis (Speed Traces)
* **Distance-based Plotting:** Telemetry data (Speed, Throttle, Brake) is mapped against track distance (meters) rather than time. This corrects for cornering speed differences, ensuring accurate comparison between cars.
* **Delta Calculation:** Calculates sector-by-sector time loss to pinpoint exactly where a driver is losing performance.

### 2. Tyre Degradation Analysis
* **Trend Modeling:** Uses linear regression to quantify tyre wear (seconds lost per lap).
* **Smart Filtering:** The algorithm automatically filters out "In-Laps," "Out-Laps," and Safety Car periods (`IsAccurate == True`) to ensure the regression model is trained only on representative clean-air racing laps.

### 3. Race Position Predictor (Machine Learning)
* **Model:** `GradientBoostingRegressor` (Scikit-learn).
* **Logic:** Predicts future track position ($t+3$ laps) based on current Tyre Age, Compound, and Lap Number.
* **Validation:** Displays Mean Absolute Error (MAE) backtested against the current session to provide confidence intervals for the prediction.

---

## 🧠 Advanced Statistical Methodology

This section details the rigorous probabilistic models used in the "Strategy Forecast" and "Bayesian Inference" tabs.

### 1. Monte Carlo Simulation (Overtake Probability)
* **Problem:** Deterministic linear extrapolation (e.g., "Gap / Delta = Laps to Catch") fails to account for lap time variance and degradation.
* **Solution:** We model future lap times as a **Stochastic Process**. Future laps $L_t$ are treated as random variables drawn from a Normal Distribution based on the driver's recent consistency:
    $$L_{driver} \sim \mathcal{N}(\mu_{recent}, \sigma^2_{recent})$$
* **Inference:** We run $N=1000$ race simulations. The probability of an overtake is the proportion of simulations where the chaser's cumulative time is less than the leader's.

### 2. Bayesian Pace Inference (Conjugate Priors)
* **Problem:** Early in a race ($n < 5$ laps), sample sizes are too small to determine a driver's true pace using simple averages.
* **Solution:** We use **Bayesian Inference** to update our beliefs as new data arrives.
    * **Prior ($\mu_0$):** Derived from Free Practice 2 (Long Run simulations).
    * **Likelihood ($y$):** Observed lap times during the current race.
    * **Posterior:** Calculated using a **Normal-Normal Conjugate Prior** update:
    $$\mu_{post} = \left( \frac{\tau_0 \mu_0 + \tau_{likelihood} \mu_{likelihood}}{\tau_0 + \tau_{likelihood}} \right)$$
      , *where precision is* $\tau = 1/\sigma^2$. This yields a robust pace estimate even with minimal data.

### 3. Probabilistic Pit Exit Model
* **Problem:** Calculating a "safe pit window" using a fixed pit loss time (e.g., 20s) ignores the variance of pit crews and pit limiters.
* **Solution:** We model the total time lost in a pit stop ($L_{pit}$) as a sum of independent variances:
    $$\sigma^2_{total} = \sigma^2_{entry} + \sigma^2_{mechanics} + \sigma^2_{launch}$$
* **Inference:** The probability of a safe rejoin ($P_{safe}$) is the integral of the Gaussian distribution where the gap remains positive:
    $$P_{safe} = \int_{0}^{\infty} \mathcal{N}(\text{Gap} - \mu_{pit}, \sigma^2_{total}) \, dx$$

---

## 🧐 Critical Analysis & Model Limitations

As a statistical project, it is crucial to acknowledge the simplifying assumptions made in these models and how they deviate from real-world physics.

**1. The Linearity of Tyre Degradation**
The current model assumes a **Linear Regression** ($y = mx + c$) for tyre wear. In reality, tyre physics is highly non-linear.
* **Critique:** Tyres typically exhibit a "Thermal Phase" (getting faster as they warm up), followed by a "Stable Plateau," and finally a "Cliff" (exponential decay). A linear model tends to underestimate wear at the end of a stint (the cliff) and overestimate wear during the stable phase.
* **Proposed Improvement:** A more rigorous approach would fit a **3rd-degree Polynomial** or a **Sigmoid Function** to capture the "Cliff" effect. Additionally, accounting for fuel load burn-off (approx 0.03s/lap gain) is necessary to isolate pure tyre degradation from vehicle mass reduction.

**2. Independence Assumption in Monte Carlo**
The Overtake Simulation assumes that lap times are Independent and Identically Distributed (I.I.D).
* **Critique:** In racing, lap times are **Auto-Correlated**. A mistake in the final sector of Lap $t$ often compromises the start of Lap $t+1$. Furthermore, blue flags (traffic) usually affect clusters of laps, not single random instances.
* **Proposed Improvement:** Implementing a **Markov Chain Monte Carlo (MCMC)** approach or adding an auto-correlation term to the variance generation would model these "streaks" of bad/good pace more accurately.

**3. Feature Scarcity in ML Prediction**
The Position Predictor relies primarily on Tyre Age and Lap Number.
* **Critique:** This ignores critical variables such as **Track Temperature** (affecting thermal degradation), **ERS Deployment Modes**, and **Fuel Mass**.
* **Proposed Improvement:** Integrating weather APIs for track temp and estimating fuel load based on total lap count would significantly reduce the Mean Absolute Error (MAE).

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.9+
* `pip`

### 1. Clone the Repository
```bash
git clone [https://github.com/AmalAbdirahman/f1-telemetry-dashboard.git](https://github.com/AmalAbdirahman/f1-telemetry-dashboard.git)
cd f1-telemetry-dashboard
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the Dashboard

```bash
streamlit run main.py
```

---

## 📚 Tech Stack

**Frontend:** Streamlit (Custom CSS for F1 aesthetics)

**Data Source:** FastF1 API (Live & Historical Telemetry)

**Visualisation:** Plotly Graph Objects (Interactive, Distance-based axes)

**Statistics:** SciPy (PDFs, CDFs, Norm), NumPy (Monte Carlo)

**Machine Learning:** Scikit-Learn (Gradient Boosting, Train-Test Split)
