"""Bayesian pace tab."""

import streamlit as st

from f1dash import data
from f1dash.models import pace
from f1dash.ui import charts, formatting, layout


def render(session: data.Session) -> None:
    st.markdown("Estimates a driver's true pace from only a few laps by starting from how the whole field ran.")
    clean = data.clean_laps(session.laps)
    prior = pace.field_prior(clean) if not clean.empty else None
    if prior is None:
        st.info("This needs at least three drivers with several clean laps each.")
        return
    sigma = pace.pooled_sd(clean)

    counts = clean["Driver"].value_counts()
    names = [d for d in session.drivers["Driver"] if counts.get(d, 0) >= 1]
    key = session.key
    driver = st.selectbox("Driver", names, key=f"bayes_driver_{key}")
    observations = pace.driver_observations(clean, driver)
    total = len(observations)
    if total > 1:
        n = st.slider("Laps observed so far", 1, total, min(5, total), key=f"bayes_laps_{key}")
    else:
        n = 1
    sample_mean = float(observations.iloc[:n].mean())
    result = pace.curves(prior, sample_mean, n, sigma)

    estimate, interval, used = st.columns(3)
    estimate.metric("Posterior mean", formatting.lap_time(result.posterior.mean))
    interval.metric("90% credible interval", f"±{1.645 * result.posterior.sd:.3f} s")
    used.metric("Average of laps seen", formatting.lap_time(sample_mean))
    layout.chart(charts.pace_figure(result, driver, n), key=f"pace_{key}")
    st.caption(
        "Prior in use: the field in this session "
        f"(mean {formatting.lap_time(prior.mean)}, SD {prior.sd:.3f} s). Lap times are fuel-corrected."
    )

    with st.expander("Method"):
        st.markdown(
            "This is empirical Bayes. The prior is the spread of per-driver mean laps across this session's "
            f"field. The lap-to-lap σ is {sigma:.3f} s, the within-driver SD pooled across the field. All three "
            "curves describe uncertainty about the driver's *mean* lap, so the likelihood's SD is σ/√n. With few "
            "laps the estimate leans on the field, and as laps accumulate it moves to the driver's own average.\n\n"
            "**Limitation:** laps are treated as independent given the mean, and the field prior includes the "
            "driver being estimated. In a race with changing weather, σ is large and the estimate is less useful."
        )
