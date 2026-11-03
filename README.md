# Formula 1 Telemetry & Strategy Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://formula1-telemetry-dashboard.streamlit.app/)
[![Tests](https://github.com/AmalAbdirahman/f1-telemetry-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/AmalAbdirahman/f1-telemetry-dashboard/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)




A Streamlit app that compares two drivers' laps and then tests the strategy behind them with statistical models: tyre degradation, Monte Carlo overtake simulation, Bayesian pace and pit-exit safety, plus a small position predictor that is always scored against a naive baseline.

Data comes from [OpenF1](https://openf1.org), a free, unofficial API covering 2023 onwards. This project is not associated with Formula 1.

## Features

| Section | What it shows | Method |
|---|---|---|
| Driver Face-Off | Speed, throttle, brake and a time delta along the lap, plus sector deltas | Traces aligned on a shared distance grid |
| Tyres | Seconds lost per lap of tyre age, per driver and stint | Huber robust fit with a bootstrap 90% interval |
| Position Predictor | Position 3 laps ahead, next to a "nothing changes" baseline | Gradient boosting, tested on later laps |
| Overtake Forecast | Chance a chasing car gets ahead on the clock | Monte Carlo with AR(1) correlated laps |
| Bayesian Pace | A driver's true pace from only a few laps | Empirical-Bayes Normal-Normal update |
| Pit Safety | Pit loss measured from the race, and the chance of a clean rejoin | Normal model of the gap after the stop |

Race-only tabs explain themselves in qualifying and practice. The sidebar offers finished sessions only, leaving out testing and cancelled events.

## Method

Only representative laps feed the models: green-flag, not a pit lap, accurately timed, and within 107% of that driver's stint median.

**Lap table.** OpenF1 serves separate tables, so `f1dash/data.py` rebuilds tyre age from stints, position from the position feed at the end of each lap, and track status from race-control messages.

**Time delta.** OpenF1 has no distance channel, so distance is speed integrated over time. Both laps are stretched to the same length and interpolated onto 500 points, then

$$\Delta(d) = t_1(d) - t_2(d)$$

A negative value means Driver 1 is ahead at that point, and the trace ends exactly on the official lap time gap.

**Tyre degradation.** Fuel-corrected pace is lap time + 0.03 × lap number, which cancels the speed gained as fuel burns. For each driver and stint, pace is fitted against tyre age:

$$\text{pace} = \alpha + \beta \cdot \text{TyreLife}$$

R² and RMSE come from the fit's residuals, and the 90% interval for $\beta$ from 500 seeded bootstrap resamples.

**Overtake simulation.** The gap after $n$ laps is

$$\text{gap}_n = \text{gap}_0 + \sum_{i=1}^{n} (\text{chaser}_i - \text{leader}_i)$$

and the chaser is ahead when it is negative. Each future lap $k$ is the recent mean (last 10 clean laps at most) plus the tyre trend × $k$, plus an AR(1) deviation:

$$e_k = \rho e_{k-1} + \sqrt{1-\rho^2}\,\sigma z_k$$

$\rho$ is the lag-1 autocorrelation of detrended recent laps, clipped to [0, 0.8], and can be overridden.

**Bayesian pace.** The prior is the spread of per-driver mean laps across the session's field. The lap-to-lap $\sigma$ is the within-driver SD, pooled across the field. After $n$ laps with mean $\bar{y}$:

$$\tau = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}, \qquad \mu_{post} = \frac{\mu_0/\sigma_0^2 + n\bar{y}/\sigma^2}{\tau}, \qquad \sigma_{post}^2 = \frac{1}{\tau}$$

**Pit safety.** The loss for one stop is (in-lap + out-lap) − 2 × the driver's median clean lap, pooled over green-flag stops. Then

$$P_{safe} = P(\text{gap} - L > \text{clearance}), \quad L \sim \mathcal{N}(\mu, \sigma^2)$$

By default $\sigma^2$ is the spread across stops plus the rival's lap variance. That spread already contains the entry, stationary and launch phases, so a separate manual mode replaces it with three assumed phase variances instead of adding them on top.

**Position predictor.** A `GradientBoostingRegressor` uses lap number, tyre age, position and compound. The target is the same driver's position 3 laps later, and laps with no known future are dropped. The last 20% of the race is held out, and training rows whose target lands in that period are removed.

## Limitations

- **The predictor loses to the baseline.** In five real races tested (2023 to 2026) "position stays the same" won every time, for example 0.29 places of error against 1.84 at the 2026 Mexico City Grand Prix. Positions rarely change over three laps, and the model sees a single race. The app says so whenever it happens.
- **The fuel figure is a constant assumption.** Tyre age and lap number rise together within a stint, so 0.03 s/lap adds directly to every degradation slope. Compare slopes with each other instead of reading them as absolute figures.
- **Degradation is linear.** A tyre cliff is not modelled, and "laps to lose 1 s" is a heuristic.
- **Ahead on the clock is not a pass.** The overtake simulation leaves out dirty air, defending, the overtaking aid of the era (DRS before 2026, the new overtake mode from 2026), pit stops and safety cars. It also uses each driver's end-of-session form, whatever gap you enter.
- **Telemetry is coarse.** About four samples a second means short events can be missed, and integrated distance drifts by a few metres.
- **Measured pit spread can be wide.** Slow stops and changing weather inflate it.
- **Data limits.** Nothing before 2023, and OpenF1's free tier allows 30 requests a minute and locks data while a session is live. Rebuilt columns are close to, but not the same as, official timing.

## Run it

Requires Python 3.13.

```bash
git clone https://github.com/AmalAbdirahman/f1-telemetry-dashboard.git
cd f1-telemetry-dashboard
pip install -r requirements-dev.txt
streamlit run main.py
```

No account or API key is needed.

## Test it

```bash
pytest -q
```

Tests make no network calls: `tests/fake_openf1.py` is a synthetic stand-in for the API. `tests/test_app_smoke.py` runs the whole app against it for race, qualifying and empty sessions.

## Structure

```
main.py             entry point
f1dash/
  config.py         colours, design tokens, modelling constants
  openf1.py         REST client with rate limits, retries and a disk cache
  data.py           session loading and the rebuilt lap table
  models/           pure, tested maths: no Streamlit or network code
  ui/               theme, charts, sidebar, Face-Off and one module per tab
assets/style.css    brand details and motion
.streamlit/         theme configuration
tests/              one test file per module, plus the app smoke test
```

## License

[MIT](LICENSE). Data credit: [OpenF1](https://openf1.org).
