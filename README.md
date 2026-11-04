


# Formula 1 Telemetry & Strategy Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://formula1-telemetry-dashboard.streamlit.app/)
[![Tests](https://github.com/AmalAbdirahman/f1-telemetry-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/AmalAbdirahman/f1-telemetry-dashboard/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)




A Streamlit app that works like a pit wall. Pick a race and a lap, and the strategy tabs see only what was known at the end of that lap: the real order, gaps, tyres, pace and wear. From there they test the decisions with statistical models: pit-exit safety with a rejoin projector, tyre degradation, Monte Carlo overtake simulation and Bayesian pace, plus a small position predictor that is always scored against a naive baseline.

https://github.com/user-attachments/assets/57b4cff2-afd4-4e26-bcff-bbb48859f166

Data comes from [OpenF1](https://openf1.org), a free, unofficial API covering 2023 onwards. This project is not associated with Formula 1.

## Features

| Section | What it shows | Method |
|---|---|---|
| Driver Face-Off | Speed, throttle, brake and a time delta along the lap, plus sector deltas | Traces aligned on a shared distance grid |
| Race lap slider | One lap N, shared by every race tab | Race state rebuilt from line crossings up to lap N |
| Race | Race trace with SC/VSC/red-flag laps and pit stops, tyre strategy bars, and the order at lap N | Time behind the winner's average pace; stints from the lap table |
| Pit Safety | Pit loss from stops before lap N, the chance of rejoining ahead of a rival, and the rejoin position and traffic for a stop on laps N+1 to N+11 | Normal model of the gap after the stop; field re-ranked with projected pace |
| Tyres | Seconds lost per lap of tyre age, per driver and stint | Huber robust fit with a bootstrap 90% interval |
| Overtake Forecast | From lap N, the chance the car behind gets ahead on the clock | Monte Carlo with AR(1) correlated laps |
| Bayesian Pace | A driver's true pace from only a few laps | Empirical-Bayes Normal-Normal update |
| Position Predictor | Position 3 laps after lap N, next to a "nothing changes" baseline | Gradient boosting, trained and tested only on outcomes known by lap N |

The strategy tabs fill in the gap, laps remaining, pace and tyre trend from the data. Manual overrides sit in a collapsed expander and reset when the lap or the drivers change. On a phone, legends sit under the charts, the Plotly toolbar is hidden, charts are shorter, tyre panels stack and tables show fewer columns.

Race-only tabs explain themselves in qualifying and practice. The sidebar offers finished sessions only, leaving out testing and cancelled events.

## Method

Only representative laps feed the models: green-flag, not a pit lap, accurately timed, and within 107% of that driver's stint median.

**Lap table.** OpenF1 serves separate tables, so `f1dash/data.py` rebuilds tyre age from stints, position from the position feed at the end of each lap, and track status from race-control messages.

**Race state at lap N.** `f1dash/models/race_state.py` first cuts the lap table to laps 1..N, clean-lap filter included, so nothing later can leak in. A test changes every later lap and checks that the state does not move. Cars are ordered by when they crossed the line to finish lap N, and the gap between two cars is the difference between those crossings. Checking exactly whether a car is lapped would need lap N+1, so it is estimated:

$$\text{laps down} = \left\lfloor \frac{\text{gap to leader}}{\text{leader's lap } N \text{ time}} \right\rfloor$$

Each car also gets its compound, tyre age, stops so far, recent pace (last 10 clean laps at most, on the current set of tyres once it has 5 clean laps, because a fresh set can be a second a lap different; before that, earlier laps stand in and the app marks the pace with *) and the current stint's degradation slope, fitted only on laps up to N. The slider stops at the last lap with recorded times, because OpenF1 often leaves the final lap blank.

**Race trace.** For lap $n$, a car's value is its time to finish lap $n$ minus $n \cdot \bar{t}_{ref}$, where $\bar{t}_{ref}$ is the winner's race time divided by its laps, so the winner ends on zero. A lap is shaded when most cars' laps overlapped a safety car, virtual safety car or red flag.

**Time delta.** OpenF1 has no distance channel, so distance is speed integrated over time. Both laps are stretched to the same length and interpolated onto 500 points, then

$$\Delta(d) = t_1(d) - t_2(d)$$

A negative value means Driver 1 is ahead at that point, and the trace ends exactly on the official lap time gap.

**Tyre degradation.** Fuel-corrected pace is lap time + 0.03 × lap number, which cancels the speed gained as fuel burns. For each driver and stint, pace is fitted against tyre age:

$$\text{pace} = \alpha + \beta \cdot \text{TyreLife}$$

R² and RMSE come from the fit's residuals, and the 90% interval for $\beta$ from 500 seeded bootstrap resamples.

**Overtake simulation.** The gap after $n$ laps is

$$\text{gap}_n = \text{gap}_0 + \sum_{i=1}^{n} (\text{chaser}_i - \text{leader}_i)$$

and the chaser is ahead when it is negative. $\text{gap}_0$, the laps remaining, each driver's recent pace and spread, and the tyre trends all come from the race state at lap N. Each future lap $k$ is the recent mean (last 10 clean laps at most) plus the tyre trend × $k$, plus an AR(1) deviation:

$$e_k = \rho e_{k-1} + \sqrt{1-\rho^2}\,\sigma z_k$$

$\rho$ is the lag-1 autocorrelation of detrended recent laps, clipped to [0, 0.8], and can be overridden.

**Bayesian pace.** The prior is the spread of per-driver mean laps across the session's field. The lap-to-lap $\sigma$ is the within-driver SD, pooled across the field. After $n$ laps with mean $\bar{y}$:

$$\tau = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}, \qquad \mu_{post} = \frac{\mu_0/\sigma_0^2 + n\bar{y}/\sigma^2}{\tau}, \qquad \sigma_{post}^2 = \frac{1}{\tau}$$

**Pit safety.** The loss for one stop is (in-lap + out-lap) − 2 × the driver's median clean lap, pooled over green-flag stops completed by lap N. If none are complete yet, the whole race's stops are used and the app says so. A stop under a safety car or VSC really is cheaper, because the field is slow while the pit lane is not, but measured against the driver's normal lap it would look dearer. So when lap N ends under one, stops with both laps under a safety car are measured against the median of the other cars on the same two laps:

$$L_{SC} = (\text{in} + \text{out}) - (\tilde{t}_{field,\,in} + \tilde{t}_{field,\,out})$$

The cheaper loss is used for the next lap only, since nobody knows at lap N how long the safety car stays out. Under a red flag the app says that tyres can usually be changed without losing time. Red-flag "stops" are never measured. The gap to the rival comes from lap N. Then

$$P_{safe} = P(\text{gap} - L > \text{clearance}), \quad L \sim \mathcal{N}(\mu, \sigma^2)$$

By default $\sigma^2$ is the spread across stops plus the rival's lap variance. That spread already contains the entry, stationary and launch phases, so a separate manual mode replaces it with three assumed phase variances instead of adding them on top.

**Rejoin projector.** Lap N is over, so the first possible in-lap is $N+1$. For an in-lap $L = N+1, \dots, N+11$, the stop is judged at the line after the out-lap, lap $L+1 = N+k$. Each car's time at the line is projected from lap N with its recent pace $\mu_i$ and tyre trend $\beta_i$:

$$c_i(N+k) = c_i(N) + \sum_{j=1}^{k} (\mu_i + \beta_i j)$$

The pitting car gets the mean pit loss added and everyone is re-ranked. Clean air is judged on the road, not in race order, so lapped traffic counts. The time gap $\delta$ to every car is wrapped into half a lap either side,

$$\delta_{road} = \big((\delta + T/2) \bmod T\big) - T/2$$

with $T$ the field's median lap, and the rejoin is clear when the nearest car in front on the road is more than the clearance away (dirty air) and so is the nearest car behind, which could pass while the new tyres are still cold.

**Position predictor.** A `GradientBoostingRegressor` uses lap number, tyre age, position and compound. The target is the same driver's position 3 laps later, and laps with no known future are dropped. At lap N only outcomes up to lap N exist, so rows whose target lap is later are removed first. The last 20% of what is left is held out, and training rows whose target lands in that period are removed. The inputs start from the chosen driver's state at lap N, and the result is shown as a whole position.

## Limitations

- **The predictor loses to the baseline.** In five real races tested (2023 to 2026) "position stays the same" won every time, for example 0.29 places of error against 1.84 at the 2026 Mexico City Grand Prix. Positions rarely change over three laps, and the model sees a single race. The app says so whenever it happens.
- **The fuel figure is a constant assumption.** Tyre age and lap number rise together within a stint, so 0.03 s/lap adds directly to every degradation slope. Compare slopes with each other instead of reading them as absolute figures.
- **Degradation is linear.** A tyre cliff is not modelled, and "laps to lose 1 s" is a heuristic.
- **Ahead on the clock is not a pass.** The overtake simulation leaves out dirty air, defending, the overtaking aid of the era (DRS before 2026, the new overtake mode from 2026), pit stops and safety cars after lap N.
- **The pit window can't see ahead.** Laps after N are projected from lap N pace and tyre trend, so other cars' later stops, safety cars and traffic are unknown, as they would be on the pit wall. It gets less reliable further out.
- **Lapped status is estimated.** Laps down come from the gap and the leader's lap N time. That avoids reading lap N+1, but can be off by one at the moment a car is being lapped.
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

Tests make no network calls: `tests/fake_openf1.py` is a synthetic stand-in for the API. `tests/test_app_smoke.py` runs the whole app against it for race, qualifying and empty sessions, including moving the Race lap slider.

## Structure

```
main.py             entry point
f1dash/
  config.py         colours, design tokens, modelling constants
  openf1.py         REST client with rate limits, retries and a disk cache
  data.py           session loading and the rebuilt lap table
  models/           pure, tested maths: no Streamlit or network code
    race_state.py   the race at lap N and the pace projection
    race_overview.py  race trace, neutralised laps, tyre stints
    pit.py          pit loss, P_safe and the rejoin projector
  ui/               theme, charts, sidebar, Face-Off and one module per tab
assets/style.css    brand details and motion
.streamlit/         theme configuration
tests/              one test file per module, plus the app smoke test
```

## License

[MIT](LICENSE). Data credit: [OpenF1](https://openf1.org).
