# CLAUDE.md — F1 Telemetry & Strategy Dashboard

Streamlit app that combines OpenF1 race data with statistical models (tyre degradation, Monte Carlo
overtake simulation, Bayesian pace, pit-exit safety) and a small ML position predictor.
Live: https://formula1-telemetry-dashboard.streamlit.app · Repo: AmalAbdirahman/f1-telemetry-dashboard

This is a **portfolio piece**. Every change should make it more correct, more readable to a hiring
manager, and more polished to use. Correct maths beats more features.

## Data source

FastF1 stopped working without an F1TV sign-in (F1's timing server answers 401), so the app uses the
free public OpenF1 API. It has no data before 2023, allows 30 requests a minute, and locks data while a
session is live. `f1dash/data.py` rebuilds tyre age, race position and track status from OpenF1's
separate tables. The `openf1/` folder is a local copy of OpenF1's server code: it is gitignored and the
app does not use it.

## Commands

```bash
pip install -r requirements-dev.txt     # runtime + pytest
streamlit run main.py                   # root main.py must stay the entry point (Streamlit Cloud uses it)
pytest -q                               # unit tests + app smoke test; must pass before every commit
```

## Target structure

```
main.py                     thin entry point: page config, theme, layout, calls ui/*
f1dash/
  config.py                 colours, design tokens, documented modelling constants, cache path
  openf1.py                 REST client: rate limits, retries, disk cache
  data.py                   session loading, per-lap table, lap filters (green-flag, no pit, accurate)
  models/                   PURE functions only: no streamlit, no network imports
    telemetry.py            distance alignment, speed delta, time-delta-along-lap, sector deltas
    degradation.py          fuel correction, robust per-stint fit, bootstrap CI
    position.py             dataset builder, temporal split, model + baseline
    overtake.py             Monte Carlo gap simulation (AR(1))
    pace.py                 empirical-Bayes pace
    pit.py                  pit-loss estimation from data + rejoin probability
  ui/
    theme.py                CSS injection, lights-out loader, reusable HTML snippets
    state.py                cached loaders and session-state helpers
    formatting.py           numbers to display text
    sidebar.py, faceoff.py  season/GP/session picker and the lap comparison
    charts.py               Plotly figure builders (one shared dark template)
    tabs/*.py               one module per tab, each exposing render(session, ...)
assets/style.css
.streamlit/config.toml
tests/                      test_<module>.py per module, fake_openf1.py, test_app_smoke.py
```

Rules: models never touch Streamlit; UI contains no maths beyond formatting. Every number shown in the
UI comes from a tested function.

## Fixed bugs

The bugs found in the original single-file app are fixed, each guarded by a `test_regression_*` test.
See the "Phase 2" commit for the list. Keep those tests.

## Do NOT "fix" these (they are correct; an earlier AI review claimed otherwise)

- **Fuel correction sign is right.** Cars get faster as fuel burns, so `LapTime + 0.03 × LapNumber`
  removes that speed-up. Do not change it to subtraction.
- `except Exception` does not catch `KeyboardInterrupt`. Narrowing it is nicer, but not a bug.
- There are no hard-coded secrets. Injecting a static CSS string with `unsafe_allow_html` is not a
  security hole. Historical session data doesn't go stale; the cache concern is memory, not freshness.

## Model specs (implement as written; document assumptions in docstrings)

- **Telemetry:** common grid `np.linspace(0, min(d1.max(), d2.max()), 500)`. Add a
  **time-delta-along-the-lap** trace: interpolate each lap's elapsed time vs distance onto the grid,
  delta = t1 − t2. This is the classic "where is the time won and lost" chart. OpenF1 has no distance
  channel, so distance is integrated speed; both laps are stretched to the same length before
  interpolating, which makes the trace end exactly on the lap time gap.
- **Degradation:** fuel-corrected pace = lap time + `FUEL_EFFECT × LapNumber` (0.03 s/lap, in config).
  Huber `sm.RLM` per driver × stint, x = TyreLife, on green-flag (`TrackStatus == '1'`), non-pit,
  `IsAccurate` laps within 107% of the driver's stint median. Report slope (s/lap), R² and RMSE from the fit's residuals, n laps, and a seeded
  bootstrap 90% CI for the slope (~500 resamples). "Laps to lose X s" is labelled a linear heuristic,
  not a cliff. Require ≥ 6 laps and > 1 distinct TyreLife.
- **Overtake MC:** μ, σ from the last ≤ 10 clean green-flag laps (show n used). Add each driver's
  degradation trend to future laps when available (μ + slope × k). Correlated laps via AR(1)
  deviations; estimate ρ from lag-1 autocorrelation of detrended recent laps, clip to [0, 0.8], allow
  user override. `np.random.default_rng(seed)`. Report P(ahead at any lap), P(ahead at the flag) and
  the median lap the gap first goes negative. Keep the caveat that real overtaking needs more than
  gap < 0 (dirty air, defending; DRS before 2026, the new overtake mode from 2026).
- **Bayesian pace (empirical Bayes):** prior = distribution of per-driver mean (fuel-corrected) lap
  times across the field in this session (mean, SD); likelihood σ = pooled within-driver SD; Normal-
  Normal posterior for the driver's mean, with a "laps observed so far" slider. Plot all three curves
  as uncertainty about the *mean*, x-range from the SDs. Say in the UI which prior is in use.
- **Pit safety:** estimate mean pit loss from the session: (in-lap + out-lap) − 2 × driver's median
  green-flag lap, pooled over green-flag stops only, σ across stops. Show the estimate and number of stops; sliders
  can override. Default σ² = σ²_stops + σ²_rival pace. Manual mode: σ²_entry + σ²_stationary + σ²_launch
  + σ²_rival pace, labelled as assumptions. Never add the phases to σ²_stops: that counts them twice.
  P_safe = P(gap − L > clearance); clearance slider approximates dirty air.
- **Position predictor:** features `[LapNumber, TyreLife, Position, CompoundCode]` (stint number if
  it helps), `GradientBoostingRegressor(random_state=42)`, temporal split above, baseline always shown.
  Only use features knowable at prediction time.

## Design system

**Keep the black base; use the Pirelli compound colours (red soft, yellow medium, white hard, green
inter, blue wet) as the accent palette.** Hex values below are tunable approximations; define them
once in `config.py` and as CSS variables.

| Token | Hex | Role in the UI |
|---|---|---|
| `--bg` | `#0A0A0A` | page background |
| `--surface` / `--surface-2` | `#131313` / `#1B1B1B` | cards, expanders, sidebar |
| `--border` | `#262626` | hairlines |
| `--text` / `--text-dim` | `#F5F5F5` / `#9CA3AF` | text |
| `--soft` (red) | `#E8002D` | primary buttons, active tab, danger / "unsafe", key highlights |
| `--medium` (yellow) | `#FFD600` | warnings, "risky", focus rings, hover accents |
| `--hard` (white) | `#F2F2F2` | neutral emphasis, headline numbers |
| `--inter` (green) | `#39B54A` | safe / success states |
| `--wet` (blue) | `#0072CE` | info, links, selection |

Rules:
- In any chart that shows tyre compounds, those five colours mean **compounds only**. Driver traces use
  team colours (OpenF1 `team_colour`, config fallback); teammates get a dashed second trace.
- Never let colour alone carry meaning: pair red/green with a sign, word or icon.
- One shared Plotly template: black paper/plot bg, `--border` gridlines, `--text-dim` axis text, no glows.
- Prefer `.streamlit/config.toml` theming. Check which theme keys the installed Streamlit version
  supports (recent versions cover fonts, border colour, radius) before writing CSS for the same thing.
- Typography: one display font for headings/numbers (e.g. Titillium Web or Rajdhani via Google Fonts),
  a clean sans for body, tabular numerals for timing tables.
- No official F1, FIA, Pirelli or team logos/wordmarks. Colours are fine; brand marks are not.
  Remove the emoji "team logos".

## Motion ("F1 transitions")

CSS-only (scripts in `st.markdown`/`st.html` don't run reliably), subtle and fast (150–400 ms, ease-out):
- **Lights-out loader:** while a session loads, five red lights fill one by one, then all go out and
  content fades in. Render into an `st.empty()` placeholder; clear it after load.
- **Tab change:** content slides in from the right with a short fade; the active-tab underline animates
  in; each tab may own one compound colour.
- **Kerb-stripe dividers** (thin red/white diagonal stripes) between sections; chequered strip under header.
- **Metric cards:** speed-line sweep on hover.
- Plotly transitions only if they actually animate under Streamlit's rerender model. Don't fake it.
- Everything inside `@media (prefers-reduced-motion: reduce)` → animations off.

## UX ("more seamless")

- Sidebar: season → GP → session. GP is a dropdown from OpenF1's meetings and sessions (exclude
  testing, cancelled and unfinished sessions), not free text. Main area: event header, Driver Face-Off, then analysis tabs.
- Every widget has a stable `key`; state lives in `st.session_state`; changing GP resets dependent
  selections gracefully.
- Each tab: one-line plain-English summary, then the chart, then a collapsed "Method" expander with maths.
- Empty/error states: no telemetry, sprint/quali/practice where race-only tabs don't apply, DNFs, wet
  races. Friendly explanation, never a stack trace; disable or explain tabs that can't work.
- Works at laptop width and doesn't break at phone width.

## Testing & verification

- `pytest` covers every model function, including the regression tests above (faster chaser ⇒ higher
  P than slower chaser; sector delta negative when Driver 1 faster; position targets never come from
  another driver or the test period; degradation fit recovers a known synthetic slope).
- `tests/test_app_smoke.py` runs the whole app with `streamlit.testing.v1.AppTest` against a
  **fake OpenF1 API** (`tests/fake_openf1.py`, monkeypatching `openf1.fetch`). No network in tests.
  Assert no exceptions across all tabs, for both race-like and qualifying-like data.
- After UI work, run the app and look at it (Playwright screenshots if available) at desktop and phone
  widths. Check the loader, tab transition and reduced-motion fallback actually work.
- GitHub Actions workflow runs `pytest` on push/PR.

## Housekeeping

- Pin `requirements.txt` to versions actually installed and tested; add `scipy`.
  `requirements-dev.txt` = `-r requirements.txt` + pytest.
- Add an MIT `LICENSE`. Keep `.devcontainer` working (update its commands if paths change).
- Rewrite README to match the code exactly: features table, method section with formulas, honest
  limitations, run/test instructions, structure, CI badge. Keep the existing demo video embed.
- Footer: correct tech stack, real links, year from `datetime`.

## Working rules

- Work on a branch. Small commits per phase with clear messages; `pytest` passes before each commit.
- Don't push or open a PR unless asked.
- Verify library APIs against the installed versions (Streamlit, OpenF1 responses) rather than assuming.
- If a spec here is wrong or impossible, say so and propose an alternative; don't silently deviate.
- Don't add features beyond this file without asking.
