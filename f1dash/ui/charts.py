"""Plotly figure builders, all sharing one dark template.

Compound colours mean compounds only. Driver traces use team colours, and
teammates are told apart by a dashed second trace.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from f1dash import config
from f1dash.models import degradation, overtake, telemetry

TEMPLATE = "f1dash"
FONT = "Titillium Web, sans-serif"


def _axis() -> dict:
    return {
        "gridcolor": config.BORDER,
        "zerolinecolor": config.BORDER,
        "linecolor": config.BORDER,
        "tickfont": {"color": config.TEXT_DIM},
        "title": {"font": {"color": config.TEXT_DIM}},
    }


pio.templates[TEMPLATE] = go.layout.Template(
    layout={
        "paper_bgcolor": config.BG,
        "plot_bgcolor": config.BG,
        "font": {"family": FONT, "color": config.TEXT},
        "xaxis": _axis(),
        "yaxis": _axis(),
        "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 56, "r": 16, "t": 48, "b": 48},
        "hoverlabel": {"bgcolor": config.SURFACE_2, "bordercolor": config.BORDER},
    }
)


@dataclass(frozen=True)
class DriverStyle:
    name: str
    color: str
    dash: str


def driver_styles(drivers: pd.DataFrame, driver_1: str, driver_2: str) -> tuple[DriverStyle, DriverStyle]:
    """Team colours for two drivers, with a dashed line for a teammate."""
    table = drivers.set_index("Driver")
    first, second = table.loc[driver_1], table.loc[driver_2]
    teammates = first["Team"] == second["Team"]
    return (
        DriverStyle(driver_1, first["TeamColor"], "solid"),
        DriverStyle(driver_2, second["TeamColor"], "dash" if teammates else "solid"),
    )


def _rgba(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def telemetry_figure(lap_1, lap_2, style_1: DriverStyle, style_2: DriverStyle) -> go.Figure:
    """Speed, time delta, throttle and brake, all on a shared distance axis."""
    grid, delta = telemetry.time_delta(lap_1, lap_2)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Speed", f"Time delta ({style_1.name} − {style_2.name})", "Throttle", "Brake"),
    )
    for lap, style in ((lap_1, style_1), (lap_2, style_2)):
        line = {"color": style.color, "dash": style.dash, "width": 2}
        for row, channel in ((1, "Speed"), (3, "Throttle"), (4, "Brake")):
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=telemetry.align(lap, grid, channel),
                    name=style.name,
                    legendgroup=style.name,
                    showlegend=row == 1,
                    line=line,
                ),
                row=row,
                col=1,
            )
    fig.add_trace(
        go.Scatter(x=grid, y=delta, line={"color": config.HARD, "width": 2}, showlegend=False, name="Delta"),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line={"color": config.TEXT_DIM, "dash": "dot", "width": 1}, row=2, col=1)
    fig.update_yaxes(title_text="km/h", row=1, col=1)
    fig.update_yaxes(title_text="s", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="%", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)
    fig.update_layout(template=TEMPLATE, height=760, hovermode="x unified")
    return fig


def tyre_figure(clean_laps: pd.DataFrame, fits: pd.DataFrame, drivers: list[str]) -> go.Figure:
    """Fuel-corrected pace against tyre age, one panel per driver."""
    fig = make_subplots(rows=1, cols=len(drivers), shared_yaxes=True, subplot_titles=drivers, horizontal_spacing=0.04)
    shown = set()
    for column, driver in enumerate(drivers, start=1):
        own = clean_laps[clean_laps["Driver"] == driver]
        for compound, group in own.groupby("Compound"):
            colour = config.COMPOUND_COLORS.get(compound, config.TEXT_DIM)
            fig.add_trace(
                go.Scatter(
                    x=group["TyreLife"],
                    y=degradation.fuel_corrected(group),
                    mode="markers",
                    name=compound.title(),
                    legendgroup=compound,
                    showlegend=compound not in shown,
                    marker={"color": colour, "size": 7, "line": {"width": 0}},
                    customdata=group[["LapNumber", "Stint"]],
                    hovertemplate="Tyre age %{x}<br>%{y:.3f} s<br>Lap %{customdata[0]}, stint %{customdata[1]}"
                    "<extra></extra>",
                ),
                row=1,
                col=column,
            )
            shown.add(compound)
        for fit in fits[fits["Driver"] == driver].itertuples():
            stint = own[own["Stint"] == fit.Stint]
            ends = np.array([stint["TyreLife"].min(), stint["TyreLife"].max()])
            fig.add_trace(
                go.Scatter(
                    x=ends,
                    y=fit.Intercept + fit.Slope * ends,
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line={"color": config.COMPOUND_COLORS.get(fit.Compound, config.TEXT_DIM), "width": 2},
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(title_text="Tyre age (laps)", row=1, col=column)
    fig.update_yaxes(title_text="Fuel-corrected lap (s)", row=1, col=1)
    fig.update_layout(template=TEMPLATE, height=460)
    return fig


def gap_fan_figure(summary: overtake.GapSummary, chaser: str, leader: str) -> go.Figure:
    laps = list(range(1, len(summary.median_gap) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=laps + laps[::-1],
            y=list(summary.band_high) + list(summary.band_low)[::-1],
            fill="toself",
            fillcolor=_rgba(config.WET, 0.25),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=f"{overtake.BAND_COVERAGE}% of simulations",
        )
    )
    fig.add_trace(go.Scatter(x=laps, y=summary.median_gap, line={"color": config.HARD, "width": 2}, name="Median gap"))
    fig.add_hline(
        y=0,
        line={"color": config.SOFT, "dash": "dash"},
        annotation_text=f"{chaser} ahead below this line",
        annotation_position="bottom left",
    )
    fig.update_layout(
        template=TEMPLATE,
        height=440,
        hovermode="x unified",
        xaxis_title="Laps from now",
        yaxis_title=f"Gap {chaser} to {leader} (s)",
    )
    return fig


def pace_figure(curves, driver: str, n_laps: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=curves.x, y=curves.prior_pdf, name="Prior (field)", line={"color": config.TEXT_DIM, "dash": "dot"})
    )
    fig.add_trace(
        go.Scatter(
            x=curves.x,
            y=curves.likelihood_pdf,
            name=f"Likelihood ({n_laps} laps)",
            line={"color": config.WET, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curves.x,
            y=curves.posterior_pdf,
            name="Posterior",
            line={"color": config.HARD, "width": 3},
            fill="tozeroy",
            fillcolor=_rgba(config.HARD, 0.08),
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        height=420,
        xaxis_title=f"{driver}'s mean fuel-corrected lap (s)",
        yaxis_title="Density",
    )
    return fig


def rejoin_figure(x, y, clearance: float, rival: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="tozeroy",
            fillcolor=_rgba(config.WET, 0.2),
            line={"color": config.HARD, "width": 2},
            name="Gap after the stop",
            showlegend=False,
        )
    )
    fig.add_vline(
        x=clearance,
        line={"color": config.SOFT, "dash": "dash"},
        annotation_text=f"Clearance to {rival}",
        annotation_position="top right",
    )
    fig.update_layout(template=TEMPLATE, height=400, xaxis_title="Gap after the stop (s)", yaxis_title="Density")
    return fig
