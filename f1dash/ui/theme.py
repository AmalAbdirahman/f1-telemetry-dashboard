"""CSS injection, the lights-out loader and reusable HTML snippets."""

from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

STYLE_SHEET = Path(__file__).resolve().parents[2] / "assets" / "style.css"
APP_URL = "https://formula1-telemetry-dashboard.streamlit.app"
REPO_URL = "https://github.com/AmalAbdirahman/f1-telemetry-dashboard"
LINKEDIN_URL = "https://www.linkedin.com/in/amalabdirahman"


def inject_css() -> None:
    st.html(f"<style>{STYLE_SHEET.read_text(encoding='utf-8')}</style>")


def lights_out_html(label: str) -> str:
    lights = "".join('<span class="f1-light"></span>' for _ in range(5))
    return (
        '<div class="f1-loader" role="status">'
        f'<div class="f1-lights">{lights}</div>'
        f'<p class="f1-loader-label">{escape(label)}</p>'
        "</div>"
    )


def show_header() -> None:
    st.html(
        '<header class="f1-header"><h1>F1 Telemetry &amp; Strategy</h1>'
        "<p>Compare laps, then test the strategy behind them.</p>"
        '<div class="f1-chequer"></div></header>'
    )


def show_event(event_name: str, session_name: str, year: int, detail: str) -> None:
    st.html(
        f'<section class="f1-event"><p class="f1-event-kicker">{year} · {escape(session_name)}</p>'
        f"<h2>{escape(event_name)}</h2><p>{escape(detail)}</p></section>"
    )


def divider() -> None:
    st.html('<div class="f1-kerb"></div>')


def show_footer() -> None:
    divider()
    st.markdown(
        "**Built with** Streamlit · Plotly · scikit-learn · statsmodels · SciPy  \n"
        "**Data** [OpenF1](https://openf1.org), an unofficial source not associated with Formula 1  \n"
        f"[Open the app]({APP_URL}) · [Source code]({REPO_URL}) · [LinkedIn]({LINKEDIN_URL})"
    )
    st.caption(f"© {datetime.now().year} Amal Abdirahman")
