"""Screen-size helpers: phone detection and one way to draw every chart.

Plotly figures have a fixed pixel height and Streamlit can't report the
viewport width, so phones are spotted from the browser's User-Agent ("Mobi" is
the token browsers use for phones). CSS covers what it can at narrow widths,
such as hiding the Plotly toolbar.
"""

import plotly.graph_objects as go
import streamlit as st

from f1dash.ui import charts

PHONE_HEIGHT_RATIO = 0.65
MIN_PHONE_HEIGHT = 260


def is_phone() -> bool:
    try:
        agent = st.context.headers.get("User-Agent", "") or ""
    except Exception:  # No browser request, e.g. in AppTest.
        return False
    return "Mobi" in agent


def chart(fig: go.Figure, key: str) -> None:
    """Draw a figure, shorter and without the toolbar on a phone."""
    phone = is_phone()
    if phone:
        height = fig.layout.height or 450
        fig.update_layout(height=max(MIN_PHONE_HEIGHT, int(height * PHONE_HEIGHT_RATIO)), **charts.PHONE_LAYOUT)
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": not phone, "displaylogo": False})
