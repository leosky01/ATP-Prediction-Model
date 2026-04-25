"""Plotly: Confidence gauge — go.Indicator with gauge display."""

import plotly.graph_objects as go
from src.config import BG_COLOR, TEXT_COLOR, P1_COLOR, WINNER_COLOR, ACCENT_COLOR


def create_confidence_gauge(confidence: float, upset_prob: float) -> go.Figure:
    fig = go.Figure()

    # Confidence gauge
    gauge_color = WINNER_COLOR if confidence > 0.6 else (P1_COLOR if confidence > 0.4 else "#FF5252")

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        delta=dict(reference=0.5, increasing=dict(color=WINNER_COLOR), decreasing=dict(color="#FF5252")),
        number=dict(font=dict(size=40, color=TEXT_COLOR), suffix=""),
        title=dict(text="<b>Confidenza</b>", font=dict(size=16, color=TEXT_COLOR)),
        gauge=dict(
            axis=dict(range=[0, 1], tickformat=".0%", tickfont=dict(color=TEXT_COLOR)),
            bar=dict(color=gauge_color, thickness=0.4),
            bgcolor=BG_COLOR,
            borderwidth=2,
            bordercolor="#1E2733",
            steps=[
                dict(range=[0, 0.3], color="rgba(255,82,82,0.2)"),
                dict(range=[0.3, 0.6], color="rgba(255,213,79,0.2)"),
                dict(range=[0.6, 1.0], color="rgba(105,240,174,0.2)"),
            ],
            threshold=dict(
                line=dict(color=ACCENT_COLOR, width=4),
                thickness=0.8,
                value=0.5,
            ),
        ),
    ))

    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=300,
        margin=dict(l=30, r=30, t=60, b=30),
        annotations=[
            dict(
                text=f"Probabilita Upset: <b>{upset_prob:.0%}</b>",
                xref="paper", yref="paper",
                x=0.5, y=-0.05, showarrow=False,
                font=dict(size=14, color=ACCENT_COLOR),
            ),
        ],
    )
    return fig
