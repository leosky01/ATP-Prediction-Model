"""Plotly: Elo trend — line chart showing Elo ratings over the last 12 months."""

import plotly.graph_objects as go
from src.config import BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR


def create_elo_trend(
    p1_name: str, p2_name: str,
    p1_dates: list[str], p1_elos: list[float],
    p2_dates: list[str], p2_elos: list[float],
    surface: str = "Overall",
) -> go.Figure:
    fig = go.Figure()

    if p1_dates:
        fig.add_trace(go.Scatter(
            x=p1_dates, y=p1_elos,
            mode="lines+markers",
            name=p1_name,
            line=dict(color=P1_COLOR, width=3),
            marker=dict(size=6),
            hovertemplate=f"<b>{p1_name}</b><br>Data: %{{x}}<br>Elo: %{{y:.0f}}<extra></extra>",
        ))

    if p2_dates:
        fig.add_trace(go.Scatter(
            x=p2_dates, y=p2_elos,
            mode="lines+markers",
            name=p2_name,
            line=dict(color=P2_COLOR, width=3),
            marker=dict(size=6),
            hovertemplate=f"<b>{p2_name}</b><br>Data: %{{x}}<br>Elo: %{{y:.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>Trend Elo — Ultimi 12 Mesi</b><br>"
            f"<span style='font-size:12px;color:#8B949E'>Superficie: {surface}</span>",
            x=0.5, font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Data", gridcolor="#1E2733", color=TEXT_COLOR,
        ),
        yaxis=dict(
            title="Elo Rating", gridcolor="#1E2733", color=TEXT_COLOR,
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=400,
        margin=dict(l=60, r=30, t=80, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig
