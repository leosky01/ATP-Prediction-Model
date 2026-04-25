"""Plotly: Surface radar — Scatterpolar radar chart comparing surface performance."""

import plotly.graph_objects as go
from src.config import BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR


def create_surface_radar(
    p1_name: str, p2_name: str,
    p1_stats: dict[str, float],
    p2_stats: dict[str, float],
) -> go.Figure:
    """
    Radar chart with axes: Hard, Clay, Grass, Win Rate, Momentum.
    p1_stats/p2_stats: {"Hard": 0.65, "Clay": 0.72, "Grass": 0.58, ...}
    """
    categories = ["Hard", "Clay", "Grass", "Win Rate", "Momentum"]

    p1_values = [p1_stats.get(c, 0.5) for c in categories]
    p2_values = [p2_stats.get(c, 0.5) for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p1_values,
        theta=categories,
        fill="toself",
        name=p1_name,
        fillcolor="rgba(79,195,247,0.27)",
        line=dict(color=P1_COLOR, width=2),
        hovertemplate="%{theta}: %{r:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=p2_values,
        theta=categories,
        fill="toself",
        name=p2_name,
        fillcolor="rgba(255,82,82,0.27)",
        line=dict(color=P2_COLOR, width=2),
        hovertemplate="%{theta}: %{r:.1%}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
                gridcolor="#1E2733",
                color=TEXT_COLOR,
            ),
            angularaxis=dict(
                gridcolor="#1E2733",
                color=TEXT_COLOR,
            ),
            bgcolor=BG_COLOR,
        ),
        title=dict(
            text="<b>Radar Superficie</b><br>"
            "<span style='font-size:12px;color:#8B949E'>Win rate per superficie + statistiche</span>",
            x=0.5, font=dict(size=16, color=TEXT_COLOR),
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=450,
        margin=dict(l=50, r=50, t=80, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig
