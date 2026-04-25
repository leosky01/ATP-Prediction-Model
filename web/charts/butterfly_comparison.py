"""Plotly: Butterfly comparison — opposing horizontal bars for P1 vs P2."""

import plotly.graph_objects as go
from src.config import BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR


def create_butterfly_comparison(
    p1_name: str, p2_name: str,
    stats: dict[str, tuple[float, float]],
) -> go.Figure:
    """
    Butterfly chart: P1 bars go left (negative), P2 bars go right (positive).
    stats: {"Metric": (p1_value, p2_value)} — values in [0, 1].
    """
    metrics = list(stats.keys())
    p1_vals = [stats[m][0] for m in metrics]
    p2_vals = [stats[m][1] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=metrics,
        x=[-v for v in p1_vals],
        orientation="h",
        name=p1_name,
        marker_color=P1_COLOR,
        customdata=p1_vals,
        hovertemplate=f"{p1_name}: %{{customdata:.1%}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=metrics,
        x=p2_vals,
        orientation="h",
        name=p2_name,
        marker_color=P2_COLOR,
        customdata=p2_vals,
        hovertemplate=f"{p2_name}: %{{customdata:.1%}}<extra></extra>",
    ))

    fig.update_layout(
        barmode="relative",
        title=dict(
            text=f"<b>{p1_name} vs {p2_name}</b><br>"
            "<span style='font-size:12px;color:#8B949E'>Confronto statistiche</span>",
            x=0.5, font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            tickformat=".0%",
            gridcolor="#1E2733", color=TEXT_COLOR,
            zeroline=True, zerolinecolor="#8B949E",
        ),
        yaxis=dict(gridcolor="#1E2733", color=TEXT_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=350,
        margin=dict(l=140, r=30, t=80, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig
