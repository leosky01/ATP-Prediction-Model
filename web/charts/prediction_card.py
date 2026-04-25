"""Plotly: Prediction card — horizontal stacked bar showing P1 vs P2 probability."""

import plotly.graph_objects as go
from src.config import P1_COLOR, P2_COLOR, WINNER_COLOR, BG_COLOR, TEXT_COLOR


def create_prediction_card(
    p1_name: str, p2_name: str,
    rank1: int, rank2: int,
    prob_p1: float, prob_p2: float,
    confidence: float,
    tournament: str = "", surface: str = "",
    winner: str = "",
) -> go.Figure:
    fig = go.Figure()

    # Stacked horizontal bar
    fig.add_trace(go.Bar(
        y=["Probabilita"],
        x=[prob_p1],
        orientation="h",
        name=p1_name,
        marker_color=P1_COLOR,
        hovertemplate=f"{p1_name}: {prob_p1:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=["Probabilita"],
        x=[prob_p2],
        orientation="h",
        name=p2_name,
        marker_color=P2_COLOR,
        hovertemplate=f"{p2_name}: {prob_p2:.1%}<extra></extra>",
    ))

    winner_color = WINNER_COLOR if confidence > 0.6 else "#FFD54F"
    title_text = (
        f"<b>{p1_name}</b> (#{rank1}) vs <b>{p2_name}</b> (#{rank2})<br>"
        f"<span style='font-size:14px;color:{winner_color}'>"
        f"Vincitore previsto: {winner} — Confidenza: {confidence:.0%}</span>"
    )
    if tournament:
        title_text += f"<br><span style='font-size:12px;color:#8B949E'>{tournament} | {surface}</span>"

    fig.update_layout(
        barmode="stack",
        title=dict(text=title_text, x=0.5, font=dict(size=18, color=TEXT_COLOR)),
        xaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#1E2733", color=TEXT_COLOR),
        yaxis=dict(visible=False),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=200,
        margin=dict(l=20, r=20, t=80, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig
