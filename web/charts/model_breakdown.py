"""Plotly: Model breakdown — horizontal bars showing MLP, Elo, and Blend probabilities."""

import plotly.graph_objects as go
from src.config import BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR, ACCENT_COLOR


def create_model_breakdown(
    mlp_prob: float, elo_prob: float, blend_prob: float,
    alpha: float, p1_name: str = "P1",
) -> go.Figure:
    models = ["MLP", "Elo", f"Blend (a={alpha:.2f})"]
    probs = [mlp_prob, elo_prob, blend_prob]
    colors = [P1_COLOR, ACCENT_COLOR, "#69F0AE"]

    fig = go.Figure()
    for model, prob, color in zip(models, probs, colors):
        fig.add_trace(go.Bar(
            y=[model],
            x=[prob],
            orientation="h",
            name=model,
            marker_color=color,
            text=[f"{p1_name}: {prob:.1%} | Avversario: {1 - prob:.1%}"],
            textposition="inside",
            textfont=dict(size=12, color="#FFFFFF"),
            hovertemplate=f"<b>{model}</b><br>{p1_name}: {prob:.1%}<br>Avversario: {1 - prob:.1%}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Breakdown del Modello</b><br>"
            "<span style='font-size:12px;color:#8B949E'>"
            "Probabilita vittoria per ciascun modello</span>",
            x=0.5, font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            tickformat=".0%", range=[0, 1],
            gridcolor="#1E2733", color=TEXT_COLOR,
        ),
        yaxis=dict(color=TEXT_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=250,
        margin=dict(l=120, r=30, t=80, b=40),
        showlegend=False,
    )
    return fig
