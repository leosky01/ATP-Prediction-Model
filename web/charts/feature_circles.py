"""Plotly: Feature contribution circles — bubble chart showing feature importance."""

import plotly.graph_objects as go
from src.config import (
    BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR, ACCENT_COLOR, FEATURE_LABELS_IT,
)


def create_feature_circles(contributions: dict) -> go.Figure:
    """Bubble chart where x=contribution value, y=feature, size=abs(contribution)."""
    features = list(FEATURE_LABELS_IT.keys())
    labels = [FEATURE_LABELS_IT[f] for f in features]
    values = [contributions.get(f, 0) for f in features]
    abs_values = [abs(v) for v in values]
    colors = [P1_COLOR if v > 0 else P2_COLOR for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values,
        y=labels,
        mode="markers+text",
        marker=dict(
            size=[max(a * 80, 15) for a in abs_values],
            color=colors,
            opacity=0.85,
            line=dict(width=2, color="#FFFFFF"),
        ),
        text=[f"{v:+.2f}" for v in values],
        textposition="middle center",
        textfont=dict(size=10, color=TEXT_COLOR),
        hovertemplate="<b>%{y}</b><br>Valore: %{x:+.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Contributo delle Feature</b><br>"
            "<span style='font-size:12px;color:#8B949E'>"
            f"<span style='color:{P1_COLOR}'>Blu</span> = favorevole a P1 | "
            f"<span style='color:{P2_COLOR}'>Rosso</span> = favorevole a P2</span>",
            x=0.5, font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Direzione (favorevole P1 ↔ P2)",
            gridcolor="#1E2733", color=TEXT_COLOR,
            zeroline=True, zerolinecolor="#8B949E",
        ),
        yaxis=dict(gridcolor="#1E2733", color=TEXT_COLOR),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=400,
        margin=dict(l=140, r=30, t=80, b=40),
        showlegend=False,
    )
    return fig
