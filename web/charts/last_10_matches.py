"""Plotly: Last 10 matches — scatter plot showing W/L sequence for both players."""

import plotly.graph_objects as go
from src.config import BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR, WINNER_COLOR


def create_last_10_matches(
    p1_name: str, p2_name: str,
    p1_results: list[dict],
    p2_results: list[dict],
) -> go.Figure:
    """
    p1_results/p2_results: list of {"opponent", "won", "score", "surface"}.
    Shows W (green circle) / L (red x) sequence for each player.
    """
    fig = go.Figure()

    # P1 results
    for i, r in enumerate(p1_results):
        color = WINNER_COLOR if r["won"] else "#FF5252"
        symbol = "circle" if r["won"] else "x"
        result_text = "V" if r["won"] else "S"
        fig.add_trace(go.Scatter(
            x=[i + 1], y=[1],
            mode="markers+text",
            marker=dict(size=18, color=color, symbol=symbol, line=dict(width=2, color="#FFFFFF")),
            text=[result_text],
            textfont=dict(size=12, color=TEXT_COLOR),
            hovertemplate=(
                f"<b>{p1_name}</b><br>"
                f"Avversario: {r['opponent']}<br>"
                f"Risultato: {'Vittoria' if r['won'] else 'Sconfitta'}<br>"
                f"Score: {r['score']}<br>"
                f"Superficie: {r['surface']}<extra></extra>"
            ),
            showlegend=(i == 0),
            name=p1_name,
            legendgroup=p1_name,
        ))

    # P2 results
    for i, r in enumerate(p2_results):
        color = WINNER_COLOR if r["won"] else "#FF5252"
        symbol = "circle" if r["won"] else "x"
        result_text = "V" if r["won"] else "S"
        fig.add_trace(go.Scatter(
            x=[i + 1], y=[0],
            mode="markers+text",
            marker=dict(size=18, color=color, symbol=symbol, line=dict(width=2, color="#FFFFFF")),
            text=[result_text],
            textfont=dict(size=12, color=TEXT_COLOR),
            hovertemplate=(
                f"<b>{p2_name}</b><br>"
                f"Avversario: {r['opponent']}<br>"
                f"Risultato: {'Vittoria' if r['won'] else 'Sconfitta'}<br>"
                f"Score: {r['score']}<br>"
                f"Superficie: {r['surface']}<extra></extra>"
            ),
            showlegend=(i == 0),
            name=p2_name,
            legendgroup=p2_name,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Ultime 10 Partite</b><br>"
            "<span style='font-size:12px;color:#8B949E'>"
            "V = Vittoria | S = Sconfitta</span>",
            x=0.5, font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(
            title="Partita #", dtick=1,
            gridcolor="#1E2733", color=TEXT_COLOR,
        ),
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=[p2_name, p1_name],
            gridcolor="#1E2733", color=TEXT_COLOR,
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_color=TEXT_COLOR,
        height=300,
        margin=dict(l=120, r=30, t=80, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig
