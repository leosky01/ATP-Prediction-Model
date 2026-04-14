"""
ATP Tennis Match Predictor – Visualisation Module

Generates 16:9 (1920x1080) PNG frames with a dark theme for video production.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

from .config import (
    BG_COLOR, TEXT_COLOR, P1_COLOR, P2_COLOR, WINNER_COLOR,
    ACCENT_COLOR, GRID_COLOR, VIDEO_WIDTH, VIDEO_HEIGHT,
    THUMB_WIDTH, THUMB_HEIGHT, TEMP_DIR, FEATURE_LABELS_IT,
)


# ── Theme setup ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "font.family": "sans-serif",
    "font.size": 14,
})

DPI = 100
FIG_W = VIDEO_WIDTH / DPI   # 19.2
FIG_H = VIDEO_HEIGHT / DPI  # 10.8


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", pad_inches=0.3,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ── 1. Match Prediction Card ────────────────────────────────────────────────

def match_prediction_card(
    p1_name: str, p2_name: str,
    rank1: int, rank2: int,
    prob_p1: float, prob_p2: float,
    confidence: float,
    tournament: str = "Torneo ATP",
    surface: str = "Hard",
    winner: str = "",
    save_path: Optional[Path] = None,
) -> Path:
    """Hero card: names, ranks, probability bar, confidence badge."""

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Tournament / surface header
    ax.text(5, 5.5, tournament, fontsize=28, fontweight="bold",
            ha="center", va="center", color=ACCENT_COLOR)
    ax.text(5, 5.0, f"Superficie: {surface}", fontsize=16,
            ha="center", va="center", color=TEXT_COLOR, alpha=0.7)

    # Player 1 (left)
    p1_color = WINNER_COLOR if winner == p1_name else P1_COLOR
    ax.text(2.2, 3.8, p1_name, fontsize=30, fontweight="bold",
            ha="center", va="center", color=p1_color)
    ax.text(2.2, 3.1, f"Rank #{rank1}", fontsize=20,
            ha="center", va="center", color=TEXT_COLOR, alpha=0.8)
    ax.text(2.2, 2.5, f"{prob_p1:.0%}", fontsize=44, fontweight="bold",
            ha="center", va="center", color=p1_color)

    # VS
    ax.text(5, 3.2, "VS", fontsize=36, fontweight="bold",
            ha="center", va="center", color=TEXT_COLOR, alpha=0.5)

    # Player 2 (right)
    p2_color = WINNER_COLOR if winner == p2_name else P2_COLOR
    ax.text(7.8, 3.8, p2_name, fontsize=30, fontweight="bold",
            ha="center", va="center", color=p2_color)
    ax.text(7.8, 3.1, f"Rank #{rank2}", fontsize=20,
            ha="center", va="center", color=TEXT_COLOR, alpha=0.8)
    ax.text(7.8, 2.5, f"{prob_p2:.0%}", fontsize=44, fontweight="bold",
            ha="center", va="center", color=p2_color)

    # Probability bar
    bar_y = 1.3
    bar_h = 0.45
    bar_w = 7.0
    bar_x = 1.5
    p1_frac = prob_p1

    # Background bar
    ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w, bar_h,
                                boxstyle="round,pad=0.05",
                                facecolor=GRID_COLOR, edgecolor="none"))
    # P1 fill
    ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w * p1_frac, bar_h,
                                boxstyle="round,pad=0.05",
                                facecolor=p1_color, edgecolor="none", alpha=0.85))

    ax.text(bar_x + bar_w * p1_frac / 2, bar_y + bar_h / 2,
            f"{prob_p1:.0%}", fontsize=16, fontweight="bold",
            ha="center", va="center", color="white")
    ax.text(bar_x + bar_w * (1 + p1_frac) / 2, bar_y + bar_h / 2,
            f"{prob_p2:.0%}", fontsize=16, fontweight="bold",
            ha="center", va="center", color="white")

    # Confidence badge
    conf_color = WINNER_COLOR if confidence > 0.6 else "#E3B341" if confidence > 0.35 else P2_COLOR
    ax.text(5, 0.5, f"Confidenza: {confidence:.0%}", fontsize=20,
            fontweight="bold", ha="center", va="center", color=conf_color)

    # Winner announcement
    if winner:
        ax.text(5, 4.6, f"Predizione: {winner}", fontsize=22,
                fontweight="bold", ha="center", va="center", color=WINNER_COLOR,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR,
                          edgecolor=WINNER_COLOR, linewidth=2, alpha=0.9))

    save_path = save_path or TEMP_DIR / "01_prediction_card.png"
    return _save(fig, save_path)


# ── 2. Feature Contribution Circles ─────────────────────────────────────────

def feature_contribution_circles(
    contributions: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Interlocking circles chart: each feature = one circle.
    Size = importance, colour = direction (blue=P1 favour, red=P2 favour).
    P1 circles cluster left, P2 right, with cross-cluster overlap at centre.
    """

    if not contributions:
        contributions = {"rank_diff": 0.3, "odds_ratio": -0.2, "win_rate_diff": 0.15,
                         "streak_diff": -0.1, "h2h_win_rate": 0.25, "momentum_diff": 0.05}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0, 1.6, "Analisi Feature per Feature", fontsize=28,
            fontweight="bold", ha="center", va="center", color=TEXT_COLOR)

    # Separate features by direction
    p1_features = {k: v for k, v in contributions.items() if v >= 0}
    p2_features = {k: v for k, v in contributions.items() if v < 0}

    max_abs = max(abs(v) for v in contributions.values()) if contributions else 1
    max_abs = max(max_abs, 0.01)

    def _radius(value: float) -> float:
        return 0.15 + 0.6 * abs(value) / max_abs

    def _layout_circles(features: dict, center_x: float) -> List[Tuple[float, float, float, str, float]]:
        """Simple iterative layout: spread circles around centre to reduce same-cluster overlap."""
        items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        placed = []
        for key, val in items:
            r = _radius(val)
            if not placed:
                placed.append((center_x, 0, r, key, val))
                continue
            # Try positions in expanding spiral
            best_pos = None
            best_overlap = float("inf")
            for angle_deg in range(0, 360, 30):
                for dist in [0.0, 0.4, 0.8, 1.2, 1.6]:
                    angle = math.radians(angle_deg)
                    cx = center_x + dist * math.cos(angle)
                    cy = dist * math.sin(angle)
                    # Only check overlap with same-cluster circles
                    overlap = sum(
                        max(0, r + pr - math.hypot(cx - px, cy - py))
                        for px, py, pr, _, _ in placed
                    )
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_pos = (cx, cy)
            placed.append((best_pos[0], best_pos[1], r, key, val))
        return placed

    p1_circles = _layout_circles(p1_features, -0.7)
    p2_circles = _layout_circles(p2_features, 0.7)

    # Draw circles with glow
    for circles, base_color in [
        (p1_circles, P1_COLOR),
        (p2_circles, P2_COLOR),
    ]:
        for cx, cy, r, key, val in circles:
            # Glow
            for i in range(3):
                ax.add_patch(Circle((cx, cy), r * (1.1 + i * 0.08),
                                    facecolor=base_color, edgecolor="none",
                                    alpha=0.06))
            # Main circle
            alpha = 0.35 + 0.55 * abs(val) / max_abs
            ax.add_patch(Circle((cx, cy), r,
                                facecolor=base_color, edgecolor="white",
                                linewidth=1.5, alpha=alpha))
            # Label
            label = FEATURE_LABELS_IT.get(key, key)
            ax.text(cx, cy, label, fontsize=11, fontweight="bold",
                    ha="center", va="center", color="white")

    # P1 / P2 labels
    ax.text(-1.8, -1.5, "P1", fontsize=24, fontweight="bold",
            ha="center", va="center", color=P1_COLOR, alpha=0.6)
    ax.text(1.8, -1.5, "P2", fontsize=24, fontweight="bold",
            ha="center", va="center", color=P2_COLOR, alpha=0.6)

    save_path = save_path or TEMP_DIR / "02_feature_circles.png"
    return _save(fig, save_path)


# ── 3. Player Comparison Bars (Butterfly) ────────────────────────────────────

def player_comparison_bars(
    p1_name: str, p2_name: str,
    stats: Dict[str, Tuple[float, float]],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Butterfly chart: P1 bars ← | → P2 bars.
    stats: dict of metric_name → (p1_value, p2_value) in [0, 1] or comparable scale.
    """

    if not stats:
        stats = {
            "Win Rate": (0.65, 0.70),
            "Serie Vittorie": (0.5, 0.6),
            "Momentum": (0.55, 0.72),
            "Superficie": (0.6, 0.65),
            "Scontri Diretti": (0.58, 0.42),
        }

    n = len(stats)
    labels = list(stats.keys())
    p1_vals = [stats[k][0] for k in labels]
    p2_vals = [stats[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    y_pos = np.arange(n)
    bar_h = 0.6

    # P1 bars (left, negative)
    ax.barh(y_pos, [-v for v in p1_vals], height=bar_h,
            color=P1_COLOR, alpha=0.85, edgecolor="none")
    # P2 bars (right, positive)
    ax.barh(y_pos, p2_vals, height=bar_h,
            color=P2_COLOR, alpha=0.85, edgecolor="none")

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=18, fontweight="bold")
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1.5, alpha=0.5)

    # Value annotations
    for i in range(n):
        ax.text(-p1_vals[i] - 0.03, i, f"{p1_vals[i]:.0%}",
                fontsize=14, ha="right", va="center", color=P1_COLOR, fontweight="bold")
        ax.text(p2_vals[i] + 0.03, i, f"{p2_vals[i]:.0%}",
                fontsize=14, ha="left", va="center", color=P2_COLOR, fontweight="bold")

    # Player headers
    ax.text(-0.55, n + 0.3, p1_name, fontsize=22, fontweight="bold",
            ha="center", va="center", color=P1_COLOR)
    ax.text(0.55, n + 0.3, p2_name, fontsize=22, fontweight="bold",
            ha="center", va="center", color=P2_COLOR)

    ax.set_title("Confronto Giocatori", fontsize=28, fontweight="bold",
                 color=TEXT_COLOR, pad=40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.grid(axis="x", alpha=0.15)

    save_path = save_path or TEMP_DIR / "03_comparison_bars.png"
    return _save(fig, save_path)


# ── 4. Confidence & Upset Gauge ─────────────────────────────────────────────

def confidence_upset_gauge(
    confidence: float,
    upset_prob: float,
    save_path: Optional[Path] = None,
) -> Path:
    """Two semicircular gauges: confidence (left) and upset probability (right)."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    def _draw_gauge(ax, value, title, color_good, color_bad, label_fmt="{:.0%}"):
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        # Background arc
        theta_bg = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta_bg), np.sin(theta_bg), color=GRID_COLOR,
                linewidth=20, solid_capstyle="round", alpha=0.5)

        # Value arc
        theta_val = np.linspace(0, np.pi * value, 100)
        color = color_good if value < 0.5 else color_bad
        ax.plot(np.cos(theta_val), np.sin(theta_val), color=color,
                linewidth=20, solid_capstyle="round", alpha=0.9)

        # Value text
        ax.text(0, 0.35, label_fmt.format(value), fontsize=48, fontweight="bold",
                ha="center", va="center", color=color)
        ax.text(0, -0.15, title, fontsize=22, fontweight="bold",
                ha="center", va="center", color=TEXT_COLOR)

    _draw_gauge(ax1, confidence, "Confidenza Modello",
                WINNER_COLOR, "#E3B341")
    _draw_gauge(ax2, upset_prob, "Probabilita Upset",
                P2_COLOR, "#E3B341")

    fig.suptitle("Affidabilita e Sorpresa", fontsize=28,
                 fontweight="bold", color=TEXT_COLOR, y=0.92)

    save_path = save_path or TEMP_DIR / "04_confidence_gauge.png"
    return _save(fig, save_path)


# ── 5. Model Breakdown ──────────────────────────────────────────────────────

def model_breakdown(
    mlp_prob: float,
    elo_prob: float,
    blend_prob: float,
    alpha: float,
    p1_name: str = "P1",
    save_path: Optional[Path] = None,
) -> Path:
    """Horizontal bars showing MLP, Elo and Blend probabilities with alpha weight."""

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    models = ["MLP (Neural Network)", "Elo Rating", f"Blend (alpha={alpha:.2f})"]
    probs = [mlp_prob, elo_prob, blend_prob]
    colors = [P1_COLOR, ACCENT_COLOR, WINNER_COLOR]

    y_pos = np.arange(len(models))
    bar_h = 0.55

    # Background bars
    ax.barh(y_pos, [1.0] * 3, height=bar_h,
            color=GRID_COLOR, alpha=0.4, edgecolor="none")
    # Value bars
    ax.barh(y_pos, probs, height=bar_h,
            color=colors, alpha=0.85, edgecolor="none")

    # Annotations
    for i, (prob, color) in enumerate(zip(probs, colors)):
        ax.text(prob + 0.02, i, f"{prob:.1%}",
                fontsize=22, fontweight="bold", ha="left", va="center", color=color)
        # Also show P(P2) on the right
        ax.text(0.98, i, f"P2: {1 - prob:.1%}",
                fontsize=14, ha="right", va="center", color=TEXT_COLOR, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=20, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.set_title(f"Analisi Modelli — P({p1_name})", fontsize=28,
                 fontweight="bold", color=TEXT_COLOR, pad=25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.grid(axis="x", alpha=0.15)

    # Alpha visualisation inset
    ax.text(0.5, -0.7,
            f"MLP x {alpha:.0%}  +  Elo x {1 - alpha:.0%}  =  Blend",
            fontsize=18, ha="center", va="center", color=TEXT_COLOR, alpha=0.7,
            transform=ax.get_yaxis_transform())

    save_path = save_path or TEMP_DIR / "05_model_breakdown.png"
    return _save(fig, save_path)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW VISUALISATIONS (v2 — daily video, richer analytics)
# ═══════════════════════════════════════════════════════════════════════════════


# ── 6. Elo Trend 12 Mesi ────────────────────────────────────────────────────

def elo_trend(
    p1_name: str, p2_name: str,
    p1_dates: List[str],
    p1_elos: List[float],
    p2_dates: List[str],
    p2_elos: List[float],
    surface: str = "Overall",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Dual line-chart showing Elo evolution for both players over 12 months.
    Each list should contain dates (str YYYY-MM-DD) and Elo values of equal length.
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Convert to datetime (use simple list to preserve length)
    dates1 = [pd.Timestamp(d) for d in p1_dates] if p1_dates else []
    dates2 = [pd.Timestamp(d) for d in p2_dates] if p2_dates else []

    if len(dates1) > 0:
        ax.plot(dates1, p1_elos, color=P1_COLOR, linewidth=2.5, label=p1_name,
                marker="o", markersize=4, markerfacecolor=P1_COLOR, alpha=0.9)
        # End label
        ax.annotate(f"{p1_elos[-1]:.0f}", (dates1[-1], p1_elos[-1]),
                    fontsize=14, fontweight="bold", color=P1_COLOR,
                    xytext=(10, 0), textcoords="offset points", va="center")

    if len(dates2) > 0:
        ax.plot(dates2, p2_elos, color=P2_COLOR, linewidth=2.5, label=p2_name,
                marker="o", markersize=4, markerfacecolor=P2_COLOR, alpha=0.9)
        ax.annotate(f"{p2_elos[-1]:.0f}", (dates2[-1], p2_elos[-1]),
                    fontsize=14, fontweight="bold", color=P2_COLOR,
                    xytext=(10, 0), textcoords="offset points", va="center")

    # Cross point highlight
    if len(dates1) > 0 and len(dates2) > 0:
        min_len = min(len(p1_elos), len(p2_elos))
        for i in range(1, min_len):
            if (p1_elos[i-1] - p2_elos[min(i-1, len(p2_elos)-1)]) * \
               (p1_elos[i] - p2_elos[min(i, len(p2_elos)-1)]) < 0:
                ax.axvline(dates1[i] if i < len(dates1) else dates1[-1],
                           color=ACCENT_COLOR, linestyle="--", alpha=0.4, linewidth=1)

    ax.set_title(f"Andamento Elo — {surface}", fontsize=28, fontweight="bold",
                 color=TEXT_COLOR, pad=20)
    ax.set_ylabel("Elo Rating", fontsize=16, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=18, loc="upper left", facecolor=BG_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.tick_params(axis="x", rotation=30)

    save_path = save_path or TEMP_DIR / "06_elo_trend.png"
    return _save(fig, save_path)


# ── 7. Ultime 10 Partite ────────────────────────────────────────────────────

def last_10_matches(
    p1_name: str, p2_name: str,
    p1_results: List[Dict],
    p2_results: List[Dict],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Visual W/L sequence for last 10 matches per player.
    Each result: {"opponent": str, "won": bool, "score": str, "surface": str}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H),
                                    gridspec_kw={"hspace": 0.45})

    def _draw_sequence(ax, player_name, results, color_win, color_loss):
        n = len(results)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-1.0, 1.8)
        ax.axis("off")

        ax.text(4.5, 1.55, player_name, fontsize=22, fontweight="bold",
                ha="center", va="center", color=TEXT_COLOR)

        for i, r in enumerate(results[:10]):
            won = r.get("won", False)
            bg_color = color_win if won else color_loss
            symbol = "V" if won else "S"
            opponent = r.get("opponent", "?")[:10]
            score = r.get("score", "")

            # Circle
            circle = Circle((i, 0.3), 0.38, facecolor=bg_color,
                             edgecolor="white", linewidth=1.5, alpha=0.85)
            ax.add_patch(circle)
            ax.text(i, 0.3, symbol, fontsize=16, fontweight="bold",
                    ha="center", va="center", color="white")

            # Opponent name below
            ax.text(i, -0.35, opponent, fontsize=9, ha="center", va="center",
                    color=TEXT_COLOR, alpha=0.7)
            # Score even smaller
            ax.text(i, -0.65, score, fontsize=7, ha="center", va="center",
                    color=TEXT_COLOR, alpha=0.5)

        # Win rate label
        wins = sum(1 for r in results[:10] if r.get("won"))
        wr = wins / max(n, 1)
        wr_color = color_win if wr >= 0.6 else "#E3B341" if wr >= 0.4 else color_loss
        ax.text(9.8, 1.55, f"{wr:.0%}", fontsize=20, fontweight="bold",
                ha="right", va="center", color=wr_color)

    p1_default = [{"opponent": "?", "won": i % 2 == 0, "score": "6-4 6-3"}
                  for i in range(10)]
    p2_default = [{"opponent": "?", "won": i % 3 != 0, "score": "6-4 4-6 6-3"}
                  for i in range(10)]

    _draw_sequence(ax1, p1_name, p1_results or p1_default, WINNER_COLOR, P2_COLOR)
    _draw_sequence(ax2, p2_name, p2_results or p2_default, WINNER_COLOR, P2_COLOR)

    fig.suptitle("Ultime 10 Partite", fontsize=28, fontweight="bold",
                 color=TEXT_COLOR, y=0.98)
    fig.text(0.5, 0.49, "V = Vittoria    S = Sconfitta", fontsize=14,
             ha="center", color=TEXT_COLOR, alpha=0.4)

    save_path = save_path or TEMP_DIR / "07_last_10.png"
    return _save(fig, save_path)


# ── 8. Radar Superficie ─────────────────────────────────────────────────────

def surface_radar(
    p1_name: str, p2_name: str,
    p1_stats: Dict[str, float],
    p2_stats: Dict[str, float],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Radar/spider chart comparing P1 vs P2 across dimensions.
    p1_stats/p2_stats: {"Hard": 0.65, "Clay": 0.72, "Grass": 0.55,
                         "Win Rate": 0.68, "Momentum": 0.70}
    """
    categories = list(p1_stats.keys()) if p1_stats else ["Hard", "Clay", "Grass", "Win Rate", "Momentum"]
    p1_values = [p1_stats.get(c, 0.5) for c in categories]
    p2_values = [p2_stats.get(c, 0.5) for c in categories]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the polygon
    p1_values_closed = p1_values + [p1_values[0]]
    p2_values_closed = p2_values + [p2_values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), subplot_kw=dict(polar=True))

    # Style the radar
    ax.set_facecolor(BG_COLOR)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid circles
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=["20%", "40%", "60%", "80%", "100%"],
                   fontsize=9, color=TEXT_COLOR, alpha=0.4)
    ax.set_rlabel_position(0)
    ax.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax.xaxis.grid(True, color=GRID_COLOR, alpha=0.3)

    # Plot
    ax.plot(angles_closed, p1_values_closed, color=P1_COLOR, linewidth=2.5,
            label=p1_name, linestyle="-")
    ax.fill(angles_closed, p1_values_closed, color=P1_COLOR, alpha=0.15)

    ax.plot(angles_closed, p2_values_closed, color=P2_COLOR, linewidth=2.5,
            label=p2_name, linestyle="-")
    ax.fill(angles_closed, p2_values_closed, color=P2_COLOR, alpha=0.15)

    # Category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=15, fontweight="bold", color=TEXT_COLOR)

    ax.legend(fontsize=18, loc="lower right", facecolor=BG_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              bbox_to_anchor=(1.15, -0.05))

    ax.set_title("Rendimento per Superficie", fontsize=28, fontweight="bold",
                 color=TEXT_COLOR, pad=30)

    save_path = save_path or TEMP_DIR / "08_surface_radar.png"
    return _save(fig, save_path)


# ── 9. Daily Batch Card ─────────────────────────────────────────────────────

def daily_batch_card(
    matches: List[Dict],
    tournament: str = "Torneo ATP",
    date_str: str = "Oggi",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Multi-match prediction table: all matches of the day at a glance.
    Each match: {"p1": str, "p2": str, "prob_winner": float,
                 "winner": str, "confidence": float, "surface": str}
    """
    n = len(matches)
    row_h = 0.9
    header_h = 2.0
    total_h = header_h + n * row_h + 0.8

    fig_w = FIG_W
    fig_h = max(FIG_H, total_h / DPI + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # Header
    ax.text(5, total_h - 0.6, tournament, fontsize=32, fontweight="bold",
            ha="center", va="center", color=ACCENT_COLOR)
    ax.text(5, total_h - 1.2, f"Predizioni del giorno — {date_str}",
            fontsize=18, ha="center", va="center", color=TEXT_COLOR, alpha=0.7)
    ax.text(5, total_h - 1.7, f"{n} match", fontsize=16,
            ha="center", va="center", color=TEXT_COLOR, alpha=0.5)

    # Column headers
    y_header = total_h - header_h + 0.3
    ax.text(0.3, y_header, "#", fontsize=14, fontweight="bold", color=TEXT_COLOR, alpha=0.5)
    ax.text(1.0, y_header, "Match", fontsize=14, fontweight="bold", color=TEXT_COLOR, alpha=0.5)
    ax.text(5.5, y_header, "Superficie", fontsize=14, fontweight="bold", color=TEXT_COLOR, alpha=0.5)
    ax.text(7.0, y_header, "Predizione", fontsize=14, fontweight="bold", color=TEXT_COLOR, alpha=0.5)
    ax.text(8.5, y_header, "Prob.", fontsize=14, fontweight="bold", color=TEXT_COLOR, alpha=0.5)
    ax.text(9.5, y_header, "Conf.", fontsize=14, fontweight="bold", color=TEXT_COLOR, alpha=0.5)

    # Separator
    ax.plot([0.2, 9.8], [y_header - 0.25, y_header - 0.25],
            color=GRID_COLOR, linewidth=1.5)

    # Rows
    for i, m in enumerate(matches):
        y = total_h - header_h - i * row_h

        # Alternating row background
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((0.1, y - 0.3), 9.8, row_h - 0.05,
                                        boxstyle="round,pad=0.02",
                                        facecolor=GRID_COLOR, edgecolor="none",
                                        alpha=0.3))

        p1 = m.get("p1", "?")
        p2 = m.get("p2", "?")
        winner = m.get("winner", "?")
        prob = m.get("prob_winner", 0.5)
        conf = m.get("confidence", 0.5)
        surface = m.get("surface", "Hard")

        conf_color = WINNER_COLOR if conf > 0.6 else "#E3B341" if conf > 0.35 else P2_COLOR

        # Row content
        ax.text(0.3, y, str(i + 1), fontsize=16, fontweight="bold",
                ha="center", va="center", color=TEXT_COLOR, alpha=0.5)
        ax.text(1.0, y, f"{p1}  vs  {p2}", fontsize=15,
                va="center", color=TEXT_COLOR)
        ax.text(5.5, y, surface, fontsize=14, va="center",
                color=ACCENT_COLOR, ha="center")
        ax.text(7.0, y, winner, fontsize=16, fontweight="bold",
                va="center", color=WINNER_COLOR, ha="center")
        ax.text(8.5, y, f"{prob:.0%}", fontsize=16, fontweight="bold",
                va="center", ha="center", color=TEXT_COLOR)
        ax.text(9.5, y, f"{conf:.0%}", fontsize=14, fontweight="bold",
                va="center", ha="center", color=conf_color)

    # Summary footer
    y_footer = total_h - header_h - n * row_h - 0.3
    high_conf = sum(1 for m in matches if m.get("confidence", 0) > 0.6)
    ax.text(5, y_footer, f"Alta confidenza: {high_conf}/{n} match",
            fontsize=16, ha="center", va="center", color=TEXT_COLOR, alpha=0.6)

    save_path = save_path or TEMP_DIR / "09_daily_card.png"
    return _save(fig, save_path)


# ── 10. Track Record Accuracy ───────────────────────────────────────────────

def track_record_accuracy(
    dates: List[str],
    accuracies: List[float],
    n_matches: List[int],
    overall_acc: Optional[float] = None,
    period_label: str = "Ultimi 90 giorni",
    save_path: Optional[Path] = None,
) -> Path:
    """
    Rolling accuracy chart showing model performance over time.
    dates: date labels (str), accuracies: rolling accuracy %, n_matches: cumulative count.
    """
    import pandas as pd

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))

    if dates:
        x = [pd.Timestamp(d) for d in dates]
    else:
        x = list(range(len(accuracies)))

    # Accuracy line
    ax1.plot(x, [a * 100 for a in accuracies], color=WINNER_COLOR, linewidth=3,
             marker="o", markersize=5, markerfacecolor=WINNER_COLOR, alpha=0.9,
             label="Accuratezza")
    ax1.fill_between(x, [a * 100 for a in accuracies], alpha=0.1, color=WINNER_COLOR)

    # 50% reference line
    ax1.axhline(50, color=TEXT_COLOR, linestyle="--", alpha=0.3, linewidth=1)
    ax1.text(x[0], 51, "50% (caso)",
             fontsize=12, color=TEXT_COLOR, alpha=0.4)

    ax1.set_ylabel("Accuratezza (%)", fontsize=16, fontweight="bold",
                   color=WINNER_COLOR)
    ax1.set_ylim(30, 85)
    ax1.tick_params(axis="y", labelcolor=WINNER_COLOR)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.15)

    # Volume on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(x, n_matches, width=2,
            color=ACCENT_COLOR, alpha=0.25, label="Match analizzati")
    ax2.set_ylabel("Match", fontsize=14, color=ACCENT_COLOR, alpha=0.6)
    ax2.tick_params(axis="y", labelcolor=ACCENT_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Overall accuracy badge
    if overall_acc is not None:
        ax1.text(0.98, 0.95, f"Accuratezza totale: {overall_acc:.1%}",
                 fontsize=22, fontweight="bold", ha="right", va="top",
                 transform=ax1.transAxes, color=WINNER_COLOR,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                           edgecolor=WINNER_COLOR, linewidth=2, alpha=0.9))

    ax1.set_title(f"Track Record Modello — {period_label}", fontsize=28,
                  fontweight="bold", color=TEXT_COLOR, pad=20)
    ax1.tick_params(axis="x", rotation=30)

    save_path = save_path or TEMP_DIR / "10_accuracy_tracker.png"
    return _save(fig, save_path)
