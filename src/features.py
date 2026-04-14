"""
Feature engineering pipeline for ATP match prediction.

Produces 14 numerical + one-hot categorical features from raw match data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import NUM_COLS, CAT_COLS, ROLLING_WINDOW


# ── Rolling statistics per player ────────────────────────────────────────────

def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling win-rate (window=ROLLING_WINDOW) and current streak
    for every player, returned as a DataFrame indexed by (player, Date).

    Uses shift(1) to avoid look-ahead bias.
    """
    df = df.copy()
    df["p1_win"] = (df["Winner"] == df["Player_1"]).astype(int)
    df["p2_win"] = (df["Winner"] == df["Player_2"]).astype(int)

    all_players = set(df["Player_1"]).union(df["Player_2"])
    frames: list[pd.DataFrame] = []

    for player in tqdm(all_players, desc="Rolling stats per player"):
        matches = pd.concat([
            df[df["Player_1"] == player][["Date", "p1_win"]].rename(columns={"p1_win": "win"}),
            df[df["Player_2"] == player][["Date", "p2_win"]].rename(columns={"p2_win": "win"}),
        ]).sort_values("Date")

        matches["win_rate"] = (
            matches["win"]
            .rolling(window=ROLLING_WINDOW, min_periods=1)
            .mean()
            .shift(1)
        )
        matches["streak"] = (
            matches["win"]
            .groupby((matches["win"] != matches["win"].shift()).cumsum())
            .cumcount() + 1
        ) * matches["win"]

        frames.append(matches[["Date", "win_rate", "streak"]].assign(player=player))

    return pd.concat(frames).sort_values(["player", "Date"])


def _last_stat_before(stats_df: pd.DataFrame, player: str, date: pd.Timestamp) -> Tuple[float, float]:
    """Return (win_rate, streak) for *player* from the most recent row before *date*."""
    mask = (stats_df["player"] == player) & (stats_df["Date"] < date)
    subset = stats_df.loc[mask]
    if subset.empty:
        return 0.5, 0
    row = subset.iloc[-1]
    return float(row["win_rate"]), int(row["streak"])


def merge_rolling_stats(df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rolling win-rate and streak onto the match-level DataFrame.
    Uses vectorised approach when possible, falls back to per-row lookup.
    """
    stats_sorted = stats_df.set_index(["player", "Date"]).sort_index()

    results_p1, results_p2 = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging rolling stats"):
        p1_wr, p1_st = _last_stat_before(stats_sorted, row["Player_1"], row["Date"])
        p2_wr, p2_st = _last_stat_before(stats_sorted, row["Player_2"], row["Date"])
        results_p1.append((p1_wr, p1_st))
        results_p2.append((p2_wr, p2_st))

    df = df.copy()
    df[["p1_win_rate", "p1_streak"]] = results_p1
    df[["p2_win_rate", "p2_streak"]] = results_p2
    return df


# ── Head-to-Head ─────────────────────────────────────────────────────────────

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append h2h_win_rate and h2h_matches columns.
    H2H is accumulated row-by-row so only past matches are used.
    """
    h2h: Dict[Tuple[str, str], dict] = {}
    features: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="H2H features"):
        key = (row["Player_1"], row["Player_2"])
        record = h2h.get(key, {"p1_wins": 0, "matches": 0})

        features.append({
            "h2h_win_rate": record["p1_wins"] / record["matches"] if record["matches"] > 0 else 0.5,
            "h2h_matches": record["matches"],
        })

        if row["Winner"] == row["Player_1"]:
            record["p1_wins"] += 1
        record["matches"] += 1
        h2h[key] = record

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)


# ── Invariant (order-agnostic) features ──────────────────────────────────────

def build_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create match-invariant numerical features from the base columns.
    These are symmetric w.r.t. player swap (min/max instead of P1/P2).
    """
    df = df.copy()
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["odds_ratio"] = df["Odd_1"] / df["Odd_2"]
    df["win_rate_diff"] = df["p1_win_rate"] - df["p2_win_rate"]
    df["streak_diff"] = df["p1_streak"] - df["p2_streak"]

    df["min_rank"] = np.minimum(df["Rank_1"], df["Rank_2"])
    df["max_rank"] = np.maximum(df["Rank_1"], df["Rank_2"])
    df["min_odds"] = np.minimum(df["Odd_1"], df["Odd_2"])
    df["max_odds"] = np.maximum(df["Odd_1"], df["Odd_2"])
    df["max_win_rate"] = np.maximum(df["p1_win_rate"], df["p2_win_rate"])
    df["min_win_rate"] = np.minimum(df["p1_win_rate"], df["p2_win_rate"])
    df["max_streak"] = np.maximum(df["p1_streak"], df["p2_streak"])
    df["min_streak"] = np.minimum(df["p1_streak"], df["p2_streak"])

    return df


# ── Upset signal scoring ─────────────────────────────────────────────────────

def upset_signal_score(row: pd.Series) -> float:
    """
    Heuristic score (0-1) indicating the likelihood of an upset.
    Used both during training-time calibration and live prediction.
    """
    score = 0.0

    # 1) Form differential
    form_diff = row.get("win_rate_diff", 0)
    if form_diff > 0.2:
        score += 0.3
    elif form_diff > 0.1:
        score += 0.15

    # 2) Positive streak for underdog (P1 perspective)
    if row.get("p1_streak", 0) > 3:
        score += 0.2

    # 3) Favourable H2H
    if row.get("h2h_win_rate", 0.5) > 0.6 and row.get("h2h_matches", 0) >= 3:
        score += 0.25

    # 4) Odds vs ranking inconsistency
    rank_1 = max(row.get("Rank_1", 100), 1)
    rank_2 = max(row.get("Rank_2", 100), 1)
    odd_1 = max(row.get("Odd_1", 1.5), 1.01)
    odd_2 = max(row.get("Odd_2", 1.5), 1.01)

    rank_ratio = rank_2 / rank_1
    odds_ratio = odd_2 / odd_1
    if odds_ratio < rank_ratio * 0.8:
        score += 0.15

    return min(score, 1.0)


# ── Full pipeline ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline:
    rolling stats → H2H → invariant features.
    """
    print("Computing rolling statistics...")
    stats_df = compute_rolling_stats(df)

    print("Merging rolling stats onto matches...")
    df = merge_rolling_stats(df, stats_df)

    print("Computing head-to-head features...")
    df = compute_h2h_features(df)

    print("Building invariant features...")
    df = build_invariant_features(df)

    return df
