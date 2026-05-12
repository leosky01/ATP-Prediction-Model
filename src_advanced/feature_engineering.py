"""
Advanced feature engineering for the parallel prediction system.

Produces ~80-100 features across 6 categories:
  1. Match Context (rank, odds, surface, round, bio differences)
  2. Rolling Performance (3 windows × per-player stats → diff + ratio)
  3. Surface-Specific rolling stats
  4. Head-to-Head advanced (win rate, avg games, avg minutes, surface-specific)
  5. Bio features (hand match, backhand, BMI diff)
  6. Match format features (best_of, draw_size)

All rolling stats use shift(1) to avoid look-ahead bias.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CAT_COLS_ADVANCED, ROLLING_WINDOWS


# ── Feature name helpers ─────────────────────────────────────────────────────

def _rolling_features_names() -> List[str]:
    """Generate names for rolling features (diff + ratio for each stat × window)."""
    base_stats = [
        "win", "ace", "df", "bp_save_rate",
        "1st_serve_pct", "1st_serve_win_pct", "2nd_serve_win_pct",
        "minutes", "streak",
        "total_games", "sv_gms",
    ]
    feats: List[str] = []
    for w in ROLLING_WINDOWS:
        for stat in base_stats:
            feats.append(f"{stat}_diff_w{w}")
            feats.append(f"{stat}_ratio_w{w}")
    return feats


def _surface_features_names() -> List[str]:
    base_stats = ["win_rate", "avg_aces", "avg_bp_save", "avg_1st_pct", "matches"]
    feats: List[str] = []
    for stat in base_stats:
        feats.append(f"surf_{stat}_diff")
    return feats


def _h2h_features_names() -> List[str]:
    return [
        "h2h_win_rate", "h2h_matches", "h2h_avg_total_games",
        "h2h_avg_minutes", "h2h_surface_win_rate",
    ]


def _bio_features_names() -> List[str]:
    return ["hand_match", "backhand_match", "bmi_diff"]


def _tournament_features_names() -> List[str]:
    return [
        "tourney_avg_games", "tourney_std_games", "tourney_game_trend",
    ]


def _fatigue_features_names() -> List[str]:
    return [
        "p1_matches_7d", "p2_matches_7d", "fatigue_diff_7d",
        "p1_matches_14d", "p2_matches_14d", "fatigue_diff_14d",
        "p1_matches_30d", "p2_matches_30d", "fatigue_diff_30d",
    ]


def _context_features_names() -> List[str]:
    return [
        "rank_diff", "log_odds_ratio", "rank_points_diff",
        "age_diff", "height_diff", "seed_diff", "best_of", "draw_size",
        "rank_diff_abs", "odds_ratio",
        # Odds features (11)
        "odd_1", "odd_2",
        "log_odd_1", "log_odd_2",
        "implied_prob_1", "implied_prob_2",
        "prob_diff",
        "min_odds", "max_odds",
        "favorite_strength",
        "match_competitiveness",
    ]


def _raw_rolling_features_names() -> List[str]:
    """Per-player raw rolling features (p1_<stat>_w<W>, p2_<stat>_w<W>)."""
    base_stats = [
        "win", "ace", "df", "bp_save_rate",
        "1st_serve_pct", "1st_serve_win_pct", "2nd_serve_win_pct",
        "minutes", "streak",
        "total_games", "sv_gms",
    ]
    feats: List[str] = []
    for w in ROLLING_WINDOWS:
        for stat in base_stats:
            feats.append(f"p1_{stat}_w{w}")
            feats.append(f"p2_{stat}_w{w}")
    return feats


def _raw_surface_features_names() -> List[str]:
    """Per-player raw surface stats."""
    return [
        "p1_surf_win_rate", "p2_surf_win_rate",
        "p1_surf_avg_aces", "p2_surf_avg_aces",
        "p1_surf_avg_bp_save", "p2_surf_avg_bp_save",
        "p1_surf_avg_1st_pct", "p2_surf_avg_1st_pct",
        "p1_surf_matches", "p2_surf_matches",
    ]


def build_all_feature_names() -> List[str]:
    """Return the complete ordered list of numerical feature names."""
    names = (
        _context_features_names()
        + _rolling_features_names()
        + _raw_rolling_features_names()
        + _surface_features_names()
        + _raw_surface_features_names()
        + _h2h_features_names()
        + _bio_features_names()
        + _tournament_features_names()
        + _fatigue_features_names()
    )
    return names


# Exported constant for other modules
NUM_COLS_ADVANCED = build_all_feature_names()


# ── Per-player rolling statistics ────────────────────────────────────────────

def _compute_player_stats(row: pd.Series, is_winner: bool) -> Dict[str, float]:
    """Extract per-player stats from a match row."""
    prefix = "p1_" if is_winner else "p2_"
    # Note: after alignment, p1_* are always Player_1's stats
    # Here we use the actual player perspective
    ace = row.get(f"{prefix}ace", np.nan)
    df = row.get(f"{prefix}df", np.nan)
    svpt = row.get(f"{prefix}svpt", np.nan)
    first_in = row.get(f"{prefix}1stIn", np.nan)
    first_won = row.get(f"{prefix}1stWon", np.nan)
    second_won = row.get(f"{prefix}2ndWon", np.nan)
    bp_saved = row.get(f"{prefix}bpSaved", np.nan)
    bp_faced = row.get(f"{prefix}bpFaced", np.nan)

    stats = {
        "ace": ace,
        "df": df,
        "bp_save_rate": bp_saved / bp_faced if pd.notna(bp_saved) and pd.notna(bp_faced) and bp_faced > 0 else np.nan,
        "1st_serve_pct": first_in / svpt if pd.notna(first_in) and pd.notna(svpt) and svpt > 0 else np.nan,
        "1st_serve_win_pct": first_won / first_in if pd.notna(first_won) and pd.notna(first_in) and first_in > 0 else np.nan,
        "2nd_serve_win_pct": second_won / (svpt - first_in) if pd.notna(second_won) and pd.notna(svpt) and pd.notna(first_in) and (svpt - first_in) > 0 else np.nan,
        "minutes": row.get("minutes", np.nan),
        "win": 1.0,
    }
    return stats


def compute_rolling_stats_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistics for each player across multiple windows.

    For each player, builds a time series of match stats and computes rolling
    means over ROLLING_WINDOWS = [10, 20, 50], using shift(1) to avoid look-ahead.
    """
    df = df.copy()

    # Determine winner flag per player perspective
    df["_p1_won"] = (df["Winner"] == df["Player_1"]).astype(float)
    df["_p2_won"] = 1 - df["_p1_won"]

    all_players = set(df["Player_1"].dropna()) | set(df["Player_2"].dropna())

    # Collect per-player time series
    player_series: Dict[str, pd.DataFrame] = {}

    for player in tqdm(all_players, desc="Computing advanced rolling stats"):
        # Get all matches for this player
        is_p1 = df["Player_1"] == player
        is_p2 = df["Player_2"] == player

        p1_rows = df.loc[is_p1, ["Date"]].copy()
        p1_rows["win"] = df.loc[is_p1, "_p1_won"]
        for col in ["p1_ace", "p1_df", "p1_bpSaved", "p1_bpFaced",
                     "p1_1stIn", "p1_svpt", "p1_1stWon", "p1_2ndWon", "minutes"]:
            p1_rows[col] = df.loc[is_p1, col] if col in df.columns else np.nan
        # O/U-specific: total_games (match-level, same for both players)
        p1_rows["total_games"] = df.loc[is_p1, "total_games"] if "total_games" in df.columns else np.nan
        p1_rows["p1_SvGms"] = df.loc[is_p1, "p1_SvGms"] if "p1_SvGms" in df.columns else np.nan

        p2_rows = df.loc[is_p2, ["Date"]].copy()
        p2_rows["win"] = df.loc[is_p2, "_p2_won"]
        for col in ["p2_ace", "p2_df", "p2_bpSaved", "p2_bpFaced",
                     "p2_1stIn", "p2_svpt", "p2_1stWon", "p2_2ndWon", "minutes"]:
            src_col = col
            p2_rows[col] = df.loc[is_p2, src_col] if col in df.columns else np.nan
        # O/U-specific: total_games (match-level), SvGms from p2 perspective
        p2_rows["total_games"] = df.loc[is_p2, "total_games"] if "total_games" in df.columns else np.nan
        p2_rows["p2_SvGms"] = df.loc[is_p2, "p2_SvGms"] if "p2_SvGms" in df.columns else np.nan

        # Rename to canonical names
        rename_map = {
            "p1_ace": "ace", "p2_ace": "ace",
            "p1_df": "df", "p2_df": "df",
            "p1_bpSaved": "bp_saved", "p2_bpSaved": "bp_saved",
            "p1_bpFaced": "bp_faced", "p2_bpFaced": "bp_faced",
            "p1_1stIn": "first_in", "p2_1stIn": "first_in",
            "p1_svpt": "svpt", "p2_svpt": "svpt",
            "p1_1stWon": "first_won", "p2_1stWon": "first_won",
            "p1_2ndWon": "second_won", "p2_2ndWon": "second_won",
            "p1_SvGms": "sv_gms", "p2_SvGms": "sv_gms",
        }
        p1_rows = p1_rows.rename(columns=rename_map)
        p2_rows = p2_rows.rename(columns=rename_map)

        matches = pd.concat([p1_rows, p2_rows], ignore_index=True).sort_values("Date")

        # Compute derived stats
        matches["bp_save_rate"] = np.where(
            (matches["bp_faced"].notna()) & (matches["bp_faced"] > 0),
            matches["bp_saved"] / matches["bp_faced"],
            np.nan,
        )
        matches["1st_serve_pct"] = np.where(
            (matches["svpt"].notna()) & (matches["svpt"] > 0),
            matches["first_in"] / matches["svpt"],
            np.nan,
        )
        matches["1st_serve_win_pct"] = np.where(
            (matches["first_in"].notna()) & (matches["first_in"] > 0),
            matches["first_won"] / matches["first_in"],
            np.nan,
        )
        matches["2nd_serve_win_pct"] = np.where(
            (matches["svpt"].notna()) & (matches["first_in"].notna()) &
            ((matches["svpt"] - matches["first_in"]) > 0),
            matches["second_won"] / (matches["svpt"] - matches["first_in"]),
            np.nan,
        )

        # Compute rolling stats for each window
        roll_cols = ["win", "ace", "df", "bp_save_rate", "1st_serve_pct",
                     "1st_serve_win_pct", "2nd_serve_win_pct", "minutes",
                     "total_games", "sv_gms"]
        for w in ROLLING_WINDOWS:
            for col in roll_cols:
                matches[f"{col}_w{w}"] = matches[col].rolling(w, min_periods=1).mean().shift(1)

        # Streak (positive-only, like existing code)
        matches["streak"] = (
            matches["win"]
            .groupby((matches["win"] != matches["win"].shift()).cumsum())
            .cumcount() + 1
        ) * matches["win"]
        for w in ROLLING_WINDOWS:
            matches[f"streak_w{w}"] = matches["streak"].shift(1)

        matches["player"] = player
        player_series[player] = matches

    # Now merge rolling stats back to match-level DataFrame
    # Build lookup: (player, Date) -> rolling stats
    all_rolling = pd.concat(player_series.values(), ignore_index=True)

    return all_rolling


def merge_rolling_stats_advanced(
    df: pd.DataFrame,
    rolling_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pre-computed rolling stats onto match DataFrame."""
    rolling_sorted = rolling_df.set_index(["player", "Date"]).sort_index()

    base_stats = ["win", "ace", "df", "bp_save_rate", "1st_serve_pct",
                  "1st_serve_win_pct", "2nd_serve_win_pct", "minutes", "streak",
                  "total_games", "sv_gms"]

    results: Dict[str, list] = {}
    for w in ROLLING_WINDOWS:
        for stat in base_stats:
            results[f"p1_{stat}_w{w}"] = []
            results[f"p2_{stat}_w{w}"] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging rolling stats"):
        for w in ROLLING_WINDOWS:
            for stat in base_stats:
                col = f"{stat}_w{w}"

                # Player 1
                p1_val = _lookup_last(rolling_sorted, row["Player_1"], row["Date"], col)
                results[f"p1_{stat}_w{w}"].append(p1_val)

                # Player 2
                p2_val = _lookup_last(rolling_sorted, row["Player_2"], row["Date"], col)
                results[f"p2_{stat}_w{w}"].append(p2_val)

    out = df.copy()
    for col, vals in results.items():
        out[col] = vals

    return out


def _lookup_last(
    rolling_sorted: pd.DataFrame,
    player: str,
    date: pd.Timestamp,
    col: str,
) -> float:
    """Look up the most recent rolling stat for a player before a date."""
    try:
        mask = rolling_sorted.index.get_loc((player, ))
        # This is too slow with MultiIndex; use a simpler approach
        pass
    except (KeyError, TypeError):
        pass

    sub = rolling_sorted.loc[
        rolling_sorted.index.get_level_values("player") == player
    ]
    sub = sub[sub.index.get_level_values("Date") < date]
    if sub.empty or col not in sub.columns:
        return np.nan
    val = sub[col].iloc[-1]
    return float(val) if pd.notna(val) else np.nan


# ── Faster merge using vectorised approach ────────────────────────────────────

def merge_rolling_fast(
    df: pd.DataFrame,
    rolling_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge rolling stats using merge_asof for speed.
    Both DataFrames are sorted by Date globally (required by merge_asof).
    """
    base_stats = ["win", "ace", "df", "bp_save_rate", "1st_serve_pct",
                  "1st_serve_win_pct", "2nd_serve_win_pct", "minutes", "streak",
                  "total_games", "sv_gms"]

    select_cols = []
    for w in ROLLING_WINDOWS:
        for stat in base_stats:
            select_cols.append(f"{stat}_w{w}")

    available_cols = [c for c in select_cols if c in rolling_df.columns]
    rolling_clean = rolling_df[["player", "Date"] + available_cols].copy()
    rolling_clean = rolling_clean.dropna(subset=["player"])

    # Drop duplicate (player, Date) — keep last (includes all matches that day)
    rolling_clean = rolling_clean.drop_duplicates(subset=["player", "Date"], keep="last")

    # Build lookup tables sorted by Date (required by merge_asof)
    p1_lookup = rolling_clean.rename(columns={"player": "Player_1"}).sort_values("Date")
    p2_lookup = rolling_clean.rename(columns={"player": "Player_2"}).sort_values("Date")

    # Rename stat columns with p1_/p2_ prefix
    p1_rename = {c: f"p1_{c}" for c in available_cols}
    p2_rename = {c: f"p2_{c}" for c in available_cols}
    p1_lookup = p1_lookup.rename(columns=p1_rename)
    p2_lookup = p2_lookup.rename(columns=p2_rename)

    df_sorted = df.sort_values("Date")

    # Single merge_asof for all P1 stats
    p1_merged = pd.merge_asof(
        df_sorted[["Date", "Player_1"]].reset_index(),
        p1_lookup,
        left_on="Date",
        right_on="Date",
        by="Player_1",
        direction="backward",
    ).set_index("index").sort_index()

    # Single merge_asof for all P2 stats
    p2_merged = pd.merge_asof(
        df_sorted[["Date", "Player_2"]].reset_index(),
        p2_lookup,
        left_on="Date",
        right_on="Date",
        by="Player_2",
        direction="backward",
    ).set_index("index").sort_index()

    # Assign all columns at once
    for c in available_cols:
        df[f"p1_{c}"] = p1_merged[f"p1_{c}"].values
        df[f"p2_{c}"] = p2_merged[f"p2_{c}"].values

    return df


# ── Compute diff and ratio features ──────────────────────────────────────────

def compute_diff_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-player rolling stats to diff + ratio features."""
    df = df.copy()

    base_stats = ["win", "ace", "df", "bp_save_rate", "1st_serve_pct",
                  "1st_serve_win_pct", "2nd_serve_win_pct", "minutes", "streak",
                  "total_games", "sv_gms"]

    for w in ROLLING_WINDOWS:
        for stat in base_stats:
            p1_col = f"p1_{stat}_w{w}"
            p2_col = f"p2_{stat}_w{w}"

            if p1_col not in df.columns:
                df[f"{stat}_diff_w{w}"] = np.nan
                df[f"{stat}_ratio_w{w}"] = np.nan
                continue

            p1 = df[p1_col].astype(float)
            p2 = df[p2_col].astype(float)

            df[f"{stat}_diff_w{w}"] = p1 - p2

            # Safe ratio: (p1 + eps) / (p2 + eps)
            eps = 1e-8
            df[f"{stat}_ratio_w{w}"] = (p1 + eps) / (p2 + eps)

    return df


# ── Surface-specific features ────────────────────────────────────────────────

def compute_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute surface-specific rolling stats for each player."""
    df = df.copy()

    # Check if we have surface info
    surface_col = "Surface" if "Surface" in df.columns else "surface"
    if surface_col not in df.columns:
        for name in _surface_features_names():
            df[name] = np.nan
        return df

    # Build per-player surface stats
    all_players = set(df["Player_1"].dropna()) | set(df["Player_2"].dropna())

    # Accumulate surface stats row-by-row (no look-ahead)
    surf_stats: Dict[Tuple[str, str], Dict] = {}
    results: Dict[str, list] = {
        "p1_surf_win_rate": [], "p2_surf_win_rate": [],
        "p1_surf_avg_aces": [], "p2_surf_avg_aces": [],
        "p1_surf_avg_bp_save": [], "p2_surf_avg_bp_save": [],
        "p1_surf_avg_1st_pct": [], "p2_surf_avg_1st_pct": [],
        "p1_surf_matches": [], "p2_surf_matches": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Surface features"):
        surface = str(row.get(surface_col, "")).strip().lower()
        p1, p2 = row["Player_1"], row["Player_2"]
        p1_won = row.get("Winner", "") == p1

        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            key = (player, surface)
            rec = surf_stats.get(key, {"wins": 0, "aces": [], "bp_save": [], "1st_pct": [], "n": 0})

            win_rate = rec["wins"] / rec["n"] if rec["n"] > 0 else 0.5
            avg_aces = np.mean(rec["aces"]) if rec["aces"] else 0.0
            avg_bp = np.mean(rec["bp_save"]) if rec["bp_save"] else 0.0
            avg_1st = np.mean(rec["1st_pct"]) if rec["1st_pct"] else 0.0

            results[f"{prefix}_surf_win_rate"].append(win_rate)
            results[f"{prefix}_surf_avg_aces"].append(avg_aces)
            results[f"{prefix}_surf_avg_bp_save"].append(avg_bp)
            results[f"{prefix}_surf_avg_1st_pct"].append(avg_1st)
            results[f"{prefix}_surf_matches"].append(rec["n"])

        # Update stats after this match
        is_p1_winner = row.get("Winner", "") == p1
        p1_stats = {
            "ace": row.get("p1_ace", np.nan),
            "bp_save_rate": _safe_divide(row.get("p1_bpSaved"), row.get("p1_bpFaced")),
            "1st_pct": _safe_divide(row.get("p1_1stIn"), row.get("p1_svpt")),
        }
        p2_stats = {
            "ace": row.get("p2_ace", np.nan),
            "bp_save_rate": _safe_divide(row.get("p2_bpSaved"), row.get("p2_bpFaced")),
            "1st_pct": _safe_divide(row.get("p2_1stIn"), row.get("p2_svpt")),
        }

        for player, stats, won in [(p1, p1_stats, is_p1_winner), (p2, p2_stats, not is_p1_winner)]:
            key = (player, surface)
            rec = surf_stats.get(key, {"wins": 0, "aces": [], "bp_save": [], "1st_pct": [], "n": 0})
            rec["wins"] += int(won)
            rec["n"] += 1
            if pd.notna(stats["ace"]):
                rec["aces"].append(float(stats["ace"]))
            if pd.notna(stats["bp_save_rate"]):
                rec["bp_save"].append(float(stats["bp_save_rate"]))
            if pd.notna(stats["1st_pct"]):
                rec["1st_pct"].append(float(stats["1st_pct"]))
            surf_stats[key] = rec

    for col, vals in results.items():
        df[col] = vals

    # Compute diffs
    df["surf_win_rate_diff"] = df.get("p1_surf_win_rate", 0) - df.get("p2_surf_win_rate", 0)
    df["surf_avg_aces_diff"] = df.get("p1_surf_avg_aces", 0) - df.get("p2_surf_avg_aces", 0)
    df["surf_avg_bp_save_diff"] = df.get("p1_surf_avg_bp_save", 0) - df.get("p2_surf_avg_bp_save", 0)
    df["surf_avg_1st_pct_diff"] = df.get("p1_surf_avg_1st_pct", 0) - df.get("p2_surf_avg_1st_pct", 0)
    df["surf_matches_diff"] = df.get("p1_surf_matches", 0) - df.get("p2_surf_matches", 0)

    return df


# ── H2H Advanced features ────────────────────────────────────────────────────

def compute_h2h_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced H2H features: win rate, avg games, avg minutes, surface-specific."""
    df = df.copy()

    surface_col = "Surface" if "Surface" in df.columns else "surface"

    h2h: Dict[Tuple[str, str], Dict] = {}
    results: Dict[str, list] = {
        "h2h_win_rate": [], "h2h_matches": [],
        "h2h_avg_total_games": [], "h2h_avg_minutes": [],
        "h2h_surface_win_rate": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="H2H advanced"):
        p1, p2 = row["Player_1"], row["Player_2"]
        key = (p1, p2)
        key_rev = (p2, p1)

        record = h2h.get(key, h2h.get(key_rev, {
            "p1_wins": 0, "matches": 0, "total_games": [], "minutes": [],
            "surface_wins": {}, "surface_matches": {},
        }))

        # If we stored it reversed, flip perspective
        if key_rev in h2h and key not in h2h:
            record = {
                "p1_wins": record["matches"] - record["p1_wins"],
                "matches": record["matches"],
                "total_games": record["total_games"],
                "minutes": record["minutes"],
                "surface_wins": {k: record.get("surface_matches", {}).get(k, 0) - v
                                 for k, v in record.get("surface_wins", {}).items()},
                "surface_matches": record.get("surface_matches", {}),
            }

        # Emit features (before updating)
        m = record["matches"]
        results["h2h_win_rate"].append(record["p1_wins"] / m if m > 0 else 0.5)
        results["h2h_matches"].append(m)
        results["h2h_avg_total_games"].append(np.mean(record["total_games"]) if record["total_games"] else np.nan)
        results["h2h_avg_minutes"].append(np.mean(record["minutes"]) if record["minutes"] else np.nan)

        surface = str(row.get(surface_col, "")).strip()
        surf_rec = record.get("surface_wins", {}).get(surface, 0)
        surf_m = record.get("surface_matches", {}).get(surface, 0)
        results["h2h_surface_win_rate"].append(surf_rec / surf_m if surf_m > 0 else 0.5)

        # Update record
        is_p1_winner = row.get("Winner", "") == p1
        tg = row.get("total_games", np.nan)
        mins = row.get("minutes", np.nan)

        record["p1_wins"] += int(is_p1_winner)
        record["matches"] += 1
        if pd.notna(tg):
            record["total_games"].append(float(tg))
        if pd.notna(mins):
            record["minutes"].append(float(mins))
        if surface:
            record.setdefault("surface_wins", {})
            record.setdefault("surface_matches", {})
            record["surface_wins"][surface] = record["surface_wins"].get(surface, 0) + int(is_p1_winner)
            record["surface_matches"][surface] = record["surface_matches"].get(surface, 0) + 1

        h2h[key] = record

    for col, vals in results.items():
        df[col] = vals

    return df


# ── Bio features ─────────────────────────────────────────────────────────────

def compute_bio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bio-based features: hand match, BMI diff."""
    df = df.copy()

    # Hand match (1 if same hand, 0 otherwise)
    if "p1_hand" in df.columns and "p2_hand" in df.columns:
        df["hand_match"] = (df["p1_hand"] == df["p2_hand"]).astype(float)
    else:
        df["hand_match"] = 0.5

    # Height diff
    if "p1_ht" in df.columns and "p2_ht" in df.columns:
        df["height_diff"] = pd.to_numeric(df["p1_ht"], errors="coerce") - pd.to_numeric(df["p2_ht"], errors="coerce")
    else:
        df["height_diff"] = 0.0

    # Age diff
    if "p1_age" in df.columns and "p2_age" in df.columns:
        df["age_diff"] = pd.to_numeric(df["p1_age"], errors="coerce") - pd.to_numeric(df["p2_age"], errors="coerce")
    else:
        df["age_diff"] = 0.0

    # BMI diff (if height and weight available)
    # For now, we don't have weight, so use height^2 as proxy
    df["backhand_match"] = 0.0  # Placeholder; not available in TML
    df["bmi_diff"] = 0.0  # Placeholder

    return df


# ── Context features ─────────────────────────────────────────────────────────

def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute match context features from base columns."""
    df = df.copy()

    rank1 = pd.to_numeric(df.get("Rank_1", np.nan), errors="coerce")
    rank2 = pd.to_numeric(df.get("Rank_2", np.nan), errors="coerce")
    odds1 = pd.to_numeric(df.get("Odd_1", np.nan), errors="coerce")
    odds2 = pd.to_numeric(df.get("Odd_2", np.nan), errors="coerce")

    # Replace -1.0 (missing odds) with NaN
    odds1 = odds1.where(odds1 > 0, np.nan)
    odds2 = odds2.where(odds2 > 0, np.nan)

    df["rank_diff"] = rank1 - rank2
    df["rank_diff_abs"] = (rank1 - rank2).abs()
    df["log_odds_ratio"] = np.log(odds1 / odds2.clip(lower=0.01))
    df["odds_ratio"] = odds1 / odds2.clip(lower=0.01)

    # ── New odds features (11) ──
    # Raw odds
    df["odd_1"] = odds1
    df["odd_2"] = odds2

    # Log odds (compress extreme values, max ~183)
    df["log_odd_1"] = np.log(odds1)
    df["log_odd_2"] = np.log(odds2)

    # Implied probabilities (bookmaker's estimate)
    df["implied_prob_1"] = 1.0 / odds1
    df["implied_prob_2"] = 1.0 / odds2

    # Probability difference (who is favourite and by how much)
    df["prob_diff"] = df["implied_prob_1"] - df["implied_prob_2"]

    # Order-agnostic features
    df["min_odds"] = np.minimum(odds1, odds2)
    df["max_odds"] = np.maximum(odds1, odds2)

    # How strong is the favourite (higher = stronger favourite)
    df["favorite_strength"] = 1.0 / df["min_odds"]

    # How competitive is the match (1 = perfectly even, ~0 = mismatch)
    df["match_competitiveness"] = df["implied_prob_1"].combine(
        df["implied_prob_2"],
        lambda a, b: min(a, b) / max(a, b) if pd.notna(a) and pd.notna(b) and max(a, b) > 0 else np.nan,
    )

    # Rank points diff (from existing dataset if available)
    pts1 = pd.to_numeric(df.get("Pts_1", np.nan), errors="coerce")
    pts2 = pd.to_numeric(df.get("Pts_2", np.nan), errors="coerce")
    df["rank_points_diff"] = pts1 - pts2

    # Seed diff (from TML)
    if "p1_seed" in df.columns and "p2_seed" in df.columns:
        seed1 = pd.to_numeric(df["p1_seed"], errors="coerce").fillna(0)
        seed2 = pd.to_numeric(df["p2_seed"], errors="coerce").fillna(0)
        df["seed_diff"] = seed1 - seed2
    else:
        df["seed_diff"] = 0.0

    # Best of
    if "best_of" in df.columns:
        df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3)
    else:
        df["best_of"] = 3

    # Draw size
    if "draw_size" in df.columns:
        df["draw_size"] = pd.to_numeric(df["draw_size"], errors="coerce").fillna(32)
    else:
        df["draw_size"] = 32

    return df


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_divide(num, den):
    """Safe division returning nan if either is nan or den is 0."""
    try:
        n = float(num) if pd.notna(num) else np.nan
        d = float(den) if pd.notna(den) else np.nan
        if pd.isna(n) or pd.isna(d) or d == 0:
            return np.nan
        return n / d
    except (TypeError, ValueError):
        return np.nan


# ── Tournament features (venue effect on total games) ────────────────────────

def compute_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute tournament-level avg total games, std, and recent trend (no look-ahead)."""
    df = df.copy()

    tourney_col = "Tournament" if "Tournament" in df.columns else "tourney_name"
    if tourney_col not in df.columns or "total_games" not in df.columns:
        for name in _tournament_features_names():
            df[name] = np.nan
        return df

    # Build cumulative stats per tournament (chronological, no look-ahead)
    tg = df["total_games"]
    tourney: Dict[str, Dict] = {}
    avg_games_list = []
    std_games_list = []
    trend_list = []

    for i, row in df.iterrows():
        t = str(row.get(tourney_col, ""))
        rec = tourney.get(t, {"games": [], "sum": 0, "sum_sq": 0, "n": 0,
                               "last_5": []})

        n = rec["n"]
        if n >= 2:
            mean_g = rec["sum"] / n
            avg_games_list.append(mean_g)
            var = (rec["sum_sq"] / n) - mean_g ** 2
            std_games_list.append(np.sqrt(max(var, 0)))
        else:
            avg_games_list.append(np.nan)
            std_games_list.append(np.nan)

        # Trend: avg of last 5 vs overall avg
        if len(rec["last_5"]) >= 3:
            recent_avg = np.mean(rec["last_5"])
            overall_avg = rec["sum"] / n if n > 0 else np.nan
            trend_list.append(recent_avg - overall_avg if pd.notna(overall_avg) else np.nan)
        else:
            trend_list.append(np.nan)

        # Update after this match
        val = tg.iloc[i] if i < len(tg) else np.nan
        if pd.notna(val):
            rec["sum"] += val
            rec["sum_sq"] += val ** 2
            rec["n"] += 1
            rec["last_5"].append(val)
            if len(rec["last_5"]) > 5:
                rec["last_5"] = rec["last_5"][-5:]
        tourney[t] = rec

    df["tourney_avg_games"] = avg_games_list
    df["tourney_std_games"] = std_games_list
    df["tourney_game_trend"] = trend_list

    return df


# ── Fatigue features (recent match load) ─────────────────────────────────────

def compute_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player fatigue: matches played in last 7/14/30 days."""
    df = df.copy()

    all_players = set(df["Player_1"].dropna()) | set(df["Player_2"].dropna())
    # Track match dates per player
    player_dates: Dict[str, list] = {p: [] for p in all_players}

    results: Dict[str, list] = {name: [] for name in _fatigue_features_names()}
    windows = [7, 14, 30]

    for _, row in df.iterrows():
        p1, p2 = row["Player_1"], row["Player_2"]
        date = row["Date"]

        for player, prefix in [(p1, "p1"), (p2, "p2")]:
            dates = player_dates.get(player, [])
            for w in windows:
                cutoff = date - pd.Timedelta(days=w)
                n_matches = sum(1 for d in dates if d > cutoff)
                results[f"{prefix}_matches_{w}d"].append(n_matches)

        # Compute diffs
        for w in windows:
            p1_n = results[f"p1_matches_{w}d"][-1]
            p2_n = results[f"p2_matches_{w}d"][-1]
            results[f"fatigue_diff_{w}d"].append(p1_n - p2_n)

        # Update dates after computing features
        player_dates[p1].append(date)
        player_dates[p2].append(date)

    for col, vals in results.items():
        df[col] = vals

    return df


# ── Full pipeline ────────────────────────────────────────────────────────────

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete advanced feature engineering pipeline.

    Steps:
    1. Context features (rank, odds, etc.)
    2. Rolling stats (3 windows, per-player)
    3. Diff + ratio transformations
    4. Surface-specific features
    5. H2H advanced features
    6. Bio features
    """
    print("\n" + "=" * 60)
    print("BUILDING ADVANCED FEATURES")
    print("=" * 60)

    # 1. Context features
    print("\n1) Computing context features...")
    df = compute_context_features(df)

    # 2. Rolling stats
    print("\n2) Computing rolling statistics...")
    rolling_df = compute_rolling_stats_advanced(df)

    print("   Merging rolling stats...")
    df = merge_rolling_fast(df, rolling_df)

    # 3. Diff + ratio
    print("\n3) Computing diff/ratio features...")
    df = compute_diff_ratio_features(df)

    # 4. Surface-specific
    print("\n4) Computing surface-specific features...")
    df = compute_surface_features(df)

    # 5. H2H advanced
    print("\n5) Computing H2H advanced features...")
    df = compute_h2h_advanced(df)

    # 6. Bio features
    print("\n6) Computing bio features...")
    df = compute_bio_features(df)

    # 7. Tournament features (venue effect on total games)
    print("\n7) Computing tournament features...")
    df = compute_tournament_features(df)

    # 8. Fatigue features (recent match load)
    print("\n8) Computing fatigue features...")
    df = compute_fatigue_features(df)

    print(f"\n   Total features: {len(NUM_COLS_ADVANCED)} numerical + {len(CAT_COLS_ADVANCED)} categorical")
    print(f"   DataFrame shape: {df.shape}")

    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (num_cols, cat_cols) that actually exist in the DataFrame.
    Filters NUM_COLS_ADVANCED to only those present.
    """
    num_cols = [c for c in NUM_COLS_ADVANCED if c in df.columns]
    cat_cols = [c for c in CAT_COLS_ADVANCED if c in df.columns]
    return num_cols, cat_cols
