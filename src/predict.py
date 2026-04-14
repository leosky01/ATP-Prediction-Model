#!/usr/bin/env python3
"""
ATP Tennis Match Predictor – Live Prediction System

Provides the PredizioneATPCompleta class that loads a trained model and
produces match-level predictions blending MLP probability with Elo ratings.

Usage (CLI):
    python -m src.predict predict \\
        --player1 "Sinner J." --player2 "Alcaraz C." \\
        --rank1 1 --rank2 3 --odds1 1.90 --odds2 1.95 \\
        --surface Clay --tournament "Roland Garros" --date 2025-06-08

    python -m src.predict optimize-alpha   # find best MLP/Elo blend weight
"""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from .config import (CAT_COLS, DATASET_PATH, DATE_END, DATE_START,
                     ELO_STATE_PATH, ENCODER_PATH, MODEL_PATH, NUM_COLS,
                     SCALER_PATH)
from .elo_tracker import AdvancedEloSystem, EloTracker
from .model import EnhancedMLP


# ── Historical statistics calculator ─────────────────────────────────────────

class ATPHistoryCalculator:
    """Compute rolling statistics, surface stats, and H2H from historical data."""

    def __init__(self, csv_path: Path = DATASET_PATH):
        self.df = pd.read_csv(csv_path, parse_dates=["Date"])
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        start = pd.to_datetime(DATE_START)
        end = pd.to_datetime(DATE_END)
        self.df = self.df[(self.df["Date"] >= start) & (self.df["Date"] <= end)]
        self.df["p1_win"] = (self.df["Winner"] == self.df["Player_1"]).astype(int)
        self.df["p2_win"] = (self.df["Winner"] == self.df["Player_2"]).astype(int)

    # ── name helpers ──────────────────────────────────────────────────────

    def resolve_player_name(self, name: str) -> str:
        candidates = set(self.df["Player_1"]).union(self.df["Player_2"])
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        return matches[0] if matches else name

    # ── per-player lookups ────────────────────────────────────────────────

    def _player_matches(self, player: str, before_date: pd.Timestamp,
                        days: int = 365) -> pd.DataFrame:
        mask_p1 = (self.df["Player_1"] == player)
        mask_p2 = (self.df["Player_2"] == player)
        sub = self.df[mask_p1 | mask_p2].copy()
        cutoff = before_date - pd.Timedelta(days=days)
        sub = sub[(sub["Date"] < before_date) & (sub["Date"] >= cutoff)]
        return sub.sort_values("Date")

    def rolling_stats(self, player: str, before_date: pd.Timestamp,
                      window: int = 10, days: int = 365 * 5) -> dict:
        """Rolling win-rate, streak, momentum (5-year lookback)."""
        sub = self._player_matches(player, before_date, days)
        if sub.empty:
            return {"win_rate": 0.5, "streak": 0, "matches_played": 0, "momentum": 0.5}

        wins_col = []
        for _, r in sub.iterrows():
            if r["Player_1"] == player:
                wins_col.append(r["p1_win"])
            else:
                wins_col.append(r["p2_win"])
        wins = pd.Series(wins_col)

        win_rate = wins.rolling(window, min_periods=1).mean().iloc[-1]
        # current streak
        streak = 0
        for v in reversed(wins.values):
            if v == wins.iloc[-1]:
                streak += 1
            else:
                break
        streak = int(streak * wins.iloc[-1])  # negative if losing
        momentum = 0.7 * win_rate + 0.3 * (min(max(streak, 0), 10) / 10)

        return {
            "win_rate": float(win_rate),
            "streak": streak,
            "matches_played": len(sub),
            "momentum": float(momentum),
        }

    def surface_performance(self, player: str, surface: str,
                            before_date: pd.Timestamp,
                            days: int = 365 * 5) -> dict:
        sub = self._player_matches(player, before_date, days)
        sub = sub[sub["Surface"].str.strip().str.title() == surface.title()]
        if sub.empty:
            return {"wins": 0, "losses": 0, "win_rate": 0.5}
        wins = 0
        for _, r in sub.iterrows():
            if r["Player_1"] == player:
                wins += r["p1_win"]
            else:
                wins += r["p2_win"]
        total = len(sub)
        return {"wins": int(wins), "losses": total - int(wins),
                "win_rate": wins / total}

    def h2h_record(self, p1: str, p2: str, before_date: pd.Timestamp) -> dict:
        sub = self.df[
            ((self.df["Player_1"] == p1) & (self.df["Player_2"] == p2)) |
            ((self.df["Player_1"] == p2) & (self.df["Player_2"] == p1))
        ]
        sub = sub[sub["Date"] < before_date]
        if sub.empty:
            return {"p1_wins": 0, "p2_wins": 0, "matches": 0, "p1_win_rate": 0.5}
        p1_wins = int(((sub["Winner"] == p1) & (sub["Player_1"] == p1)).sum() +
                      ((sub["Winner"] == p1) & (sub["Player_2"] == p1)).sum())
        p2_wins = int(((sub["Winner"] == p2) & (sub["Player_1"] == p2)).sum() +
                      ((sub["Winner"] == p2) & (sub["Player_2"] == p2)).sum())
        total = p1_wins + p2_wins
        return {"p1_wins": p1_wins, "p2_wins": p2_wins, "matches": total,
                "p1_win_rate": p1_wins / total if total else 0.5}


# ── Upset signal (enhanced, bidirectional) ───────────────────────────────────

def _upset_signal_enhanced(row: dict) -> float:
    """Enhanced upset signal with bidirectional H2H, momentum diff, etc."""
    score = 0.0
    form_diff = row.get("win_rate_diff", 0)
    if abs(form_diff) > 0.2:
        score += 0.3
    elif abs(form_diff) > 0.1:
        score += 0.15

    streak_diff = abs(row.get("streak_diff", 0))
    if streak_diff > 3:
        score += 0.15

    h2h_wr = row.get("h2h_win_rate", 0.5)
    if (h2h_wr > 0.6 or h2h_wr < 0.4) and row.get("h2h_matches", 0) >= 3:
        score += 0.25

    rank_1 = max(row.get("Rank_1", 100), 1)
    rank_2 = max(row.get("Rank_2", 100), 1)
    odd_1 = max(row.get("Odd_1", 1.5), 1.01)
    odd_2 = max(row.get("Odd_2", 1.5), 1.01)
    rank_ratio = rank_2 / rank_1
    odds_ratio = odd_2 / odd_1
    if odds_ratio < rank_ratio * 0.7:
        score += 0.15

    # High win-rate with inverted ranking
    wr1 = row.get("p1_win_rate", 0.5)
    wr2 = row.get("p2_win_rate", 0.5)
    if wr1 > 0.7 and rank_1 > rank_2:
        score += 0.1
    if wr2 > 0.7 and rank_2 > rank_1:
        score += 0.1

    return min(score, 1.0)


# ── Confidence and upset probability ─────────────────────────────────────────

def _calculate_confidence(raw_prob: float, cal_prob: float,
                          odds_ratio: float, rank_diff: float,
                          h2h_matches: int, upset_score: float) -> float:
    decision_conf = abs(cal_prob - 0.5) * 2
    cal_agreement = 1 - abs(raw_prob - cal_prob) * 2
    signal_strength = (abs(np.log(max(odds_ratio, 0.01))) + abs(rank_diff) / 100) / 2
    h2h_rel = min(h2h_matches / 10, 1)
    confidence = (decision_conf * 0.35 + cal_agreement * 0.25 +
                  signal_strength * 0.20 + h2h_rel * 0.20) - upset_score * 0.3
    return float(np.clip(confidence, 0, 1))


def _calculate_upset_probability(segnali: float, rank_diff: float,
                                 odds_ratio: float) -> float:
    rank_factor = 1 / (1 + abs(rank_diff) / 50)
    odds_factor = 1 / (1 + abs(np.log(max(odds_ratio, 0.01))))
    return float(np.clip(
        0.15 * 0.4 + segnali * 0.3 +
        (1 - rank_factor) * 0.15 + (1 - odds_factor) * 0.15, 0, 1))


# ── Main predictor ───────────────────────────────────────────────────────────

class ATPPredictor:
    """
    Load a trained model + Elo state and produce match predictions.

    Supports three modes: ``mlp``, ``elo``, ``blend``.
    """

    def __init__(self,
                 model_path: Path = MODEL_PATH,
                 scaler_path: Path = SCALER_PATH,
                 encoder_path: Path = ENCODER_PATH,
                 elo_path: Path = ELO_STATE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = ATPHistoryCalculator()

        # Load scaler & encoder
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

        # Load model
        n_num = len(NUM_COLS)
        n_cat = self.encoder.categories_[0].size + self.encoder.categories_[1].size
        input_dim = n_num + n_cat
        self.model = EnhancedMLP(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # Load Elo
        if elo_path.exists():
            self.elo_tracker = EloTracker.load(elo_path)
        else:
            self.elo_tracker = EloTracker()

    # ── internal helpers ──────────────────────────────────────────────────

    def _normalize_surface(self, surface: str) -> str:
        s = surface.strip().title() if isinstance(surface, str) else "Hard"
        return s if s in {"Hard", "Clay", "Grass", "Carpet"} else "Hard"

    def _mlp_prob(self, features: dict) -> Tuple[float, np.ndarray]:
        """Run the MLP on a feature dict → (calibrated_prob, raw_features)."""
        num_vals = [features.get(c, 0) for c in NUM_COLS]
        X_num = self.scaler.transform(np.array([num_vals]))

        surface = self._normalize_surface(features.get("Surface", "Hard"))
        tournament = features.get("Tournament", "Unknown")
        X_cat = self.encoder.transform(pd.DataFrame([[surface, tournament]],
                                                    columns=CAT_COLS))

        X = np.hstack([X_num, X_cat])
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logit = self.model(X_t).item()
            raw_prob = float(torch.sigmoid(torch.tensor(logit)).item())

        # Bias correction (legacy factor 0.92)
        corrected = raw_prob * 0.92 + 0.5 * 0.08
        return float(np.clip(corrected, 0.01, 0.99)), X_num.flatten()

    # ── public API ────────────────────────────────────────────────────────

    def predict(self,
                player1: str, player2: str,
                rank1: int, rank2: int,
                odds1: float, odds2: float,
                surface: str,
                tournament: str = "Unknown",
                date_str: Optional[str] = None,
                mode: str = "blend",
                alpha: float = 0.65) -> dict:
        """
        Predict a single match.

        Parameters
        ----------
        mode : ``mlp`` | ``elo`` | ``blend``
        alpha : weight for MLP in blend mode (1-alpha for Elo)

        Returns dict with keys: prediction, winner, prob_p1, prob_p2,
        confidence, upset_probability, mode, alpha, plus per-player stats.
        """
        date = pd.to_datetime(date_str) if date_str else pd.Timestamp.now()

        # Resolve names
        p1 = self.history.resolve_player_name(player1)
        p2 = self.history.resolve_player_name(player2)

        # Historical stats
        st1 = self.history.rolling_stats(p1, date)
        st2 = self.history.rolling_stats(p2, date)
        surf1 = self.history.surface_performance(p1, surface, date)
        surf2 = self.history.surface_performance(p2, surface, date)
        h2h = self.history.h2h_record(p1, p2, date)

        # Feature dict
        feat = {
            "Rank_1": rank1, "Rank_2": rank2,
            "Odd_1": odds1, "Odd_2": odds2,
            "p1_win_rate": st1["win_rate"], "p2_win_rate": st2["win_rate"],
            "p1_streak": st1["streak"], "p2_streak": st2["streak"],
            "h2h_win_rate": h2h["p1_win_rate"], "h2h_matches": h2h["matches"],
            "Surface": surface, "Tournament": tournament,
            "rank_diff": rank1 - rank2,
            "odds_ratio": odds1 / max(odds2, 0.01),
            "win_rate_diff": st1["win_rate"] - st2["win_rate"],
            "streak_diff": st1["streak"] - st2["streak"],
            "min_rank": min(rank1, rank2), "max_rank": max(rank1, rank2),
            "min_odds": min(odds1, odds2), "max_odds": max(odds1, odds2),
            "momentum_p1": st1["momentum"], "momentum_p2": st2["momentum"],
            "momentum_diff": st1["momentum"] - st2["momentum"],
        }

        # MLP probability
        mlp_prob, _ = self._mlp_prob(feat)

        # Elo probability
        elo_p1 = self.elo_tracker.get_player_elo(p1, surface)
        elo_p2 = self.elo_tracker.get_player_elo(p2, surface)
        elo_prob = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))

        # Upset signal & confidence
        upset_score = _upset_signal_enhanced(feat)
        confidence = _calculate_confidence(
            mlp_prob, mlp_prob, feat["odds_ratio"],
            feat["rank_diff"], feat["h2h_matches"], upset_score,
        )
        upset_prob = _calculate_upset_probability(
            upset_score, feat["rank_diff"], feat["odds_ratio"],
        )

        # Blend
        if mode == "mlp":
            final_prob = mlp_prob
        elif mode == "elo":
            final_prob = elo_prob
        else:  # blend
            final_prob = alpha * mlp_prob + (1 - alpha) * elo_prob

        winner = p1 if final_prob > 0.5 else p2

        return {
            "prediction": winner,
            "winner": winner,
            "prob_p1": round(final_prob, 4),
            "prob_p2": round(1 - final_prob, 4),
            "mlp_prob": round(mlp_prob, 4),
            "elo_prob": round(elo_prob, 4),
            "blend_prob": round(alpha * mlp_prob + (1 - alpha) * elo_prob, 4),
            "confidence": round(confidence, 4),
            "upset_probability": round(upset_prob, 4),
            "upset_signal_score": round(upset_score, 4),
            "mode": mode,
            "alpha": alpha,
            "player1_stats": st1,
            "player2_stats": st2,
            "surface_p1": surf1,
            "surface_p2": surf2,
            "h2h": h2h,
            "momentum_diff": round(feat["momentum_diff"], 4),
        }

    def optimize_alpha(self, test_df: pd.DataFrame,
                       alphas: np.ndarray = np.arange(0.3, 0.95, 0.05)) -> float:
        """Grid-search for the MLP/Elo blend weight that maximises accuracy."""
        best_alpha, best_acc = 0.65, 0
        for a in alphas:
            correct = 0
            total = 0
            for _, row in test_df.iterrows():
                try:
                    res = self.predict(
                        row["Player_1"], row["Player_2"],
                        int(row["Rank_1"]), int(row["Rank_2"]),
                        float(row["Odd_1"]), float(row["Odd_2"]),
                        row["Surface"], row.get("Tournament", "Unknown"),
                        str(row["Date"].date()), mode="blend", alpha=a,
                    )
                    pred = 1 if res["prob_p1"] > 0.5 else 0
                    actual = 1 if row["Winner"] == row["Player_1"] else 0
                    correct += (pred == actual)
                    total += 1
                except Exception:
                    continue
            acc = correct / total if total else 0
            if acc > best_acc:
                best_acc, best_alpha = acc, a
        print(f"Best alpha={best_alpha:.2f} → accuracy={best_acc:.2%}")
        return best_alpha


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ATP Match Predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Predict a single match")
    p.add_argument("--player1", required=True)
    p.add_argument("--player2", required=True)
    p.add_argument("--rank1", type=int, required=True)
    p.add_argument("--rank2", type=int, required=True)
    p.add_argument("--odds1", type=float, required=True)
    p.add_argument("--odds2", type=float, required=True)
    p.add_argument("--surface", default="Hard")
    p.add_argument("--tournament", default="Unknown")
    p.add_argument("--date", default=None)
    p.add_argument("--mode", default="blend", choices=["mlp", "elo", "blend"])
    p.add_argument("--alpha", type=float, default=0.65)

    p2 = sub.add_parser("optimize-alpha", help="Find best blend weight")
    p2.add_argument("--csv", default=str(DATASET_PATH))
    p2.add_argument("--n-samples", type=int, default=200)

    args = parser.parse_args()
    predictor = ATPPredictor()

    if args.cmd == "predict":
        result = predictor.predict(
            args.player1, args.player2,
            args.rank1, args.rank2,
            args.odds1, args.odds2,
            args.surface, args.tournament,
            args.date, args.mode, args.alpha,
        )
        print(f"\n  Prediction: {result['prediction']}")
        print(f"  P({args.player1}): {result['prob_p1']:.1%}")
        print(f"  P({args.player2}): {result['prob_p2']:.1%}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Upset prob: {result['upset_probability']:.1%}")
        print(f"  Mode: {result['mode']} (alpha={result['alpha']})")
        print(f"  MLP prob: {result['mlp_prob']:.1%}  Elo prob: {result['elo_prob']:.1%}")
        print(f"  Momentum diff: {result['momentum_diff']:+.3f}")
        print(f"  H2H: {result['h2h']}")

    elif args.cmd == "optimize-alpha":
        df = pd.read_csv(args.csv, parse_dates=["Date"])
        df = df.sample(min(args.n_samples, len(df)), random_state=42)
        predictor.optimize_alpha(df)


if __name__ == "__main__":
    main()
