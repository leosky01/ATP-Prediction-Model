#!/usr/bin/env python3
"""
ATP Tennis Match Predictor – Live Prediction System

Provides ATPPredictor that loads a trained model and produces match-level
predictions blending MLP probability with Elo ratings, aligned with the
notebook's refined prediction logic (isotonic calibration, dynamic
thresholds, bidirectional upset signals, Elo blend).

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
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from .config import (CAT_COLS, DATASET_PATH, ELO_STATE_PATH, ENCODER_PATH,
                     MODEL_PATH, NUM_COLS, OPTIMIZATION_CONFIG_PATH,
                     SCALER_PATH)
from .elo_tracker import AdvancedEloSystem, EloTracker
from .model import EnhancedMLP


# ── Historical statistics calculator ─────────────────────────────────────────

class ATPHistoryCalculator:
    """Compute rolling statistics, surface stats, and H2H from historical data.

    Aligned with notebook: 5-year lookback, last ``window`` matches only,
    positive-only streak, momentum formula.
    """

    def __init__(self, csv_path: Path = DATASET_PATH):
        self.df = pd.read_csv(csv_path, parse_dates=["Date"])
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        self.df.sort_values("Date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Fast player-name lookup structures
        self._players = pd.Index(
            pd.unique(pd.concat([self.df["Player_1"], self.df["Player_2"]],
                                ignore_index=True))
        )
        self._players_lower: Dict[str, str] = {p.lower(): p for p in self._players}

    # ── name helpers ──────────────────────────────────────────────────────

    def resolve_player_name(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            return name
        name = name.strip()
        # Exact match
        if name in self._players:
            return name
        # Case-insensitive match
        key = name.lower()
        if key in self._players_lower:
            return self._players_lower[key]
        # Fuzzy match with cutoff 0.85 (stricter than legacy 0.6)
        matches = difflib.get_close_matches(
            name, self._players.tolist(), n=1, cutoff=0.85
        )
        return matches[0] if matches else name

    # ── per-player lookups ────────────────────────────────────────────────

    def get_player_matches_before_date(
        self, player: str, before_date: pd.Timestamp,
        days_lookback: int = 1825,
    ) -> pd.DataFrame:
        """Return matches of *player* before *before_date* within lookback."""
        player = self.resolve_player_name(player)
        cutoff = before_date - pd.Timedelta(days=days_lookback)
        mask = (
            ((self.df["Player_1"] == player) | (self.df["Player_2"] == player))
            & (self.df["Date"] < before_date)
            & (self.df["Date"] >= cutoff)
        )
        return self.df.loc[mask].sort_values("Date", ascending=False)

    def calculate_rolling_stats(
        self, player: str, before_date: pd.Timestamp, window: int = 10,
    ) -> dict:
        """Rolling stats using only last *window* matches (5-year lookback).

        Returns dict with keys: win_rate, streak, matches_played, momentum.
        """
        recent = self.get_player_matches_before_date(player, before_date,
                                                     days_lookback=1825)
        if recent.empty:
            return {"win_rate": 0.5, "streak": 0,
                    "matches_played": 0, "momentum": 0.5}

        last_matches = recent.head(window)
        resolved = self.resolve_player_name(player)
        wins = sum(1 for _, m in last_matches.iterrows() if m["Winner"] == resolved)
        win_rate = wins / len(last_matches)

        streak = self._calculate_current_streak(resolved, recent)

        streak_scaled = min(max(float(streak), 0.0), 10.0) / 10.0
        momentum = float(np.clip(
            0.7 * float(win_rate) + 0.3 * streak_scaled, 0.0, 1.0
        ))

        return {
            "win_rate": float(win_rate),
            "streak": int(streak),
            "matches_played": int(len(last_matches)),
            "momentum": momentum,
        }

    def _calculate_current_streak(
        self, player: str, recent_matches: pd.DataFrame,
    ) -> int:
        """Positive-only streak (``max(streak, 0)``)."""
        if recent_matches.empty:
            return 0
        player = self.resolve_player_name(player)
        streak = 0
        last_result = None
        for _, match in recent_matches.iterrows():
            won = match["Winner"] == player
            if last_result is None:
                last_result = won
                streak = 1 if won else -1
            elif won == last_result:
                streak += 1 if won else -1
            else:
                break
        return max(streak, 0)

    def calculate_surface_performance(
        self, player: str, surface: str, before_date: pd.Timestamp,
    ) -> dict:
        """Surface stats (5-year lookback). Returns {surface_wins, surface_matches}."""
        recent = self.get_player_matches_before_date(player, before_date,
                                                     days_lookback=1825)
        surface = surface.strip().title() if isinstance(surface, str) else ""
        recent = recent[recent["Surface"].str.strip().str.title() == surface]

        if recent.empty:
            return {"surface_wins": 5, "surface_matches": 10}

        resolved = self.resolve_player_name(player)
        wins = int((recent["Winner"] == resolved).sum())
        return {"surface_wins": wins, "surface_matches": len(recent)}

    def calculate_h2h_record(
        self, p1: str, p2: str, before_date: pd.Timestamp,
    ) -> Tuple[float, int]:
        """Return ``(p1_winrate, total_matches)`` tuple."""
        p1 = self.resolve_player_name(p1)
        p2 = self.resolve_player_name(p2)
        mask = (
            ((self.df["Player_1"] == p1) & (self.df["Player_2"] == p2))
            | ((self.df["Player_1"] == p2) & (self.df["Player_2"] == p1))
        ) & (self.df["Date"] < before_date)
        h2h = self.df.loc[mask]
        if h2h.empty:
            return 0.5, 0
        p1_wins = int((h2h["Winner"] == p1).sum())
        total = len(h2h)
        return (p1_wins / total, total)

    # ── Backward-compatible aliases (used by make_video.py & web app) ───

    def _player_matches(self, player: str, before_date: pd.Timestamp,
                        days: int = 365) -> pd.DataFrame:
        """Alias used by make_video helpers."""
        return self.get_player_matches_before_date(player, before_date,
                                                   days_lookback=days)

    def rolling_stats(self, player: str, before_date: pd.Timestamp,
                      window: int = 10, days: int = 365 * 5) -> dict:
        """Alias for calculate_rolling_stats."""
        return self.calculate_rolling_stats(player, before_date, window)

    def surface_performance(self, player: str, surface: str,
                            before_date: pd.Timestamp,
                            days: int = 365 * 5) -> dict:
        """Alias that returns dict with 'win_rate' key for make_video/web."""
        sp = self.calculate_surface_performance(player, surface, before_date)
        wins = sp["surface_wins"]
        total = sp["surface_matches"]
        return {
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.5,
            "surface_wins": wins,
            "surface_matches": total,
        }

    def h2h_record(self, p1: str, p2: str,
                   before_date: pd.Timestamp) -> dict:
        """Alias that returns dict for make_video/web compatibility."""
        wr, m = self.calculate_h2h_record(p1, p2, before_date)
        return {
            "p1_wins": int(round(wr * m)),
            "p2_wins": int(round((1 - wr) * m)),
            "matches": m,
            "p1_win_rate": wr,
        }


# ── Main predictor ───────────────────────────────────────────────────────────

class ATPPredictor:
    """
    Load a trained model + Elo state and produce match predictions.

    Supports three modes: ``mlp``, ``elo``, ``blend``.
    Uses isotonic calibration (favoriti / upset paths), dynamic thresholds,
    bidirectional upset signals, and ``0.35 * overall + 0.65 * surface`` Elo blend.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        encoder_path: Path = ENCODER_PATH,
        elo_path: Path = ELO_STATE_PATH,
        config_path: Path = OPTIMIZATION_CONFIG_PATH,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = ATPHistoryCalculator()

        # Load scaler & encoder
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

        # Load model (handle bn1 → input_bn key remapping for invariant model)
        n_num = len(NUM_COLS)
        n_cat = sum(cat.size for cat in self.encoder.categories_)
        input_dim = n_num + n_cat
        self.model = EnhancedMLP(input_dim).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device,
                                weights_only=True)
        if "bn1.weight" in state_dict and "input_bn.weight" not in state_dict:
            for old_key in list(state_dict.keys()):
                if old_key.startswith("bn1."):
                    new_key = "input_bn." + old_key[4:]
                    state_dict[new_key] = state_dict.pop(old_key)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load isotonic calibrators from optimization config
        self.isotonic_favoriti = None
        self.isotonic_upset = None
        self.upset_bias_factor = 1.0
        try:
            cfg = joblib.load(config_path)
            if isinstance(cfg, dict):
                self.isotonic_favoriti = cfg.get("isotonic_favoriti")
                self.isotonic_upset = cfg.get("isotonic_upset")
                self.upset_bias_factor = float(cfg.get("upset_bias_factor", 1.0))
        except Exception:
            pass

        # Identity calibrator fallback
        self._identity_calibrator = IsotonicRegression(out_of_bounds="clip")
        self._identity_calibrator.fit([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])

        # Load Elo
        if elo_path.exists():
            self.elo_tracker = EloTracker.load(elo_path)
        else:
            self.elo_tracker = EloTracker()

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_surface(surface: str) -> str:
        s = surface.strip().title() if isinstance(surface, str) else "Hard"
        return s if s in {"Hard", "Clay", "Grass", "Carpet"} else "Hard"

    def _calcola_segnali_upset(self, features: dict) -> float:
        """Bidirectional upset signals (notebook's _legacy version)."""
        segnali = 0.0

        form_diff = features["win_rate_diff"]
        if form_diff < -0.2:
            segnali += 0.35
        elif form_diff < -0.1:
            segnali += 0.2

        streak_diff = features["streak_diff"]
        if streak_diff < -3:
            segnali += 0.25
        elif streak_diff < -1:
            segnali += 0.15

        # H2H direction-aware
        if features["rank_diff"] > 0:
            if features["h2h_win_rate"] > 0.6 and features["h2h_matches"] >= 3:
                segnali += 0.3
            elif features["h2h_win_rate"] > 0.5 and features["h2h_matches"] >= 2:
                segnali += 0.15
        else:
            if features["h2h_win_rate"] < 0.4 and features["h2h_matches"] >= 3:
                segnali += 0.3
            elif features["h2h_win_rate"] < 0.5 and features["h2h_matches"] >= 2:
                segnali += 0.15

        rank_ratio = min(features["max_rank"], 200) / max(features["min_rank"], 1)
        odds_ratio = features["max_odds"] / max(features["min_odds"], 1.01)
        if odds_ratio < rank_ratio * 0.7:
            segnali += 0.25
        elif odds_ratio < rank_ratio * 0.85:
            segnali += 0.15

        if features["max_win_rate"] > 0.7 and features["max_rank"] > features["min_rank"]:
            segnali += 0.15

        return min(segnali, 1.0)

    def _apply_dynamic_threshold(
        self, prob: float, rank1: int, rank2: int,
        odds1: float, odds2: float, surface: str,
        segnali_upset: float,
    ) -> float:
        """Dynamic decision threshold based on rank brackets + surface + upset."""
        rank_diff = rank1 - rank2

        if rank_diff <= -50:
            base = 0.22
        elif rank_diff <= -20:
            base = 0.32
        elif rank_diff <= 0:
            base = 0.47
        elif rank_diff <= 20:
            base = 0.45
        elif rank_diff <= 50:
            base = 0.52
        else:
            base = 0.55

        surface_adj = {"Hard": 0.002, "Clay": 0.001, "Grass": 0.002}.get(surface, 0)
        base += surface_adj

        if segnali_upset > 0.45:
            base -= 0.20
        elif segnali_upset > 0.25:
            base -= 0.19
        elif segnali_upset > 0.1:
            base -= 0.12

        return max(base, 0.22)

    def _calculate_confidence_improved(
        self, calibrated_prob: float, raw_prob: float,
        features: dict, segnali_upset: float,
    ) -> float:
        """Confidence score from decision/confidence + calibration agreement
        + signal strength + H2H reliability – upset penalty."""
        decision_confidence = abs(calibrated_prob - 0.5) * 2
        calibration_agreement = 1 - abs(raw_prob - calibrated_prob) * 2

        odds_strength = abs(np.log(max(features["odds_ratio"], 0.01))) / 2
        rank_strength = abs(features["rank_diff"]) / 100
        signal_strength = min((odds_strength + rank_strength) / 2, 1.0)

        h2h_matches = features.get("h2h_matches", 0)
        h2h_reliability = (
            min(h2h_matches / 10, 1.0) if h2h_matches > 0 else 0.5
        )

        upset_penalty = 0.0
        if segnali_upset > 0.5 and abs(calibrated_prob - 0.5) < 0.15:
            upset_penalty = 0.2

        confidence = (
            decision_confidence * 0.35
            + calibration_agreement * 0.25
            + signal_strength * 0.20
            + h2h_reliability * 0.20
        ) - upset_penalty
        return float(np.clip(confidence, 0.0, 1.0))

    def _calculate_upset_probability(
        self, features: dict, calibrated_prob: float,
        rank1: int, rank2: int, odds1: float, odds2: float,
    ) -> float:
        """Upset probability from base_upset + segnali + rank/odds factors."""
        if odds1 < odds2:
            base_upset = 1 - calibrated_prob
        else:
            base_upset = calibrated_prob

        segnali = self._calcola_segnali_upset(features)

        rank_factor = min(abs(rank1 - rank2) / 100, 1.0)
        odds_factor = min(abs(np.log(odds1 / max(odds2, 0.01))) / 2, 1.0)

        upset_prob = (
            base_upset * 0.4
            + segnali * 0.3
            + (1 - rank_factor) * 0.15
            + (1 - odds_factor) * 0.15
        )
        return float(np.clip(upset_prob, 0.0, 1.0))

    def _mlp_prob(self, features: dict) -> float:
        """Run MLP → sigmoid → isotonic calibration. Returns calibrated prob."""
        num_vals = [features.get(c, 0) for c in NUM_COLS]
        X_num = self.scaler.transform(np.array([num_vals]))

        surface = self._normalize_surface(features.get("Surface", "Hard"))
        tournament = features.get("Tournament", "Unknown")
        X_cat = self.encoder.transform(
            pd.DataFrame([[surface, tournament]], columns=CAT_COLS)
        )

        X = np.hstack([X_num, X_cat])
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            raw_logit = self.model(X_t).item()
            raw_prob = float(torch.sigmoid(torch.tensor(raw_logit)).item())

        # Isotonic calibration: favoriti vs upset path
        odds1 = features.get("Odd_1", 1.9)
        odds2 = features.get("Odd_2", 1.9)
        is_upset_potential = odds1 > odds2

        if is_upset_potential:
            prob_for_cal = float(np.clip(
                raw_prob * self.upset_bias_factor, 0.0, 1.0
            ))
            cal = self.isotonic_upset or self._identity_calibrator
        else:
            prob_for_cal = raw_prob
            cal = self.isotonic_favoriti or self._identity_calibrator

        try:
            calibrated_prob = float(cal.predict([prob_for_cal])[0])
        except Exception:
            calibrated_prob = float(raw_prob)

        return float(np.clip(calibrated_prob, 0.01, 0.99))

    # ── public API ────────────────────────────────────────────────────────

    def predict(
        self,
        player1: str, player2: str,
        rank1: int, rank2: int,
        odds1: float, odds2: float,
        surface: str,
        tournament: str = "Unknown",
        date_str: Optional[str] = None,
        mode: str = "blend",
        alpha: float = 0.65,
    ) -> dict:
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

        # Normalize mode
        mode_n = str(mode).strip().lower()
        if mode_n == "blended":
            mode_n = "blend"
        if mode_n not in {"mlp", "elo", "blend"}:
            mode_n = "blend"

        # Resolve names
        p1 = self.history.resolve_player_name(player1)
        p2 = self.history.resolve_player_name(player2)

        # Historical stats (with momentum)
        st1 = self.history.calculate_rolling_stats(p1, date)
        st2 = self.history.calculate_rolling_stats(p2, date)
        surf1 = self.history.calculate_surface_performance(p1, surface, date)
        surf2 = self.history.calculate_surface_performance(p2, surface, date)
        h2h_wr, h2h_m = self.history.calculate_h2h_record(p1, p2, date)

        m1 = st1.get("momentum", st1.get("win_rate", 0.5))
        m2 = st2.get("momentum", st2.get("win_rate", 0.5))

        # Build full 14-feature dict (including max/min_win_rate, max/min_streak)
        feat = {
            "rank_diff": rank1 - rank2,
            "odds_ratio": odds1 / max(odds2, 0.01),
            "win_rate_diff": st1["win_rate"] - st2["win_rate"],
            "streak_diff": st1["streak"] - st2["streak"],
            "min_rank": min(rank1, rank2),
            "max_rank": max(rank1, rank2),
            "min_odds": min(odds1, odds2),
            "max_odds": max(odds1, odds2),
            "max_win_rate": max(st1["win_rate"], st2["win_rate"]),
            "min_win_rate": min(st1["win_rate"], st2["win_rate"]),
            "max_streak": max(st1["streak"], st2["streak"]),
            "min_streak": min(st1["streak"], st2["streak"]),
            "h2h_win_rate": h2h_wr,
            "h2h_matches": h2h_m,
            "Surface": surface,
            "Tournament": tournament,
            "Odd_1": odds1,
            "Odd_2": odds2,
        }

        # Upset signals
        segnali_upset = self._calcola_segnali_upset(feat)

        # MLP probability (with isotonic calibration)
        mlp_prob = self._mlp_prob(feat)

        # Elo probability: 0.35 * overall + 0.65 * surface
        elo_prob = 0.5
        try:
            elo_p1_overall = float(
                self.elo_tracker.get_player_elo(p1)
            )
            elo_p2_overall = float(
                self.elo_tracker.get_player_elo(p2)
            )
            surf_norm = self._normalize_surface(surface)
            elo_p1_surf = float(
                self.elo_tracker.get_player_elo(p1, surface=surf_norm)
            )
            elo_p2_surf = float(
                self.elo_tracker.get_player_elo(p2, surface=surf_norm)
            )
            elo_overall = AdvancedEloSystem.expected_outcome(
                elo_p1_overall, elo_p2_overall
            )
            elo_surface = AdvancedEloSystem.expected_outcome(
                elo_p1_surf, elo_p2_surf
            )
            elo_prob = float(np.clip(
                0.35 * elo_overall + 0.65 * elo_surface, 0.01, 0.99
            ))
        except Exception:
            pass

        # Blend probability
        blend_prob = float(np.clip(
            alpha * mlp_prob + (1 - alpha) * elo_prob, 0.01, 0.99
        ))

        # Mode selection
        if mode_n == "mlp":
            final_prob = mlp_prob
        elif mode_n == "elo":
            final_prob = elo_prob
        else:  # blend
            final_prob = blend_prob

        # Dynamic threshold for final decision
        threshold = self._apply_dynamic_threshold(
            final_prob, rank1, rank2, odds1, odds2, surface, segnali_upset,
        )
        prediction = 1 if final_prob > threshold else 0
        winner = p1 if prediction == 1 else p2

        # Confidence & upset (computed on MLP calibrated prob for consistency)
        raw_prob_for_conf = mlp_prob  # MLP raw = calibrated for confidence calc
        confidence = self._calculate_confidence_improved(
            mlp_prob, raw_prob_for_conf, feat, segnali_upset,
        )
        upset_probability = self._calculate_upset_probability(
            feat, mlp_prob, rank1, rank2, odds1, odds2,
        )

        # Build surface dicts with win_rate for web app compatibility
        surf1_out = {
            "wins": surf1["surface_wins"],
            "losses": surf1["surface_matches"] - surf1["surface_wins"],
            "win_rate": (
                surf1["surface_wins"] / surf1["surface_matches"]
                if surf1["surface_matches"] > 0 else 0.5
            ),
            "surface_wins": surf1["surface_wins"],
            "surface_matches": surf1["surface_matches"],
        }
        surf2_out = {
            "wins": surf2["surface_wins"],
            "losses": surf2["surface_matches"] - surf2["surface_wins"],
            "win_rate": (
                surf2["surface_wins"] / surf2["surface_matches"]
                if surf2["surface_matches"] > 0 else 0.5
            ),
            "surface_wins": surf2["surface_wins"],
            "surface_matches": surf2["surface_matches"],
        }

        return {
            "prediction": winner,
            "winner": winner,
            "prob_p1": round(final_prob, 4),
            "prob_p2": round(1 - final_prob, 4),
            "mlp_prob": round(mlp_prob, 4),
            "elo_prob": round(elo_prob, 4),
            "blend_prob": round(blend_prob, 4),
            "confidence": round(confidence, 4),
            "upset_probability": round(upset_probability, 4),
            "upset_signal_score": round(segnali_upset, 4),
            "mode": mode_n,
            "alpha": alpha,
            "player1_stats": st1,
            "player2_stats": st2,
            "surface_p1": surf1_out,
            "surface_p2": surf2_out,
            "h2h": {
                "p1_wins": int(round(h2h_wr * h2h_m)),
                "p2_wins": int(round((1 - h2h_wr) * h2h_m)),
                "matches": h2h_m,
                "p1_win_rate": round(h2h_wr, 4),
            },
            "momentum_diff": round(m1 - m2, 4),
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
        print(f"Best alpha={best_alpha:.2f} -> accuracy={best_acc:.2%}")
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
