#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


class AdvancedEloSystem:
    def __init__(
        self,
        initial_elo: float = 1500,
        k_multiplier: float = 250,
        k_offset: float = 5,
        k_shape: float = 0.4,
        decay_lambda: float = 0.01,
        decay_threshold: int = 90,
        comeback_k_boost: float = 1.5,
    ):
        from collections import defaultdict

        self.initial_elo = initial_elo
        self.k_multiplier = k_multiplier
        self.k_offset = k_offset
        self.k_shape = k_shape
        self.decay_lambda = decay_lambda
        self.decay_threshold = decay_threshold
        self.comeback_k_boost = comeback_k_boost

        self.elo_overall: Dict[str, float] = defaultdict(lambda: initial_elo)
        self.elo_surface: Dict[str, Dict[str, float]] = {
            "Hard": defaultdict(lambda: initial_elo),
            "Clay": defaultdict(lambda: initial_elo),
            "Grass": defaultdict(lambda: initial_elo),
            "Carpet": defaultdict(lambda: initial_elo),
        }

        self.matches_played: Dict[str, int] = defaultdict(int)
        self.last_match_date: Dict[str, pd.Timestamp] = {}
        self.is_comeback: Dict[str, bool] = defaultdict(bool)

    def _calculate_k(self, player: str, is_comeback: bool = False) -> float:
        n_matches = self.matches_played[player]
        k = self.k_multiplier / ((n_matches + self.k_offset) ** self.k_shape)
        if is_comeback:
            k *= self.comeback_k_boost
        return float(k)

    def _apply_decay(self, player: str, current_date: pd.Timestamp) -> None:
        if player not in self.last_match_date:
            return

        days_idle = (current_date - self.last_match_date[player]).days
        if days_idle > self.decay_threshold:
            old_elo = self.elo_overall[player]
            decay_factor = np.exp(-self.decay_lambda * (days_idle - self.decay_threshold))
            new_elo = self.initial_elo + (old_elo - self.initial_elo) * decay_factor
            self.elo_overall[player] = float(new_elo)
            self.is_comeback[player] = True

    @staticmethod
    def expected_outcome(elo1: float, elo2: float) -> float:
        return float(1 / (1 + 10 ** ((elo2 - elo1) / 400)))

    def update(
        self,
        winner: str,
        loser: str,
        surface: str,
        match_date: pd.Timestamp,
    ) -> Tuple[float, float, float, float]:
        self._apply_decay(winner, match_date)
        self._apply_decay(loser, match_date)

        winner_elo_before = float(self.elo_overall[winner])
        loser_elo_before = float(self.elo_overall[loser])

        k_winner = self._calculate_k(winner, self.is_comeback.get(winner, False))
        k_loser = self._calculate_k(loser, self.is_comeback.get(loser, False))

        expected_winner = self.expected_outcome(winner_elo_before, loser_elo_before)
        expected_loser = 1 - expected_winner

        self.elo_overall[winner] = float(winner_elo_before + k_winner * (1 - expected_winner))
        self.elo_overall[loser] = float(loser_elo_before + k_loser * (0 - expected_loser))

        if surface in self.elo_surface:
            k_surface = 32.0
            winner_surf_elo = float(self.elo_surface[surface][winner])
            loser_surf_elo = float(self.elo_surface[surface][loser])
            expected_surf = self.expected_outcome(winner_surf_elo, loser_surf_elo)

            self.elo_surface[surface][winner] = float(winner_surf_elo + k_surface * (1 - expected_surf))
            self.elo_surface[surface][loser] = float(loser_surf_elo + k_surface * (0 - (1 - expected_surf)))

        self.matches_played[winner] += 1
        self.matches_played[loser] += 1
        self.last_match_date[winner] = match_date
        self.last_match_date[loser] = match_date
        self.is_comeback[winner] = False
        self.is_comeback[loser] = False

        return (winner_elo_before, loser_elo_before, float(self.elo_overall[winner]), float(self.elo_overall[loser]))

    def get_elo(self, player: str, surface: Optional[str] = None) -> float:
        if surface and surface in self.elo_surface:
            return float(self.elo_surface[surface][player])
        return float(self.elo_overall[player])


@dataclass
class EloStateCursor:
    last_date: pd.Timestamp
    last_orig_idx: int


class EloTracker:
    def __init__(self, elo_system: Optional[AdvancedEloSystem] = None, cursor: Optional[EloStateCursor] = None):
        self.elo = elo_system or AdvancedEloSystem()
        self.cursor = cursor

    def to_state_dict(self) -> dict:
        return {
            "elo_overall": dict(self.elo.elo_overall),
            "elo_surface": {s: dict(d) for s, d in self.elo.elo_surface.items()},
            "matches_played": dict(self.elo.matches_played),
            "last_match_date": {k: pd.Timestamp(v) for k, v in self.elo.last_match_date.items()},
            "cursor": None
            if self.cursor is None
            else {"last_date": pd.Timestamp(self.cursor.last_date), "last_orig_idx": int(self.cursor.last_orig_idx)},
            "params": {
                "initial_elo": self.elo.initial_elo,
                "k_multiplier": self.elo.k_multiplier,
                "k_offset": self.elo.k_offset,
                "k_shape": self.elo.k_shape,
                "decay_lambda": self.elo.decay_lambda,
                "decay_threshold": self.elo.decay_threshold,
                "comeback_k_boost": self.elo.comeback_k_boost,
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "EloTracker":
        params = state.get("params", {})
        elo = AdvancedEloSystem(**params)  # type: ignore[arg-type]

        elo.elo_overall.update(state.get("elo_overall", {}))
        for surface, d in state.get("elo_surface", {}).items():
            if surface not in elo.elo_surface:
                continue
            elo.elo_surface[surface].update(d)

        elo.matches_played.update(state.get("matches_played", {}))
        elo.last_match_date.update({k: pd.Timestamp(v) for k, v in state.get("last_match_date", {}).items()})

        cursor_obj = None
        cursor = state.get("cursor")
        if cursor:
            cursor_obj = EloStateCursor(last_date=pd.Timestamp(cursor["last_date"]), last_orig_idx=int(cursor["last_orig_idx"]))

        return cls(elo_system=elo, cursor=cursor_obj)

    @classmethod
    def load(cls, path: Path) -> "EloTracker":
        state = joblib.load(path)
        return cls.from_state_dict(state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_state_dict(), path)

    @staticmethod
    def _normalize_surface(surface: str) -> str:
        if not isinstance(surface, str):
            return "Hard"
        s = surface.strip().title()
        if s in {"Hard", "Clay", "Grass", "Carpet"}:
            return s
        return "Hard"

    @staticmethod
    def _validate_match_row(row: pd.Series) -> bool:
        for col in ["Date", "Player_1", "Player_2", "Winner"]:
            if col not in row or pd.isna(row[col]):
                return False
        if row["Winner"] not in (row["Player_1"], row["Player_2"]):
            return False
        return True

    def apply_match(self, player1: str, player2: str, winner: str, surface: str, date: pd.Timestamp) -> None:
        surface_n = self._normalize_surface(surface)
        loser = player2 if winner == player1 else player1
        self.elo.update(winner, loser, surface_n, pd.Timestamp(date))

        # Update cursor monotonically when using apply_match()
        # (no orig_idx info: we just store date and -1)
        if self.cursor is None:
            self.cursor = EloStateCursor(last_date=pd.Timestamp(date), last_orig_idx=-1)
        else:
            if pd.Timestamp(date) > self.cursor.last_date:
                self.cursor = EloStateCursor(last_date=pd.Timestamp(date), last_orig_idx=-1)

    def update_from_dataframe(self, df: pd.DataFrame, end_date: Optional[pd.Timestamp] = None) -> int:
        if "Date" not in df.columns:
            raise ValueError("Missing Date column")

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Player_1", "Player_2", "Winner"])
        df["_orig_idx"] = np.arange(len(df), dtype=int)

        df = df.sort_values(["Date", "_orig_idx"]).reset_index(drop=True)
        if end_date is not None:
            df = df[df["Date"] <= pd.Timestamp(end_date)].reset_index(drop=True)

        if self.cursor is not None:
            last_date = pd.Timestamp(self.cursor.last_date)
            last_orig = int(self.cursor.last_orig_idx)
            df = df[(df["Date"] > last_date) | ((df["Date"] == last_date) & (df["_orig_idx"] > last_orig))]
            df = df.reset_index(drop=True)

        processed = 0
        last_seen_date = None
        last_seen_orig = None
        for _, row in df.iterrows():
            if not self._validate_match_row(row):
                continue
            p1 = str(row["Player_1"]).strip()
            p2 = str(row["Player_2"]).strip()
            winner = str(row["Winner"]).strip()
            loser = p2 if winner == p1 else p1
            surface = self._normalize_surface(str(row.get("Surface", "Hard")))
            d = pd.Timestamp(row["Date"])

            self.elo.update(winner, loser, surface, d)
            processed += 1
            last_seen_date = d
            last_seen_orig = int(row["_orig_idx"])

        if processed > 0 and last_seen_date is not None and last_seen_orig is not None:
            self.cursor = EloStateCursor(last_date=last_seen_date, last_orig_idx=last_seen_orig)

        return processed

    def update_from_csv(self, csv_path: Path, end_date: Optional[str] = None) -> int:
        df = pd.read_csv(csv_path)
        end_ts = pd.to_datetime(end_date) if end_date else None
        return self.update_from_dataframe(df, end_date=end_ts)

    def get_player_elo(self, player: str, surface: Optional[str] = None) -> float:
        return self.elo.get_elo(player, surface)


def _default_state_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "elo_tracker" / "elo_state.joblib"


def _default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "atp_tennis_cleaned.csv"


def main() -> None:
    parser = argparse.ArgumentParser(prog="elo_tracker", description="Dryja-style Elo tracker (overall + surface + decay).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build Elo state from scratch up to a given end date")
    p_build.add_argument("--csv", type=str, default=str(_default_dataset_path()))
    p_build.add_argument("--end-date", type=str, default="2025-12-31")
    p_build.add_argument("--out", type=str, default=str(_default_state_path()))

    p_update = sub.add_parser("update", help="Update existing Elo state from CSV (incremental)")
    p_update.add_argument("--csv", type=str, default=str(_default_dataset_path()))
    p_update.add_argument("--state", type=str, default=str(_default_state_path()))
    p_update.add_argument("--out", type=str, default=str(_default_state_path()))

    p_add = sub.add_parser("add-match", help="Apply a single match update and persist")
    p_add.add_argument("--state", type=str, default=str(_default_state_path()))
    p_add.add_argument("--out", type=str, default=str(_default_state_path()))
    p_add.add_argument("--date", type=str, required=True)
    p_add.add_argument("--surface", type=str, default="Hard")
    p_add.add_argument("--player1", type=str, required=True)
    p_add.add_argument("--player2", type=str, required=True)
    p_add.add_argument("--winner", type=str, required=True)

    p_query = sub.add_parser("query", help="Query current Elo for a player")
    p_query.add_argument("--state", type=str, default=str(_default_state_path()))
    p_query.add_argument("--player", type=str, required=True)
    p_query.add_argument("--surface", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "build":
        tracker = EloTracker()
        processed = tracker.update_from_csv(Path(args.csv), end_date=args.end_date)
        tracker.save(Path(args.out))
        print(f"processed_matches={processed}")
        if tracker.cursor is not None:
            print(f"cursor_last_date={tracker.cursor.last_date.date()} cursor_last_orig_idx={tracker.cursor.last_orig_idx}")

    elif args.cmd == "update":
        tracker = EloTracker.load(Path(args.state))
        processed = tracker.update_from_csv(Path(args.csv))
        tracker.save(Path(args.out))
        print(f"processed_matches={processed}")
        if tracker.cursor is not None:
            print(f"cursor_last_date={tracker.cursor.last_date.date()} cursor_last_orig_idx={tracker.cursor.last_orig_idx}")

    elif args.cmd == "add-match":
        tracker = EloTracker.load(Path(args.state))
        d = pd.to_datetime(args.date)
        tracker.apply_match(args.player1, args.player2, args.winner, args.surface, d)
        tracker.save(Path(args.out))
        print("ok")

    elif args.cmd == "query":
        tracker = EloTracker.load(Path(args.state))
        elo = tracker.get_player_elo(args.player, surface=args.surface)
        print(f"elo={elo:.2f}")


if __name__ == "__main__":
    main()
