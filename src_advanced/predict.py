#!/usr/bin/env python3
"""
ATP Advanced Prediction System – AdvancedPredictor

Loads all trained models and provides a unified prediction interface.

Usage:
    from src_advanced.predict import AdvancedPredictor
    predictor = AdvancedPredictor()
    result = predictor.predict(match_features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from .config import (
    CAT_COLS_ADVANCED, GAMES_THRESHOLD_BO3, GAMES_THRESHOLD_BO5,
    MERGED_DATASET_PATH, MODELS_ADV_DIR,
)
from .feature_engineering import get_feature_columns
from .models.games_predictor import GamesPredictor
from .models.score_predictor import ScorePredictor3, ScorePredictor5
from .models.duration_predictor import DurationPredictor
from .models.stats_predictor import StatsPredictor, StatsDataset


class AdvancedPredictor:
    """
    Unified predictor that loads all trained models and produces
    comprehensive match predictions.

    Output format:
    {
        'games': {'over_under': 'OVER', 'line': 22.5, 'probability': 0.63, 'predicted_total': 23.4},
        'set_score': {'predicted': '2-1', 'probabilities': {'2-0': 0.25, '2-1': 0.42}},
        'duration': {'predicted_minutes': 142, 'ci_lower': 118, 'ci_upper': 167},
        'stats': {'p1': {'aces': 8.2, ...}, 'p2': {...}},
    }
    """

    def __init__(self, models_dir: Path = MODELS_ADV_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = models_dir
        self._models: Dict[str, torch.nn.Module] = {}
        self._scalers: Dict[str, object] = {}
        self._encoders: Dict[str, object] = {}
        self._meta: Dict[str, dict] = {}
        self._feature_cols: Optional[tuple] = None

        # Try loading the merged dataset for feature column info
        if MERGED_DATASET_PATH.exists():
            df = pd.read_csv(MERGED_DATASET_PATH, nrows=5)
            self._feature_cols = get_feature_columns(df)

        # Load available models
        self._load_all()

    def _load_all(self):
        """Load all available trained models."""
        model_names = ["games", "score_bo3", "score_bo5", "duration", "stats"]
        model_classes = {
            "games": GamesPredictor,
            "score_bo3": ScorePredictor3,
            "score_bo5": ScorePredictor5,
            "duration": DurationPredictor,
            "stats": StatsPredictor,
        }

        for name in model_names:
            model_path = self.models_dir / f"{name}_model.pt"
            if not model_path.exists():
                continue

            try:
                meta = joblib.load(self.models_dir / f"{name}_meta.joblib")
                scaler = joblib.load(self.models_dir / f"{name}_scaler.joblib")
                encoder = joblib.load(self.models_dir / f"{name}_encoder.joblib")

                input_dim = meta["input_dim"]
                model = model_classes[name](input_dim).to(self.device)
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.eval()

                self._models[name] = model
                self._scalers[name] = scaler
                self._encoders[name] = encoder
                self._meta[name] = meta
                print(f"   Loaded {name} model (input_dim={input_dim})")
            except Exception as e:
                print(f"   Warning: failed to load {name}: {e}")

    def _prepare_input(self, features: dict, model_name: str) -> torch.Tensor:
        """Convert a features dict to a model input tensor."""
        num_cols, cat_cols = self._feature_cols or ([], [])
        scaler = self._scalers[model_name]
        encoder = self._encoders[model_name]

        num_vals = [float(features.get(c, 0)) for c in num_cols]
        X_num = scaler.transform(np.array([num_vals])).astype(np.float32)

        if cat_cols:
            cat_df = pd.DataFrame(
                [[features.get(c, "NA") for c in cat_cols]],
                columns=cat_cols,
            )
            X_cat = encoder.transform(cat_df.astype(str))
        else:
            X_cat = np.zeros((1, 0), dtype=np.float32)

        X = np.hstack([X_num, X_cat])
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def predict_games(self, features: dict, best_of: int = 3) -> dict:
        """Predict over/under and total games."""
        if "games" not in self._models:
            return {"over_under": "N/A", "line": 0, "probability": 0, "predicted_total": 0}

        model = self._models["games"]
        X = self._prepare_input(features, "games")

        with torch.no_grad():
            ou_logit, total_pred = model(X)
            ou_prob = torch.sigmoid(ou_logit).item()
            total_games = total_pred.item()

        threshold = GAMES_THRESHOLD_BO5 if best_of == 5 else GAMES_THRESHOLD_BO3
        over_under = "OVER" if ou_prob > 0.5 else "UNDER"

        return {
            "over_under": over_under,
            "line": threshold,
            "probability": round(ou_prob, 4),
            "predicted_total": round(total_games, 1),
        }

    def predict_score(self, features: dict, best_of: int = 3) -> dict:
        """Predict set score."""
        name = f"score_bo{best_of}"
        if name not in self._models:
            return {"predicted": "N/A", "probabilities": {}}

        model = self._models[name]
        X = self._prepare_input(features, name)

        class_names = {
            3: ["2-0", "2-1"],
            5: ["3-0", "3-1", "3-2"],
        }

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        class_n = class_names.get(best_of, ["2-0", "2-1"])
        predicted_idx = int(np.argmax(probs))
        probabilities = {class_n[i]: round(float(probs[i]), 4) for i in range(len(class_n))}

        return {
            "predicted": class_n[predicted_idx],
            "probabilities": probabilities,
        }

    def predict_duration(self, features: dict) -> dict:
        """Predict match duration with confidence interval."""
        if "duration" not in self._models:
            return {"predicted_minutes": 0, "ci_lower": 0, "ci_upper": 0}

        model = self._models["duration"]
        X = self._prepare_input(features, "duration")

        with torch.no_grad():
            mean, log_var = model(X)
            mean_minutes = mean.item()
            std_minutes = np.exp(0.5 * log_var.item())

        return {
            "predicted_minutes": round(mean_minutes, 1),
            "ci_lower": round(mean_minutes - 1.96 * std_minutes, 1),
            "ci_upper": round(mean_minutes + 1.96 * std_minutes, 1),
            "uncertainty_std": round(std_minutes, 1),
        }

    def predict_stats(self, features: dict) -> dict:
        """Predict player statistics."""
        if "stats" not in self._models:
            return {"p1": {}, "p2": {}}

        model = self._models["stats"]
        X = self._prepare_input(features, "stats")

        with torch.no_grad():
            preds = model(X)

        # Map predictions to named stats
        ace_preds = preds["aces"].cpu().numpy().flatten()
        df_preds = preds["df"].cpu().numpy().flatten()
        bp_preds = preds["bp"].cpu().numpy().flatten()
        serve_preds = preds["serve"].cpu().numpy().flatten()

        return {
            "p1": {
                "aces": round(float(ace_preds[0]), 1),
                "df": round(float(df_preds[0]), 1),
                "bp_saved": round(float(bp_preds[0]), 1),
                "bp_faced": round(float(bp_preds[1]), 1),
                "1st_serve_pct": round(float(serve_preds[0]), 3),
                "1st_serve_win_pct": round(float(serve_preds[1]), 3),
            },
            "p2": {
                "aces": round(float(ace_preds[1]), 1),
                "df": round(float(df_preds[1]), 1),
                "bp_saved": round(float(bp_preds[2]), 1),
                "bp_faced": round(float(bp_preds[3]), 1),
                "1st_serve_pct": round(float(serve_preds[2]), 3),
                "1st_serve_win_pct": round(float(serve_preds[3]), 3),
            },
        }

    def predict(self, features: dict, best_of: int = 3) -> dict:
        """
        Produce a complete prediction for a match.

        Parameters
        ----------
        features : dict
            Feature dictionary with all numerical and categorical features.
        best_of : int
            3 for best-of-3, 5 for best-of-5.

        Returns
        -------
        dict with keys: games, set_score, duration, stats
        """
        return {
            "games": self.predict_games(features, best_of),
            "set_score": self.predict_score(features, best_of),
            "duration": self.predict_duration(features),
            "stats": self.predict_stats(features),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """CLI demo: predict from a sample row of the merged dataset."""
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Predictor CLI")
    parser.add_argument("--sample", type=int, default=0, help="Row index from merged dataset")
    parser.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    args = parser.parse_args()

    if not MERGED_DATASET_PATH.exists():
        print("ERROR: Merged dataset not found. Run training first.")
        return

    from .feature_engineering import build_advanced_features
    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["Date"])
    df = build_advanced_features(df)

    if args.sample >= len(df):
        print(f"ERROR: sample index {args.sample} out of range (max {len(df)-1})")
        return

    row = df.iloc[args.sample]
    num_cols, cat_cols = get_feature_columns(df)
    features = {}
    for c in num_cols:
        features[c] = row.get(c, 0)
    for c in cat_cols:
        features[c] = row.get(c, "NA")

    print(f"\nPredicting match: {row['Player_1']} vs {row['Player_2']}")
    print(f"Date: {row['Date']}, Surface: {row.get('Surface', 'N/A')}")
    if "Score" in row.index:
        print(f"Actual Score: {row['Score']}")

    predictor = AdvancedPredictor()
    result = predictor.predict(features, best_of=args.best_of)

    import json
    print("\nPrediction Results:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
