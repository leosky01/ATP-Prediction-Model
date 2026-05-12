#!/usr/bin/env python3
"""
ATP Advanced Prediction System — Predictor V3.

Loads the V3 trained models (LightGBM + NN ensembles with isotonic
calibration) and exposes a single `.predict(features, best_of)` interface.

Outputs:
{
    'games': {
        'over_under': 'OVER'|'UNDER',
        'line': 22.5 or 38.5,
        'probability_over': 0.63,
        'predicted_total': 23.4,
    },
    'n_sets': {
        'predicted_n_sets': 2|3|4|5,
        'probabilities': {2: 0.62, 3: 0.38} or {3:.., 4:.., 5:..},
    },
    'duration': {
        'predicted_minutes': 142,
        'ci_lower': 118, 'ci_upper': 167,
    },
    'stats': {
        'p1': {'aces':.., 'df':.., 'bp_saved':.., 'bp_faced':.., '1st_serve_pct':.., '1st_serve_win_pct':..},
        'p2': {...},
    }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .config import (
    GAMES_THRESHOLD_BO3, GAMES_THRESHOLD_BO5,
    MERGED_DATASET_PATH, MODELS_ADV_DIR,
)
from .feature_engineering import get_feature_columns
from .train_v3 import OUNet, DurationNet


class AdvancedPredictorV3:
    """High-accuracy ensemble predictor combining LightGBM + NN with calibration."""

    def __init__(self, models_dir: Path = MODELS_ADV_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = models_dir

        # ── feature columns ──
        # NOTE: we use the static canonical list, NOT get_feature_columns(df),
        # because the raw merged CSV doesn't have engineered features available.
        from .feature_engineering import NUM_COLS_ADVANCED
        from .config import CAT_COLS_ADVANCED
        self.num_cols = list(NUM_COLS_ADVANCED)
        self.cat_cols = list(CAT_COLS_ADVANCED)

        # ── games (OU + total) ──
        self.games_scaler = self._maybe_load("games_v3_scaler.joblib")
        self.games_encoder = self._maybe_load("games_v3_encoder.joblib")
        self.games_meta = self._maybe_load("games_v3_meta.joblib") or {}

        self.games_models: Dict[int, dict] = {}
        for bo in (3, 5):
            artifacts = {}
            lgb_ou_path = models_dir / f"games_v3_bo{bo}_lgb_ou.txt"
            if not lgb_ou_path.exists():
                continue
            artifacts["lgb_ou"] = lgb.Booster(model_file=str(lgb_ou_path))
            artifacts["lgb_total"] = lgb.Booster(model_file=str(models_dir / f"games_v3_bo{bo}_lgb_total.txt"))

            nn_models = []
            for si in range(self.games_meta.get(f"bo{bo}_n_nn_seeds", 3)):
                p = models_dir / f"games_v3_bo{bo}_nn_s{si}.pt"
                if not p.exists():
                    continue
                m = OUNet(self.games_meta.get("input_dim", 196)).to(self.device)
                m.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                m.eval()
                nn_models.append(m)
            artifacts["nn_seeds"] = nn_models
            artifacts["stacker"] = self._maybe_load(f"games_v3_bo{bo}_stacker.joblib")
            artifacts["calibrator"] = self._maybe_load(f"games_v3_bo{bo}_calibrator.joblib")
            self.games_models[bo] = artifacts

        # ── duration ──
        self.duration_scaler = self._maybe_load("duration_v3_scaler.joblib")
        self.duration_encoder = self._maybe_load("duration_v3_encoder.joblib")
        self.duration_meta = self._maybe_load("duration_v3_meta.joblib") or {}
        self.duration_lgb = None
        p = models_dir / "duration_v3_lgb.txt"
        if p.exists():
            self.duration_lgb = lgb.Booster(model_file=str(p))
        self.duration_nn = None
        p = models_dir / "duration_v3_nn.pt"
        if p.exists():
            self.duration_nn = DurationNet(self.duration_meta.get("input_dim", 196)).to(self.device)
            self.duration_nn.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            self.duration_nn.eval()

        # ── n_sets ──
        self.nsets_scaler = self._maybe_load("nsets_v3_scaler.joblib")
        self.nsets_encoder = self._maybe_load("nsets_v3_encoder.joblib")
        self.nsets_meta = self._maybe_load("nsets_v3_meta.joblib") or {}
        self.nsets_models: Dict[int, lgb.Booster] = {}
        for bo in (3, 5):
            p = models_dir / f"nsets_v3_bo{bo}_lgb.txt"
            if p.exists():
                self.nsets_models[bo] = lgb.Booster(model_file=str(p))

        # ── stats ──
        self.stats_scaler = self._maybe_load("stats_v3_scaler.joblib")
        self.stats_encoder = self._maybe_load("stats_v3_encoder.joblib")
        self.stats_meta = self._maybe_load("stats_v3_meta.joblib") or {}
        self.stats_models: Dict[str, lgb.Booster] = {}
        for tgt in self.stats_meta.get("targets", []):
            p = models_dir / f"stats_v3_{tgt}.txt"
            if p.exists():
                self.stats_models[tgt] = lgb.Booster(model_file=str(p))

        print(f"[V3 predictor] loaded:  games={list(self.games_models.keys())}  "
              f"duration={'OK' if self.duration_lgb else 'NO'}  "
              f"nsets={list(self.nsets_models.keys())}  "
              f"stats={len(self.stats_models)} targets")

    def _maybe_load(self, name: str):
        p = self.models_dir / name
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                print(f"   warn: failed loading {name}: {e}")
        return None

    # ── Feature preparation ──
    def _build_X(self, features: dict, scaler, encoder) -> np.ndarray:
        # Determine effective num_cols from the scaler if it cached them
        num_cols = getattr(scaler, "num_cols_", None) or self.num_cols
        num_vals = np.array(
            [float(features.get(c, np.nan)) for c in num_cols], dtype=np.float32
        )
        # Fill NaN with training-median
        if scaler is not None and hasattr(scaler, "train_medians_"):
            medians = scaler.train_medians_
            mask = np.isnan(num_vals)
            num_vals[mask] = medians[mask]
        else:
            num_vals = np.nan_to_num(num_vals, nan=0.0)
        X_num = scaler.transform(num_vals.reshape(1, -1)).astype(np.float32) if scaler else num_vals.reshape(1, -1)

        # Use the encoder's actual feature names (may differ from CAT_COLS_ADVANCED)
        if encoder is not None and hasattr(encoder, "feature_names_in_") and len(encoder.feature_names_in_):
            cat_names = list(encoder.feature_names_in_)
            cat_df = pd.DataFrame(
                [[features.get(c, "NA") for c in cat_names]],
                columns=cat_names,
            )
            X_cat = encoder.transform(cat_df.astype(str)).astype(np.float32)
        else:
            X_cat = np.zeros((1, 0), dtype=np.float32)
        return np.hstack([X_num, X_cat]).astype(np.float32)

    # ── Games ──
    def predict_games(self, features: dict, best_of: int = 3) -> dict:
        models = self.games_models.get(best_of)
        if models is None:
            return {"over_under": "N/A", "line": 0, "probability_over": 0.0, "predicted_total": 0.0}
        X = self._build_X(features, self.games_scaler, self.games_encoder)

        # LGBM OU prob
        lgb_p = float(models["lgb_ou"].predict(X)[0])

        # NN ensemble
        nn_models = models["nn_seeds"]
        if nn_models:
            probs = []
            with torch.no_grad():
                xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
                for m in nn_models:
                    logit = m(xb)
                    probs.append(torch.sigmoid(logit).cpu().numpy().flatten()[0])
            nn_p = float(np.mean(probs))
        else:
            nn_p = lgb_p

        # Stacker
        stacker = models["stacker"]
        if stacker is not None:
            stack_in = np.array([[lgb_p, nn_p]])
            ens_p_raw = float(stacker.predict_proba(stack_in)[0, 1])
        else:
            ens_p_raw = 0.5 * (lgb_p + nn_p)

        # Isotonic calibration
        calibrator = models["calibrator"]
        if calibrator is not None:
            ens_p = float(np.clip(calibrator.predict([ens_p_raw])[0], 1e-6, 1 - 1e-6))
        else:
            ens_p = ens_p_raw

        # Total games
        total = float(models["lgb_total"].predict(X)[0])

        threshold = GAMES_THRESHOLD_BO5 if best_of == 5 else GAMES_THRESHOLD_BO3
        return {
            "over_under": "OVER" if ens_p > 0.5 else "UNDER",
            "line": threshold,
            "probability_over": round(ens_p, 4),
            "predicted_total": round(total, 2),
            "_components": {
                "lgbm_p": round(lgb_p, 4), "nn_p": round(nn_p, 4),
                "stacker_p": round(ens_p_raw, 4),
            },
        }

    # ── Duration ──
    def predict_duration(self, features: dict) -> dict:
        if self.duration_lgb is None:
            return {"predicted_minutes": 0, "ci_lower": 0, "ci_upper": 0}
        X = self._build_X(features, self.duration_scaler, self.duration_encoder)
        lgb_min = float(self.duration_lgb.predict(X)[0])
        if self.duration_nn is not None:
            with torch.no_grad():
                xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
                m, lv = self.duration_nn(xb)
                nn_min = float(m.cpu().numpy().flatten()[0])
                nn_std = float(np.exp(0.5 * lv.cpu().numpy().flatten()[0]))
            mean = 0.5 * (lgb_min + nn_min)
            std = nn_std  # use NN's predictive uncertainty
        else:
            mean = lgb_min
            std = 25.0  # constant fallback
        return {
            "predicted_minutes": round(mean, 1),
            "ci_lower": round(mean - 1.96 * std, 1),
            "ci_upper": round(mean + 1.96 * std, 1),
            "uncertainty_std": round(std, 1),
        }

    # ── N-sets ──
    def predict_n_sets(self, features: dict, best_of: int = 3) -> dict:
        model = self.nsets_models.get(best_of)
        if model is None:
            return {"predicted_n_sets": None, "probabilities": {}}
        X = self._build_X(features, self.nsets_scaler, self.nsets_encoder)
        # LightGBM Booster returns raw probabilities of class 1 in binary
        raw = model.predict(X)
        raw = np.array(raw).reshape(1, -1)
        classes = [2, 3] if best_of == 3 else [3, 4, 5]
        if raw.shape[1] == 1:
            # binary case: model returns P(class=1)
            p1 = float(raw[0, 0])
            probs = {classes[0]: round(1 - p1, 4), classes[1]: round(p1, 4)}
        else:
            probs = {classes[i]: round(float(raw[0, i]), 4) for i in range(len(classes))}
        predicted = max(probs.items(), key=lambda kv: kv[1])[0]
        return {"predicted_n_sets": int(predicted), "probabilities": probs}

    # ── Stats ──
    def predict_stats(self, features: dict) -> dict:
        if not self.stats_models:
            return {"p1": {}, "p2": {}}
        X = self._build_X(features, self.stats_scaler, self.stats_encoder)
        out: Dict[str, float] = {}
        for tgt, model in self.stats_models.items():
            out[tgt] = float(model.predict(X)[0])
        def _get(key, default=0.0):
            return round(out.get(key, default), 3)
        return {
            "p1": {
                "aces": _get("p1_ace"),
                "df": _get("p1_df"),
                "bp_saved": _get("p1_bpSaved"),
                "bp_faced": _get("p1_bpFaced"),
                "1st_serve_pct": _get("p1_1st_pct"),
                "1st_serve_win_pct": _get("p1_1st_win_pct"),
            },
            "p2": {
                "aces": _get("p2_ace"),
                "df": _get("p2_df"),
                "bp_saved": _get("p2_bpSaved"),
                "bp_faced": _get("p2_bpFaced"),
                "1st_serve_pct": _get("p2_1st_pct"),
                "1st_serve_win_pct": _get("p2_1st_win_pct"),
            },
        }

    def predict(self, features: dict, best_of: int = 3) -> dict:
        return {
            "games": self.predict_games(features, best_of),
            "n_sets": self.predict_n_sets(features, best_of),
            "duration": self.predict_duration(features),
            "stats": self.predict_stats(features),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V3 Advanced Predictor CLI")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--best-of", type=int, default=None, choices=[3, 5])
    args = parser.parse_args()

    from .feature_cache import load_or_build_features
    df = load_or_build_features()

    if args.sample >= len(df):
        print(f"ERROR: sample index out of range")
        return
    row = df.iloc[args.sample]
    bo = args.best_of or int(row.get("best_of", 3) or 3)
    print(f"\nMatch: {row.get('Player_1')} vs {row.get('Player_2')}  "
          f"Date={row.get('Date')}  Surface={row.get('Surface') or row.get('surface')}  BO{bo}")
    if "score" in row.index:
        print(f"Actual score: {row['score']}  total_games={row.get('total_games')}  minutes={row.get('minutes')}")

    feats = row.to_dict()
    pred = AdvancedPredictorV3().predict(feats, best_of=bo)
    print(json.dumps(pred, indent=2, default=str))


if __name__ == "__main__":
    main()
