#!/usr/bin/env python3
"""
ATP Tennis Match Predictor – Video Generation CLI

Orchestrates: predict → viz → script → TTS → compose → thumbnail.

Usage:
    # Single match
    python -m src.make_video \
        --player1 "Sinner J." --player2 "Alcaraz C." \
        --rank1 1 --rank2 3 --odds1 1.90 --odds2 1.95 \
        --surface Clay --tournament "Roland Garros"

    # Batch from JSON
    python -m src.make_video --matches matches_monte_carlo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import TEMP_DIR, VIDEOS_DIR, THUMBNAILS_DIR
from . import viz
from . import video


# ── Single match pipeline ────────────────────────────────────────────────────

def run_single_match(
    predictor: ATPPredictor,
    player1: str, player2: str,
    rank1: int, rank2: int,
    odds1: float, odds2: float,
    surface: str,
    tournament: str = "Unknown",
    date_str: Optional[str] = None,
    mode: str = "blend",
    alpha: float = 0.65,
    output_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Full pipeline for one match:
        predict → 5 viz frames → narration script → TTS → compose video → thumbnail
    """
    print(f"\n{'='*60}")
    print(f"  Match: {player1} vs {player2} ({tournament})")
    print(f"{'='*60}")

    # ── Step 1: Predict ──────────────────────────────────────────────────
    print("[1/6] Running prediction...")
    pred = predictor.predict(
        player1, player2, rank1, rank2,
        odds1, odds2, surface, tournament,
        date_str, mode, alpha,
    )
    print(f"  → Winner: {pred['prediction']} "
          f"(P1={pred['prob_p1']:.1%}, P2={pred['prob_p2']:.1%})")

    # ── Step 2: Visualisations ───────────────────────────────────────────
    print("[2/6] Generating visualisations...")
    match_slug = output_name or f"{player1.replace(' ', '_')}_vs_{player2.replace(' ', '_')}"
    frame_dir = TEMP_DIR / match_slug
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    # 2a – Prediction card
    frames.append(viz.match_prediction_card(
        p1_name=player1, p2_name=player2,
        rank1=rank1, rank2=rank2,
        prob_p1=pred["prob_p1"], prob_p2=pred["prob_p2"],
        confidence=pred["confidence"],
        tournament=tournament, surface=surface,
        winner=pred["prediction"],
        save_path=frame_dir / "01_prediction_card.png",
    ))

    # 2b – Feature contribution circles
    contributions = _compute_contributions(pred, rank1, rank2, odds1, odds2)
    frames.append(viz.feature_contribution_circles(
        contributions=contributions,
        save_path=frame_dir / "02_feature_circles.png",
    ))

    # 2c – Player comparison butterfly
    st1 = pred.get("player1_stats", {})
    st2 = pred.get("player2_stats", {})
    surf1 = pred.get("surface_p1", {})
    surf2 = pred.get("surface_p2", {})
    h2h = pred.get("h2h", {})

    comparison_stats = {
        "Win Rate": (st1.get("win_rate", 0.5), st2.get("win_rate", 0.5)),
        "Serie Vittorie": (
            min(max(st1.get("streak", 0), 0) / 10, 1.0),
            min(max(st2.get("streak", 0), 0) / 10, 1.0),
        ),
        "Momentum": (st1.get("momentum", 0.5), st2.get("momentum", 0.5)),
        "Superficie": (surf1.get("win_rate", 0.5), surf2.get("win_rate", 0.5)),
        "Scontri Diretti": (
            h2h.get("p1_win_rate", 0.5),
            1 - h2h.get("p1_win_rate", 0.5),
        ),
    }
    frames.append(viz.player_comparison_bars(
        p1_name=player1, p2_name=player2,
        stats=comparison_stats,
        save_path=frame_dir / "03_comparison_bars.png",
    ))

    # 2d – Confidence / upset gauge
    frames.append(viz.confidence_upset_gauge(
        confidence=pred["confidence"],
        upset_prob=pred["upset_probability"],
        save_path=frame_dir / "04_confidence_gauge.png",
    ))

    # 2e – Model breakdown
    frames.append(viz.model_breakdown(
        mlp_prob=pred["mlp_prob"],
        elo_prob=pred["elo_prob"],
        blend_prob=pred.get("blend_prob", pred["prob_p1"]),
        alpha=pred["alpha"],
        p1_name=player1,
        save_path=frame_dir / "05_model_breakdown.png",
    ))

    print(f"  → {len(frames)} frames saved to {frame_dir}")

    # ── Step 3: Narration script ─────────────────────────────────────────
    print("[3/6] Generating narration script...")
    sections = video.generate_script(pred, tournament, surface)
    for s in sections:
        print(f"  [{s['section']}] ({s['duration']}s) {s['text'][:60]}...")

    # ── Step 4: TTS ──────────────────────────────────────────────────────
    print("[4/6] Generating TTS audio...")
    audio_dir = TEMP_DIR / match_slug / "audio"
    audio_files = video.generate_tts(sections, output_dir=audio_dir)
    print(f"  → {len(audio_files)} audio files generated")

    # ── Step 5: Compose video ────────────────────────────────────────────
    print("[5/6] Composing video...")
    video_path = VIDEOS_DIR / f"{match_slug}.mp4"
    video.compose_video(frames, audio_files, output_path=video_path)
    print(f"  → Video saved: {video_path}")

    # ── Step 6: Thumbnail ────────────────────────────────────────────────
    print("[6/6] Generating thumbnail...")
    thumb_path = THUMBNAILS_DIR / f"{match_slug}_thumb.png"
    video.generate_thumbnail(
        frame_path=frames[0],  # use prediction card as base
        winner=pred["prediction"],
        prob_winner=max(pred["prob_p1"], pred["prob_p2"]),
        tournament=tournament,
        output_path=thumb_path,
    )
    print(f"  → Thumbnail saved: {thumb_path}")

    return {
        "prediction": pred,
        "frames": frames,
        "audio": audio_files,
        "video": video_path,
        "thumbnail": thumb_path,
    }


def _compute_contributions(pred: dict, rank1: int, rank2: int,
                           odds1: float, odds2: float) -> Dict[str, float]:
    """
    Compute per-feature directional contributions.
    Positive = favours P1, Negative = favours P2.
    """
    st1 = pred.get("player1_stats", {})
    st2 = pred.get("player2_stats", {})
    surf1 = pred.get("surface_p1", {})
    surf2 = pred.get("surface_p2", {})

    rank_diff = rank1 - rank2  # negative = P1 higher ranked (favours P1)
    odds_ratio = odds1 / max(odds2, 0.01)
    win_rate_diff = st1.get("win_rate", 0.5) - st2.get("win_rate", 0.5)
    streak_diff = st1.get("streak", 0) - st2.get("streak", 0)
    h2h_wr = pred.get("h2h", {}).get("p1_win_rate", 0.5)
    h2h_contribution = (h2h_wr - 0.5) * 2  # scale to [-1, 1]
    momentum_diff = pred.get("momentum_diff", 0.0)
    surface_diff = surf1.get("win_rate", 0.5) - surf2.get("win_rate", 0.5)

    # Normalise rank_diff to reasonable scale
    rank_norm = -rank_diff / max(max(abs(rank_diff), 1), 1) * 0.5

    # Normalise odds_ratio
    odds_norm = -(odds_ratio - 1) * 0.5

    # Normalise streak
    streak_norm = max(min(streak_diff / 10, 1), -1) * 0.3

    return {
        "rank_diff": rank_norm,
        "odds_ratio": odds_norm,
        "win_rate_diff": win_rate_diff,
        "streak_diff": streak_norm,
        "h2h_win_rate": h2h_contribution,
        "momentum_diff": momentum_diff,
        "surface_perf_diff": surface_diff,
    }


# ── Batch pipeline (JSON) ────────────────────────────────────────────────────

def run_batch(predictor: ATPPredictor, matches_path: Path) -> List[Dict]:
    """
    Run the pipeline for each match in a JSON file.

    JSON format:
    [
        {
            "player1": "Sinner J.", "player2": "Alcaraz C.",
            "rank1": 1, "rank2": 3, "odds1": 1.90, "odds2": 1.95,
            "surface": "Clay", "tournament": "Roland Garros",
            "date": "2025-06-08" (optional)
        },
        ...
    ]
    """
    with open(matches_path) as f:
        matches = json.load(f)

    results = []
    for i, m in enumerate(matches):
        print(f"\n{'─'*60}")
        print(f"  Match {i+1}/{len(matches)}")
        print(f"{'─'*60}")
        result = run_single_match(
            predictor,
            player1=m["player1"], player2=m["player2"],
            rank1=int(m["rank1"]), rank2=int(m["rank2"]),
            odds1=float(m["odds1"]), odds2=float(m["odds2"]),
            surface=m.get("surface", "Hard"),
            tournament=m.get("tournament", "Unknown"),
            date_str=m.get("date"),
            output_name=f"match_{i+1:02d}_{m.get('tournament', 'atp').replace(' ', '_')}",
        )
        results.append(result)

    print(f"\n{'='*60}")
    print(f"  Batch complete: {len(results)} videos generated")
    print(f"{'='*60}")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ATP Prediction Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single match options
    parser.add_argument("--player1", help="Player 1 name")
    parser.add_argument("--player2", help="Player 2 name")
    parser.add_argument("--rank1", type=int, help="Player 1 ATP rank")
    parser.add_argument("--rank2", type=int, help="Player 2 ATP rank")
    parser.add_argument("--odds1", type=float, help="Player 1 betting odds")
    parser.add_argument("--odds2", type=float, help="Player 2 betting odds")
    parser.add_argument("--surface", default="Hard",
                        choices=["Hard", "Clay", "Grass", "Carpet"])
    parser.add_argument("--tournament", default="Unknown")
    parser.add_argument("--date", default=None)
    parser.add_argument("--mode", default="blend", choices=["mlp", "elo", "blend"])
    parser.add_argument("--alpha", type=float, default=0.65)

    # Batch mode
    parser.add_argument("--matches", type=Path, default=None,
                        help="Path to JSON file with match list")

    args = parser.parse_args()

    # Validate arguments
    if args.matches:
        if not args.matches.exists():
            print(f"Error: file not found: {args.matches}")
            sys.exit(1)
    elif not all([args.player1, args.player2, args.rank1, args.rank2,
                  args.odds1, args.odds2]):
        print("Error: provide either --matches JSON or all player args "
              "(--player1, --player2, --rank1, --rank2, --odds1, --odds2)")
        sys.exit(1)

    # Initialise predictor (lazy import to avoid loading torch at module level)
    from .predict import ATPPredictor
    print("Loading prediction model...")
    predictor = ATPPredictor()

    if args.matches:
        run_batch(predictor, args.matches)
    else:
        run_single_match(
            predictor,
            player1=args.player1, player2=args.player2,
            rank1=args.rank1, rank2=args.rank2,
            odds1=args.odds1, odds2=args.odds2,
            surface=args.surface, tournament=args.tournament,
            date_str=args.date, mode=args.mode, alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
