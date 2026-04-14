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

    # Batch from JSON (one video per match)
    python -m src.make_video --matches matches_monte_carlo.json

    # Daily video (all matches in one video)
    python -m src.make_video --daily matches_monte_carlo.json \
        --daily-tournament "Monte Carlo Masters" --daily-date "2026-04-10"
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
from .predict import ATPHistoryCalculator


# ── Single match pipeline ────────────────────────────────────────────────────

def _get_elo_trend(history: ATPHistoryCalculator, player: str,
                   months: int = 12) -> Tuple[List[str], List[float]]:
    """
    Rebuild Elo for a single player from historical data.
    Returns (dates, elo_values) for the last `months` months.
    """
    from .elo_tracker import AdvancedEloSystem
    import pandas as pd

    before_date = pd.Timestamp.now()
    cutoff = before_date - pd.DateOffset(months=months)
    sub = history._player_matches(player, before_date, days=months * 31)
    sub = sub[sub["Date"] >= cutoff]

    if sub.empty:
        return [], []

    # Build Elo from scratch using the player's matches
    elo_sys = AdvancedEloSystem()
    dates_out = []
    elos_out = []

    for _, row in sub.iterrows():
        p1 = row["Player_1"]
        p2 = row["Player_2"]
        winner = row["Winner"]
        loser = p2 if winner == p1 else p1
        surface = str(row.get("Surface", "Hard")).strip().title()
        if surface not in {"Hard", "Clay", "Grass", "Carpet"}:
            surface = "Hard"
        elo_sys.update(winner, loser, surface, row["Date"])

        current_elo = elo_sys.elo_overall.get(player, 1500)
        dates_out.append(row["Date"].strftime("%Y-%m-%d"))
        elos_out.append(current_elo)

    return dates_out, elos_out


def _get_last_10(history: ATPHistoryCalculator, player: str,
                 before_date=None) -> List[Dict]:
    """Return last 10 match results for a player."""
    import pandas as pd
    if before_date is None:
        before_date = pd.Timestamp.now()
    sub = history._player_matches(player, before_date, days=365).tail(10)

    results = []
    for _, row in sub.iterrows():
        is_p1 = row["Player_1"] == player
        won = (row["Winner"] == player)
        opponent = row["Player_2"] if is_p1 else row["Player_1"]
        score = str(row.get("Score", ""))
        surface = str(row.get("Surface", ""))
        results.append({
            "opponent": opponent,
            "won": won,
            "score": score,
            "surface": surface,
        })
    return results


def _get_surface_stats_all(history: ATPHistoryCalculator, player: str,
                           before_date=None) -> Dict[str, float]:
    """Return win rates per surface + overall win rate + momentum."""
    import pandas as pd
    if before_date is None:
        before_date = pd.Timestamp.now()

    stats = {}
    for surface in ["Hard", "Clay", "Grass"]:
        sp = history.surface_performance(player, surface, before_date)
        stats[surface] = sp.get("win_rate", 0.5)

    rs = history.rolling_stats(player, before_date)
    stats["Win Rate"] = rs.get("win_rate", 0.5)
    stats["Momentum"] = rs.get("momentum", 0.5)

    return stats


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
    history = predictor.history
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

    # 2f – Elo trend 12 mesi
    print("  Generating Elo trend...")
    try:
        p1_dates, p1_elos = _get_elo_trend(history, player1)
        p2_dates, p2_elos = _get_elo_trend(history, player2)
        frames.append(viz.elo_trend(
            player1, player2, p1_dates, p1_elos, p2_dates, p2_elos,
            surface=surface, save_path=frame_dir / "06_elo_trend.png",
        ))
    except Exception as e:
        print(f"  [skip] Elo trend: {e}")

    # 2g – Ultime 10 partite
    print("  Generating last 10 matches...")
    try:
        p1_last10 = _get_last_10(history, player1)
        p2_last10 = _get_last_10(history, player2)
        frames.append(viz.last_10_matches(
            player1, player2, p1_last10, p2_last10,
            save_path=frame_dir / "07_last_10.png",
        ))
    except Exception as e:
        print(f"  [skip] Last 10: {e}")

    # 2h – Radar superficie
    print("  Generating surface radar...")
    try:
        p1_surf = _get_surface_stats_all(history, player1)
        p2_surf = _get_surface_stats_all(history, player2)
        frames.append(viz.surface_radar(
            player1, player2, p1_surf, p2_surf,
            save_path=frame_dir / "08_surface_radar.png",
        ))
    except Exception as e:
        print(f"  [skip] Surface radar: {e}")

    print(f"  → {len(frames)} total frames")

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


# ── Daily video pipeline (all matches in one video) ─────────────────────────

def run_daily(
    predictor,
    matches_path: Path,
    tournament: str = "Torneo ATP",
    date_str: str = "oggi",
    track_record: Optional[Dict] = None,
) -> Dict:
    """
    Generate a single YouTube video covering ALL matches of the day.
    Pipeline: predict all → daily card + per-match frames → daily narration → video.
    """
    with open(matches_path) as f:
        matches = json.load(f)

    print(f"\n{'='*60}")
    print(f"  DAILY VIDEO — {tournament} ({len(matches)} match)")
    print(f"{'='*60}")

    history = predictor.history

    # Step 1: Predict all matches
    print("[1/6] Running predictions...")
    predictions = []
    match_cards = []
    for i, m in enumerate(matches):
        p1 = m["player1"]
        p2 = m["player2"]
        pred = predictor.predict(
            p1, p2,
            int(m["rank1"]), int(m["rank2"]),
            float(m["odds1"]), float(m["odds2"]),
            m.get("surface", "Hard"), m.get("tournament", tournament),
            m.get("date"), m.get("mode", "blend"), m.get("alpha", 0.65),
        )
        # Enrich prediction with player names for narration
        pred["player1"] = p1
        pred["player2"] = p2
        predictions.append(pred)

        match_cards.append({
            "p1": p1, "p2": p2,
            "winner": pred["prediction"],
            "prob_winner": max(pred["prob_p1"], pred["prob_p2"]),
            "confidence": pred["confidence"],
            "surface": m.get("surface", "Hard"),
        })
        print(f"  {i+1}. {p1} vs {p2} → {pred['prediction']} "
              f"({max(pred['prob_p1'], pred['prob_p2']):.0%})")

    # Step 2: Generate frames
    print("[2/6] Generating frames...")
    daily_dir = TEMP_DIR / f"daily_{tournament.replace(' ', '_')}_{date_str}"
    daily_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []

    # 2a – Daily batch card (overview)
    all_frames.append(viz.daily_batch_card(
        matches=match_cards,
        tournament=tournament,
        date_str=date_str,
        save_path=daily_dir / "00_daily_card.png",
    ))

    # 2b – Per-match frames (3 per match: prediction card, last 10, surface radar)
    for i, (m, pred) in enumerate(zip(matches, predictions)):
        p1 = m["player1"]
        p2 = m["player2"]
        surface = m.get("surface", "Hard")
        prefix = f"{i+1:02d}"

        # Prediction card
        all_frames.append(viz.match_prediction_card(
            p1_name=p1, p2_name=p2,
            rank1=int(m["rank1"]), rank2=int(m["rank2"]),
            prob_p1=pred["prob_p1"], prob_p2=pred["prob_p2"],
            confidence=pred["confidence"],
            tournament=tournament, surface=surface,
            winner=pred["prediction"],
            save_path=daily_dir / f"{prefix}_card.png",
        ))

        # Last 10
        try:
            p1_last10 = _get_last_10(history, p1)
            p2_last10 = _get_last_10(history, p2)
            all_frames.append(viz.last_10_matches(
                p1, p2, p1_last10, p2_last10,
                save_path=daily_dir / f"{prefix}_last10.png",
            ))
        except Exception:
            pass

        # Surface radar
        try:
            p1_surf = _get_surface_stats_all(history, p1)
            p2_surf = _get_surface_stats_all(history, p2)
            all_frames.append(viz.surface_radar(
                p1, p2, p1_surf, p2_surf,
                save_path=daily_dir / f"{prefix}_radar.png",
            ))
        except Exception:
            pass

    # 2c – Track record
    if track_record:
        all_frames.append(viz.track_record_accuracy(
            dates=track_record.get("dates", []),
            accuracies=track_record.get("accuracies", []),
            n_matches=track_record.get("n_matches_list", []),
            overall_acc=track_record.get("accuracy"),
            period_label=track_record.get("period", "Ultimi 90 giorni"),
            save_path=daily_dir / "99_track_record.png",
        ))

    print(f"  → {len(all_frames)} total frames")

    # Step 3: Narration
    print("[3/6] Generating narration...")
    sections = video.generate_daily_script(
        predictions=predictions,
        tournament=tournament,
        date_str=date_str,
        track_record=track_record,
    )
    print(f"  → {len(sections)} sections")

    # Step 4: TTS
    print("[4/6] Generating TTS...")
    audio_dir = daily_dir / "audio"
    audio_files = video.generate_tts(sections, output_dir=audio_dir)
    print(f"  → {len(audio_files)} audio files")

    # Step 5: Compose
    print("[5/6] Composing video...")
    video_name = f"daily_{tournament.replace(' ', '_')}_{date_str}.mp4"
    video_path = VIDEOS_DIR / video_name
    video.compose_video(all_frames, audio_files, output_path=video_path)
    print(f"  → {video_path}")

    # Step 6: Thumbnail
    print("[6/6] Generating thumbnail...")
    thumb_path = THUMBNAILS_DIR / f"daily_{tournament.replace(' ', '_')}_{date_str}_thumb.png"
    best_match = max(match_cards, key=lambda x: x["confidence"])
    video.generate_thumbnail(
        frame_path=all_frames[0],
        winner=best_match["winner"],
        prob_winner=best_match["prob_winner"],
        tournament=f"{tournament} — {len(matches)} match",
        output_path=thumb_path,
    )
    print(f"  → {thumb_path}")

    print(f"\n{'='*60}")
    print(f"  DAILY VIDEO COMPLETE")
    print(f"  {len(all_frames)} frames | {len(audio_files)} audio | 1 video | 1 thumb")
    print(f"{'='*60}")

    return {
        "predictions": predictions,
        "frames": all_frames,
        "audio": audio_files,
        "video": video_path,
        "thumbnail": thumb_path,
    }


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

    # Daily video mode (all matches in one video)
    parser.add_argument("--daily", type=Path, default=None,
                        help="Path to JSON file → single daily video with all matches")
    parser.add_argument("--daily-tournament", default="Torneo ATP")
    parser.add_argument("--daily-date", default="oggi")

    args = parser.parse_args()

    # Validate arguments
    if args.daily:
        if not args.daily.exists():
            print(f"Error: file not found: {args.daily}")
            sys.exit(1)
    elif args.matches:
        if not args.matches.exists():
            print(f"Error: file not found: {args.matches}")
            sys.exit(1)
    elif not all([args.player1, args.player2, args.rank1, args.rank2,
                  args.odds1, args.odds2]):
        print("Error: provide --daily JSON, --matches JSON, or all player args "
              "(--player1, --player2, --rank1, --rank2, --odds1, --odds2)")
        sys.exit(1)

    # Initialise predictor (lazy import to avoid loading torch at module level)
    from .predict import ATPPredictor
    print("Loading prediction model...")
    predictor = ATPPredictor()

    if args.daily:
        run_daily(predictor, args.daily,
                  tournament=args.daily_tournament,
                  date_str=args.daily_date)
    elif args.matches:
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
