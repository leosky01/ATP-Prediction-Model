"""
ATP Tennis Match Predictor – Video Generation Module

Handles Italian narration script generation, TTS via edge-tts,
video composition via moviepy, and thumbnail creation via Pillow.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    TEMP_DIR, VIDEOS_DIR, THUMBNAILS_DIR,
    VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, THUMB_WIDTH, THUMB_HEIGHT,
    TTS_VOICE, TTS_RATE, SECTION_DURATIONS,
)


# ── 0. Daily multi-match script ─────────────────────────────────────────────

def generate_daily_script(
    predictions: List[dict],
    tournament: str = "Torneo ATP",
    surface: str = "Hard",
    date_str: str = "oggi",
    track_record: Optional[dict] = None,
) -> List[Dict[str, str]]:
    """
    Generate narration sections for a daily multi-match video.

    predictions: list of prediction dicts (as returned by ATPPredictor.predict).
    track_record: optional {"accuracy": float, "n_matches": int, "period": str}
    """
    n = len(predictions)
    sections = []

    # 1 – Daily intro
    sections.append({
        "section": "daily_intro",
        "text": (
            f"Benvenuti su ATP Predictions! Oggi analizziamo {n} match di {tournament}. "
            f"Il nostro modello di intelligenza artificiale ha elaborato tutte le partite. "
            f"Scopriamo insieme le predizioni!"
        ),
        "duration": 10,
    })

    # 2 – Daily card overview
    high_conf = sum(1 for p in predictions if p.get("confidence", 0) > 0.6)
    sections.append({
        "section": "daily_card",
        "text": (
            f"Ecco il quadro completo della giornata: {n} match, "
            f"di cui {high_conf} ad alta confidenza. "
            f"Il modello è più sicuro su alcune partite rispetto ad altre. Analizziamole nel dettaglio."
        ),
        "duration": 12,
    })

    # 3..N – Per-match sections (abbreviated for daily video)
    for i, pred in enumerate(predictions):
        p1 = pred.get("player1", f"P1_{i}")
        p2 = pred.get("player2", f"P2_{i}")
        winner = pred.get("prediction", "?")
        prob = max(pred.get("prob_p1", 0.5), pred.get("prob_p2", 0.5))
        conf = pred.get("confidence", 0.5)
        upset = pred.get("upset_probability", 0.15)
        mlp = pred.get("mlp_prob", 0.5)
        elo = pred.get("elo_prob", 0.5)

        sections.append({
            "section": f"match_{i+1:02d}",
            "text": (
                f"Match numero {i+1}: {p1} contro {p2}. "
                f"Il modello prevede {winner} al {prob:.0%} di probabilità. "
                f"Confidenza: {conf:.0%}. "
                f"La rete neurale dà il {mlp:.0%}, il sistema Elo il {elo:.0%}. "
                f"Probabilità di sorpresa: {upset:.0%}."
            ),
            "duration": 15,
        })

    # Track record section
    if track_record:
        acc = track_record.get("accuracy", 0.68)
        n_hist = track_record.get("n_matches", 100)
        period = track_record.get("period", "ultimi 90 giorni")
        sections.append({
            "section": "track_record",
            "text": (
                f"E il nostro track record? Negli {period}, "
                f"il modello ha azzeccato il {acc:.0%} delle predizioni, "
                f"su un totale di {n_hist} partite analizzate. "
                f"Un risultato significativamente superiore al 50 per cento del caso!"
            ),
            "duration": 12,
        })

    # Outro
    sections.append({
        "section": "daily_outro",
        "text": (
            f"Questo è tutto per le predizioni di oggi! "
            f"Iscrivetevi al canale e mettete like per supportare il progetto. "
            f"Alla prossima giornata ATP!"
        ),
        "duration": 6,
    })

    return sections


# ── 1. Narration Script Generation ──────────────────────────────────────────

def generate_script(pred: dict, tournament: str = "Torneo ATP",
                    surface: str = "Hard") -> List[Dict[str, str]]:
    """
    Generate 7 Italian narration sections from a prediction dict.

    Returns list of {"section": name, "text": narration, "duration": seconds}.
    """
    p1_name = "P1"
    p2_name = "P2"

    st1 = pred.get("player1_stats", {})
    st2 = pred.get("player2_stats", {})

    prob_p1 = pred.get("prob_p1", 0.5)
    prob_p2 = pred.get("prob_p2", 0.5)
    winner = pred.get("prediction", "P1")
    confidence = pred.get("confidence", 0.5)
    upset = pred.get("upset_probability", 0.15)
    mlp_prob = pred.get("mlp_prob", 0.5)
    elo_prob = pred.get("elo_prob", 0.5)
    alpha = pred.get("alpha", 0.65)
    momentum_diff = pred.get("momentum_diff", 0.0)
    h2h = pred.get("h2h", {})

    favoured_pct = max(prob_p1, prob_p2)

    sections = []

    # 1 – Intro
    sections.append({
        "section": "intro",
        "text": (
            f"Benvenuti su ATP Predictions! Oggi analizziamo il match di {tournament}, "
            f"su superficie {surface}. Vediamo cosa dice il nostro modello di intelligenza artificiale."
        ),
        "duration": SECTION_DURATIONS["intro"],
    })

    # 2 – Prediction Card
    sections.append({
        "section": "prediction_card",
        "text": (
            f"Il modello prevede la vittoria di {winner} con una probabilità del {favoured_pct:.0%}. "
            f"Analizziamo nel dettaglio le ragioni di questa previsione."
        ),
        "duration": SECTION_DURATIONS["prediction_card"],
    })

    # 3 – Feature Circles
    ranking_fav = p1_name if prob_p1 > prob_p2 else p2_name
    sections.append({
        "section": "feature_circles",
        "text": (
            f"Analizziamo le feature chiave. Il ranking favorisce {ranking_fav}. "
            f"Il momentum ha un differenziale di {abs(momentum_diff):.2f}. "
            f"Negli scontri diretti, il bilancio è di "
            f"{h2h.get('p1_wins', 0)} a {h2h.get('p2_wins', 0)}. "
            f"Ogni cerchio rappresenta una feature: più grande, più importante."
        ),
        "duration": SECTION_DURATIONS["feature_circles"],
    })

    # 4 – Player Comparison
    wr1 = st1.get("win_rate", 0.5)
    wr2 = st2.get("win_rate", 0.5)
    streak1 = st1.get("streak", 0)
    streak2 = st2.get("streak", 0)
    sections.append({
        "section": "player_comparison",
        "text": (
            f"Confrontiamo le statistiche. "
            f"{p1_name} ha un win rate del {wr1:.0%} con una serie di {streak1} partite. "
            f"{p2_name} ha un win rate del {wr2:.0%} con una serie di {streak2} partite. "
            f"Il grafico a farfalla mostra chi domina ogni aspetto."
        ),
        "duration": SECTION_DURATIONS["player_comparison"],
    })

    # 5 – Confidence / Upset
    sections.append({
        "section": "confidence_upset",
        "text": (
            f"La confidenza del modello è del {confidence:.0%}. "
            f"La probabilità di upset, ovvero che vinca il giocatore sfavorito, è del {upset:.0%}. "
            f"Un valore alto indica un match più incerto e spettacolare."
        ),
        "duration": SECTION_DURATIONS["confidence_upset"],
    })

    # 6 – Model Breakdown
    sections.append({
        "section": "model_breakdown",
        "text": (
            f"Vediamo il contributo dei singoli modelli. "
            f"La rete neurale MLP dà il {mlp_prob:.0%} per {p1_name}. "
            f"Il sistema Elo rating dà il {elo_prob:.0%}. "
            f"La combinazione blend, con alpha pari a {alpha:.2f}, produce la previsione finale."
        ),
        "duration": SECTION_DURATIONS["model_breakdown"],
    })

    # 7 – Outro
    sections.append({
        "section": "outro",
        "text": (
            f"Questo è tutto per oggi! Il nostro modello punta su {winner}. "
            f"Seguite il canale per le prossime predizioni ATP. Alla prossima!"
        ),
        "duration": SECTION_DURATIONS["outro"],
    })

    return sections


# ── 2. TTS Generation ────────────────────────────────────────────────────────

async def _tts_single(text: str, output_path: Path) -> Path:
    """Generate a single TTS audio file via edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
    await communicate.save(str(output_path))
    return output_path


def generate_tts(sections: List[Dict[str, str]],
                 output_dir: Optional[Path] = None) -> List[Path]:
    """
    Generate one MP3 per section using edge-tts (Italian neural voice).

    Returns list of MP3 paths in order.
    """
    output_dir = output_dir or TEMP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = []
    for i, section in enumerate(sections):
        path = output_dir / f"tts_{i:02d}_{section['section']}.mp3"
        asyncio.run(_tts_single(section["text"], path))
        audio_paths.append(path)

    return audio_paths


# ── 3. Video Composition ────────────────────────────────────────────────────

def compose_video(
    frames: List[Path],
    audio_files: List[Path],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Compose MP4 from PNG frames + MP3 audio sections using moviepy.
    If fewer frames than audio sections, the last frame is repeated for
    remaining sections.
    """
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

    output_path = output_path or VIDEOS_DIR / "prediction.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map each audio section to a frame (repeat last frame for extra audio sections)
    if len(audio_files) <= len(frames):
        mapped_frames = frames[:len(audio_files)]
    else:
        mapped_frames = [frames[min(i, len(frames) - 1)] for i in range(len(audio_files))]

    clips = []
    audio_clips = []

    for frame_path, audio_path in zip(mapped_frames, audio_files):
        audio_clip = AudioFileClip(str(audio_path))
        duration = audio_clip.duration

        img_clip = (ImageClip(str(frame_path))
                    .with_duration(duration)
                    .resized((VIDEO_WIDTH, VIDEO_HEIGHT))
                    .with_audio(audio_clip))

        clips.append(img_clip)
        audio_clips.append(audio_clip)

    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(str(output_path), fps=VIDEO_FPS, codec="libx264",
                          audio_codec="aac", logger=None)

    # Cleanup
    for c in clips:
        c.close()
    for a in audio_clips:
        a.close()
    video.close()

    return output_path


# ── 4. Thumbnail Generation ─────────────────────────────────────────────────

def generate_thumbnail(
    frame_path: Path,
    winner: str,
    prob_winner: float,
    tournament: str = "ATP",
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a 1280x720 YouTube thumbnail from the prediction card frame."""
    from PIL import Image, ImageDraw, ImageFont

    output_path = output_path or THUMBNAILS_DIR / "thumbnail.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and resize the prediction card
    try:
        img = Image.open(str(frame_path)).resize((THUMB_WIDTH, THUMB_HEIGHT),
                                                  Image.LANCZOS)
    except Exception:
        img = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), "#0D1117")

    draw = ImageDraw.Draw(img)

    # Semi-transparent overlay at bottom
    overlay = Image.new("RGBA", (THUMB_WIDTH, THUMB_HEIGHT), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, THUMB_HEIGHT - 160), (THUMB_WIDTH, THUMB_HEIGHT)],
                           fill=(13, 17, 23, 200))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Bold text
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except (OSError, IOError):
        font_big = ImageFont.load_default()
        font_med = font_big
        font_sm = font_big

    # Winner text
    draw.text((40, THUMB_HEIGHT - 145), f"WIN: {winner}",
              fill=(63, 185, 80), font=font_big)
    draw.text((40, THUMB_HEIGHT - 85), f"{prob_winner:.0%} probabilita",
              fill=(230, 237, 243), font=font_med)
    draw.text((40, THUMB_HEIGHT - 48), tournament,
              fill=(210, 168, 255), font=font_sm)

    img.save(str(output_path))
    return output_path
