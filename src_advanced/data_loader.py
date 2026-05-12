"""
Data loading, TML download, name mapping, merge, and target parsing.

Downloads TML match CSVs from GitHub, builds a name mapping between the
existing dataset ("Dimitrov G.") and TML ("Grigor Dimitrov"), merges on
date + sorted player names, and parses prediction targets.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import (
    EXISTING_DATASET, MERGED_DATASET_PATH, MODELS_ADV_DIR, NAME_MAPPING_PATH,
    TML_BASE_URL, TML_BIO_URL, TML_DATABASE_CSV, TML_DIR, TML_RAW_DIR, TML_YEARS,
)


# ── TML CSV download ─────────────────────────────────────────────────────────

def download_tml_years(
    years: Optional[List[int]] = None,
    force: bool = False,
) -> List[Path]:
    """Download TML match CSVs for *years* into data/tml/raw/."""
    TML_RAW_DIR.mkdir(parents=True, exist_ok=True)
    years = years or TML_YEARS
    paths: List[Path] = []
    for year in tqdm(years, desc="Downloading TML CSVs"):
        dest = TML_RAW_DIR / f"{year}.csv"
        if dest.exists() and not force:
            paths.append(dest)
            continue
        url = TML_BASE_URL.format(year=year)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            paths.append(dest)
        except requests.RequestException as e:
            print(f"  Warning: failed to download {year}: {e}")
    return paths


def download_tml_bio(force: bool = False) -> Path:
    """Download ATP_Database.csv (player bios) into data/tml/."""
    TML_DIR.mkdir(parents=True, exist_ok=True)
    dest = TML_DATABASE_CSV
    if dest.exists() and not force:
        return dest
    r = requests.get(TML_BIO_URL, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# ── Load TML data ────────────────────────────────────────────────────────────

def load_tml_matches(years: Optional[List[int]] = None) -> pd.DataFrame:
    """Load and concatenate TML match CSVs. Returns DataFrame with all TML columns."""
    TML_RAW_DIR.mkdir(parents=True, exist_ok=True)
    years = years or TML_YEARS
    frames: List[pd.DataFrame] = []
    for year in years:
        path = TML_RAW_DIR / f"{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Parse tourney_date to datetime
    out["tourney_date"] = pd.to_datetime(
        out["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    return out


def load_tml_bio() -> pd.DataFrame:
    """Load ATP_Database.csv with player bio info."""
    if not TML_DATABASE_CSV.exists():
        download_tml_bio()
    # TML bio file may use Windows-1252 encoding
    try:
        return pd.read_csv(TML_DATABASE_CSV, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(TML_DATABASE_CSV, low_memory=False, encoding="windows-1252")


# ── Name mapping ─────────────────────────────────────────────────────────────

def _normalise_existing_name(name: str) -> str:
    """Normalise existing dataset name: "Dimitrov G." -> "dimitrov g"."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower().rstrip(".")


def _normalise_tml_name(name: str) -> str:
    """Normalise TML name: "Grigor Dimitrov" -> "grigor dimitrov"."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def build_name_mapping(
    existing_df: pd.DataFrame,
    tml_bio: pd.DataFrame,
) -> Dict[str, str]:
    """
    Build mapping: existing_name -> tml_name.

    Existing dataset uses "LastName Initial." format (e.g. "Dimitrov G.")
    TML uses "First Last" format (e.g. "Grigor Dimitrov")

    Strategy:
    1. For each existing name, extract last name and optional initial
    2. Look up in bio where player/atpname's last word matches
    3. If multiple matches, use initial to disambiguate
    4. Fall back to fuzzy matching
    """
    mapping: Dict[str, str] = {}

    # Collect all existing player names
    existing_names = set(existing_df["Player_1"].dropna()) | set(existing_df["Player_2"].dropna())

    # Build lookup from bio: (last_name_lower -> [(full_name, first_initial)])
    bio_player_col = "player" if "player" in tml_bio.columns else None
    if bio_player_col is None:
        return mapping

    bio_lookup: Dict[str, List[Tuple[str, str]]] = {}  # last_lower -> [(full, first_initial)]
    for name in tml_bio[bio_player_col].dropna().astype(str):
        name = name.strip()
        if not name:
            continue
        parts = name.split()
        last_lower = parts[-1].lower()
        first_initial = parts[0][0].lower() if parts else ""
        bio_lookup.setdefault(last_lower, []).append((name, first_initial))

    # Also build a set of all TML names for fuzzy fallback
    all_tml_names = list(tml_bio[bio_player_col].dropna().astype(str))

    for ename in tqdm(existing_names, desc="Building name mapping"):
        ename_str = str(ename).strip()
        if not ename_str:
            continue

        # Parse existing format: "Dimitrov G." -> last="Dimitrov", initial="G"
        # Format is always: "LastName X." where X is one or more initials
        clean = ename_str.rstrip(".")
        parts = clean.split()
        if len(parts) < 1:
            continue

        last_name = parts[0]
        # Extract initial(s) after last name
        existing_initial = parts[1][0].lower() if len(parts) > 1 and parts[1] else None

        last_lower = last_name.lower()

        # Find candidates in bio with matching last name
        candidates = bio_lookup.get(last_lower, [])

        if len(candidates) == 0:
            # No match by last name -> fuzzy fallback
            matches = difflib.get_close_matches(
                ename_str, all_tml_names, n=1, cutoff=0.6
            )
            if matches:
                mapping[ename_str] = matches[0]
            continue

        if len(candidates) == 1:
            mapping[ename_str] = candidates[0][0]
            continue

        # Multiple candidates: disambiguate by first initial
        if existing_initial is not None:
            initial_matches = [
                (full, init) for full, init in candidates
                if init == existing_initial
            ]
            if len(initial_matches) == 1:
                mapping[ename_str] = initial_matches[0][0]
                continue
            if len(initial_matches) > 1:
                # Still ambiguous, pick first
                mapping[ename_str] = initial_matches[0][0]
                continue

        # Fallback: pick the most common name (first in list)
        mapping[ename_str] = candidates[0][0]

    return mapping


def get_name_mapping(
    existing_df: pd.DataFrame,
    force: bool = False,
) -> Dict[str, str]:
    """Get or build + cache the name mapping."""
    if NAME_MAPPING_PATH.exists() and not force:
        return joblib.load(NAME_MAPPING_PATH)

    tml_bio = load_tml_bio()
    mapping = build_name_mapping(existing_df, tml_bio)
    NAME_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mapping, NAME_MAPPING_PATH)
    print(f"Name mapping: {len(mapping)} players mapped "
          f"(saved to {NAME_MAPPING_PATH})")
    return mapping


# ── Merge ────────────────────────────────────────────────────────────────────

def _prepare_existing(existing_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Prepare existing dataset for merge: add merge keys using TML names."""
    df = existing_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Normalise to Monday of the week for merge (matches TML tourney_date convention)
    df["_merge_week"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["_merge_week"] = df["_merge_week"].dt.date

    # Map existing player names -> TML names for pair key
    def to_tml_key(name: str) -> str:
        if not isinstance(name, str):
            return str(name)
        return mapping.get(name, name).strip().lower()

    df["_p1_tml"] = df["Player_1"].apply(to_tml_key)
    df["_p2_tml"] = df["Player_2"].apply(to_tml_key)
    df["_pair_key"] = df.apply(
        lambda r: "||".join(sorted([r["_p1_tml"], r["_p2_tml"]])),
        axis=1,
    )
    df.drop(columns=["_p1_tml", "_p2_tml"], inplace=True)
    return df


def _prepare_tml(tml_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Prepare TML data for merge: normalise names, create merge keys."""
    df = tml_df.copy()

    # Use TML names directly (lower-cased) for pair key — same space as existing
    def tml_key(name: str) -> str:
        if not isinstance(name, str):
            return str(name)
        return name.strip().lower()

    # Normalise to Monday of the week (tourney_date should already be Monday,
    # but we normalise defensively)
    df["_merge_week"] = df["tourney_date"] - pd.to_timedelta(
        df["tourney_date"].dt.weekday, unit="D"
    )
    df["_merge_week"] = df["_merge_week"].dt.date

    # Duplicate rows with +7 days to cover second week of 2-week tournaments
    df_week2 = df.copy()
    df_week2["_merge_week"] = df_week2["_merge_week"].apply(
        lambda d: d + pd.Timedelta(days=7) if pd.notna(d) else d
    )
    df = pd.concat([df, df_week2], ignore_index=True)

    df["_pair_key"] = df.apply(
        lambda r: "||".join(sorted([tml_key(r["winner_name"]), tml_key(r["loser_name"])])),
        axis=1,
    )
    return df


def merge_datasets(
    existing_df: Optional[pd.DataFrame] = None,
    tml_df: Optional[pd.DataFrame] = None,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Left-join existing dataset with TML stats on (date, sorted player pair).

    Returns the merged DataFrame with all existing columns + TML stat columns.
    """
    if existing_df is None:
        existing_df = pd.read_csv(EXISTING_DATASET, parse_dates=["Date"])
    if mapping is None:
        mapping = get_name_mapping(existing_df)

    # Download + load TML if not provided
    if tml_df is None:
        download_tml_years(force=False)
        tml_df = load_tml_matches()

    existing = _prepare_existing(existing_df, mapping)
    tml = _prepare_tml(tml_df, mapping)

    # Select columns to bring from TML (stats + match info)
    tml_merge_cols = [
        "_merge_week", "_pair_key",
        "best_of", "minutes", "score", "round",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "winner_name", "loser_name", "winner_hand", "loser_hand",
        "winner_ht", "loser_ht", "winner_age", "loser_age",
        "tourney_level", "draw_size", "indoor",
    ]
    # Only keep columns that exist
    tml_merge_cols = [c for c in tml_merge_cols if c in tml.columns]
    tml_subset = tml[["_merge_week", "_pair_key"] + [c for c in tml_merge_cols if c not in ("_merge_week", "_pair_key")]].copy()

    # Drop duplicate pairs (keep first match in same week)
    tml_subset = tml_subset.drop_duplicates(subset=["_merge_week", "_pair_key"], keep="first")

    merged = existing.merge(
        tml_subset,
        on=["_merge_week", "_pair_key"],
        how="left",
    )

    # Drop helper columns
    merged.drop(columns=["_merge_week", "_pair_key"], inplace=True, errors="ignore")

    # Align winner/loser stats with Player_1/Player_2 perspective
    merged = _align_stats(merged)

    match_pct = merged["p1_ace"].notna().mean() * 100
    print(f"Merged dataset: {len(merged):,} rows, TML stats coverage: {match_pct:.1f}%")

    return merged


def _align_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align winner/loser stats to Player_1/Player_2 perspective.

    If Player_1 == winner_name (mapped), winner stats -> p1 stats.
    Otherwise, swap them.
    """
    df = df.copy()

    # Determine which rows have Player_1 as winner
    # We use the original Winner column: if Winner == Player_1, then p1 = winner
    is_p1_winner = df["Winner"] == df["Player_1"]

    stat_pairs = [
        ("w_ace", "l_ace", "p1_ace", "p2_ace"),
        ("w_df", "l_df", "p1_df", "p2_df"),
        ("w_svpt", "l_svpt", "p1_svpt", "p2_svpt"),
        ("w_1stIn", "l_1stIn", "p1_1stIn", "p2_1stIn"),
        ("w_1stWon", "l_1stWon", "p1_1stWon", "p2_1stWon"),
        ("w_2ndWon", "l_2ndWon", "p1_2ndWon", "p2_2ndWon"),
        ("w_SvGms", "l_SvGms", "p1_SvGms", "p2_SvGms"),
        ("w_bpSaved", "l_bpSaved", "p1_bpSaved", "p2_bpSaved"),
        ("w_bpFaced", "l_bpFaced", "p1_bpFaced", "p2_bpFaced"),
        ("winner_hand", "loser_hand", "p1_hand", "p2_hand"),
        ("winner_ht", "loser_ht", "p1_ht", "p2_ht"),
        ("winner_age", "loser_age", "p1_age", "p2_age"),
    ]

    for w_col, l_col, p1_col, p2_col in stat_pairs:
        if w_col not in df.columns:
            continue
        df[p1_col] = np.where(is_p1_winner, df[w_col], df[l_col])
        df[p2_col] = np.where(is_p1_winner, df[l_col], df[w_col])

    # Keep minutes as-is (match-level, not player-level)
    # Keep score, best_of as-is
    return df


# ── Target parsing ───────────────────────────────────────────────────────────

def parse_total_games(score: str) -> Optional[int]:
    """
    Parse a tennis score string and return total games played.

    Examples: "6-4 6-2" -> 18, "6-4 4-6 7-5" -> 32
    Handles retirements ("6-2 3-0 RET") by summing only valid sets.
    """
    if not isinstance(score, str) or not score.strip():
        return None
    total = 0
    for set_score in score.split():
        set_score = set_score.strip().upper()
        if set_score in ("RET", "W/O", "W/O", "DEF", "ABD", "WALKOVER"):
            break
        # Match patterns like "6-4", "7-6(3)", "6-4"
        m = re.match(r"(\d+)-(\d+)", set_score)
        if m:
            total += int(m.group(1)) + int(m.group(2))
        else:
            break  # stop at first unparsable token (e.g., RET)
    return total if total > 0 else None


def parse_set_score(score: str, best_of: int = 3) -> Optional[str]:
    """
    Parse score string and return set result like "2-0", "2-1", "3-2".

    Returns the result from Player_1 (winner of the match) perspective.
    For the existing dataset, Player_1 is always the winner, so it's
    winner_sets-loser_sets.
    """
    if not isinstance(score, str) or not score.strip():
        return None
    sets_won = 0
    sets_lost = 0
    for set_score in score.split():
        set_score = set_score.strip().upper()
        if set_score in ("RET", "W/O", "W/O", "DEF", "ABD", "WALKOVER"):
            break
        m = re.match(r"(\d+)-(\d+)", set_score)
        if m:
            g1, g2 = int(m.group(1)), int(m.group(2))
            if g1 > g2:
                sets_won += 1
            else:
                sets_lost += 1
    if sets_won == 0 and sets_lost == 0:
        return None
    return f"{sets_won}-{sets_lost}"


def compute_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all target columns to the merged DataFrame."""
    df = df.copy()

    # Total games
    df["total_games"] = df["score"].apply(parse_total_games)

    # Set score (from winner perspective)
    best_of = df["best_of"].fillna(3).astype(int)
    df["set_score"] = [parse_set_score(s, b) for s, b in zip(df["score"], best_of)]

    # Over/under (configurable thresholds)
    from .config import GAMES_THRESHOLD_BO3, GAMES_THRESHOLD_BO5
    df["games_threshold"] = np.where(
        df["best_of"].fillna(3) == 5,
        GAMES_THRESHOLD_BO5,
        GAMES_THRESHOLD_BO3,
    )
    df["over_under"] = (
        df["total_games"] > df["games_threshold"]
    ).astype(int)  # 1 = OVER

    # Duration is already in "minutes" column

    return df


# ── Full pipeline ────────────────────────────────────────────────────────────

def build_merged_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Full pipeline: download -> name mapping -> merge -> parse targets.
    """
    print("=" * 60)
    print("BUILDING MERGED DATASET")
    print("=" * 60)

    # 1. Download
    print("\n1) Downloading TML data...")
    download_tml_years(force=force_download)
    download_tml_bio(force=force_download)

    # 2. Load existing
    print("\n2) Loading existing dataset...")
    existing = pd.read_csv(EXISTING_DATASET, parse_dates=["Date"])
    print(f"   {len(existing):,} existing matches")

    # 3. Name mapping
    print("\n3) Building name mapping...")
    mapping = get_name_mapping(existing, force=force_download)
    print(f"   {len(mapping):,} names mapped")

    # 4. Load TML
    print("\n4) Loading TML matches...")
    tml = load_tml_matches()
    print(f"   {len(tml):,} TML matches loaded")

    # 5. Merge
    print("\n5) Merging datasets...")
    merged = merge_datasets(existing, tml, mapping)

    # 6. Parse targets
    print("\n6) Parsing targets...")
    merged = compute_target_columns(merged)

    # 7. Save
    MODELS_ADV_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_DATASET_PATH, index=False)
    print(f"\n   Saved to {MERGED_DATASET_PATH}")

    # Summary
    n_total = len(merged)
    n_with_stats = merged["p1_ace"].notna().sum()
    n_with_duration = merged["minutes"].notna().sum()
    n_with_score = merged["total_games"].notna().sum()

    print(f"\n   Summary:")
    print(f"   Total matches:        {n_total:,}")
    print(f"   With stats:           {n_with_stats:,} ({n_with_stats/n_total*100:.1f}%)")
    print(f"   With duration:        {n_with_duration:,} ({n_with_duration/n_total*100:.1f}%)")
    print(f"   With score parsed:    {n_with_score:,} ({n_with_score/n_total*100:.1f}%)")

    return merged


if __name__ == "__main__":
    build_merged_dataset()
