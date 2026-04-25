"""Helper functions for loading player/tournament lists from CSV for autocomplete."""

import pandas as pd
from pathlib import Path
import streamlit as st


@st.cache_data
def load_match_data():
    """Load the ATP dataset once and cache it."""
    csv_path = Path(__file__).resolve().parents[2] / "data" / "atp_tennis_cleaned.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df


@st.cache_data
def get_player_list() -> list[str]:
    """Return sorted list of unique player names for selectbox autocomplete."""
    df = load_match_data()
    players = sorted(set(df["Player_1"].dropna().unique()) | set(df["Player_2"].dropna().unique()))
    return players


@st.cache_data
def get_tournament_list() -> list[str]:
    """Return sorted list of unique tournament names."""
    df = load_match_data()
    tournaments = sorted(df["Tournament"].dropna().unique().tolist())
    return tournaments


@st.cache_data
def get_series_list() -> list[str]:
    """Return sorted list of unique tournament series/categories."""
    df = load_match_data()
    series = sorted(df["Series"].dropna().unique().tolist())
    return series


@st.cache_data
def get_surface_list() -> list[str]:
    """Return sorted list of surfaces."""
    return ["Hard", "Clay", "Grass"]
