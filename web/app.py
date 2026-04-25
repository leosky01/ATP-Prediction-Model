"""ATP Predictor Web App — Entry point.

Run with: streamlit run web/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from web.db.database import init_db
from web.auth.manager import ensure_admin_exists

# ── Initialize database and admin ────────────────────────────────────────
init_db()
ensure_admin_exists()

# ── Load custom CSS ──────────────────────────────────────────────────────
css_path = Path(__file__).parent / "static" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state defaults ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())

# ── Main page ────────────────────────────────────────────────────────────
st.title("ATP Match Predictor")
st.markdown(
    "Modello di intelligenza artificiale per la predizione delle partite ATP.<br>"
    "Inserisci i dati del match nella pagina **Predizione** per ottenere analisi dettagliate.",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Modalita", "MLP + Elo + Blend")
with col2:
    st.metric("Grafici", "8 interattivi (Plotly)")
with col3:
    st.metric("Free Tier", "3 predizioni gratuite")

st.divider()

st.markdown(
    "### Come funziona\n"
    "1. Vai alla pagina **Predizione** e inserisci i dati del match\n"
    "2. Il modello calcola le probabilita usando MLP (rete neurale) + Elo\n"
    "3. Visualizza 8 grafici interattivi: trend Elo, radar superficie, e altro\n\n"
    "**Piano gratuito**: 3 predizioni senza registrazione. "
    "Registrati per sbloccare piu predizioni o attiva l'abbonamento settimanale."
)

# Quick-start buttons
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Inizia a Predire", type="primary", use_container_width=True):
        st.switch_page("pages/1_Predict.py")
with col_b:
    if st.button("Accedi / Registrati", use_container_width=True):
        st.switch_page("pages/3_Account.py")
