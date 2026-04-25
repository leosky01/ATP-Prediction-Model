"""History page: shows user's past predictions."""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from web.auth.manager import is_logged_in, get_current_user_id, is_admin
from web.db.database import get_session
from web.db.models import PredictionLog

st.set_page_config(page_title="Storico", page_icon="📋", layout="wide")

st.title("Storico Predizioni")

if not is_logged_in():
    st.warning("Effettua il login per vedere il tuo storico.")
    if st.button("Vai ad Account"):
        st.switch_page("pages/3_Account.py")
    st.stop()

user_id = get_current_user_id()
session = get_session()

try:
    query = session.query(PredictionLog).order_by(PredictionLog.created_at.desc())

    if not is_admin():
        query = query.filter_by(user_id=user_id)

    logs = query.limit(200).all()

    if not logs:
        st.info("Nessuna predizione nel database.")
        st.stop()

    # Convert to DataFrame
    rows = []
    for log in logs:
        rows.append({
            "Data": log.created_at.strftime("%Y-%m-%d %H:%M"),
            "Giocatore 1": log.player1,
            "Giocatore 2": log.player2,
            "Rank 1": log.rank1,
            "Rank 2": log.rank2,
            "Quota 1": log.odds1,
            "Quota 2": log.odds2,
            "Superficie": log.surface,
            "Torneo": log.tournament,
            "Confidenza": f"{log.confidence:.0%}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(f"Mostrate le ultime {len(rows)} predizioni")

    # Stats summary
    if is_admin():
        st.divider()
        st.subheader("Statistiche Admin")
        total = session.query(PredictionLog).count()
        unique_users = session.query(PredictionLog.user_id).distinct().count()
        st.metric("Totale predizioni", total)
        st.metric("Utenti unici", unique_users)

finally:
    session.close()
