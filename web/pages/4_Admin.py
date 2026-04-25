"""Admin dashboard: users, revenue, prediction statistics."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from web.auth.manager import is_admin
from web.db.database import init_db, get_session
init_db()
from web.db.models import User, Subscription, PredictionLog, FreeTierUsage

st.set_page_config(page_title="Admin", page_icon="⚙️", layout="wide")

if not is_admin():
    st.error("Accesso riservato agli amministratori.")
    st.stop()

st.title("Admin Dashboard")

session = get_session()

try:
    # ── Overview metrics ──────────────────────────────────────────────────
    st.subheader("Panoramica")

    total_users = session.query(User).count()
    total_preds = session.query(PredictionLog).count()
    active_subs = session.query(Subscription).filter_by(status="active", plan_type="weekly").count()

    # Revenue calculation
    all_subs = session.query(Subscription).all()
    total_revenue = sum(s.amount_paid_eur for s in all_subs if s.amount_paid_eur)

    # Today's predictions
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_preds = session.query(PredictionLog).filter(
        PredictionLog.created_at >= today
    ).count()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Utenti", total_users)
    with col2:
        st.metric("Predizioni Totali", total_preds)
    with col3:
        st.metric("Oggi", today_preds)
    with col4:
        st.metric("Abbonamenti Attivi", active_subs)
    with col5:
        st.metric("Revenue (EUR)", f"{total_revenue:.2f}")

    st.divider()

    # ── Recent users ──────────────────────────────────────────────────────
    st.subheader("Utenti Recenti")
    users = session.query(User).order_by(User.created_at.desc()).limit(50).all()
    user_rows = []
    for u in users:
        sub_status = "Nessuno"
        sub = session.query(Subscription).filter_by(user_id=u.id, status="active").first()
        if sub:
            sub_status = f"{sub.plan_type} (scade: {sub.expires_at.strftime('%Y-%m-%d') if sub.expires_at else 'N/A'})"

        pred_count = session.query(PredictionLog).filter_by(user_id=u.id).count()
        user_rows.append({
            "ID": u.id,
            "Username": u.username,
            "Email": u.email,
            "Admin": "Si" if u.is_admin else "No",
            "Registrato": u.created_at.strftime("%Y-%m-%d"),
            "Abbonamento": sub_status,
            "Predizioni": pred_count,
        })
    st.dataframe(pd.DataFrame(user_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Revenue breakdown ─────────────────────────────────────────────────
    st.subheader("Revenue")
    subs = session.query(Subscription).order_by(Subscription.created_at.desc()).limit(100).all()
    sub_rows = []
    for s in subs:
        user = session.query(User).get(s.user_id)
        sub_rows.append({
            "ID": s.id,
            "Utente": user.username if user else "?",
            "Piano": s.plan_type,
            "Stato": s.status,
            "EUR": f"{s.amount_paid_eur:.2f}",
            "Crediti Rimanenti": s.credits_remaining,
            "Scadenza": s.expires_at.strftime("%Y-%m-%d %H:%M") if s.expires_at else "N/A",
            "Data": s.created_at.strftime("%Y-%m-%d %H:%M"),
        })
    if sub_rows:
        st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Free tier tracking ────────────────────────────────────────────────
    st.subheader("Free Tier Tracking")
    free_users = session.query(FreeTierUsage).order_by(FreeTierUsage.predictions_used.desc()).limit(50).all()
    free_rows = [{
        "IP": f.ip_address,
        "Email": f.email or "Anonimo",
        "Usate": f.predictions_used,
        "Ultima": f.last_prediction.strftime("%Y-%m-%d %H:%M") if f.last_prediction else "N/A",
    } for f in free_users]
    if free_rows:
        st.dataframe(pd.DataFrame(free_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Recent predictions ────────────────────────────────────────────────
    st.subheader("Predizioni Recenti")
    recent = session.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(100).all()
    pred_rows = [{
        "Data": p.created_at.strftime("%Y-%m-%d %H:%M"),
        "Utente": p.user_id or "Anonimo",
        "Match": f"{p.player1} vs {p.player2}",
        "Superficie": p.surface,
        "Confidenza": f"{p.confidence:.0%}",
    } for p in recent]
    if pred_rows:
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

finally:
    session.close()
