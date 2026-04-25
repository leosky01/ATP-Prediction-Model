"""User authentication and free-tier management."""

import bcrypt
import streamlit as st
from datetime import datetime

from web.db.database import get_session, init_db
from web.db.models import User, Subscription, FreeTierUsage, PredictionLog


FREE_TIER_LIMIT = 3


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def register_user(email: str, username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    session = get_session()
    try:
        if session.query(User).filter_by(email=email).first():
            return False, "Email gia registrata."
        if session.query(User).filter_by(username=username).first():
            return False, "Username gia in uso."

        user = User(
            email=email,
            username=username,
            password_hash=_hash_password(password),
            is_admin=False,
        )
        session.add(user)
        session.commit()
        return True, "Registrazione completata!"
    except Exception as e:
        session.rollback()
        return False, f"Errore: {e}"
    finally:
        session.close()


def login_user(email: str, password: str) -> tuple[bool, str]:
    """Authenticate user. Returns (success, message). Sets session state."""
    session = get_session()
    try:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return False, "Email non trovata."
        if not _check_password(password, user.password_hash):
            return False, "Password errata."

        st.session_state["user_id"] = user.id
        st.session_state["username"] = user.username
        st.session_state["email"] = user.email
        st.session_state["is_admin"] = user.is_admin
        return True, f"Bentornato, {user.username}!"
    finally:
        session.close()


def logout_user():
    """Clear session state."""
    for key in ["user_id", "username", "email", "is_admin"]:
        st.session_state.pop(key, None)


def is_logged_in() -> bool:
    return "user_id" in st.session_state


def get_current_user_id() -> int | None:
    return st.session_state.get("user_id")


def is_admin() -> bool:
    return st.session_state.get("is_admin", False)


# ── Subscription / credits ──────────────────────────────────────────────

def has_active_subscription(user_id: int) -> bool:
    """Check if user has an active weekly subscription."""
    session = get_session()
    try:
        sub = (
            session.query(Subscription)
            .filter_by(user_id=user_id, status="active")
            .filter(Subscription.plan_type == "weekly")
            .filter(Subscription.expires_at > datetime.utcnow())
            .first()
        )
        return sub is not None
    finally:
        session.close()


def get_credits_remaining(user_id: int) -> int:
    """Get total pay-per-use credits for a user."""
    session = get_session()
    try:
        subs = (
            session.query(Subscription)
            .filter_by(user_id=user_id, status="active")
            .filter(Subscription.plan_type == "pay_per_use")
            .filter(Subscription.credits_remaining > 0)
            .all()
        )
        return sum(s.credits_remaining for s in subs)
    finally:
        session.close()


def consume_credit(user_id: int) -> bool:
    """Consume one pay-per-use credit. Returns True if successful."""
    session = get_session()
    try:
        sub = (
            session.query(Subscription)
            .filter_by(user_id=user_id, status="active")
            .filter(Subscription.plan_type == "pay_per_use")
            .filter(Subscription.credits_remaining > 0)
            .first()
        )
        if sub:
            sub.credits_remaining -= 1
            session.commit()
            return True
        return False
    finally:
        session.close()


def add_credits(user_id: int, amount: int, paypal_order_id: str = "", euros: float = 0):
    """Add pay-per-use credits after PayPal payment."""
    session = get_session()
    try:
        sub = Subscription(
            user_id=user_id,
            plan_type="pay_per_use",
            paypal_order_id=paypal_order_id,
            status="active",
            credits_remaining=amount,
            amount_paid_eur=euros,
        )
        session.add(sub)
        session.commit()
    finally:
        session.close()


def activate_weekly_subscription(user_id: int, paypal_order_id: str = ""):
    """Activate a weekly subscription."""
    session = get_session()
    try:
        # Deactivate any existing active subscription
        existing = session.query(Subscription).filter_by(
            user_id=user_id, status="active", plan_type="weekly"
        ).all()
        for s in existing:
            s.status = "cancelled"

        from datetime import timedelta
        sub = Subscription(
            user_id=user_id,
            plan_type="weekly",
            paypal_order_id=paypal_order_id,
            status="active",
            expires_at=datetime.utcnow() + timedelta(days=7),
            amount_paid_eur=5.00,
        )
        session.add(sub)
        session.commit()
    finally:
        session.close()


# ── Free tier (anonymous, IP-based) ────────────────────────────────────

def get_free_predictions_used(ip_address: str) -> int:
    session = get_session()
    try:
        record = session.query(FreeTierUsage).filter_by(ip_address=ip_address).first()
        return record.predictions_used if record else 0
    finally:
        session.close()


def increment_free_usage(ip_address: str):
    session = get_session()
    try:
        record = session.query(FreeTierUsage).filter_by(ip_address=ip_address).first()
        if record:
            record.predictions_used += 1
            record.last_prediction = datetime.utcnow()
        else:
            record = FreeTierUsage(
                ip_address=ip_address,
                predictions_used=1,
            )
            session.add(record)
        session.commit()
    finally:
        session.close()


# ── Can predict? ────────────────────────────────────────────────────────

def can_predict(ip_address: str = "") -> tuple[bool, str]:
    """
    Check if the current user/session can make a prediction.
    Returns (can_predict, reason).
    """
    if is_logged_in():
        user_id = get_current_user_id()
        if has_active_subscription(user_id):
            return True, "Abbonamento attivo"
        credits = get_credits_remaining(user_id)
        if credits > 0:
            return True, f"{credits} predizioni rimaste"
        return False, "Nessun credito — acquista predizioni o attiva l'abbonamento"
    else:
        if not ip_address:
            return True, "Primo accesso"
        used = get_free_predictions_used(ip_address)
        if used < FREE_TIER_LIMIT:
            return True, f"{FREE_TIER_LIMIT - used} predizioni gratuite rimaste"
        return False, "Limite gratuito raggiunto — registrati per continuare"


def record_prediction_after(user_id: int | None, ip_address: str,
                             player1: str, player2: str,
                             rank1: int, rank2: int,
                             odds1: float, odds2: float,
                             surface: str, tournament: str,
                             result_json: str, confidence: float):
    """Log a prediction and consume credits / free tier as needed."""
    session = get_session()
    try:
        log = PredictionLog(
            user_id=user_id,
            session_id=st.session_state.get("session_id", ""),
            ip_address=ip_address,
            player1=player1, player2=player2,
            rank1=rank1, rank2=rank2,
            odds1=odds1, odds2=odds2,
            surface=surface, tournament=tournament,
            result_json=result_json,
            confidence=confidence,
        )
        session.add(log)
        session.commit()
    finally:
        session.close()

    # Consume credit or free tier
    if user_id:
        if not has_active_subscription(user_id):
            consume_credit(user_id)
    else:
        increment_free_usage(ip_address)


# ── Seed admin ──────────────────────────────────────────────────────────

def ensure_admin_exists():
    """Create default admin user if no admin exists."""
    session = get_session()
    try:
        if not session.query(User).filter_by(is_admin=True).first():
            import os
            admin_email = os.environ.get("ADMIN_EMAIL", "admin@atppredictor.com")
            admin_pass = os.environ.get("ADMIN_PASSWORD", "changeme123")
            admin = User(
                email=admin_email,
                username="admin",
                password_hash=_hash_password(admin_pass),
                is_admin=True,
            )
            session.add(admin)
            session.commit()
    finally:
        session.close()
