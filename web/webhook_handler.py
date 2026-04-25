"""Flask webhook handler for PayPal events (runs on separate port).

Run alongside Streamlit:
    python -m web.webhook_handler

Or via supervisor / systemd in production.
"""

import json
import os
import sys
from pathlib import Path

from flask import Flask, request, jsonify

# Ensure project root on path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from web.db.database import init_db, get_session
from web.db.models import User, Subscription
from web.auth.paypal_client import verify_webhook_signature

app = Flask(__name__)

# Init DB on startup
init_db()


@app.route("/webhooks/paypal", methods=["POST"])
def paypal_webhook():
    """Handle incoming PayPal webhook events."""
    payload = request.get_data(as_text=True)
    headers = dict(request.headers)

    # Verify signature
    if not verify_webhook_signature(headers, payload.encode()):
        return jsonify({"error": "Invalid signature"}), 401

    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON"}), 400

    event_type = event.get("event_type", "")

    if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
        handle_subscription_activated(event)
    elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
        handle_subscription_cancelled(event)
    elif event_type == "BILLING.SUBSCRIPTION.EXPIRED":
        handle_subscription_expired(event)
    elif event_type == "PAYMENT.SALE.COMPLETED":
        handle_payment_completed(event)
    else:
        print(f"Unhandled PayPal event: {event_type}")

    return jsonify({"status": "ok"}), 200


def handle_subscription_activated(event: dict):
    """Activate weekly subscription for user."""
    resource = event.get("resource", {})
    sub_id = resource.get("id", "")
    custom_id = resource.get("custom_id", "")

    # In production, custom_id should be the user_id
    if not custom_id:
        print(f"No custom_id in subscription {sub_id}")
        return

    session = get_session()
    try:
        user_id = int(custom_id)
        user = session.query(User).get(user_id)
        if not user:
            print(f"User {user_id} not found for subscription {sub_id}")
            return

        # Cancel existing active weekly subs
        existing = session.query(Subscription).filter_by(
            user_id=user_id, status="active", plan_type="weekly"
        ).all()
        for s in existing:
            s.status = "cancelled"

        from datetime import datetime, timedelta
        sub = Subscription(
            user_id=user_id,
            plan_type="weekly",
            paypal_order_id=sub_id,
            status="active",
            expires_at=datetime.utcnow() + timedelta(days=7),
            amount_paid_eur=5.00,
        )
        session.add(sub)
        session.commit()
        print(f"Activated weekly subscription for user {user_id}")
    except Exception as e:
        session.rollback()
        print(f"Error activating subscription: {e}")
    finally:
        session.close()


def handle_subscription_cancelled(event: dict):
    resource = event.get("resource", {})
    sub_id = resource.get("id", "")
    session = get_session()
    try:
        sub = session.query(Subscription).filter_by(paypal_order_id=sub_id).first()
        if sub:
            sub.status = "cancelled"
            session.commit()
    finally:
        session.close()


def handle_subscription_expired(event: dict):
    resource = event.get("resource", {})
    sub_id = resource.get("id", "")
    session = get_session()
    try:
        sub = session.query(Subscription).filter_by(paypal_order_id=sub_id).first()
        if sub:
            sub.status = "expired"
            session.commit()
    finally:
        session.close()


def handle_payment_completed(event: dict):
    """Handle recurring payment for subscription."""
    print(f"Payment completed: {event.get('id', 'unknown')}")
    # Could extend subscription here if needed


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "paypal-webhook"})


if __name__ == "__main__":
    port = int(os.environ.get("WEBHOOK_PORT", 5001))
    print(f"PayPal webhook handler starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
