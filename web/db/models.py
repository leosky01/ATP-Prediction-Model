"""SQLAlchemy ORM models for the ATP Predictor web app."""

from datetime import datetime, timedelta

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    subscriptions = relationship("Subscription", back_populates="user", lazy="dynamic")
    predictions = relationship("PredictionLog", back_populates="user", lazy="dynamic")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    plan_type = Column(String(50), nullable=False)  # "weekly" or "pay_per_use"
    paypal_order_id = Column(String(255), nullable=True)
    status = Column(String(50), default="active", nullable=False)  # active/cancelled/expired
    expires_at = Column(DateTime, nullable=True)
    credits_remaining = Column(Integer, default=0, nullable=False)
    amount_paid_eur = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="subscriptions")


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    player1 = Column(String(200), nullable=False)
    player2 = Column(String(200), nullable=False)
    rank1 = Column(Integer, nullable=False)
    rank2 = Column(Integer, nullable=False)
    odds1 = Column(Float, nullable=False)
    odds2 = Column(Float, nullable=False)
    surface = Column(String(50), nullable=False)
    tournament = Column(String(255), nullable=False)
    result_json = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="predictions")


class FreeTierUsage(Base):
    __tablename__ = "free_tier_tracking"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ip_address = Column(String(45), nullable=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    predictions_used = Column(Integer, default=0, nullable=False)
    last_prediction = Column(DateTime, default=datetime.utcnow, nullable=False)
