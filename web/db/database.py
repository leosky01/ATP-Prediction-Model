"""SQLite database engine + session factory."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

DB_DIR = Path(__file__).resolve().parents[2] / "web_db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "atp_predictor.db"

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Return a new DB session. Caller must close it."""
    return SessionLocal()
