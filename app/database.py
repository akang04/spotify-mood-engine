import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.models import Base

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mood_engine.db")
_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    # Migrate existing databases that predate the genres column
    with engine.connect() as conn:
        for ddl in [
            "ALTER TABLE tracks ADD COLUMN genres TEXT",
            "ALTER TABLE tracks ADD COLUMN lastfm_tags TEXT",
        ]:
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception:
                pass  # column already exists


def get_session() -> Session:
    return SessionLocal()
