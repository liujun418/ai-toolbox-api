from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_tables_created = False


def _ensure_tables():
    """Create tables on first DB request if they don't exist."""
    global _tables_created
    if _tables_created:
        return
    try:
        with engine.connect() as conn:
            existing = inspect(conn).get_table_names()
        if not any(t in existing for t in ("users", "tasks", "transactions")):
            Base.metadata.create_all(bind=engine)
        _tables_created = True
    except Exception:
        pass  # Tables may not exist yet, will retry


def get_db():
    _ensure_tables()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
