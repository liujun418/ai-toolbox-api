from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.routers import auth, tasks, uploads, payments


def _ensure_db_columns():
    """Add missing columns to users table (lightweight alternative to alembic)."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("users")}
    needed = {
        "email_verified": "BOOLEAN DEFAULT FALSE",
        "verification_token": "VARCHAR(255)",
        "reset_token": "VARCHAR(255)",
        "reset_token_expires": "TIMESTAMP WITH TIME ZONE",
    }
    with engine.begin() as conn:
        for col_name, col_type in needed.items():
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_db_columns()
    yield


app = FastAPI(
    title="AI ToolBox API",
    description="Backend API for AI ToolBox Online",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(uploads.router)
app.include_router(payments.router)


@app.get("/")
async def root():
    return {"message": "AI ToolBox API is running"}
