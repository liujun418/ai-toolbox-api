import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.routers import auth, tasks, uploads, payments, admin, lateral_thinking, bing_wallpaper, nasa_apod, crypto_price, random_quote

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _ensure_db_columns():
    """Add missing columns and tables (lightweight alternative to alembic)."""
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
        "role": "VARCHAR(20) DEFAULT 'user'",
        "subscription_tier": "VARCHAR(50)",
        "subscription_end_date": "TIMESTAMP WITH TIME ZONE",
        "last_checkin": "TIMESTAMP WITH TIME ZONE",
        "checkin_streak": "INTEGER DEFAULT 0",
        "referral_code": "VARCHAR(20)",
        "referred_by": "VARCHAR(36)",
    }
    with engine.begin() as conn:
        for col_name, col_type in needed.items():
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
        # Normalize role column
        col_type = conn.execute(text(
            "SELECT data_type FROM information_schema.columns WHERE table_name='users' AND column_name='role'"
        )).scalar()
        if col_type == "USER-DEFINED":
            conn.execute(text("ALTER TABLE users ALTER COLUMN role TYPE VARCHAR(20) USING role::text"))
        conn.execute(text("UPDATE users SET role = 'admin' WHERE role = 'ADMIN'"))
        conn.execute(text("UPDATE users SET role = 'user' WHERE role = 'USER'"))

        # Ensure suggestions table exists
        if "suggestions" not in inspector.get_table_names():
            conn.execute(text(
                "CREATE TABLE suggestions ("
                "id VARCHAR(36) PRIMARY KEY, "
                "text TEXT NOT NULL, "
                "read BOOLEAN DEFAULT FALSE, "
                "created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
                ")"
            ))

        # Ensure blog_posts table exists
        if "blog_posts" not in inspector.get_table_names():
            conn.execute(text(
                "CREATE TABLE blog_posts ("
                "id VARCHAR(36) PRIMARY KEY, "
                "slug VARCHAR(255) UNIQUE NOT NULL, "
                "title VARCHAR(500) NOT NULL, "
                "description VARCHAR(1000) NOT NULL, "
                "content TEXT NOT NULL, "
                "category VARCHAR(100) NOT NULL, "
                "tags TEXT DEFAULT '', "
                "related_tools TEXT, "
                "published BOOLEAN DEFAULT TRUE, "
                "created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), "
                "updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
                ")"
            ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_db_columns()
    try:
        from app.services.style_references import init_style_references
        await init_style_references()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to initialize style reference images. Style Transfer will be unavailable.",
            exc_info=True,
        )
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
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "https://toolboxonline.club",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(uploads.router)
app.include_router(payments.router)
app.include_router(admin.router)
app.include_router(admin.blog_public)
app.include_router(lateral_thinking.router)
app.include_router(bing_wallpaper.router)
app.include_router(nasa_apod.router)
app.include_router(crypto_price.router)
app.include_router(random_quote.router)


@app.get("/")
async def root():
    return {"message": "AI ToolBox API is running"}
