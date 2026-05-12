from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from alembic.config import Config
from alembic import command

from app.config import settings
from app.routers import auth, tasks, uploads, payments

app = FastAPI(
    title="AI ToolBox API",
    description="Backend API for AI ToolBox Online",
    version="0.1.0",
)


@app.on_event("startup")
def run_migrations():
    """Auto-run Alembic migrations on startup."""
    import os
    cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
    command.upgrade(cfg, "head")

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
