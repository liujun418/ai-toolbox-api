from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.models import Base
from app.routers import tasks, uploads

app = FastAPI(
    title="AI ToolBox API",
    description="Backend API for AI ToolBox Online",
    version="0.1.0",
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
app.include_router(tasks.router)
app.include_router(uploads.router)


@app.get("/")
async def root():
    return {"message": "AI ToolBox API is running"}


@app.get("/api/init-db")
async def init_db():
    """Temporary endpoint to create tables. Remove after initial setup."""
    try:
        from sqlalchemy import inspect
        with engine.connect() as conn:
            existing = inspect(conn).get_table_names()
        if not any(t in existing for t in ("users", "tasks", "transactions")):
            Base.metadata.create_all(bind=engine)
            return {"status": "created", "message": "Tables created successfully"}
        return {"status": "exists", "tables": existing}
    except Exception as e:
        return {"status": "error", "message": str(e)}
