from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import auth, tasks, uploads

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
app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(uploads.router)


@app.get("/")
async def root():
    return {"message": "AI ToolBox API is running"}
