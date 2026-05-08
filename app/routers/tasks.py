from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.database import get_db
from app.models import TaskStatus
from app.schemas import TaskResponse

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "AI ToolBox API"}


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str, db=Depends(get_db)):
    """Get task status and result URL."""
    from sqlalchemy.orm import joinedload
    from app.models import Task

    task = (
        db.query(Task)
        .filter(Task.id == task_id)
        .options(joinedload(Task.user))
        .first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse(
        id=task.id,
        user_id=task.user_id,
        tool_type=task.tool_type,
        status=task.status,
        output_file_url=task.output_file_url,
        error_message=task.error_message,
        credits_cost=task.credits_cost,
        created_at=task.created_at,
        completed_at=task.completed_at,
    )
