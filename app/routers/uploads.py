"""File upload and AI processing endpoints."""

import uuid
from datetime import datetime, UTC

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Task, TaskStatus, User
from app.routers.auth import get_current_user
from app.schemas import TaskResponse, CREDIT_COSTS
from app.services.image_processing import run_background_remover_local
from app.services.storage import (
    generate_upload_key,
    generate_download_key,
    upload_file,
    generate_presigned_url,
)

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/{tool_type}", response_model=TaskResponse)
async def upload_and_process(
    tool_type: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a file and start AI processing. Requires authentication.

    Supported tool_types: background-remover, avatar-generator
    """
    # Validate tool type
    if tool_type not in CREDIT_COSTS:
        raise HTTPException(status_code=400, detail=f"Unknown tool type: {tool_type}")

    # Validate user has enough credits
    credits_needed = CREDIT_COSTS[tool_type]
    if user.credits < credits_needed:
        raise HTTPException(
            status_code=402,
            detail=f"Not enough credits. Need {credits_needed}, have {user.credits}",
        )

    # Read file
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Upload to storage
    upload_key = generate_upload_key(file.filename or "upload.png", user.id)
    content_type = file.content_type or "image/png"
    await upload_file(file_bytes, upload_key, content_type)

    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        id=task_id,
        user_id=user.id,
        tool_type=tool_type,
        status=TaskStatus.PROCESSING,
        input_file_url=upload_key,
        credits_cost=credits_needed,
        created_at=datetime.now(UTC),
    )
    db.add(task)
    db.commit()

    # Process based on tool type
    try:
        if tool_type == "background-remover":
            output_bytes = await run_background_remover_local(file_bytes)
            # Upload result to R2
            output_key = generate_download_key(user.id, task_id)
            await upload_file(output_bytes, output_key, "image/png")
            # Generate presigned URL
            task.output_file_url = generate_presigned_url(output_key)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "avatar-generator":
            # TODO: Implement avatar generation with local model or Replicate
            raise NotImplementedError("Avatar generator is coming soon")

        # Deduct credits
        user.credits -= credits_needed
        db.commit()

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.completed_at = datetime.now(UTC)
        db.commit()

    return TaskResponse(
        id=task.id,
        user_id=task.user_id,
        tool_type=task.tool_type,
        status=task.status.value if isinstance(task.status, TaskStatus) else task.status,
        output_file_url=task.output_file_url,
        error_message=task.error_message,
        credits_cost=task.credits_cost,
        created_at=task.created_at,
        completed_at=task.completed_at,
    )
