"""File upload and AI processing endpoints."""

import json
import uuid
from datetime import datetime, UTC

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Task, TaskStatus, User
from app.routers.auth import get_current_user
from app.schemas import TaskResponse, CREDIT_COSTS
from app.services.storage import (
    generate_upload_key,
    generate_download_key,
    upload_file,
    generate_presigned_url,
)
from app.services.replicate_service import (
    run_background_remover,
    run_image_upscaler,
    run_style_transfer,
    run_text_polish,
)

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/{tool_type}", response_model=TaskResponse)
async def upload_and_process(
    tool_type: str,
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a file and start AI processing. Requires authentication.

    Supported tool_types: background-remover, avatar-generator, image-upscaler,
    style-transfer, text-polish
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
            output = await run_background_remover(upload_key)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "avatar-generator":
            # TODO: Implement with Replicate or local model
            raise NotImplementedError("Avatar generator is coming soon")

        elif tool_type == "image-upscaler":
            # Parse scale from prompt: "Upscale image by 2x. ..." or "Upscale image by 4x. ..."
            scale = "2x"
            if prompt and "4x" in prompt:
                scale = "4x"
            output = await run_image_upscaler(upload_key, scale)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "style-transfer":
            # Parse style from prompt
            style = "oil-painting"
            if prompt:
                for s in ["oil-painting", "watercolor", "anime", "sketch"]:
                    if s in prompt.lower():
                        style = s
                        break
            output = await run_style_transfer(upload_key, style)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "text-polish":
            # Parse mode and text from prompt
            mode = "polish"
            text_content = ""
            if prompt:
                # Format: "Mode: polish. Text: <actual text>"
                mode_part, _, text_part = prompt.partition(". Text: ")
                mode = mode_part.replace("Mode: ", "").strip().lower()
                text_content = text_part

            if not text_content:
                # Fallback: read text from uploaded file
                text_content = file_bytes.decode("utf-8", errors="replace")

            output = await run_text_polish(text_content, mode)
            # Save text output to storage
            output_key = generate_download_key(user.id, task_id, "txt")
            await upload_file(output.encode("utf-8"), output_key, "text/plain")
            task.output_file_url = generate_presigned_url(output_key, expires_in=3600)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

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
