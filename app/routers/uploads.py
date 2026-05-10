"""File upload and AI processing endpoints."""

import io as io_module
import json
import uuid
from datetime import datetime, UTC

from PIL import Image

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
from app.services.pdf_service import convert_pdf_to_word, get_pdf_page_count
from app.services.replicate_service import (
    run_avatar_generation,
    run_background_remover,
    run_image_upscaler,
    run_photo_restoration,
    run_style_transfer,
    run_text_polish,
    run_watermark_removal,
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

    Supported tool_types: background-remover, watermark-remover, photo-restorer,
    avatar-generator, pdf-to-word, image-upscaler, style-transfer, text-polish
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

    # Create task first (so we can track failures)
    task_id = str(uuid.uuid4())
    upload_key = generate_upload_key(file.filename or "upload.png", user.id)
    content_type = file.content_type or "image/png"
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

    # Wrap all external calls so failures are caught and recorded
    try:
        # Upload to storage
        await upload_file(file_bytes, upload_key, content_type)
        image_url = generate_presigned_url(upload_key, expires_in=3600)

        if tool_type == "background-remover":
            output = await run_background_remover(image_url)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "watermark-remover":
            # Generate white mask for LaMa inpainting (whole image = inpaint area)
            mask_key = f"masks/{user.id}/{uuid.uuid4().hex}.png"
            img = Image.open(io_module.BytesIO(file_bytes))
            mask = Image.new("L", img.size, 255)
            mask_buffer = io_module.BytesIO()
            mask.save(mask_buffer, format="PNG")
            await upload_file(mask_buffer.getvalue(), mask_key, "image/png")
            output = await run_watermark_removal(image_url, generate_presigned_url(mask_key, expires_in=3600))
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "photo-restorer":
            colorize = "colorize" in prompt.lower() if prompt else False
            output = await run_photo_restoration(image_url, colorize)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "avatar-generator":
            style = "cartoon"
            if prompt:
                for s in ["cartoon", "anime", "professional", "pixel-art", "watercolor", "oil-painting"]:
                    if s in prompt.lower():
                        style = s
                        break
            outputs = await run_avatar_generation(image_url, style)
            task.output_file_url = outputs[0] if isinstance(outputs, list) and outputs else str(outputs)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "image-upscaler":
            # Parse scale from prompt: "Upscale image by 2x. ..." or "Upscale image by 4x. ..."
            scale = "2x"
            if prompt and "4x" in prompt:
                scale = "4x"
            output = await run_image_upscaler(image_url, scale)
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
            output = await run_style_transfer(image_url, style)
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

        elif tool_type == "pdf-to-word":
            page_count = await get_pdf_page_count(file_bytes)
            if page_count <= 5:
                actual_cost = 1
            elif page_count <= 20:
                actual_cost = 2
            else:
                actual_cost = 3

            if user.credits < actual_cost:
                raise HTTPException(
                    status_code=402,
                    detail=f"Not enough credits. This PDF has {page_count} pages "
                    f"and requires {actual_cost} credits, but you have {user.credits}.",
                )

            task.credits_cost = actual_cost
            credits_needed = actual_cost

            docx_bytes = await convert_pdf_to_word(
                file_bytes, file.filename or "document.pdf"
            )
            output_key = generate_download_key(user.id, task_id, "docx")
            await upload_file(
                docx_bytes,
                output_key,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
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
