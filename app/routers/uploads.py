"""File upload and AI processing endpoints."""

import io as io_module
import json
import uuid
from datetime import datetime, UTC

import httpx
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
    mask: UploadFile | None = File(default=None),
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

    # Read file and validate size
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    MAX_SIZES = {"pdf-to-word": 20 * 1024 * 1024}
    default_max = 5 * 1024 * 1024
    size_limit = MAX_SIZES.get(tool_type, default_max)
    if len(file_bytes) > size_limit:
        limit_mb = size_limit // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {limit_mb}MB.")

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

    # Track result content for tools that return text
    result_content = None

    # Wrap all external calls so failures are caught and recorded
    try:
        # Upload to storage
        await upload_file(file_bytes, upload_key, content_type)
        image_url = generate_presigned_url(upload_key, expires_in=3600)

        if tool_type == "background-remover":
            if mask is not None and mask.filename and mask.size and mask.size > 0:
                # Manual: keep painted area, clear rest, then AI processes only kept area
                mask_bytes = await mask.read()
                user_mask = Image.open(io_module.BytesIO(mask_bytes)).convert("L")
                orig_img = Image.open(io_module.BytesIO(file_bytes)).convert("RGBA")
                if user_mask.size != orig_img.size:
                    user_mask = user_mask.resize(orig_img.size, Image.LANCZOS)
                # Clear unmarked pixels (black in mask = remove)
                px = orig_img.load()
                mp = user_mask.load()
                for y in range(orig_img.height):
                    for x in range(orig_img.width):
                        if mp[x, y] < 128:
                            px[x, y] = (0, 0, 0, 0)
                # Upload masked image for AI processing
                buf = io_module.BytesIO()
                orig_img.save(buf, format="PNG")
                masked_key = f"masked/{user.id}/{uuid.uuid4().hex}.png"
                await upload_file(buf.getvalue(), masked_key, "image/png")
                masked_url = generate_presigned_url(masked_key, expires_in=3600)
                result_url = await run_background_remover(masked_url)
                task.output_file_url = result_url
            else:
                result_url = await run_background_remover(image_url)
                task.output_file_url = result_url
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "watermark-remover":
            orig_img = Image.open(io_module.BytesIO(file_bytes)).convert("RGB")
            orig_w, orig_h = orig_img.size

            # Build mask from user input or default to full white
            if mask is not None and mask.filename:
                mask_bytes = await mask.read()
                user_mask = Image.open(io_module.BytesIO(mask_bytes)).convert("L")
                if user_mask.size != (orig_w, orig_h):
                    user_mask = user_mask.resize((orig_w, orig_h), Image.LANCZOS)
                if user_mask.getextrema()[1] < 128:
                    user_mask = None
            else:
                user_mask = None

            if user_mask is None:
                user_mask = Image.new("L", (orig_w, orig_h), 255)

            # Get SDXL cleaned version
            output_url = await run_watermark_removal(image_url)

            # Download SDXL output and composite with original using mask
            resp = httpx.get(output_url, timeout=30)
            cleaned = Image.open(io_module.BytesIO(resp.content)).convert("RGB")
            cleaned = cleaned.resize((orig_w, orig_h), Image.LANCZOS)

            # Composite: mask white(255) = cleaned, mask black(0) = original
            # Image.composite(image1=mask_0, image2=mask_255, mask)
            result_img = Image.composite(orig_img, cleaned, user_mask)

            out_buf = io_module.BytesIO()
            result_img.save(out_buf, format="PNG")
            output_key = generate_download_key(user.id, task_id, "png")
            await upload_file(out_buf.getvalue(), output_key, "image/png")
            task.output_file_url = generate_presigned_url(output_key, expires_in=3600)

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
            scale = 4 if (prompt and "4x" in prompt) else 2
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
            result_content = output
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

            try:
                docx_bytes = await convert_pdf_to_word(
                    file_bytes, file.filename or "document.pdf"
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

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
        result_content=result_content,
        error_message=task.error_message,
        credits_cost=task.credits_cost,
        created_at=task.created_at,
        completed_at=task.completed_at,
    )
