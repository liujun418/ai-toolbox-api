"""File upload and AI processing endpoint — full pipeline with validation,
preprocessing, postprocessing, transaction logging, and structured error handling."""

import io as io_module
import logging
import uuid
from datetime import datetime, UTC

from PIL import Image

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Task, TaskStatus, Transaction, User
from app.routers.auth import get_current_user
from app.schemas import TaskResponse, CREDIT_COSTS
from app.services.image_processing import preprocess_image, preprocess_avatar, preprocess_style_transfer, postprocess_image
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
    auto_detect_watermark,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Image tools that need preprocessing/postprocessing
IMAGE_TOOL_TYPES = {
    "background-remover",
    "watermark-remover",
    "photo-restorer",
    "avatar-generator",
    "image-upscaler",
    "style-transfer",
}

ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}

# Friendly error messages for known failure patterns
ERROR_MESSAGES = {
    "timeout": "Processing timed out — this may be due to input size or server load. Try again with smaller input.",
    "rate_limit": "Too many requests right now. Please wait a moment and try again.",
    "model_error": "The AI model couldn't process this request. The input may be too long or in an unexpected format.",
    "validation": "This file doesn't appear to be a valid image. Please use PNG, JPG, or WebP.",
    "image_too_small": "The image is too small to upscale meaningfully. Minimum recommended: 100×100 pixels.",
    "gpu_oom": "The input is too large for the AI model. Try a smaller image (under 3000px) or shorter text.",
    "face_not_found": "No clear face detected in this image. Face Pro works best with visible, front-facing portraits.",
    "text_too_long": "The text is too long for the AI to process. Try shortening it or splitting into smaller sections.",
}


def _friendly_error(message: str) -> str:
    """Map raw error messages to user-friendly versions."""
    msg_lower = message.lower()
    if "timeout" in msg_lower:
        return ERROR_MESSAGES["timeout"]
    if "rate limit" in msg_lower or "429" in msg_lower:
        return ERROR_MESSAGES["rate_limit"]
    if "memory" in msg_lower or "oom" in msg_lower or "cuda" in msg_lower:
        return ERROR_MESSAGES["gpu_oom"]
    if "face" in msg_lower and ("detect" in msg_lower or "not found" in msg_lower or "no face" in msg_lower):
        return ERROR_MESSAGES["face_not_found"]
    # Only match "invalid" for image-related errors
    if ("invalid" in msg_lower and ("image" in msg_lower or "format" in msg_lower or "file" in msg_lower)) \
            or "corrupt" in msg_lower or "cannot identify" in msg_lower:
        return ERROR_MESSAGES["validation"]
    if "too small" in msg_lower or "resolution" in msg_lower:
        return ERROR_MESSAGES["image_too_small"]
    if "too long" in msg_lower or "too many" in msg_lower:
        return ERROR_MESSAGES["text_too_long"]
    if "replicate" in msg_lower or "model" in msg_lower or "prediction" in msg_lower:
        return ERROR_MESSAGES["model_error"]
    # Default: truncate long messages
    return message if len(message) < 200 else message[:200] + "..."


@router.post("/{tool_type}", response_model=TaskResponse)
async def upload_and_process(
    tool_type: str,
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    style: str | None = Form(default=None),
    bg_color: str | None = Form(default=None),
    mask: UploadFile | None = File(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a file and start AI processing."""
    if tool_type not in CREDIT_COSTS:
        raise HTTPException(status_code=400, detail=f"Unknown tool type: {tool_type}")

    credits_needed = CREDIT_COSTS[tool_type]
    if user.credits < credits_needed:
        raise HTTPException(
            status_code=402,
            detail=f"Not enough credits. Need {credits_needed}, have {user.credits:.0f}",
        )

    # Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Size validation
    if tool_type == "pdf-to-word":
        size_limit = 20 * 1024 * 1024
    elif tool_type in IMAGE_TOOL_TYPES:
        size_limit = settings.IMAGE_TOOLS_MAX_FILE_SIZE
    else:
        size_limit = 5 * 1024 * 1024

    if len(file_bytes) > size_limit:
        limit_mb = size_limit // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {limit_mb}MB.",
        )

    # Content-type validation for image tools
    if tool_type in IMAGE_TOOL_TYPES:
        content_type = file.content_type or ""
        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            # Also allow if PIL can open it
            try:
                Image.open(io_module.BytesIO(file_bytes))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Please use PNG, JPG, or WebP.",
                )

    # Text length validation for text-polish
    if tool_type == "text-polish" and prompt and len(prompt) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum length is {settings.MAX_TEXT_LENGTH} characters.",
        )

    # Create task
    task_id = str(uuid.uuid4())
    upload_key = generate_upload_key(file.filename or "upload.png", user.id)
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

    result_content = None

    try:
        # Preprocess images for image tools
        if tool_type in IMAGE_TOOL_TYPES:
            keep_alpha = tool_type == "background-remover"
            try:
                if tool_type == "avatar-generator":
                    file_bytes, img_meta = preprocess_avatar(file_bytes)
                elif tool_type == "style-transfer":
                    file_bytes, img_meta = preprocess_style_transfer(file_bytes)
                else:
                    file_bytes, img_meta = preprocess_image(file_bytes, keep_alpha=keep_alpha)
                logger.info(
                    "Preprocessed image for task %s: %dx%d -> %s (%d bytes)",
                    task_id,
                    img_meta["original_width"],
                    img_meta["original_height"],
                    img_meta.get("output_format", "unknown"),
                    img_meta.get("processed_size", 0),
                )
            except Exception as e:
                logger.warning("Preprocessing failed for task %s: %s", task_id, str(e))
                # Continue with original bytes

        content_type = file.content_type or "image/png"
        if tool_type in IMAGE_TOOL_TYPES:
            content_type = "image/png"  # preprocess outputs PNG or JPEG

        # Upload to storage
        await upload_file(file_bytes, upload_key, content_type)
        image_url = generate_presigned_url(upload_key, expires_in=3600)

        replicate_id = None

        if tool_type == "background-remover":
            # Parse background color
            bg_tuple: tuple[int, int, int] | None = None
            if bg_color and bg_color != "transparent":
                if bg_color == "white":
                    bg_tuple = (255, 255, 255)
                elif bg_color == "black":
                    bg_tuple = (0, 0, 0)
                elif bg_color.startswith("#") and len(bg_color) == 7:
                    try:
                        r = int(bg_color[1:3], 16)
                        g = int(bg_color[3:5], 16)
                        b = int(bg_color[5:7], 16)
                        bg_tuple = (r, g, b)
                    except ValueError:
                        pass

            if mask is not None and mask.filename:
                # Manual keep mode: user-painted mask
                mask_bytes = await mask.read()
                user_mask = Image.open(io_module.BytesIO(mask_bytes)).convert("L")
                orig_img = Image.open(io_module.BytesIO(file_bytes)).convert("RGBA")
                if user_mask.size != orig_img.size:
                    user_mask = user_mask.resize(orig_img.size, Image.LANCZOS)
                import numpy as np
                img_arr = np.array(orig_img)
                mask_arr = np.array(user_mask)
                mask_channel = (mask_arr < 128).astype(np.uint8) * 255
                img_arr[:, :, 3] = np.where(mask_channel == 0, img_arr[:, :, 3], 0)
                result_img = Image.fromarray(img_arr)

                buf = io_module.BytesIO()
                result_img.save(buf, format="PNG")
                masked_key = f"masked/{user.id}/{uuid.uuid4().hex}.png"
                await upload_file(buf.getvalue(), masked_key, "image/png")
                masked_url = generate_presigned_url(masked_key, expires_in=3600)
                result_url, replicate_id = await run_background_remover(masked_url)
                task.output_file_url = result_url
            else:
                # Auto mode: one-click background removal
                result_url, replicate_id = await run_background_remover(image_url)
                task.output_file_url = result_url

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "watermark-remover":
            orig_img = Image.open(io_module.BytesIO(file_bytes))
            img_w, img_h = orig_img.size

            if mask is not None and mask.filename:
                # User-painted mask → BRIA Eraser (precise)
                mask_bytes = await mask.read()
                mask_img = Image.open(io_module.BytesIO(mask_bytes)).convert("L")
                if mask_img.size != orig_img.size:
                    mask_img = mask_img.resize(orig_img.size, Image.LANCZOS)
                logger.info("Watermark removal: user-painted mask, image %dx%d", img_w, img_h)
            else:
                # No user mask → auto-detect watermark with Florence-2
                logger.info("Watermark removal: no user mask, auto-detecting with Florence-2")
                auto_mask_bytes = await auto_detect_watermark(image_url)
                if auto_mask_bytes is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not automatically detect watermark. Please paint over the watermark area and try again.",
                    )
                mask_img = Image.open(io_module.BytesIO(auto_mask_bytes)).convert("L")
                if mask_img.size != orig_img.size:
                    mask_img = mask_img.resize(orig_img.size, Image.LANCZOS)

            # BRIA Eraser convention: white (255) = erase area, black (0) = keep
            mask_buf = io_module.BytesIO()
            mask_img.save(mask_buf, format="PNG")
            mask_key = f"masks/{user.id}/{uuid.uuid4().hex}.png"
            await upload_file(mask_buf.getvalue(), mask_key, "image/png")
            mask_url = generate_presigned_url(mask_key, expires_in=3600)

            result_url, replicate_id = await run_watermark_removal(image_url, mask_url)
            task.output_file_url = result_url
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "photo-restorer":
            strength = "auto"
            if prompt and prompt.strip().lower() in ("auto", "face"):
                strength = prompt.strip().lower()
            output, replicate_id = await run_photo_restoration(image_url, strength)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "avatar-generator":
            valid_styles = {"cartoon", "anime", "professional", "pixel-art", "watercolor", "oil-painting"}
            requested_style = (style or "").lower()
            if requested_style not in valid_styles:
                requested_style = "cartoon"
            outputs, replicate_id = await run_avatar_generation(image_url, requested_style)
            task.output_file_url = outputs[0] if isinstance(outputs, list) and outputs else str(outputs)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "image-upscaler":
            # Parse "photo:2" or "anime:4" from prompt
            image_type = "photo"
            scale = 2
            if prompt:
                parts = prompt.strip().split(":")
                if len(parts) == 2:
                    image_type = parts[0].strip() if parts[0].strip() in ("photo", "anime") else "photo"
                    scale_str = parts[1].strip().rstrip("x")
                    try:
                        scale = int(scale_str)
                        if scale not in (2, 4):
                            scale = 2
                    except ValueError:
                        scale = 2
            output, replicate_id = await run_image_upscaler(image_url, scale, image_type)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "style-transfer":
            style = "oil-painting"
            if prompt:
                for s in ["oil-painting", "watercolor", "sketch", "cartoon", "cyberpunk", "fantasy"]:
                    if s in prompt.lower():
                        style = s
                        break
            output, replicate_id = await run_style_transfer(image_url, style)
            task.output_file_url = output
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(UTC)

        elif tool_type == "text-polish":
            mode = "polish"
            text_content = ""
            if prompt:
                mode_part, _, text_part = prompt.partition(". Text: ")
                mode = mode_part.replace("Mode: ", "").strip().lower()
                text_content = text_part

            if not text_content:
                text_content = file_bytes.decode("utf-8", errors="replace")

            output = await run_text_polish(text_content, mode)
            result_content = output
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
                    f"and requires {actual_cost} credits, but you have {user.credits:.0f}.",
                )

            task.credits_cost = actual_cost
            credits_needed = actual_cost

            try:
                docx_bytes = await convert_pdf_to_word(file_bytes, file.filename or "document.pdf")
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

        # Post-process output for image tools
        if tool_type in IMAGE_TOOL_TYPES and task.output_file_url:
            try:
                import httpx
                resp = httpx.get(task.output_file_url, follow_redirects=True, timeout=30)
                if resp.status_code == 200:
                    # Background remover: feather edges + optional background fill
                    if tool_type == "background-remover":
                        processed_bytes = postprocess_image(
                            resp.content,
                            feather_radius=1.5,
                            bg_color=bg_tuple,
                        )
                    else:
                        processed_bytes = postprocess_image(resp.content)
                    ext = "png" if task.output_file_url.lower().endswith(".png") else "webp"
                    output_key = f"output/{user.id}/{task_id}.{ext}"
                    ct = "image/png" if ext == "png" else "image/webp"
                    await upload_file(processed_bytes, output_key, ct)
                    task.output_file_url = generate_presigned_url(output_key, expires_in=3600)
            except Exception as e:
                logger.warning("Postprocessing failed for task %s: %s (continuing with raw output)", task_id, str(e))

        # Deduct credits
        user.credits -= credits_needed

        # Create transaction record
        transaction = Transaction(
            id=str(uuid.uuid4()),
            user_id=user.id,
            type="deduction",
            amount=-credits_needed,
            description=f"{tool_type}: task {task_id}",
        )
        db.add(transaction)

        # Store replicate_id if available
        if replicate_id:
            task.replicate_id = replicate_id

        db.commit()
        logger.info(
            "Task %s completed for user %s. Credits: -%.0f (balance: %.0f)",
            task_id, user.id, credits_needed, user.credits,
        )

    except HTTPException:
        raise
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error_message = _friendly_error(str(e))
        task.completed_at = datetime.now(UTC)
        db.commit()
        logger.error("Task %s failed: %s", task_id, str(e), exc_info=True)

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
