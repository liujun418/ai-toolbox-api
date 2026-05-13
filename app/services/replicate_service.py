"""Replicate API service — async-wrapped, prompt-templated, retry-enabled."""

import asyncio
import io as io_module
import logging

from PIL import Image, ImageDraw

import replicate

from app.config import settings
from app.services.prompt_templates import (
    TOOL_PROMPTS,
    STYLE_PROMPTS,
    AVATAR_PROMPTS,
    AVATAR_NEGATIVE_PROMPTS,
    AVATAR_PARAMS,
    PromptTemplate,
)
from app.services.retry import retry_with_backoff

logger = logging.getLogger(__name__)


def _get_client():
    """Get authenticated Replicate client."""
    return replicate.Client(api_token=settings.REPLICATE_API_TOKEN)


async def _run_model(model: str, input: dict) -> any:
    """Run a Replicate model in a thread to avoid blocking the event loop."""
    client = _get_client()
    return await asyncio.to_thread(client.run, model, input=input)


# ── Background Remover ────────────────────────────────────────────

async def run_background_remover(image_url: str) -> tuple[str, str | None]:
    """Remove background from image. Returns (output_url, replicate_id)."""
    async def _call():
        return await _run_model(
            TOOL_PROMPTS["background-remover"].model,
            input={"image": image_url},
        )
    output = await retry_with_backoff(_call)
    return str(output), None


# ── Watermark Remover ─────────────────────────────────────────────

async def run_watermark_removal(image_url: str, mask_url: str) -> tuple[str, str | None]:
    """Remove watermarks using BRIA Eraser (professional inpainting).
    Returns (output_url, replicate_id). Raises if output is empty."""
    tpl = TOOL_PROMPTS["watermark-remover"]
    inp = {
        "image": image_url,
        "mask": mask_url,
        **tpl.default_params,
    }
    async def _call():
        return await _run_model(tpl.model, input=inp)

    output = await retry_with_backoff(_call)
    if isinstance(output, list):
        if not output:
            raise ValueError("BRIA Eraser returned empty output")
        return str(output[0]), None
    return str(output), None


# ── Watermark Auto-Detect (Florence-2) ─────────────────────────────

FLORENCE2_MODEL = "lucataco/florence-2-large:da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595"

# Labels that likely indicate watermark/overlay regions
_WATERMARK_KEYWORDS = {
    "text", "watermark", "logo", "stamp", "signature", "label", "caption",
    "overlay", "brand", "copyright", "mark", "icon", "badge", "tag",
    "letter", "word", "character", "symbol", "writing", "inscription",
    "banner", "ribbon", "emblem", "crest", "seal",
}

# Short labels (1-4 chars) at edges/corners are often watermarks
_SHORT_LABEL_KEYWORDS = {
    "www", "http", ".com", ".net", ".org", "©", "®", "tm",
    "url", "site", "link",
}


async def auto_detect_watermark(image_url: str) -> bytes | None:
    """Detect watermark regions using Florence-2 OD (open-vocabulary object detection).
    Falls back to OCR. Returns mask PNG bytes (white = watermark area) or None."""

    bboxes = []
    labels = []

    # Strategy 1: Object detection targeting watermarks/logos/text
    try:
        async def _detect_od():
            return await _run_model(
                FLORENCE2_MODEL,
                input={
                    "image": image_url,
                    "task": "<OD>",
                    "prompt": "watermark, logo, text, signature, stamp",
                },
            )
        result = await retry_with_backoff(_detect_od, max_retries=1, base_delay=2)
        logger.info("Florence-2 OD result type: %s", type(result).__name__)

        if isinstance(result, dict):
            od_data = result.get("<OD>", {})
        elif isinstance(result, str):
            import json
            try:
                parsed = json.loads(result)
                od_data = parsed.get("<OD>", {})
            except (json.JSONDecodeError, AttributeError):
                od_data = {}
        else:
            od_data = {}

        bboxes = od_data.get("bboxes", [])
        labels = od_data.get("labels", [])
        logger.info("Florence-2 OD: %d detections, labels=%s",
            len(labels), labels[:10] if labels else "none")
    except Exception as e:
        logger.warning("Florence-2 OD failed: %s", str(e))

    # Strategy 2: OCR for text watermarks
    if not bboxes:
        try:
            async def _detect_ocr():
                return await _run_model(
                    FLORENCE2_MODEL,
                    input={"image": image_url, "task": "<OCR_WITH_REGION>"},
                )
            result = await retry_with_backoff(_detect_ocr, max_retries=1, base_delay=2)

            if isinstance(result, dict):
                ocr_data = result.get("<OCR_WITH_REGION>", {})
            elif isinstance(result, str):
                import json
                try:
                    parsed = json.loads(result)
                    ocr_data = parsed.get("<OCR_WITH_REGION>", {})
                except (json.JSONDecodeError, AttributeError):
                    ocr_data = {}
            else:
                ocr_data = {}

            ocr_bboxes = ocr_data.get("bboxes", [])
            ocr_labels = ocr_data.get("labels", [])
            logger.info("Florence-2 OCR: %d text regions, labels=%s",
                len(ocr_labels), ocr_labels[:5] if ocr_labels else "none")

            if ocr_bboxes:
                bboxes = ocr_bboxes
                labels = ocr_labels
        except Exception as e:
            logger.warning("Florence-2 OCR failed: %s", str(e))

    if bboxes:
        selected = bboxes  # Use all OD/OCR detections directly
        logger.info("Auto-detect: using %d detected regions for mask", len(selected))
    else:
        logger.info("Auto-detect: no watermarks detected by any method")
        return None

    # Fetch image to get dimensions
    try:
        import httpx
        resp = httpx.get(image_url, follow_redirects=True, timeout=15)
        img = Image.open(io_module.BytesIO(resp.content))
        w, h = img.size
    except Exception as e:
        logger.warning("Auto-detect: failed to fetch image for dimensions: %s", str(e))
        return None

    # Build mask
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for bbox in selected:
        x1 = max(0, int(bbox[0] / 1000 * w))
        y1 = max(0, int(bbox[1] / 1000 * h))
        x2 = min(w, int(bbox[2] / 1000 * w))
        y2 = min(h, int(bbox[3] / 1000 * h))
        pad = max(12, min(w, h) // 25)
        draw.rectangle([
            max(0, x1 - pad), max(0, y1 - pad),
            min(w, x2 + pad), min(h, y2 + pad)
        ], fill=255)

    from PIL import ImageFilter
    mask = mask.filter(ImageFilter.MaxFilter(3))

    buf = io_module.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


# ── Photo Restorer ────────────────────────────────────────────────

async def run_photo_restoration(image_url: str) -> tuple[str, str | None]:
    """Restore old/damaged photo using GFPGAN. Returns (output_url, replicate_id)."""
    tpl = TOOL_PROMPTS["photo-restorer"]
    async def _call():
        return await _run_model(
            tpl.model,
            input={"img": image_url, **tpl.default_params},
        )
    output = await retry_with_backoff(_call)
    return str(output), None


# ── Avatar Generator ──────────────────────────────────────────────

async def run_avatar_generation(
    image_url: str,
    style: str = "cartoon",
) -> tuple[list[str], str | None]:
    """Generate avatar from photo using SDXL with per-style locked params.
    Returns (url_list, replicate_id)."""
    tpl = TOOL_PROMPTS["avatar-generator"]

    # Per-style positive prompt
    avatar_style = AVATAR_PROMPTS.get(style, AVATAR_PROMPTS["cartoon"])
    full_prompt = tpl.positive_prompt.format(user_prompt=avatar_style)

    # Per-style locked generation parameters
    params = AVATAR_PARAMS.get(style, AVATAR_PARAMS["cartoon"])

    # Per-style negative prompt
    negative = AVATAR_NEGATIVE_PROMPTS.get(style, "")

    inp = {
        "prompt": full_prompt,
        "image": image_url,
        **params,
        "negative_prompt": negative,
    }

    async def _call():
        return await _run_model(tpl.model, input=inp)

    output = await retry_with_backoff(_call)
    return [str(u) for u in output], None


# ── Image Upscaler ────────────────────────────────────────────────

async def run_image_upscaler(image_url: str, scale: int = 2) -> tuple[str, str | None]:
    """Upscale image. Returns (output_url, replicate_id)."""
    tpl = TOOL_PROMPTS["image-upscaler"]
    async def _call():
        return await _run_model(
            tpl.model,
            input={"image": image_url, "scale": scale, **tpl.default_params},
        )
    output = await retry_with_backoff(_call)
    return str(output), None


# ── Style Transfer ────────────────────────────────────────────────

async def run_style_transfer(
    image_url: str,
    style: str = "oil-painting",
    user_prompt: str = "",
) -> tuple[str, str | None]:
    """Transform image into artistic style. Returns (output_url, replicate_id)."""
    tpl = TOOL_PROMPTS["style-transfer"]
    style_text = STYLE_PROMPTS.get(style, STYLE_PROMPTS["oil-painting"])
    full_prompt = style_text.format(
        user_prompt=user_prompt
    ).strip()

    inp = {
        "prompt": full_prompt,
        "image": image_url,
        **tpl.default_params,
    }
    if tpl.negative_prompt:
        inp["negative_prompt"] = tpl.negative_prompt

    async def _call():
        return await _run_model(tpl.model, input=inp)

    output = await retry_with_backoff(_call)
    urls = [str(u) for u in output]
    return urls[0] if urls else "", None


# ── Text Polish (unchanged — no image processing) ─────────────────

async def run_text_polish(text: str, mode: str = "polish") -> str:
    """Polish, rewrite, shorten, or expand text using LLM."""
    mode_instructions = {
        "polish": "Improve the grammar, spelling, and clarity of the given text while keeping the same meaning. Return only the improved text, no explanations.",
        "rewrite": "Rewrite the given text with different wording while keeping the same meaning. Return only the rewritten text, no explanations.",
        "shorten": "Make the given text more concise while keeping the key points. Return only the shortened text, no explanations.",
        "expand": "Expand the given text with more detail and explanation. Return only the expanded text, no explanations.",
    }
    instruction = mode_instructions.get(mode, mode_instructions["polish"])

    async def _call():
        client = _get_client()
        return await asyncio.to_thread(
            client.run,
            "meta/meta-llama-3.1-70b-instruct:baf226e1f0cc30952e39198a7dc1e8083d2686196464e0665e2d88108db29c61",
            input={
                "system_prompt": instruction,
                "prompt": text,
                "max_tokens": 4096,
                "temperature": 0.7,
            },
        )

    output = await retry_with_backoff(_call)
    return "".join(list(output)).strip()
