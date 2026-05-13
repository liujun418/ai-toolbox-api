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
    async def _call():
        return await _run_model(
            TOOL_PROMPTS["watermark-remover"].model,
            input={"image": image_url, "mask": mask_url},
        )
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
    """Use Florence-2 to detect watermark regions and generate a mask.
    Returns mask PNG bytes (white = watermark area) or None if no watermark found."""

    async def _detect():
        return await _run_model(
            FLORENCE2_MODEL,
            input={"image": image_url, "task": "dense region caption"},
        )

    try:
        result = await retry_with_backoff(_detect, max_retries=1, base_delay=2)
    except Exception as e:
        logger.warning("Florence-2 detection failed: %s", str(e))
        return None

    # Parse Florence-2 output: {"<DENSE_REGION_CAPTION>": {"bboxes": [...], "labels": [...]}}
    if isinstance(result, dict):
        caption_data = result.get("<DENSE_REGION_CAPTION>", {})
    elif isinstance(result, str):
        # Some Replicate wrappers return JSON string
        import json
        try:
            parsed = json.loads(result)
            caption_data = parsed.get("<DENSE_REGION_CAPTION>", {})
        except (json.JSONDecodeError, AttributeError):
            return None
    else:
        return None

    bboxes = caption_data.get("bboxes", [])
    labels = caption_data.get("labels", [])

    if not bboxes or not labels:
        return None

    # Filter: only keep regions that look like watermark/text overlays
    selected = []
    for bbox, label in zip(bboxes, labels):
        label_lower = label.lower().strip()
        if any(kw in label_lower for kw in _WATERMARK_KEYWORDS):
            selected.append(bbox)
        elif len(label) <= 4 and any(kw in label_lower for kw in _SHORT_LABEL_KEYWORDS):
            selected.append(bbox)

    if not selected:
        # No keyword match — use all detected regions as fallback
        # (Florence may label watermarks with generic terms)
        if len(bboxes) <= 8:
            selected = bboxes
            logger.info("Auto-detect: using all %d regions as fallback", len(bboxes))
        else:
            logger.info("Auto-detect: too many regions (%d), no clear watermark found", len(bboxes))
            return None

    # Build mask from selected bounding boxes
    # We need image dimensions — fetch the image to get size
    try:
        import httpx
        resp = httpx.get(image_url, follow_redirects=True, timeout=15)
        img = Image.open(io_module.BytesIO(resp.content))
        w, h = img.size
    except Exception:
        return None

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for bbox in selected:
        # Florence-2 bbox format: [x1, y1, x2, y2] — relative coordinates (0-1000)
        x1 = int(bbox[0] / 1000 * w)
        y1 = int(bbox[1] / 1000 * h)
        x2 = int(bbox[2] / 1000 * w)
        y2 = int(bbox[3] / 1000 * h)
        # Add generous padding for LaMa (better context for inpainting)
        pad = max(8, min(w, h) // 40)
        draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill=255)

    # Dilate mask slightly for cleaner inpainting edges
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
