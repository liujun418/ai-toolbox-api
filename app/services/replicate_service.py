"""Replicate API service — async-wrapped, prompt-templated, retry-enabled."""

import asyncio
import io as io_module
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import replicate

from app.config import settings
from app.services.prompt_templates import (
    TOOL_PROMPTS,
    STYLE_REFERENCE_PROMPTS,
    AVATAR_PROMPTS,
    AVATAR_NEGATIVE_PROMPTS,
    AVATAR_PARAMS,
    PromptTemplate,
)
from app.services.style_references import get_style_reference_url
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


# ── Watermark Auto-Detect (PIL heuristics) ─────────────────────────

async def auto_detect_watermark(image_url: str) -> bytes | None:
    """Generate a mask for potential watermark regions using image analysis.
    Uses edge detection in corners/edges to find text/logo patterns.
    Always returns a mask (falls back to conservative corner coverage).
    Returns mask PNG bytes (white = potential watermark area)."""

    # Fetch image
    try:
        import httpx
        resp = httpx.get(image_url, follow_redirects=True, timeout=15)
        img = Image.open(io_module.BytesIO(resp.content)).convert("RGB")
        w, h = img.size
    except Exception as e:
        logger.warning("Auto-detect: failed to fetch image: %s", str(e))
        return None

    arr = np.array(img, dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Define regions to scan for watermarks (as fraction of image)
    regions = [
        ("bottom",  0.0, 0.85, 1.0, 1.0),    # bottom 15%
        ("right",   0.85, 0.0, 1.0, 1.0),     # right 15%
        ("top",     0.0, 0.0, 1.0, 0.10),      # top 10%
        ("left",    0.0, 0.0, 0.10, 1.0),      # left 10%
    ]

    found_any = False
    for name, fx1, fy1, fx2, fy2 in regions:
        x1, y1 = int(fx1 * w), int(fy1 * h)
        x2, y2 = int(fx2 * w), int(fy2 * h)
        if x2 <= x1 or y2 <= y1:
            continue

        region = arr[y1:y2, x1:x2]

        # Compute local standard deviation (high std = text/logo detail)
        gray = np.mean(region, axis=2)
        # Use a small kernel to find high-frequency content (text edges)
        region_img = Image.fromarray(region.astype(np.uint8))
        edges = region_img.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges, dtype=np.float32)
        edge_intensity = np.mean(edge_arr, axis=2)

        # Threshold: high edge intensity = likely text/watermark
        edge_threshold = 40  # Adjust based on testing
        high_edge = edge_intensity > edge_threshold
        edge_density = high_edge.mean()

        logger.info("Auto-detect: region '%s' edge_density=%.4f", name, edge_density)

        if edge_density > 0.03:  # >3% pixels are edges = likely text/graphics
            mask[y1:y2, x1:x2] = np.where(high_edge, 255, 0).astype(np.uint8)
            found_any = True

    if not found_any:
        # Fallback: cover common watermark positions with conservative masks
        logger.info("Auto-detect: no text detected, using corner fallback masks")

        # Bottom-right corner
        br_x1, br_y1 = int(w * 0.75), int(h * 0.80)
        mask[br_y1:h, br_x1:w] = 255

        # Bottom strip (center-bottom)
        bs_y1 = int(h * 0.90)
        mask[bs_y1:h, int(w * 0.25):int(w * 0.75)] = 255

        # Top-right corner
        tr_y2 = int(h * 0.10)
        mask[0:tr_y2, int(w * 0.80):w] = 255

    # Dilate mask for better inpainting
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

    white_px = int(np.array(mask_img).sum() // 255)
    logger.info("Auto-detect: generated mask %dx%d, white px=%d", w, h, white_px)

    if white_px == 0:
        return None

    buf = io_module.BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()


# ── Photo Restorer ────────────────────────────────────────────────

async def run_photo_restoration(image_url: str, strength: str = "auto") -> tuple[str, str | None]:
    """Restore old/damaged photo.

    Strength levels:
      - auto: Topaz D&S — automatic scratch/dust/blemish removal
      - face: GFPGAN — dedicated face enhancement, preserves age features
    """
    if strength == "face":
        tpl = TOOL_PROMPTS["photo-restorer-face"]
        async def _call():
            return await _run_model(
                tpl.model,
                input={"img": image_url, "version": tpl.default_params["version"],
                       "scale": tpl.default_params["scale"], "weight": tpl.default_params["weight"]},
            )
    else:
        tpl = TOOL_PROMPTS["photo-restorer"]
        inp = {"image": image_url, "output_format": tpl.default_params["output_format"]}
        async def _call():
            return await _run_model(tpl.model, input=inp)
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

async def run_image_upscaler(image_url: str, scale: int = 2, image_type: str = "photo") -> tuple[str, str | None]:
    """Upscale image with Real-ESRGAN.

    image_type:
      - photo: face_enhance enabled (best for portraits/photos)
      - anime: face_enhance disabled (preserves line art and flat colors)
    """
    tpl = TOOL_PROMPTS["image-upscaler"]
    face_enhance = image_type != "anime"
    async def _call():
        return await _run_model(
            tpl.model,
            input={"image": image_url, "scale": scale, "face_enhance": face_enhance},
        )
    output = await retry_with_backoff(_call)
    return str(output), None


# ── Style Transfer ────────────────────────────────────────────────

async def run_style_transfer(
    image_url: str,
    style: str = "oil-painting",
    user_prompt: str = "",
) -> tuple[str, str | None]:
    """Transform image into artistic style using IP-Adapter + ControlNet.

    Uses fofr/style-transfer with a pre-generated style reference image
    and optional text prompt to guide the style application.
    Returns (output_url, replicate_id).
    """
    tpl = TOOL_PROMPTS["style-transfer"]
    style_ref_url = get_style_reference_url(style)
    if not style_ref_url:
        style_ref_url = get_style_reference_url("oil-painting")  # fallback

    prompt_text = STYLE_REFERENCE_PROMPTS.get(style, STYLE_REFERENCE_PROMPTS["oil-painting"])
    if user_prompt:
        prompt_text = f"{prompt_text}, {user_prompt}"

    inp = {
        "style_image": style_ref_url,
        "structure_image": image_url,
        "prompt": prompt_text,
        **tpl.default_params,
    }

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
