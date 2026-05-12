"""Replicate API service — async-wrapped, prompt-templated, retry-enabled."""

import asyncio
import logging

import replicate

from app.config import settings
from app.services.prompt_templates import (
    TOOL_PROMPTS,
    STYLE_PROMPTS,
    AVATAR_PROMPTS,
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
    """Remove watermarks using BRIA Eraser. Returns (output_url, replicate_id)."""
    async def _call():
        return await _run_model(
            TOOL_PROMPTS["watermark-remover"].model,
            input={"image": image_url, "mask": mask_url},
        )
    output = await retry_with_backoff(_call)
    return str(output), None


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
    user_prompt: str = "",
) -> tuple[list[str], str | None]:
    """Generate avatar from photo using SDXL. Returns (url_list, replicate_id)."""
    tpl = TOOL_PROMPTS["avatar-generator"]
    avatar_style = AVATAR_PROMPTS.get(style, AVATAR_PROMPTS["cartoon"])
    full_prompt = tpl.positive_prompt.format(
        user_prompt=f"{avatar_style}, {user_prompt}" if user_prompt else avatar_style
    )

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
