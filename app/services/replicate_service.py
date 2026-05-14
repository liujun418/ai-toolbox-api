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


# ── Text Polish ──────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English.
    Returns 'zh' if >30% CJK characters, 'en' otherwise."""
    cjk_count = sum(1 for c in text if '一' <= c <= '鿿' or '㐀' <= c <= '䶿')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return 'en'
    return 'zh' if cjk_count / total_chars > 0.3 else 'en'


def _split_text(text: str, max_chars: int = 3000) -> list[str]:
    """Split long text at paragraph boundaries for segmented processing."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split('\n')
    chunks = []
    current = ''
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + '\n'
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + '\n'
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


async def run_text_polish(text: str, mode: str = "polish") -> str:
    """Polish, rewrite, shorten, expand, or restyle text using Llama 3.1 70B.

    Modes: polish, rewrite, shorten, expand, academic, business
    Auto-detects Chinese/English for language-specific optimization.
    Supports long text via paragraph-boundary chunking.
    """
    lang = _detect_language(text)

    # ── Mode instructions per language ──
    if lang == 'zh':
        anti_ai = "像母语写作者一样自然流畅。禁止使用'值得注意的是''此外''总而言之''首先其次最后'等AI套话。不要过度使用连接词，用自然的汉语表达。"
        mode_prompts = {
            "polish": f"纠正语法错误和错别字，优化句子流畅度，保持原意和语气不变。{anti_ai}仅输出润色后的文本，不要任何解释。",
            "rewrite": f"用完全不同的措辞和句式重新表达相同含义，避免重复原文用词和结构。{anti_ai}仅输出改写后的文本，不要任何解释。",
            "shorten": f"删除冗余表述，提炼核心信息，使文本更加精炼有力。保留所有关键事实和数据。{anti_ai}仅输出精简后的文本，不要任何解释。",
            "expand": f"丰富细节，补充背景、举例或解释，使内容更加充实饱满。保持原文核心观点不变。{anti_ai}仅输出扩写后的文本，不要任何解释。",
            "academic": f"转换为正式学术论文风格：逻辑严密、用词精准、句式规范、避免口语化。保留原文主旨和核心论证。{anti_ai}仅输出改写后的文本，不要任何解释。",
            "business": f"转换为专业商务沟通风格：简洁有力、礼貌得体、重点突出、行动导向。适合邮件、报告、提案。{anti_ai}仅输出改写后的文本，不要任何解释。",
        }
    else:
        anti_ai = "Sound like a native human writer, not an AI. Avoid formulaic phrasing like 'it is worth noting', 'furthermore', 'in conclusion', 'firstly secondly lastly'. Avoid overly polite filler and robotic transitions. Write naturally."
        mode_prompts = {
            "polish": f"Fix grammar, spelling, and improve sentence flow while keeping the original meaning and tone. {anti_ai} Return only the polished text, no explanations.",
            "rewrite": f"Express the same meaning with completely different wording and sentence structure. Avoid repeating the original phrasing. {anti_ai} Return only the rewritten text, no explanations.",
            "shorten": f"Remove redundancy and distill to the core message. Keep all key facts and data points. {anti_ai} Return only the shortened text, no explanations.",
            "expand": f"Add detail, examples, and explanation to enrich the content. Keep the original core message intact. {anti_ai} Return only the expanded text, no explanations.",
            "academic": f"Transform into formal academic writing: rigorous logic, precise vocabulary, proper structure, no colloquialisms. Preserve original arguments. {anti_ai} Return only the rewritten text, no explanations.",
            "business": f"Transform into professional business communication: concise, courteous, action-oriented. Suitable for emails, reports, and proposals. {anti_ai} Return only the rewritten text, no explanations.",
        }

    instruction = mode_prompts.get(mode, mode_prompts["polish"])

    # ── Long text: chunk at paragraph boundaries ──
    chunks = _split_text(text)
    if len(chunks) == 1:
        chunks_to_process = [text]
    else:
        chunks_to_process = []
        prev_context = ""
        for i, chunk in enumerate(chunks):
            if prev_context:
                chunk = f"[Context from previous section]\n{prev_context}\n\n[Current section to process]\n{chunk}"
            chunks_to_process.append(chunk)
            # Keep last 200 chars as context for next chunk
            prev_context = chunk[-200:]

    # ── Process all chunks ──
    results = []
    for chunk in chunks_to_process:
        async def _call():
            client = _get_client()
            return await asyncio.to_thread(
                client.run,
                "meta/meta-llama-3-70b-instruct:fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d",
                input={
                    "system_prompt": instruction,
                    "prompt": chunk,
                    "max_tokens": 4096,
                    "temperature": 0.5,
                },
            )

        output = await retry_with_backoff(_call)
        results.append("".join(list(output)).strip())

    return "\n\n".join(results).strip()
