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
    PDF_RESTRUCTURE_SYSTEM_PROMPT,
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
    """Detect if text is Arabic, Spanish, or English.
    Returns 'ar', 'es', or 'en'."""
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return 'en'

    # Arabic: count characters in Arabic Unicode blocks
    ar_count = sum(1 for c in text if '؀' <= c <= 'ۿ' or 'ݐ' <= c <= 'ݿ')
    if ar_count / total_chars > 0.3:
        return 'ar'

    # Spanish: check for Spanish-specific characters
    es_markers = set('ñÑáéíóúüÁÉÍÓÚÜ¿¡')
    if any(c in es_markers for c in text):
        return 'es'

    return 'en'


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
    """Polish, rewrite, shorten, expand, or restyle text using Llama 3 70B.

    Modes: polish, rewrite, shorten, expand, academic, business
    Auto-detects English, Spanish, or Arabic for language-specific optimization.
    Supports long text via paragraph-boundary chunking.
    """
    lang = _detect_language(text)

    # ── Mode instructions per language ──
    if lang == 'ar':
        anti_ai = "اكتب كمتحدث أصلي للعربية، وليس كذكاء اصطناعي. تجنب العبارات النمطية مثل 'من الجدير بالذكر' و'في الختام' و'أولاً ثانياً ثالثاً'. تجنب الحشو المفرط والانتقالات الآلية. اكتب بشكل طبيعي."
        mode_prompts = {
            "polish": f"صحح القواعد النحوية والإملائية وحسّن تدفق الجمل مع الحفاظ على المعنى والنبرة الأصلية. {anti_ai} أعد فقط النص المنقح، بدون أي تفسيرات.",
            "rewrite": f"عبّر عن نفس المعنى بكلمات وتراكيب جمل مختلفة تماماً. تجنب تكرار الصياغة الأصلية. {anti_ai} أعد فقط النص المعاد كتابته، بدون أي تفسيرات.",
            "shorten": f"احذف التكرار واختصر إلى الرسالة الأساسية. احتفظ بجميع الحقائق والبيانات الرئيسية. {anti_ai} أعد فقط النص المختصر، بدون أي تفسيرات.",
            "expand": f"أضف التفاصيل والأمثلة والتوضيحات لإثراء المحتوى. حافظ على الرسالة الأساسية الأصلية. {anti_ai} أعد فقط النص الموسع، بدون أي تفسيرات.",
            "academic": f"حوّل إلى أسلوب أكاديمي رسمي: منطق دقيق، مفردات متخصصة، تركيب مناسب، بدون تعابير عامية. حافظ على الحجج الأصلية. {anti_ai} أعد فقط النص المعاد كتابته، بدون أي تفسيرات.",
            "business": f"حوّل إلى أسلوب تواصل مهني: موجز، مهذب، موجه نحو العمل. مناسب للرسائل والتقارير والعروض. {anti_ai} أعد فقط النص المعاد كتابته، بدون أي تفسيرات.",
        }
    elif lang == 'es':
        anti_ai = "Escribe como un hablante nativo de español, no como una IA. Evita frases formulaicas como 'es digno de mención', 'en conclusión', 'en primer lugar, en segundo lugar'. Evita los rellenos excesivos y las transiciones robóticas. Escribe con naturalidad."
        mode_prompts = {
            "polish": f"Corrige la gramática, ortografía y mejora la fluidez de las oraciones manteniendo el significado y tono original. {anti_ai} Devuelve solo el texto mejorado, sin explicaciones.",
            "rewrite": f"Expresa el mismo significado con palabras y estructuras de oraciones completamente diferentes. Evita repetir la redacción original. {anti_ai} Devuelve solo el texto reescrito, sin explicaciones.",
            "shorten": f"Elimina la redundancia y destila el mensaje central. Conserva todos los hechos y datos clave. {anti_ai} Devuelve solo el texto acortado, sin explicaciones.",
            "expand": f"Añade detalles, ejemplos y explicaciones para enriquecer el contenido. Mantén intacto el mensaje central original. {anti_ai} Devuelve solo el texto expandido, sin explicaciones.",
            "academic": f"Transforma a un estilo académico formal: lógica rigurosa, vocabulario preciso, estructura adecuada, sin coloquialismos. Conserva los argumentos originales. {anti_ai} Devuelve solo el texto reescrito, sin explicaciones.",
            "business": f"Transforma a comunicación profesional: conciso, cortés, orientado a la acción. Adecuado para correos, informes y propuestas. {anti_ai} Devuelve solo el texto reescrito, sin explicaciones.",
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


# ── PDF OCR ─────────────────────────────────────────────────────────

async def run_pdf_ocr(file_url: str) -> str:
    """Extract text from PDF (including scanned/image-based) using Datalab Marker.

    Marker is the #1 OCR model on Replicate (82.7 olmOCR-Bench).
    It natively supports PDF input and outputs markdown with structure preserved.
    Returns concatenated markdown from all pages.
    """
    async def _call():
        client = _get_client()
        return await asyncio.to_thread(
            client.run,
            "datalab-to/marker",
            input={
                "file": file_url,
                "force_ocr": True,
                "paginate": False,
            },
        )

    output = await retry_with_backoff(_call)

    # Marker returns a dict with "markdown" key
    if isinstance(output, dict) and "markdown" in output:
        return str(output["markdown"])
    # Fallback: try to extract text from whatever format
    if isinstance(output, str):
        return output
    return str(output)


async def run_pdf_restructure(ocr_text: str) -> str:
    """Use Llama 3.1 405B to correct OCR errors and rebuild document structure.

    Takes raw OCR markdown text and returns cleaned, well-structured markdown
    suitable for conversion to .docx.
    """
    async def _call():
        client = _get_client()
        return await asyncio.to_thread(
            client.run,
            "meta/meta-llama-3.1-405b-instruct",
            input={
                "system_prompt": PDF_RESTRUCTURE_SYSTEM_PROMPT,
                "prompt": f"Restore and format this document:\n\n{ocr_text}",
                "max_tokens": 4096,
                "temperature": 0.3,
                "top_p": 0.9,
            },
        )

    output = await retry_with_backoff(_call)
    return "".join(list(output)).strip()
