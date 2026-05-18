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
    AI_IMAGE_GENERATOR_NEGATIVE,
    AI_IMAGE_GENERATOR_POSITIVE_PREFIX,
    AI_IMAGE_GENERATOR_PARAMS,
    AI_IMAGE_GENERATOR_DIMENSIONS,
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


# ── AI Image Generator (SDXL text-to-image / image-to-image) ──────

async def run_ai_image_generation(
    user_prompt: str,
    quality: str = "medium",
    aspect_ratio: str = "1:1",
    num_images: int = 1,
    reference_image_url: str | None = None,
) -> list[str]:
    """Generate AI images using stability-ai/sdxl.

    Args:
        user_prompt: User's text description.
        quality: "low", "medium", or "high" — controls inference steps and CFG scale.
        aspect_ratio: "1:1", "3:2", or "2:3" — output dimensions.
        num_images: 1-4 images to generate.
        reference_image_url: Optional reference image for img2img mode.

    Returns:
        List of output image URLs.
    """
    params = AI_IMAGE_GENERATOR_PARAMS.get(quality, AI_IMAGE_GENERATOR_PARAMS["medium"])
    dims = AI_IMAGE_GENERATOR_DIMENSIONS.get(aspect_ratio, (1024, 1024))

    # Build enhanced prompt
    full_prompt = f"{AI_IMAGE_GENERATOR_POSITIVE_PREFIX}, {user_prompt}"

    inp: dict = {
        "prompt": full_prompt,
        "negative_prompt": AI_IMAGE_GENERATOR_NEGATIVE,
        "num_outputs": max(1, min(4, num_images)),
        "num_inference_steps": params["num_inference_steps"],
        "guidance_scale": params["guidance_scale"],
        "width": dims[0],
        "height": dims[1],
    }

    # Image-to-image mode when reference image provided
    if reference_image_url:
        inp["image"] = reference_image_url
        inp["prompt_strength"] = 0.65

    async def _call():
        return await _run_model(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input=inp,
        )

    output = await retry_with_backoff(_call)
    return [str(u) for u in output]


# ── PDF OCR ─────────────────────────────────────────────────────────

async def run_pdf_ocr(file_url: str) -> tuple[str, list[bytes]]:
    """Extract text from PDF pages. For scanned/image PDFs, renders pages as images.

    Returns (ocr_text, page_images) where:
    - ocr_text: empty string for scanned PDFs (OCR unavailable via Replicate)
    - page_images: list of PNG bytes for each page (for image embedding fallback)
    """
    import io as _io
    import fitz as _fitz
    import httpx as _httpx

    # Download PDF from presigned URL
    resp = _httpx.get(file_url, follow_redirects=True, timeout=30)
    pdf_bytes = resp.content

    # Convert each page to an image
    pdf_stream = _io.BytesIO(pdf_bytes)
    doc = _fitz.open(stream=pdf_stream, filetype="pdf")
    page_images = []
    ocr_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Check if page has text layer
        page_text = page.get_text().strip()
        if len(page_text) >= 20:
            ocr_text_parts.append(page_text)
        else:
            # Render page to image
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            page_images.append(img_bytes)
            ocr_text_parts.append(f"[Page {page_num + 1}: image-based, see embedded image]")

    doc.close()

    ocr_text = "\n\n".join(ocr_text_parts)
    return ocr_text, page_images


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


# ── Face Detection ──────────────────────────────────────────────────

FACE_DETECTION_MODEL = "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"


async def run_face_detection(image_url: str) -> list[dict]:
    """Detect faces using Grounding DINO on Replicate.

    Returns list of {x, y, w, h} in pixel coordinates.
    Each detection has confidence >= box_threshold.
    """
    async def _call():
        return await _run_model(
            FACE_DETECTION_MODEL,
            input={
                "image": image_url,
                "query": "human face",
                "box_threshold": 0.35,
                "text_threshold": 0.2,
                "show_visualisation": False,
            },
        )

    output = await retry_with_backoff(_call)
    detections = output.get("detections", []) if isinstance(output, dict) else []
    faces = []
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            faces.append({
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "w": int(bbox[2] - bbox[0]),
                "h": int(bbox[3] - bbox[1]),
            })
    logger.info("Face detection: found %d faces in image", len(faces))
    return faces


# ── Text to Speech ──────────────────────────────────────────────────

TTS_MODEL = "minimax/speech-2.6-turbo"

# Supported language codes for MiniMax Speech 2.6 (40+ languages)
TTS_SPEAKER_MAP: dict[str, str] = {
    "en": "male-qn-qingse", "es": "female-shaonv",
    "ar": "male-qn-qingse", "fr": "female-shaonv",
    "de": "male-qn-qingse", "it": "female-shaonv",
    "ja": "female-shaonv", "zh": "male-qn-qingse",
    "ko": "female-shaonv", "pt": "male-qn-qingse",
    "ru": "male-qn-qingse", "tr": "female-shaonv",
    "pl": "male-qn-qingse", "nl": "female-shaonv",
    "cs": "male-qn-qingse", "hi": "female-shaonv",
    "hu": "male-qn-qingse",
}


async def run_tts(text: str, language: str = "en") -> bytes:
    """Convert text to speech using MiniMax Speech 2.6 Turbo. Returns MP3 audio bytes."""
    voice = TTS_SPEAKER_MAP.get(language, "male-qn-qingse")

    async def _call():
        return await _run_model(
            TTS_MODEL,
            input={"text": text, "voice": voice, "output_format": "mp3"},
        )

    output = await retry_with_backoff(_call)
    output_url = str(output) if not isinstance(output, list) else str(output[0])

    import httpx
    resp = httpx.get(output_url, follow_redirects=True, timeout=60)
    return resp.content


# ── Image Description ───────────────────────────────────────────────

IMAGE_DESC_MODEL = "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb329ef07c4c9b93ae3ce8c9f"


async def run_image_description(image_url: str, prompt: str = "") -> str:
    """Generate a detailed description of an image using LLaVA-13b."""
    system_prompt = (
        "You are an expert image describer. Analyze the image carefully and generate "
        "1) a concise single-sentence caption for alt text / SEO, and "
        "2) a detailed 3-5 sentence description covering key elements, colors, composition, "
        "setting, people/objects, mood, and any text visible in the image. "
        "Be accurate and specific. Format:\n"
        "ALT: [one sentence]\n\nDESC: [detailed description]"
    )
    user_prompt = prompt.strip() if prompt and prompt.strip() else "Please describe this image in detail."

    async def _call():
        client = _get_client()
        return await asyncio.to_thread(
            client.run,
            IMAGE_DESC_MODEL,
            input={
                "image": image_url,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "temperature": 0.2,
                "max_tokens": 512,
            },
        )

    output = await retry_with_backoff(_call)
    return "".join(list(output)).strip()


# ── B&W Colorizer ──────────────────────────────────────────────────

async def run_colorizer(image_url: str) -> str:
    """Colorize a black & white photo using DeOldify (cneural/colorize).
    Well-established model with Artistic/Stable modes, render_factor control."""
    async def _call():
        return await _run_model(
            "cneural/colorize:1297e6b7ad8b3aa7f0f7e5c3826212b04dd83001cc2df6c4db522c602a73cffa",
            input={
                "input_image": image_url,
                "model_name": "Artistic",
                "render_factor": 35,
            },
        )

    output = await retry_with_backoff(_call)
    if isinstance(output, list) and output:
        return str(output[0])
    return str(output)


# ── Object Remover ──────────────────────────────────────────────────

async def run_object_removal(image_url: str, mask_url: str) -> str:
    """Remove unwanted objects using BRIA Eraser inpainting (same model as watermark remover).
    Takes image + user-painted mask, returns inpainted image URL."""
    tpl = TOOL_PROMPTS["watermark-remover"]
    inp = {"image": image_url, "mask": mask_url, **tpl.default_params}

    async def _call():
        return await _run_model(tpl.model, input=inp)

    output = await retry_with_backoff(_call)
    if isinstance(output, list):
        if not output:
            raise ValueError("BRIA Eraser returned empty output")
        return str(output[0])
    return str(output)


# ── Article Generator ──────────────────────────────────────────────

async def run_article_generation(topic: str, keywords: str = "", tone: str = "") -> str:
    """Generate a structured article using Llama 3 70B."""
    lang = _detect_language(topic + " " + keywords)

    if lang == "ar":
        instruction = (
            f"اكتب مقالة متكاملة باللغة العربية حول: {topic}\n\n"
            f"{'الكلمات المفتاحية: ' + keywords if keywords else ''}\n"
            f"{'النبرة: ' + tone if tone else ''}\n\n"
            "يجب أن تتضمن المقالة:\n"
            "1. عنوان جذاب\n2. مقدمة مشوقة\n3. 3-5 أقسام رئيسية (كل قسم بعنوان فرعي)\n4. خاتمة\n\n"
            "اكتب ككاتب محترف، وليس كذكاء اصطناعي. قدم محتوى قيماً وعملياً."
        )
    elif lang == "es":
        instruction = (
            f"Escribe un artículo completo en español sobre: {topic}\n\n"
            f"{'Palabras clave: ' + keywords if keywords else ''}\n"
            f"{'Tono: ' + tone if tone else ''}\n\n"
            "El artículo debe incluir:\n"
            "1. Título atractivo\n2. Introducción convincente\n3. 3-5 secciones principales (subtituladas)\n4. Conclusión\n\n"
            "Escribe como un escritor profesional, no como una IA. Proporciona contenido valioso."
        )
    else:
        instruction = (
            f"Write a complete, well-structured article about: {topic}\n\n"
            f"{'Keywords: ' + keywords if keywords else ''}\n"
            f"{'Tone: ' + tone if tone else ''}\n\n"
            "The article must include:\n"
            "1. An engaging title (as a heading)\n"
            "2. A compelling introduction\n"
            "3. 3-5 main sections (each with a subheading)\n"
            "4. A conclusion with key takeaways\n\n"
            "Write as a professional human writer, not an AI. "
            "Provide valuable, actionable content."
        )

    async def _call():
        client = _get_client()
        return await asyncio.to_thread(
            client.run,
            "meta/meta-llama-3-70b-instruct:fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d",
            input={
                "system_prompt": instruction,
                "prompt": "Write the article now.",
                "max_tokens": 4096,
                "temperature": 0.7,
            },
        )

    output = await retry_with_backoff(_call)
    return "".join(list(output)).strip()
