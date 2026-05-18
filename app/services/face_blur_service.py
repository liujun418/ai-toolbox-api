"""Face privacy blur service — pure PIL implementation.

Face detection is done via Replicate (Grounding DINO). Blur processing uses
PIL/Pillow only — no OpenCV dependency, avoiding system library issues.

Supports four blur styles: mosaic, gaussian, pixelate, emoji.
"""

import io
import logging

from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

EMOJI_OPTIONS = {
    "smile": "😊",
    "mask": "😷",
    "cat": "🐱",
    "dog": "🐶",
    "bear": "🐻",
    "star": "⭐",
}


def _expand_region(region: dict, factor: float = 0.15) -> dict:
    """Expand a face region slightly to cover the full head."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    dx = int(w * factor)
    dy = int(h * factor)
    return {
        "x": x - dx,
        "y": y - dy // 2,
        "w": w + 2 * dx,
        "h": h + 2 * dy,
    }


def _clamp_region(img_w: int, img_h: int, region: dict) -> dict | None:
    """Clamp region to image bounds. Returns None if region is out of bounds."""
    x = max(0, region["x"])
    y = max(0, region["y"])
    w = min(region["w"], img_w - x)
    h = min(region["h"], img_h - y)
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _apply_blur_region(img: Image.Image, region: dict, style: str) -> None:
    """Apply blur style to a single rectangular region (in-place on PIL Image)."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    crop = img.crop((x, y, x + w, y + h))

    if style == "mosaic":
        block = max(4, min(w, h) // 20)
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = crop.resize((small_w, small_h), Image.NEAREST)
        blurred = small.resize((w, h), Image.NEAREST)

    elif style == "gaussian":
        radius = max(10, min(w, h) // 6)
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=radius))

    elif style == "pixelate":
        block = max(6, min(w, h) // 10)
        blurred = crop.resize(
            (max(1, w // block), max(1, h // block)),
            Image.NEAREST,
        ).resize((w, h), Image.NEAREST)

    else:
        return

    img.paste(blurred, (x, y))


# Twemoji CDN URLs — reliable emoji rendering via PNG, no font dependency
_EMOJI_CODEPOINTS = {
    "smile": "1f60a",
    "mask": "1f637",
    "cat": "1f431",
    "dog": "1f436",
    "bear": "1f43b",
    "star": "2b50",
}
_EMOJI_CACHE: dict[str, Image.Image] = {}


def _get_emoji_image(emoji_type: str, size: int) -> Image.Image:
    """Fetch emoji PNG from twemoji CDN, with in-memory cache."""
    import httpx

    cache_key = f"{emoji_type}_{size}"
    if cache_key in _EMOJI_CACHE:
        return _EMOJI_CACHE[cache_key]

    codepoint = _EMOJI_CODEPOINTS.get(emoji_type, "1f60a")
    url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{codepoint}.png"

    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        emoji_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        emoji_img = emoji_img.resize((size, size), Image.LANCZOS)
        _EMOJI_CACHE[cache_key] = emoji_img
        return emoji_img
    except Exception:
        # Fallback: return None to signal rendering failure
        return None


def _apply_emoji_region(img: Image.Image, region: dict, emoji_type: str) -> None:
    """Overlay a twemoji PNG on a face region, matching the selector icon exactly."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    if w <= 0 or h <= 0:
        return

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # White circular background for visibility
    r = min(w, h) // 2 - 4
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(255, 255, 255, 200), outline=(255, 255, 255, 230), width=3)

    # Download and paste the actual emoji PNG from twemoji
    emoji_size = int(r * 1.3)
    emoji_img = _get_emoji_image(emoji_type, emoji_size)
    if emoji_img:
        ex = (w - emoji_img.width) // 2
        ey = (h - emoji_img.height) // 2
        overlay.paste(emoji_img, (ex, ey), emoji_img)

    img.paste(overlay, (x, y), overlay)


def apply_face_blur(
    image_bytes: bytes,
    blur_style: str = "mosaic",
    manual_regions: list[dict] | None = None,
    emoji_type: str = "smile",
    auto_only: bool = False,
    auto_regions: list[dict] | None = None,
) -> tuple[bytes, int, int]:
    """Apply face blur to an image using PIL.

    auto_regions: pre-detected face regions from client. If None, no auto blur.
    manual_regions: user-drawn supplementary regions.
    Returns (processed_image_bytes, face_count, region_count).
    """
    if blur_style not in ("mosaic", "gaussian", "pixelate", "emoji"):
        blur_style = "mosaic"

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size

    # Build region list
    regions: list[dict] = []
    if auto_regions:
        regions.extend(_expand_region(r) for r in auto_regions)
    face_count = len(auto_regions) if auto_regions else 0

    if manual_regions:
        regions.extend(manual_regions)

    # Clamp all regions to image bounds
    valid_regions = []
    for r in regions:
        clamped = _clamp_region(img_w, img_h, r)
        if clamped:
            valid_regions.append(clamped)

    # Apply style
    for region in valid_regions:
        if blur_style == "emoji":
            _apply_emoji_region(img, region, emoji_type)
        else:
            _apply_blur_region(img, region, blur_style)

    # Encode as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), face_count, len(valid_regions)
