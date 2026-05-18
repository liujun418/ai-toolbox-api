"""Face privacy blur service — pure PIL implementation.

Face detection is done via Replicate (Grounding DINO). Blur processing uses
PIL/Pillow only — no OpenCV dependency, avoiding system library issues.

Supports four blur styles: mosaic, gaussian, pixelate, emoji.
"""

import io
import logging

from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


def _apply_emoji_region(img: Image.Image, region: dict, emoji_type: str) -> None:
    """Overlay a cute icon on a face region using PIL shapes (no font dependency)."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    if w <= 0 or h <= 0:
        return

    # Color themes per emoji type
    themes = {
        "smile": ("#FDD835", "#F9A825", "#5D4037"),   # yellow
        "mask":  ("#90CAF9", "#1976D2", "#0D47A1"),   # blue
        "cat":   ("#FFAB91", "#E64A19", "#BF360C"),   # orange
        "dog":   ("#A5D6A7", "#388E3C", "#1B5E20"),   # green
        "bear":  ("#BCAAA4", "#5D4037", "#3E2723"),   # brown
        "star":  ("#FFF176", "#FBC02D", "#F57F17"),   # gold
    }
    bg, border, detail = themes.get(emoji_type, themes["smile"])

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin = max(w, h) // 25 + 2
    ex0, ey0 = margin, margin
    ex1, ey1 = w - margin, h - margin
    draw.ellipse([ex0, ey0, ex1, ey1], fill=bg, outline=border, width=max(3, min(w, h) // 40))

    cx, cy = w // 2, h // 2
    eye_r = max(w // 14, 3)
    eye_y = cy - h // 7
    # Left eye
    lx = cx - w // 5
    draw.ellipse([lx - eye_r, eye_y - eye_r, lx + eye_r, eye_y + eye_r], fill=detail)
    # Right eye
    rx = cx + w // 5
    draw.ellipse([rx - eye_r, eye_y - eye_r, rx + eye_r, eye_y + eye_r], fill=detail)

    # Smile arc
    smile_w = w // 4
    smile_h = h // 6
    smile_y = cy + h // 9
    draw.arc(
        [cx - smile_w, smile_y - smile_h, cx + smile_w, smile_y + smile_h],
        0, 180, fill=detail, width=max(3, min(w, h) // 40),
    )

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
