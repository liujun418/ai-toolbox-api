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


def _get_emoji_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font capable of rendering emoji characters."""
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux (Noto)
        "seguiemj.ttf",                                          # Windows
        "/System/Library/Fonts/Apple Color Emoji.ttc",           # macOS
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size, encoding="unic")
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _apply_emoji_region(img: Image.Image, region: dict, emoji_type: str) -> None:
    """Overlay an emoji character on a face region, matching the selector icon."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    if w <= 0 or h <= 0:
        return

    # Map emoji type to the exact same emoji character shown in the selector
    emoji_chars = {
        "smile": "😊",
        "mask": "😷",
        "cat": "🐱",
        "dog": "🐶",
        "bear": "🐻",
        "star": "⭐",
    }
    emoji_char = emoji_chars.get(emoji_type, "😊")

    # Create a white semi-transparent circular background for visibility
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw white circle background so emoji is visible on any image
    r = min(w, h) // 2 - 4
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(255, 255, 255, 180), outline=(255, 255, 255, 200), width=3)

    # Render the actual emoji character using system emoji font
    font_size = max(int(r * 1.2), 20)
    font = _get_emoji_font(font_size)

    # Center the emoji in the circle
    bbox = draw.textbbox((0, 0), emoji_char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = (h - th) // 2
    draw.text((tx, ty), emoji_char, font=font, embedded_color=True, fill=None)

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
