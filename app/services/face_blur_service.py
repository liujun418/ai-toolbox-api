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
    """Overlay a distinct cute icon on a face region using PIL shapes."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    if w <= 0 or h <= 0:
        return

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke = max(3, min(w, h) // 40)
    margin = max(w, h) // 25 + 2

    cx, cy = w // 2, h // 2
    eye_r = max(w // 14, 3)
    eye_y = cy - h // 7
    smile_w = w // 4
    smile_h = h // 6
    smile_y = cy + h // 9

    if emoji_type == "star":
        # Draw a 5-pointed star
        import math
        outer_r = min(w, h) // 2 - margin
        inner_r = outer_r * 0.45
        pts = []
        for i in range(10):
            angle = math.pi / 2 - i * math.pi / 5
            r = outer_r if i % 2 == 0 else inner_r
            pts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
        draw.polygon(pts, fill="#FDD835", outline="#F9A825")
        # Eyes inside star
        draw.ellipse([cx - eye_r * 2, eye_y - eye_r, cx - eye_r // 2, eye_y + eye_r], fill="#5D4037")
        draw.ellipse([cx + eye_r // 2, eye_y - eye_r, cx + eye_r * 2, eye_y + eye_r], fill="#5D4037")
        draw.arc([cx - smile_w, smile_y - smile_h, cx + smile_w, smile_y + smile_h],
                 0, 180, fill="#5D4037", width=stroke)

    elif emoji_type == "mask":
        # Blue circle with medical mask
        draw.ellipse([margin, margin, w - margin, h - margin],
                     fill="#90CAF9", outline="#1976D2", width=stroke)
        # White mask rectangle covering lower half
        mask_top = cy + h // 20
        mask_bottom = h - margin * 2
        mask_l = cx - w // 3
        mask_r = cx + w // 3
        draw.rounded_rectangle([mask_l, mask_top, mask_r, mask_bottom],
                               radius=w // 12, fill="white", outline="#B0BEC5", width=2)
        # Mask ear loops
        draw.arc([mask_l - w // 10, mask_top - h // 10, mask_l + w // 10, cy],
                 270, 90, fill="#B0BEC5", width=stroke)
        draw.arc([mask_r - w // 10, mask_top - h // 10, mask_r + w // 10, cy],
                 90, 270, fill="#B0BEC5", width=stroke)
        # Eyes above mask
        draw.ellipse([cx - w // 4 - eye_r, eye_y - eye_r, cx - w // 4 + eye_r, eye_y + eye_r],
                     fill="#0D47A1")
        draw.ellipse([cx + w // 4 - eye_r, eye_y - eye_r, cx + w // 4 + eye_r, eye_y + eye_r],
                     fill="#0D47A1")

    elif emoji_type == "cat":
        # Orange circle with cat ears
        ear_h = h // 4
        ear_w = w // 5
        # Left ear
        draw.polygon([cx - w // 3 - ear_w, cy - h // 4, cx - w // 5, cy - h // 5 + ear_h,
                      cx - w // 4, cy - h // 4],
                     fill="#FFAB91", outline="#E64A19", width=stroke)
        # Right ear
        draw.polygon([cx + w // 3 + ear_w, cy - h // 4, cx + w // 5, cy - h // 5 + ear_h,
                      cx + w // 4, cy - h // 4],
                     fill="#FFAB91", outline="#E64A19", width=stroke)
        # Inner ears
        draw.polygon([cx - w // 3 - ear_w // 2, cy - h // 5, cx - w // 5, cy - h // 6 + ear_h * 2 // 3,
                      cx - w // 4, cy - h // 5],
                     fill="#FFCCBC")
        draw.polygon([cx + w // 3 + ear_w // 2, cy - h // 5, cx + w // 5, cy - h // 6 + ear_h * 2 // 3,
                      cx + w // 4, cy - h // 5],
                     fill="#FFCCBC")
        # Face circle
        draw.ellipse([margin, margin, w - margin, h - margin],
                     fill="#FFAB91", outline="#E64A19", width=stroke)
        # Eyes
        draw.ellipse([cx - w // 5 - eye_r, eye_y - eye_r, cx - w // 5 + eye_r, eye_y + eye_r],
                     fill="#BF360C")
        draw.ellipse([cx + w // 5 - eye_r, eye_y - eye_r, cx + w // 5 + eye_r, eye_y + eye_r],
                     fill="#BF360C")
        # Nose
        draw.ellipse([cx - 2, cy + h // 20 - 2, cx + 2, cy + h // 20 + 2], fill="#E64A19")
        # Mouth (two arcs)
        draw.arc([cx - w // 6, cy + h // 15, cx, cy + h // 6],
                 0, 180, fill="#BF360C", width=stroke)
        draw.arc([cx, cy + h // 15, cx + w // 6, cy + h // 6],
                 0, 180, fill="#BF360C", width=stroke)
        # Whiskers
        for side, dir in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ws = cx + side * w // 5
            we = cx + side * w // 2
            wy = cy + h // 20 + dir * h // 20
            draw.line([ws, wy, we, wy + dir * h // 15], fill="#BF360C", width=max(1, stroke - 1))

    elif emoji_type == "dog":
        # Green circle with floppy dog ears
        ear_h = h // 3
        ear_w = w // 4
        # Left floppy ear
        draw.ellipse([cx - w // 3 - ear_w, cy - h // 3, cx - w // 6, cy],
                     fill="#A5D6A7", outline="#388E3C", width=stroke)
        # Right floppy ear
        draw.ellipse([cx + w // 6, cy - h // 3, cx + w // 3 + ear_w, cy],
                     fill="#A5D6A7", outline="#388E3C", width=stroke)
        # Face circle
        draw.ellipse([margin, margin, w - margin, h - margin],
                     fill="#A5D6A7", outline="#388E3C", width=stroke)
        # Eyes
        draw.ellipse([cx - w // 5 - eye_r, eye_y - eye_r, cx - w // 5 + eye_r, eye_y + eye_r],
                     fill="#1B5E20")
        draw.ellipse([cx + w // 5 - eye_r, eye_y - eye_r, cx + w // 5 + eye_r, eye_y + eye_r],
                     fill="#1B5E20")
        # Nose
        draw.ellipse([cx - w // 12, cy + h // 20 - h // 16, cx + w // 12, cy + h // 20],
                     fill="#2E7D32")
        # Tongue
        draw.ellipse([cx - w // 10, cy + h // 6, cx + w // 10, cy + h // 3],
                     fill="#E57373")
        draw.line([cx, cy + h // 6, cx, cy + h // 4], fill="#C62828", width=1)

    elif emoji_type == "bear":
        # Brown circle with round bear ears
        ear_r = w // 6
        # Left ear
        draw.ellipse([cx - w // 3 - ear_r, cy - h // 3, cx - w // 3 + ear_r, cy - h // 3 + ear_r * 2],
                     fill="#BCAAA4", outline="#5D4037", width=stroke)
        draw.ellipse([cx - w // 3 - ear_r * 2 // 3, cy - h // 3 + ear_r // 3,
                      cx - w // 3 + ear_r * 2 // 3, cy - h // 3 + ear_r],
                     fill="#D7CCC8")
        # Right ear
        draw.ellipse([cx + w // 3 - ear_r, cy - h // 3, cx + w // 3 + ear_r, cy - h // 3 + ear_r * 2],
                     fill="#BCAAA4", outline="#5D4037", width=stroke)
        draw.ellipse([cx + w // 3 - ear_r * 2 // 3, cy - h // 3 + ear_r // 3,
                      cx + w // 3 + ear_r * 2 // 3, cy - h // 3 + ear_r],
                     fill="#D7CCC8")
        # Face circle
        draw.ellipse([margin, margin, w - margin, h - margin],
                     fill="#BCAAA4", outline="#5D4037", width=stroke)
        # Muzzle (lighter oval)
        mz_w = w // 4
        mz_h = h // 5
        draw.ellipse([cx - mz_w, cy - h // 20, cx + mz_w, cy + h // 5],
                     fill="#D7CCC8")
        # Eyes
        draw.ellipse([cx - w // 5 - eye_r, eye_y - eye_r, cx - w // 5 + eye_r, eye_y + eye_r],
                     fill="#3E2723")
        draw.ellipse([cx + w // 5 - eye_r, eye_y - eye_r, cx + w // 5 + eye_r, eye_y + eye_r],
                     fill="#3E2723")
        # Nose
        draw.ellipse([cx - w // 16, cy + h // 30 - h // 30, cx + w // 16, cy + h // 30 + h // 30],
                     fill="#3E2723")
        draw.arc([cx - w // 8, cy + h // 15, cx + w // 8, cy + h // 5],
                 0, 180, fill="#3E2723", width=stroke)

    else:  # smile (default)
        # Yellow circle with smiley face
        draw.ellipse([margin, margin, w - margin, h - margin],
                     fill="#FDD835", outline="#F9A825", width=stroke)
        draw.ellipse([cx - w // 5 - eye_r, eye_y - eye_r, cx - w // 5 + eye_r, eye_y + eye_r],
                     fill="#5D4037")
        draw.ellipse([cx + w // 5 - eye_r, eye_y - eye_r, cx + w // 5 + eye_r, eye_y + eye_r],
                     fill="#5D4037")
        draw.arc([cx - smile_w, smile_y - smile_h, cx + smile_w, smile_y + smile_h],
                 0, 180, fill="#5D4037", width=stroke)

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
