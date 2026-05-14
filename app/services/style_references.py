"""Style reference image generation for fofr/style-transfer IP-Adapter.
Generates abstract style exemplars using PIL at startup and caches their R2 URLs.
"""

import io
import random

from PIL import Image, ImageDraw, ImageFilter

from app.services.storage import upload_file, generate_presigned_url

SIZE = 768
SEED = 42
random.seed(SEED)

# Cache: style_id -> presigned URL
_url_cache: dict[str, str] = {}
_initialized = False


def _oil_painting() -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), (230, 210, 180))
    draw = ImageDraw.Draw(img)
    colors = [(180, 100, 40), (200, 150, 80), (120, 60, 30), (220, 180, 130), (160, 90, 50)]
    for _ in range(200):
        x1, y1 = random.randint(0, SIZE), random.randint(0, SIZE)
        x2, y2 = x1 + random.randint(-60, 60), y1 + random.randint(-60, 60)
        draw.line([(x1, y1), (x2, y2)], fill=random.choice(colors), width=random.randint(3, 18))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _watercolor() -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), (250, 248, 240))
    colors = [(200, 180, 210), (180, 200, 220), (210, 200, 180), (190, 210, 200), (220, 180, 190)]
    for _ in range(30):
        overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        cx, cy = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(80, 250)
        c = random.choice(colors)
        for _ in range(5):
            ox, oy = cx + random.randint(-40, 40), cy + random.randint(-40, 40)
            orad = int(r * random.uniform(0.5, 1.2))
            odraw.ellipse([(ox - orad, oy - orad), (ox + orad, oy + orad)], fill=(c[0], c[1], c[2], 40))
        overlay = overlay.filter(ImageFilter.GaussianBlur(30))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(1))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _sketch() -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), (248, 246, 240))
    draw = ImageDraw.Draw(img)
    for _ in range(600):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        length = random.randint(15, 80)
        shade = random.randint(60, 160)
        draw.line([(x, y), (x + length * 0.7, y + length * 0.3)], fill=(shade, shade, shade), width=1)
    for _ in range(300):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        shade = random.randint(100, 200)
        draw.line([(x, y), (x + random.randint(10, 50), y)], fill=(shade, shade, shade), width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _cartoon() -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), (240, 230, 250))
    draw = ImageDraw.Draw(img)
    colors = [(255, 180, 180), (180, 220, 255), (255, 220, 180), (200, 255, 200), (255, 200, 255)]
    for _ in range(15):
        cx, cy = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(60, 200)
        c = random.choice(colors)
        for i in range(r, 0, -2):
            ratio = i / r
            shade = tuple(int(c[j] * 0.4 + 255 * 0.6 * ratio) for j in range(3))
            draw.ellipse([(cx - i, cy - i), (cx + i, cy + i)], fill=shade)
    img = img.filter(ImageFilter.GaussianBlur(8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _cyberpunk() -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), (15, 5, 35))
    draw = ImageDraw.Draw(img)
    for y in range(SIZE):
        r = int(15 + 10 * (y / SIZE))
        g = int(5 + 5 * (y / SIZE))
        b = int(35 + 30 * (y / SIZE))
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))
    neon_colors = [(0, 240, 255), (255, 0, 180), (0, 255, 120), (255, 100, 0), (180, 0, 255)]
    for _ in range(40):
        x1, y1 = random.randint(0, SIZE), random.randint(0, SIZE)
        x2, y2 = x1 + random.randint(-200, 200), y1 + random.randint(-200, 200)
        draw.line([(x1, y1), (x2, y2)], fill=random.choice(neon_colors), width=random.randint(1, 3))
    for _ in range(80):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(2, 6)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=random.choice(neon_colors))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _fantasy() -> bytes:
    img = Image.new("RGB", (SIZE, SIZE), (10, 5, 30))
    draw = ImageDraw.Draw(img)
    for y in range(SIZE):
        r = int(10 + 20 * (1 - abs(y - SIZE / 2) / (SIZE / 2)))
        g = int(5 + 10 * (1 - abs(y - SIZE / 2) / (SIZE / 2)))
        b = int(30 + 40 * (y / SIZE))
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))
    glow_colors = [(255, 220, 180), (180, 200, 255), (255, 200, 255), (200, 255, 220), (255, 255, 200)]
    for _ in range(60):
        cx, cy = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(10, 50)
        c = random.choice(glow_colors)
        for i in range(r, 0, -3):
            alpha = int(100 * (i / r))
            shade = tuple(int(c[j] * alpha / 255 + 10 * (1 - i / r)) for j in range(3))
            draw.ellipse([(cx - i, cy - i), (cx + i, cy + i)], fill=shade)
    for _ in range(200):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(1, 3)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=random.choice(glow_colors))
    img = img.filter(ImageFilter.GaussianBlur(2))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_GENERATORS = {
    "oil-painting": _oil_painting,
    "watercolor": _watercolor,
    "sketch": _sketch,
    "cartoon": _cartoon,
    "cyberpunk": _cyberpunk,
    "fantasy": _fantasy,
}


async def init_style_references():
    """Generate and upload all style reference images to R2. Called once at startup."""
    global _initialized
    if _initialized:
        return
    for style_id, generator in _GENERATORS.items():
        image_bytes = generator()
        key = f"style-references/{style_id}.png"
        await upload_file(image_bytes, key, "image/png")
        _url_cache[style_id] = generate_presigned_url(key, expires_in=86400 * 365)
    _initialized = True


def get_style_reference_url(style_id: str) -> str | None:
    """Get the presigned URL for a style reference image. Must call init_style_references() first."""
    return _url_cache.get(style_id)
