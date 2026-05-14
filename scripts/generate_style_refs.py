"""Generate 6 abstract style reference images for fofr/style-transfer IP-Adapter.
Run once: python scripts/generate_style_refs.py
Uploads to R2 and prints the keys to hardcode in prompt_templates.py.
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFilter
import random

from app.services.storage import upload_file, generate_presigned_url

SIZE = 768  # Good size for IP-Adapter reference
SEED = 42
random.seed(SEED)


def oil_painting_ref() -> bytes:
    """Thick brush strokes on warm canvas background."""
    img = Image.new("RGB", (SIZE, SIZE), (230, 210, 180))
    draw = ImageDraw.Draw(img)
    colors = [(180, 100, 40), (200, 150, 80), (120, 60, 30), (220, 180, 130), (160, 90, 50)]
    for _ in range(200):
        x1, y1 = random.randint(0, SIZE), random.randint(0, SIZE)
        x2, y2 = x1 + random.randint(-60, 60), y1 + random.randint(-60, 60)
        w = random.randint(3, 18)
        draw.line([(x1, y1), (x2, y2)], fill=random.choice(colors), width=w)
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def watercolor_ref() -> bytes:
    """Soft translucent color washes."""
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
            odraw.ellipse([(ox - orad, oy - orad), (ox + orad, oy + orad)],
                          fill=(c[0], c[1], c[2], 40))
        overlay = overlay.filter(ImageFilter.GaussianBlur(30))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(1))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def sketch_ref() -> bytes:
    """Pencil cross-hatching on white paper."""
    img = Image.new("RGB", (SIZE, SIZE), (248, 246, 240))
    draw = ImageDraw.Draw(img)
    for _ in range(600):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        angle = random.uniform(0, 6.28)
        length = random.randint(15, 80)
        end_x = x + length * 0.7
        end_y = y + length * 0.3
        shade = random.randint(60, 160)
        draw.line([(x, y), (end_x, end_y)], fill=(shade, shade, shade), width=1)
    for _ in range(300):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        end_x = x + random.randint(10, 50)
        end_y = y
        shade = random.randint(100, 200)
        draw.line([(x, y), (end_x, end_y)], fill=(shade, shade, shade), width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def cartoon_ref() -> bytes:
    """Smooth 3D polished gradients."""
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


def cyberpunk_ref() -> bytes:
    """Neon lights on dark synthwave background."""
    img = Image.new("RGB", (SIZE, SIZE), (15, 5, 35))
    draw = ImageDraw.Draw(img)
    # Gradient background
    for y in range(SIZE):
        r = int(15 + 10 * (y / SIZE))
        g = int(5 + 5 * (y / SIZE))
        b = int(35 + 30 * (y / SIZE))
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))
    # Neon lines
    neon_colors = [(0, 240, 255), (255, 0, 180), (0, 255, 120), (255, 100, 0), (180, 0, 255)]
    for _ in range(40):
        x1, y1 = random.randint(0, SIZE), random.randint(0, SIZE)
        x2, y2 = x1 + random.randint(-200, 200), y1 + random.randint(-200, 200)
        c = random.choice(neon_colors)
        draw.line([(x1, y1), (x2, y2)], fill=c, width=random.randint(1, 3))
    # Glow dots
    for _ in range(80):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(2, 6)
        c = random.choice(neon_colors)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=c)
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def fantasy_ref() -> bytes:
    """Magical glowing bokeh and sparkles on dark mystical background."""
    img = Image.new("RGB", (SIZE, SIZE), (10, 5, 30))
    draw = ImageDraw.Draw(img)
    # Mystical gradient
    for y in range(SIZE):
        r = int(10 + 20 * (1 - abs(y - SIZE / 2) / (SIZE / 2)))
        g = int(5 + 10 * (1 - abs(y - SIZE / 2) / (SIZE / 2)))
        b = int(30 + 40 * (y / SIZE))
        draw.line([(0, y), (SIZE, y)], fill=(r, g, b))
    # Glowing bokeh particles
    glow_colors = [(255, 220, 180), (180, 200, 255), (255, 200, 255), (200, 255, 220), (255, 255, 200)]
    for _ in range(60):
        cx, cy = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(10, 50)
        c = random.choice(glow_colors)
        for i in range(r, 0, -3):
            alpha = int(100 * (i / r))
            shade = tuple(int(c[j] * alpha / 255 + 10 * (1 - i / r)) for j in range(3))
            draw.ellipse([(cx - i, cy - i), (cx + i, cy + i)], fill=shade)
    # Sparkles
    for _ in range(200):
        x, y = random.randint(0, SIZE), random.randint(0, SIZE)
        r = random.randint(1, 3)
        c = random.choice(glow_colors)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=c)
    img = img.filter(ImageFilter.GaussianBlur(2))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


STYLES = {
    "oil-painting": oil_painting_ref,
    "watercolor": watercolor_ref,
    "sketch": sketch_ref,
    "cartoon": cartoon_ref,
    "cyberpunk": cyberpunk_ref,
    "fantasy": fantasy_ref,
}


async def main():
    for style_id, generator in STYLES.items():
        image_bytes = generator()
        key = f"style-references/{style_id}.png"
        await upload_file(image_bytes, key, "image/png")
        url = generate_presigned_url(key, expires_in=86400 * 365)  # 1 year
        print(f'"{style_id}": "{url}",')


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
