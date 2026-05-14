"""Image preprocessing and postprocessing pipeline.

Ensures all uploaded images are within safe bounds before AI processing,
and all outputs are consistently formatted with quality enhancement.
"""

import io

from PIL import Image, ImageFilter, ImageEnhance

# Safety limits
MAX_DIMENSION = 4096
MAX_FILE_BYTES = 4 * 1024 * 1024  # 4MB threshold for compression
JPEG_QUALITY = 85
WEBP_QUALITY = 85


def preprocess_image(image_bytes: bytes, keep_alpha: bool = False) -> tuple[bytes, dict]:
    """Preprocess an uploaded image before sending to AI model.

    - Validates and resizes if exceeding MAX_DIMENSION
    - Converts to RGB (unless keep_alpha=True for rembg)
    - Compresses if > MAX_FILE_BYTES
    - Strips EXIF metadata

    Returns (processed_bytes, metadata_dict).
    """
    img = Image.open(io.BytesIO(image_bytes))
    orig_w, orig_h = img.size
    orig_format = img.format

    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "original_format": str(orig_format) if orig_format else "unknown",
    }

    # Resize if exceeds max dimension
    if max(orig_w, orig_h) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(orig_w, orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        metadata["resized"] = True
        metadata["new_width"] = new_w
        metadata["new_height"] = new_h

    # Color mode
    if keep_alpha:
        img = img.convert("RGBA")
    else:
        # Remove alpha for non-rembg tools (prevents model errors)
        if img.mode in ("RGBA", "LA", "PA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

    # Compress if too large
    buf = io.BytesIO()
    if img.size[0] * img.size[1] * 3 > MAX_FILE_BYTES:
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        metadata["compressed"] = True
        metadata["output_format"] = "JPEG"
    else:
        img.save(buf, format="PNG", optimize=True)
        metadata["output_format"] = "PNG"

    buf.seek(0)
    result = buf.read()
    metadata["processed_size"] = len(result)
    return result, metadata


def preprocess_avatar(image_bytes: bytes) -> tuple[bytes, dict]:
    """Preprocess an uploaded photo for avatar generation.

    - Center square crop (shortest edge) for consistent portrait framing
    - Resize to 1024x1024 (SDXL optimal resolution)
    - Convert to RGB, white background for alpha images
    - Output as high-quality JPEG

    Returns (processed_bytes, metadata_dict).
    """
    img = Image.open(io.BytesIO(image_bytes))
    orig_w, orig_h = img.size

    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "original_format": str(img.format) if img.format else "unknown",
    }

    # Center square crop: use shortest edge
    crop_size = min(orig_w, orig_h)
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))
    metadata["cropped"] = True
    metadata["crop_size"] = crop_size

    # Resize to SDXL optimal resolution
    img = img.resize((1024, 1024), Image.LANCZOS)
    metadata["resized"] = True
    metadata["new_width"] = 1024
    metadata["new_height"] = 1024

    # Color mode: remove alpha, composite on white
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Output as JPEG with good quality
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    buf.seek(0)
    result = buf.read()
    metadata["processed_size"] = len(result)
    return result, metadata


def preprocess_style_transfer(image_bytes: bytes) -> tuple[bytes, dict]:
    """Preprocess an uploaded photo for style transfer.

    - Center square crop (shortest edge) for consistent composition
    - Resize to 1024x1024 (SDXL optimal resolution for img2img)
    - Convert to RGB, white background for alpha images
    - Output as high-quality JPEG

    Returns (processed_bytes, metadata_dict).
    """
    img = Image.open(io.BytesIO(image_bytes))
    orig_w, orig_h = img.size

    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "original_format": str(img.format) if img.format else "unknown",
    }

    # Center square crop: use shortest edge
    crop_size = min(orig_w, orig_h)
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))
    metadata["cropped"] = True
    metadata["crop_size"] = crop_size

    # Resize to SDXL optimal resolution
    img = img.resize((1024, 1024), Image.LANCZOS)
    metadata["resized"] = True
    metadata["new_width"] = 1024
    metadata["new_height"] = 1024

    # Color mode: remove alpha, composite on white
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Output as JPEG with good quality
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    buf.seek(0)
    result = buf.read()
    metadata["processed_size"] = len(result)
    return result, metadata


def feather_alpha(image_bytes: bytes, radius: float = 1.5) -> bytes:
    """Smooth the alpha channel to eliminate hard edges and halos.

    Separates the alpha channel, applies Gaussian blur to create smooth
    edge transitions, then recombines with the original RGB data.

    Returns feather-edged RGBA PNG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    r, g, b, a = img.split()
    # Gaussian blur on alpha channel for smooth edge transition
    a = a.filter(ImageFilter.GaussianBlur(radius=radius))
    img = Image.merge("RGBA", (r, g, b, a))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


def composite_on_background(
    image_bytes: bytes,
    color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """Composite an RGBA image onto a solid color background.

    Returns RGB PNG bytes (alpha channel flattened into solid color).
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    r, g, b, a = img.split()
    background = Image.new("RGB", img.size, color)
    background.paste(img, mask=a)

    buf = io.BytesIO()
    background.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


def postprocess_image(
    image_bytes: bytes,
    feather_radius: float = 0,
    bg_color: tuple[int, int, int] | None = None,
    sharpen_percent: int = 180,
    contrast_boost: float = 1.06,
) -> bytes:
    """Post-process AI output for consistent quality and format.

    - feather_radius > 0: applies Gaussian blur to alpha channel for smooth edges
    - bg_color: composites result onto solid color background (None = keep transparent)
    - sharpen_percent: UnsharpMask percent (default 180, higher = more sharpening)
    - contrast_boost: Contrast enhance factor (default 1.06, higher = more contrast)
    - Non-RGBA images: sharpening, contrast, saturation, WebP output
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # RGBA path: feather alpha, optionally fill background, output PNG
        if img.mode == "RGBA":
            if feather_radius > 0:
                r, g, b, a = img.split()
                a = a.filter(ImageFilter.GaussianBlur(radius=feather_radius))
                img = Image.merge("RGBA", (r, g, b, a))

            if bg_color is not None:
                r, g, b, a = img.split()
                background = Image.new("RGB", img.size, bg_color)
                background.paste(img, mask=a)
                img = background
                # Apply sharpening to composite result
                img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=sharpen_percent, threshold=3))
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast_boost if contrast_boost > 1.04 else 1.04)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            return buf.read()

        # Non-RGBA image (e.g., RGB from upscaler)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=sharpen_percent, threshold=3))

        # Contrast boost
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_boost)

        # Subtle saturation boost
        sat_enhancer = ImageEnhance.Color(img)
        img = sat_enhancer.enhance(1.03)

        # Output as WebP
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=90, optimize=True)
        buf.seek(0)
        return buf.read()
    except Exception:
        return image_bytes
