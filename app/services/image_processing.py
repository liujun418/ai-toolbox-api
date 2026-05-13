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


def postprocess_image(image_bytes: bytes) -> bytes:
    """Post-process AI output for consistent quality and format.

    - Applies mild unsharp mask sharpening
    - Converts to WebP for uniform output
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed (handles RGBA output from some models)
        if img.mode == "RGBA":
            # Keep alpha: save as PNG for RGBA results
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            return buf.read()
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Sharpening (enhanced for crisp output)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3))

        # Contrast boost for vibrancy
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.06)

        # Subtle saturation boost
        sat_enhancer = ImageEnhance.Color(img)
        img = sat_enhancer.enhance(1.03)

        # Output as WebP (quality 90 for crisp detail)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=90, optimize=True)
        buf.seek(0)
        return buf.read()
    except Exception:
        # If postprocessing fails, return original bytes — don't break the pipeline
        return image_bytes
