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

        # Mild sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # Subtle contrast boost for vibrancy
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.05)

        # Output as WebP
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=WEBP_QUALITY, optimize=True)
        buf.seek(0)
        return buf.read()
    except Exception:
        # If postprocessing fails, return original bytes — don't break the pipeline
        return image_bytes
