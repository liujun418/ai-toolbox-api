"""Face detection and privacy blur service.

Uses OpenCV Haar cascade for face detection (offline, no API cost).
Supports three blur styles: mosaic, gaussian, pixelate.
"""

import io
import logging
import math

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Load Haar cascade once at module level
_face_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def detect_faces(image_bytes: bytes) -> list[dict]:
    """Detect faces in image, return list of {x, y, w, h}."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]


def _apply_blur_region(
    img: np.ndarray,
    region: dict,
    style: str,
    base_block_size: int = 12,
) -> None:
    """Apply blur style to a single rectangular region (in-place)."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

    # Clamp to image bounds
    h_img, w_img = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return

    roi = img[y : y + h, x : x + w]

    if style == "mosaic":
        # Pixelate: downsample then upsample
        block = max(4, base_block_size)
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    elif style == "gaussian":
        # Strong gaussian blur
        ksize = max(15, min(w, h) // 4)
        if ksize % 2 == 0:
            ksize += 1
        roi[:] = cv2.GaussianBlur(roi, (ksize, ksize), 30)

    elif style == "pixelate":
        # Cell-based pixelation with larger blocks
        block = max(6, min(w, h) // 10)
        for by in range(0, h, block):
            for bx in range(0, w, block):
                bh = min(block, h - by)
                bw = min(block, w - bx)
                cell = roi[by : by + bh, bx : bx + bw]
                avg_color = cell.mean(axis=(0, 1)).astype(np.uint8)
                roi[by : by + bh, bx : bx + bw] = avg_color


def _expand_region(region: dict, factor: float = 0.15) -> dict:
    """Expand a face region slightly to cover the full head."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    dx = int(w * factor)
    dy = int(h * factor)
    return {
        "x": x - dx,
        "y": y - dy // 2,  # expand up less (face detection covers chin)
        "w": w + 2 * dx,
        "h": h + 2 * dy,
    }


def apply_face_blur(
    image_bytes: bytes,
    blur_style: str = "mosaic",
    manual_regions: list[dict] | None = None,
) -> tuple[bytes, int, int]:
    """Apply face blur to an image.

    Returns (processed_image_bytes, face_count, region_count).
    """
    if blur_style not in ("mosaic", "gaussian", "pixelate"):
        blur_style = "mosaic"

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    # Detect faces
    auto_regions = detect_faces(image_bytes)

    # Expand face regions to cover full head
    regions = [_expand_region(r) for r in auto_regions]

    # Add manual regions
    if manual_regions:
        regions.extend(manual_regions)

    # Apply blur
    for region in regions:
        _apply_blur_region(img, region, blur_style)

    # Encode result as PNG (lossless)
    success, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise RuntimeError("Failed to encode output image")

    return buf.tobytes(), len(auto_regions), len(regions)


def count_faces(image_bytes: bytes) -> int:
    """Quick face count without applying blur (for credit cost determination)."""
    faces = detect_faces(image_bytes)
    return len(faces)
