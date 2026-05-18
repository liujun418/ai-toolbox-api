"""Face detection and privacy blur service.

Uses Google MediaPipe (BlazeFace) for AI face detection — far more accurate
than OpenCV Haar cascade, especially for profile/side faces, partial occlusion,
and varied angles. Runs entirely offline with no API cost.

Supports four blur styles: mosaic, gaussian, pixelate, emoji.
"""

import io
import logging
import math

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import — cv2 needs system libs (libxcb etc.) that may not be available at startup
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_module
        _cv2 = _cv2_module
    return _cv2

# MediaPipe Face Detection — lazy init (loads AI model once)
_mp_face_detection = None


def _get_face_detector():
    global _mp_face_detection
    if _mp_face_detection is None:
        import mediapipe as mp
        _mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0=short-range (≤2m), 1=full-range (≤5m, better for photos)
            min_detection_confidence=0.5,
        )
    return _mp_face_detection


def detect_faces(image_bytes: bytes) -> list[dict]:
    """Detect faces in image using MediaPipe AI model.

    Returns list of {x, y, w, h} in pixel coordinates.
    Much more accurate than Haar cascade: handles profile faces, partial
    occlusion, varied lighting, and angles that Haar cascade misses.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = _get_cv2().imdecode(nparr, _get_cv2().IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    h, w = img.shape[:2]

    # MediaPipe requires RGB
    rgb = _get_cv2().cvtColor(img, _get_cv2().COLOR_BGR2RGB)
    detector = _get_face_detector()
    results = detector.process(rgb)

    faces = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            # Convert normalized (0-1) to pixel coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            faces.append({"x": max(0, x), "y": max(0, y), "w": bw, "h": bh})

    return faces


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
        small = _get_cv2().resize(roi, (max(1, w // block), max(1, h // block)), interpolation=_get_cv2().INTER_LINEAR)
        roi[:] = _get_cv2().resize(small, (w, h), interpolation=_get_cv2().INTER_NEAREST)

    elif style == "gaussian":
        # Strong gaussian blur
        ksize = max(15, min(w, h) // 4)
        if ksize % 2 == 0:
            ksize += 1
        roi[:] = _get_cv2().GaussianBlur(roi, (ksize, ksize), 30)

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

EMOJI_OPTIONS = {
    "smile": "😊",
    "mask": "😷",
    "cat": "🐱",
    "dog": "🐶",
    "bear": "🐻",
    "star": "⭐",
}


def _apply_emoji_region(img: np.ndarray, region: dict, emoji_char: str) -> None:
    """Overlay a cute emoji on a face region using PIL."""
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    h_img, w_img = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return

    # Convert ROI to PIL for text rendering
    roi_bgr = img[y : y + h, x : x + w]
    roi_rgb = _get_cv2().cvtColor(roi_bgr, _get_cv2().COLOR_BGR2RGB)
    pil_roi = Image.fromarray(roi_rgb)

    # Draw emoji centered in the region
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(pil_roi)
    font_size = max(min(w, h) // 2, 20)
    # Use default font — emoji rendering depends on OS support
    try:
        font = ImageFont.truetype("seguiemj.ttf", font_size)  # Windows Segoe UI Emoji
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Center the emoji
    bbox = draw.textbbox((0, 0), emoji_char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = (h - th) // 2
    draw.text((tx, ty), emoji_char, font=font, embedded_color=True)

    # Convert back to BGR and place in original image
    result_rgb = np.array(pil_roi)
    result_bgr = _get_cv2().cvtColor(result_rgb, _get_cv2().COLOR_RGB2BGR)
    img[y : y + h, x : x + w] = result_bgr


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
    emoji_type: str = "smile",
    auto_only: bool = False,
    auto_regions: list[dict] | None = None,
) -> tuple[bytes, int, int]:
    """Apply face blur to an image.

    auto_regions: pre-detected face regions from client. If None, detect on server.
    manual_regions: user-drawn supplementary regions.
    auto_only: deprecated legacy flag (ignored when auto_regions is provided).
    Returns (processed_image_bytes, face_count, region_count).
    """
    if blur_style not in ("mosaic", "gaussian", "pixelate", "emoji"):
        blur_style = "mosaic"

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = _get_cv2().imdecode(nparr, _get_cv2().IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    # Use client-provided regions or detect on server
    if auto_regions is not None:
        regions = [_expand_region(r) for r in auto_regions]
        face_count = len(auto_regions)
    else:
        detected = detect_faces(image_bytes)
        regions = [_expand_region(r) for r in detected]
        face_count = len(detected)

    if manual_regions:
        regions.extend(manual_regions)

    # Apply style
    emoji_char = EMOJI_OPTIONS.get(emoji_type, "😊")
    for region in regions:
        if blur_style == "emoji":
            _apply_emoji_region(img, region, emoji_char)
        else:
            _apply_blur_region(img, region, blur_style)

    # Encode result as PNG (lossless)
    success, buf = _get_cv2().imencode(".png", img, [_get_cv2().IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise RuntimeError("Failed to encode output image")

    return buf.tobytes(), len(auto_regions), len(regions)


def count_faces(image_bytes: bytes) -> int:
    """Quick face count without applying blur (for credit cost determination)."""
    faces = detect_faces(image_bytes)
    return len(faces)
