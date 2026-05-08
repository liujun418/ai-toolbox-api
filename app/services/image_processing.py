"""Local image processing using rembg (no external API needed)."""

import io

from rembg import remove


async def run_background_remover_local(file_bytes: bytes) -> bytes:
    """Remove background from image bytes. Returns output image bytes."""
    return remove(file_bytes)
