"""Replicate API service for AI image processing."""

import replicate

from app.config import settings


def get_replicate():
    """Get authenticated Replicate client."""
    return replicate.Client(api_token=settings.REPLICATE_API_TOKEN)


async def run_background_remover(image_url: str) -> str:
    """Remove background from image using rembg model.
    Returns URL of output image."""
    client = get_replicate()
    output = client.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": image_url},
    )
    # Replicate returns a file-like object or URL
    return str(output)


async def run_watermark_removal(image_url: str, mask_url: str) -> str:
    """Remove watermark using LaMa inpainting model."""
    client = get_replicate()
    output = client.run(
        "talesofai/lama:927d0c87970210e45524c3648c121961d4503047f80989a06f32928b561953cd",
        input={"image": image_url, "mask": mask_url},
    )
    return str(output)


async def run_photo_restoration(image_url: str, colorize: bool = False) -> str:
    """Restore old/damaged photo using GFPGAN."""
    client = get_replicate()
    model = "xinntiao/gfpgan:92296352d6ba42479f5c1629c5a2007e5cc09a71a08e2695d3e3d27e11069496"
    output = client.run(
        model,
        input={
            "img": image_url,
            "version": "v1.4",
            "upscale": 2,
        },
    )
    return str(output)


async def run_avatar_generation(
    image_url: str,
    style: str = "cartoon",
) -> list[str]:
    """Generate avatar/cartoon from photo using SDXL.
    Returns list of generated image URLs."""
    client = get_replicate()
    # SDXL with a cartoon/anime LoRA
    output = client.run(
        "stability-ai/sdxl:778a0147fb3535e38c831e5d42e0fa29e3c2c8c4e3b4b4b1b4b4b4b4b4b4b4b",
        input={
            "prompt": f"cartoon avatar portrait, anime style, vibrant colors, {style}",
            "image": image_url,
            "strength": 0.7,
            "num_outputs": 4,
        },
    )
    return [str(u) for u in output]
