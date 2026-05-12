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
    """Remove watermarks/logos using BRIA Eraser (purpose-built inpainting).

    Takes image + mask and outputs a cleaned image where the masked area
    is seamlessly reconstructed from surrounding context.
    """
    client = get_replicate()
    output = client.run(
        "bria/eraser:893e924eecc119a0c5fbfa5d98401118dcbf0662574eb8d2c01be5749756cbd4",
        input={
            "image": image_url,
            "mask": mask_url,
        },
    )
    return str(output)


async def run_photo_restoration(image_url: str, colorize: bool = False) -> str:
    """Restore old/damaged photo using GFPGAN."""
    client = get_replicate()
    output = client.run(
        "xinntiao/gfpgan:92296352d6ba42479f5c1629c5a2007e5cc09a71a08e2695d3e3d27e11069496",
        input={
            "img": image_url,
            "version": "v1.4",
            "scale": 2,
            "weight": 0.5,
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
    # SDXL img2img for avatar generation
    output = client.run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "prompt": f"cartoon avatar portrait, anime style, vibrant colors, {style}",
            "image": image_url,
            "prompt_strength": 0.6,
            "num_outputs": 4,
            "num_inference_steps": 25,
        },
    )
    return [str(u) for u in output]


async def run_image_upscaler(image_url: str, scale: int = 2) -> str:
    """Upscale image using Real-ESRGAN super-resolution.
    Returns URL of upscaled image. Scale: 2 or 4."""
    client = get_replicate()
    output = client.run(
        "nightmareai/real-esrgan",
        input={
            "image": image_url,
            "scale": scale,
            "face_enhance": True,
        },
    )
    return str(output)


async def run_style_transfer(image_url: str, style: str = "oil-painting") -> str:
    """Transform image into artistic style (oil-painting, watercolor, anime, sketch).
    Returns URL of stylized image."""
    client = get_replicate()

    style_prompts = {
        "oil-painting": "oil painting style, thick brush strokes, rich colors, impressionist",
        "watercolor": "watercolor painting style, soft edges, translucent colors, artistic",
        "anime": "anime art style, cel shading, vibrant colors, manga illustration",
        "sketch": "pencil sketch style, black and white drawing, detailed line art",
    }
    prompt = style_prompts.get(style, style_prompts["oil-painting"])

    output = client.run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "prompt": prompt,
            "image": image_url,
            "prompt_strength": 0.6,
            "num_outputs": 1,
            "num_inference_steps": 25,
        },
    )
    urls = [str(u) for u in output]
    return urls[0] if urls else ""


async def run_text_polish(text: str, mode: str = "polish") -> str:
    """Polish, rewrite, shorten, or expand text using LLM.
    Returns processed text."""
    client = get_replicate()

    mode_instructions = {
        "polish": "Improve the grammar, spelling, and clarity of the given text while keeping the same meaning. Return only the improved text, no explanations.",
        "rewrite": "Rewrite the given text with different wording while keeping the same meaning. Return only the rewritten text, no explanations.",
        "shorten": "Make the given text more concise while keeping the key points. Return only the shortened text, no explanations.",
        "expand": "Expand the given text with more detail and explanation. Return only the expanded text, no explanations.",
    }
    instruction = mode_instructions.get(mode, mode_instructions["polish"])

    output = client.run(
        "meta/meta-llama-3.1-70b-instruct:baf226e1f0cc30952e39198a7dc1e8083d2686196464e0665e2d88108db29c61",
        input={
            "system_prompt": instruction,
            "prompt": text,
            "max_tokens": 4096,
            "temperature": 0.7,
        },
    )
    return "".join(list(output)).strip()
