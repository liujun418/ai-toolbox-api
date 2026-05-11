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
    """Remove watermark using SDXL inpainting — only inpaints masked areas."""
    client = get_replicate()
    output = client.run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "image": image_url,
            "mask": mask_url,
            "prompt": "clean background without any text, watermark, logo, or overlay. Seamless restoration matching surrounding colors and textures.",
            "prompt_strength": 0.15,
            "num_inference_steps": 25,
        },
    )
    urls = [str(u) for u in output]
    return urls[0] if urls else ""


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
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "prompt": f"cartoon avatar portrait, anime style, vibrant colors, {style}",
            "image": image_url,
            "strength": 0.7,
            "num_outputs": 4,
        },
    )
    return [str(u) for u in output]


async def run_image_upscaler(image_url: str, scale: str = "2x") -> str:
    """Upscale image using Real-ESRGAN super-resolution.
    Returns URL of upscaled image."""
    client = get_replicate()
    upscale_factor = 2 if scale == "2x" else 4
    output = client.run(
        "stability-ai/esrgan-v2:13aa845e31c1d5a4d0067ba9351dd6b3961dc4e10e28af0f15a45450c0d4e7f0",
        input={
            "image": image_url,
            "scale": upscale_factor,
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
            "strength": 0.75,
            "num_outputs": 1,
        },
    )
    urls = [str(u) for u in output]
    return urls[0] if urls else ""


async def run_text_polish(text: str, mode: str = "polish") -> str:
    """Polish, rewrite, shorten, or expand text using LLM.
    Returns processed text."""
    client = get_replicate()

    mode_instructions = {
        "polish": "Improve the grammar, spelling, and clarity of the following text while keeping the same meaning.",
        "rewrite": "Rewrite the following text with different wording while keeping the same meaning.",
        "shorten": "Make the following text more concise while keeping the key points.",
        "expand": "Expand the following text with more detail and explanation.",
    }
    instruction = mode_instructions.get(mode, mode_instructions["polish"])

    output = client.run(
        "meta/meta-llama-3-70b-instruct:fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d",
        input={
            "prompt": f"{instruction}\n\nText: {text}\n\nResult:",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
    )
    return "".join(list(output)).strip()
