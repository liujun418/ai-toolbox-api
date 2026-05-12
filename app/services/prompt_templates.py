"""Unified prompt templates for all AI image tools.

Each template contains a fixed positive prompt, negative prompt,
pinned model version, and locked generation parameters.
"""

from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """A complete prompt template for a single tool."""
    # Replicate model identifier (name:hash or bare name for non-pinned)
    model: str
    # Fixed positive prompt (may contain {user_prompt} placeholder)
    positive_prompt: str
    # Negative prompt (passed to models that support it)
    negative_prompt: str = ""
    # Locked generation parameters
    default_params: dict = field(default_factory=dict)


# ── Negative prompts ──────────────────────────────────────────────
NEGATIVE_SDXL = (
    "deformed, ugly, blurry, low quality, lowres, watermark, text, signature, "
    "bad anatomy, extra limbs, disfigured, poorly drawn face, mutation, "
    "duplicate, cropped, worst quality"
)
NEGATIVE_GFPGAN = "blurry, noisy, artifacted, oversharpened, oversaturated"
NEGATIVE_REMBG = "low quality, blurry, artifacts, jagged edges"
NEGATIVE_ERASER = "blurry, smudged, unnatural, low quality artifacts"

# ── Tool templates ────────────────────────────────────────────────

TOOL_PROMPTS: dict[str, PromptTemplate] = {
    "background-remover": PromptTemplate(
        model="cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        positive_prompt="",
        negative_prompt=NEGATIVE_REMBG,
        default_params={},
    ),

    "watermark-remover": PromptTemplate(
        model="bria/eraser:893e924eecc119a0c5fbfa5d98401118dcbf0662574eb8d2c01be5749756cbd4",
        positive_prompt="",
        negative_prompt=NEGATIVE_ERASER,
        default_params={},
    ),

    "photo-restorer": PromptTemplate(
        model="xinntiao/gfpgan:92296352d6ba42479f5c1629c5a2007e5cc09a71a08e2695d3e3d27e11069496",
        positive_prompt="",
        negative_prompt=NEGATIVE_GFPGAN,
        default_params={
            "version": "v1.4",
            "scale": 2,
            "weight": 0.5,
        },
    ),

    "avatar-generator": PromptTemplate(
        model="stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        positive_prompt=(
            "high quality digital art portrait, vibrant colors, clean lines, "
            "professional illustration, detailed face, {user_prompt}"
        ),
        negative_prompt=NEGATIVE_SDXL,
        default_params={
            "prompt_strength": 0.6,
            "num_outputs": 4,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    ),

    "image-upscaler": PromptTemplate(
        model="nightmareai/real-esrgan:b0bb4c529bb749ac98145e70f42f88d36598e89310beeeba4ee00e14238d1b4d",
        positive_prompt="",
        negative_prompt="",
        default_params={
            "face_enhance": True,
        },
    ),

    "style-transfer": PromptTemplate(
        model="stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        positive_prompt="",  # Filled per-style below
        negative_prompt=NEGATIVE_SDXL,
        default_params={
            "prompt_strength": 0.6,
            "num_outputs": 1,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    ),
}

# ── Style-specific prompts for style-transfer ─────────────────────
STYLE_PROMPTS: dict[str, str] = {
    "oil-painting": (
        "oil painting style, thick impasto brush strokes, rich warm colors, "
        "classical impressionist composition, textured canvas, masterful artwork, "
        "high quality, {user_prompt}"
    ),
    "watercolor": (
        "watercolor painting style, soft translucent washes, delicate edges, "
        "flowing colors, fine art illustration, elegant artistic finish, "
        "high quality, {user_prompt}"
    ),
    "anime": (
        "anime art style, cel shading, crisp clean linework, vibrant saturated "
        "colors, manga illustration, professional anime key art, high quality, "
        "{user_prompt}"
    ),
    "sketch": (
        "detailed pencil sketch style, graphite drawing, fine cross-hatching, "
        "black and white line art, professional illustration, clean edges, "
        "high quality, {user_prompt}"
    ),
}

# ── Style-specific prompts for avatar-generator ───────────────────
AVATAR_PROMPTS: dict[str, str] = {
    "cartoon": "vibrant cartoon character, bold outlines, flat color style, Pixar-style 3D render, cute and expressive",
    "anime": "anime character portrait, cel-shaded, big expressive eyes, manga art style, clean linework",
    "professional": "professional corporate headshot, clean studio lighting, photorealistic, sharp focus, LinkedIn profile photo",
    "pixel-art": "16-bit pixel art character sprite, retro game art style, clean pixel edges, nostalgic game character",
    "watercolor": "watercolor portrait illustration, soft washes, delicate brush strokes, fine art gallery style",
    "oil-painting": "oil painting portrait, thick impasto brush strokes, classical impressionist style, museum quality artwork",
}
