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
NEGATIVE_REMBG = (
    "low quality, blurry, artifacts, jagged edges, halos, white edges, "
    "incomplete removal, semi-transparent residue, background bleeding, "
    "clipped details, over-smooth, over-erased, broken edges, noise"
)
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
        model="tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        positive_prompt="",
        negative_prompt="",
        default_params={
            "version": "v1.4",
            "scale": 2,
            "weight": 0.4,
        },
    ),

    "avatar-generator": PromptTemplate(
        model="stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        positive_prompt=(
            "high quality portrait, detailed face, professional lighting, {user_prompt}"
        ),
        negative_prompt="",  # Per-style negative prompts used instead
        default_params={
            "prompt_strength": 0.55,
            "num_outputs": 4,
            "num_inference_steps": 40,
            "guidance_scale": 8.0,
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

# ── Avatar Generator: per-style positive prompts ──────────────────
AVATAR_PROMPTS: dict[str, str] = {
    "cartoon": (
        "3D Pixar-style character portrait, smooth skin, expressive face, "
        "vibrant colors, soft cinematic lighting, cute charming proportions, "
        "detailed hair strands, professional digital art render, high quality, "
        "volumetric lighting, subsurface scattering on skin"
    ),
    "anime": (
        "Japanese anime portrait, cel-shaded, large sparkling expressive eyes, "
        "delicate facial features, soft gradient hair with highlights, clean crisp lineart, "
        "studio quality key visual, vibrant saturated colors, cherry blossom pink tones, "
        "Makoto Shinkai anime movie style, high quality"
    ),
    "professional": (
        "photorealistic corporate headshot, professional studio lighting with softbox, "
        "sharp focus on face, natural skin texture with visible pores, clean neutral background, "
        "Canon EOS R5, 85mm f/1.4 portrait lens, shallow depth of field, "
        "LinkedIn profile photo quality, magazine editorial style"
    ),
    "pixel-art": (
        "16-bit pixel art character portrait, clean pixel grid, limited 32-color palette, "
        "sharp pixel edges, retro RPG sprite style, SNES era game art quality, "
        "pixel-perfect rendering, nostalgic game character sprite"
    ),
    "watercolor": (
        "watercolor portrait painting, soft translucent washes, delicate flowing brush strokes, "
        "subtle color bleeding, fine art illustration on textured cold-press paper, "
        "artistic elegance, ethereal dreamy atmosphere, gallery wall art quality"
    ),
    "oil-painting": (
        "classical oil painting portrait, thick impasto brush strokes with visible texture, "
        "rich warm earthy palette, dramatic Rembrandt lighting, museum quality artwork, "
        "canvas texture visible, old master portrait style, golden ratio composition, "
        "fine art gallery piece"
    ),
}

# ── Avatar Generator: per-style negative prompts ──────────────────
AVATAR_NEGATIVE_PROMPTS: dict[str, str] = {
    "cartoon": (
        "deformed face, distorted facial features, extra fingers, mutated hands, "
        "blurry, low quality, low resolution, ugly, disfigured, bad anatomy, "
        "asymmetric eyes, crooked mouth, watermark, text, signature, "
        "worst quality, plastic skin, uncanny valley, disproportioned body, "
        "missing fingers, fused fingers, poorly drawn face, mutation"
    ),
    "anime": (
        "deformed face, distorted eyes, extra fingers, mutated hands, "
        "blurry, low quality, low resolution, realistic photo, 3D render, "
        "western cartoon style, bad anatomy, asymmetric face, crooked mouth, "
        "watermark, text, signature, worst quality, messy lineart, inconsistent style, "
        "poorly drawn eyes, missing facial features, disproportioned body"
    ),
    "professional": (
        "deformed face, distorted facial features, extra fingers, mutated hands, "
        "blurry, low quality, low resolution, cartoon, anime, illustration, "
        "painting, drawing, 3D render, sketch, bad anatomy, asymmetric eyes, "
        "crooked mouth, watermark, text, signature, worst quality, "
        "plastic skin, over-smoothed skin, airbrushed, unnatural skin texture, "
        "uncanny valley, composite face, photoshop artifacts, poorly drawn face"
    ),
    "pixel-art": (
        "deformed face, distorted facial features, extra fingers, mutated hands, "
        "blurry, low quality, low resolution, realistic photo, 3D render, "
        "smooth gradients, anti-aliased edges, high resolution, watermark, "
        "text, signature, worst quality, bad anatomy, modern graphics, vector art"
    ),
    "watercolor": (
        "deformed face, distorted facial features, extra fingers, mutated hands, "
        "blurry, low quality, low resolution, digital art, oil painting, "
        "photorealistic, 3D render, thick paint, bold outlines, cartoon, "
        "watermark, text, signature, worst quality, bad anatomy, messy execution, "
        "overworked painting, muddy colors"
    ),
    "oil-painting": (
        "deformed face, distorted facial features, extra fingers, mutated hands, "
        "blurry, low quality, low resolution, digital art, watercolor, "
        "photorealistic photograph, 3D render, cartoon, anime, flat colors, "
        "watermark, text, signature, worst quality, bad anatomy, "
        "messy brushwork, muddy palette, poor composition, overworked canvas"
    ),
}

# ── Avatar Generator: per-style locked generation parameters ──────
AVATAR_PARAMS: dict[str, dict] = {
    "cartoon": {
        "prompt_strength": 0.55,
        "num_outputs": 4,
        "num_inference_steps": 40,
        "guidance_scale": 8.0,
    },
    "anime": {
        "prompt_strength": 0.55,
        "num_outputs": 4,
        "num_inference_steps": 40,
        "guidance_scale": 8.0,
    },
    "professional": {
        "prompt_strength": 0.50,
        "num_outputs": 4,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    },
    "pixel-art": {
        "prompt_strength": 0.65,
        "num_outputs": 4,
        "num_inference_steps": 30,
        "guidance_scale": 8.5,
    },
    "watercolor": {
        "prompt_strength": 0.55,
        "num_outputs": 4,
        "num_inference_steps": 40,
        "guidance_scale": 8.0,
    },
    "oil-painting": {
        "prompt_strength": 0.50,
        "num_outputs": 4,
        "num_inference_steps": 40,
        "guidance_scale": 8.0,
    },
}
