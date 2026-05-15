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
    "deformed limbs, distorted facial features, deformed body, twisted face, "
    "blurry, low quality, low resolution, worst quality, ugly, watermark, text, "
    "signature, bad anatomy, extra limbs, disfigured, poorly drawn face, mutation, "
    "duplicate, cropped, out of frame, cut off, asymmetric, unnatural proportions"
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
        model="topazlabs/dust-and-scratch-v2:f9848c7feb1604b71c4d09a70ccfde538c86e3c82dbdacecb93cdc2513163c44",
        positive_prompt="",
        negative_prompt="",
        default_params={
            "grain": True,
            "grain_model": "silver rich",
            "grain_strength": 20,
            "grain_density": 30,
            "grain_size": 1,
            "output_format": "png",
        },
    ),

    "photo-restorer-face": PromptTemplate(
        model="tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
        positive_prompt="",
        negative_prompt="",
        default_params={
            "version": "v1.4",
            "scale": 2,
            "weight": 0.6,
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
        default_params={},
    ),

    "style-transfer": PromptTemplate(
        model="fofr/style-transfer:f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0",
        positive_prompt="",  # Filled per-style from STYLE_REFERENCE_PROMPTS
        negative_prompt="",
        default_params={
            "structure_depth_strength": 0.9,
            "structure_denoising_strength": 0.65,
            "number_of_images": 1,
        },
    ),
}

# ── Style Transfer: per-style text prompts (accompany reference images) ──
STYLE_REFERENCE_PROMPTS: dict[str, str] = {
    "oil-painting": "oil painting, thick impasto brushstrokes, canvas texture, fine art",
    "watercolor": "watercolor painting, soft delicate washes, artistic, flowing colors",
    "sketch": "pencil sketch, graphite drawing, detailed, black and white, hand-drawn",
    "cartoon": "3D Pixar style cartoon, smooth, polished, vibrant colors, cute",
    "cyberpunk": "cyberpunk, neon lights, futuristic synthwave, electric colors",
    "fantasy": "fantasy art, magical glowing, ethereal, mystical, dreamlike",
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

# ── AI Image Generator: Scene configuration (extensible) ──────────
# Each scene provides: prefix prompt, style keywords, optional negative override,
# recommended aspect ratio, and whether to lock the ratio.
# To add a new scene: just add an entry here + add metadata in the frontend scene list.
AI_IMAGE_GENERATOR_SCENES: dict[str, dict] = {
    "free": {
        "prefix": "",
        "style": "",
        "negative_override": None,
        "recommended_ratio": None,
        "lock_ratio": False,
    },
    "portrait": {
        "prefix": (
            "professional portrait photography, single person, upper body shot, "
            "clean background, soft bokeh, natural skin texture,"
        ),
        "style": (
            "studio lighting, 85mm portrait lens, Canon EOS R5, "
            "magazine editorial style, professional color grading"
        ),
        "negative_override": (
            "group photo, full body, messy background, cartoon, anime, illustration, "
            "3d render, low quality portrait, harsh lighting, overexposed, "
            "deformed face, extra fingers, bad anatomy"
        ),
        "recommended_ratio": "2:3",
        "lock_ratio": True,
    },
    "ecommerce": {
        "prefix": (
            "professional product photography, single product on pure white background, "
            "studio lighting setup, commercial product shot, clean and minimal"
        ),
        "style": (
            "softbox lighting, product photography, e-commerce hero image, "
            "high-end commercial quality, 8k product render, macro detail shot"
        ),
        "negative_override": (
            "cluttered background, multiple products, hands holding product, "
            "watermark, logo, text on product, shadow chaos, low quality, "
            "harsh reflection, dark background, messy scene, distorted product"
        ),
        "recommended_ratio": "1:1",
        "lock_ratio": True,
    },
    "social-media": {
        "prefix": (
            "eye-catching social media post design, bold visual hierarchy, "
            "modern graphic design, engaging composition, vibrant and scroll-stopping, "
            "negative space for text overlay, Instagram-optimized"
        ),
        "style": (
            "modern flat design with depth, vibrant color palette, "
            "clean typography-friendly background, social media marketing, "
            "high engagement visual, contemporary aesthetic"
        ),
        "negative_override": (
            "cluttered, messy, too much text, watermark, dated design, "
            "low contrast, boring, dull colors, corporate stock photo, "
            "small subject, busy background, unreadable text space"
        ),
        "recommended_ratio": "1:1",
        "lock_ratio": True,
    },
    "short-video-cover": {
        "prefix": (
            "YouTube thumbnail style, dramatic and click-worthy composition, "
            "bold visual impact, strong focal point centered, vibrant colors, "
            "professional video cover art, high CTR optimized"
        ),
        "style": (
            "cinematic lighting, dramatic contrast, motion blur hints, "
            "video thumbnail aesthetic, eye-catching thumbnail, "
            "professional YouTuber style, high energy visual"
        ),
        "negative_override": (
            "small subject in frame, dull colors, low contrast, "
            "text-heavy, watermark, boring composition, flat lighting, "
            "tiny details, wide landscape, empty space"
        ),
        "recommended_ratio": "16:9",
        "lock_ratio": False,  # SDXL doesn't natively support 16:9, use 3:2 as closest
    },
    "app-ui": {
        "prefix": (
            "modern mobile app UI design, clean user interface, "
            "minimalist flat design, well-organized layout, professional app screen, "
            "iOS design language, beautiful UI components"
        ),
        "style": (
            "clean minimal UI design, soft shadows, rounded corners, "
            "modern tech aesthetic, Dribbble-quality UI, "
            "light mode app interface, subtle gradient accents"
        ),
        "negative_override": (
            "cluttered interface, outdated design, dark mode unless specified, "
            "realistic photo, 3d render, skeuomorphic, complex charts, "
            "code visible, messy layout, low quality UI, web browser, desktop UI"
        ),
        "recommended_ratio": "2:3",
        "lock_ratio": True,
    },
    "live-stream-ui": {
        "prefix": (
            "professional live streaming interface design, clean streaming overlay, "
            "modern streaming dashboard, well-organized layout, "
            "broadcast-quality UI, engaging streaming screen"
        ),
        "style": (
            "gaming stream aesthetic, modern UI with neon accents, "
            "clean chat area, donation alert space, professional streamer setup, "
            "dark theme with vibrant accents, Twitch-style design"
        ),
        "negative_override": (
            "cluttered layout, messy interface, outdated design, "
            "realistic photo, 3d scene, small text, unreadable, "
            "light mode, boring layout, empty screen, broken UI"
        ),
        "recommended_ratio": "16:9",
        "lock_ratio": False,  # Use 3:2 as closest SDXL-supported ratio
    },
    "anime": {
        "prefix": (
            "high quality anime illustration, detailed character art, "
            "vibrant cel-shaded colors, clean lineart, expressive composition, "
            "Japanese anime style, studio quality key visual"
        ),
        "style": (
            "Makoto Shinkai anime movie style, studio Ghibli-level detail, "
            "beautiful lighting effects, hand-drawn anime aesthetic, "
            "professional anime production quality, rich colors"
        ),
        "negative_override": (
            "realistic photo, 3d render, western cartoon style, pixel art, "
            "sketch, rough draft, low quality, deformed face, bad anatomy, "
            "ugly, blurry, distorted, watermark, text, signature"
        ),
        "recommended_ratio": "2:3",
        "lock_ratio": True,
    },
    "landscape": {
        "prefix": (
            "breathtaking landscape photography, wide scenic vista, "
            "perfect composition with foreground interest and distant horizon, "
            "professional nature photography, rich depth and atmosphere"
        ),
        "style": (
            "golden hour lighting, National Geographic quality, "
            "cinematic composition, atmospheric haze, "
            "hyper-realistic nature photography, stunning natural beauty, "
            "professional travel photographer style"
        ),
        "negative_override": (
            "urban city, buildings, people in frame, cars, roads, "
            "artificial structures, cartoon, painting, illustration, "
            "low quality, flat lighting, overcast boring sky, "
            "watermark, text, tilt-shift miniature effect"
        ),
        "recommended_ratio": "3:2",
        "lock_ratio": True,
    },
    "business": {
        "prefix": (
            "professional business promotional design, corporate visual, "
            "clean modern layout with sophisticated aesthetic, "
            "professional marketing material, polished corporate image"
        ),
        "style": (
            "corporate Memphis style, professional business illustration, "
            "clean geometric elements with depth, modern office aesthetic, "
            "blue and white professional palette, high-end corporate quality"
        ),
        "negative_override": (
            "messy, childish, cartoon, informal, casual snapshot, "
            "low quality, cluttered, dark moody, grunge, "
            "distorted, unprofessional, handwritten font, watermark, "
            "too colorful, chaotic composition"
        ),
        "recommended_ratio": "3:2",
        "lock_ratio": True,
    },
}

AI_IMAGE_GENERATOR_NEGATIVE = (
    "deformed, distorted, disfigured, bad anatomy, extra limbs, missing limbs, "
    "floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, "
    "blurry, blur, low quality, low resolution, worst quality, jpeg artifacts, "
    "grainy, noisy, oversaturated, underexposed, overexposed, poor lighting, "
    "watermark, text, signature, username, logo, copyright, frame, border, "
    "collage, multiple views, split screen, cropped, out of frame, cut off, "
    "asymmetric eyes, cross-eyed, poorly drawn face, cloned face, extra fingers, "
    "fused fingers, too many fingers, long neck, bad proportions, unnatural colors, "
    "double image, ghosting, haze, fog, distorted perspective, warped"
)

AI_IMAGE_GENERATOR_POSITIVE_PREFIX = (
    "masterpiece, best quality, highly detailed, sharp focus, professional, "
    "stunning, beautiful, high resolution, 4k, 8k, intricate details"
)

# Per-quality SDXL generation parameters
AI_IMAGE_GENERATOR_PARAMS = {
    "low": {
        "num_inference_steps": 20,
        "guidance_scale": 7.0,
    },
    "medium": {
        "num_inference_steps": 30,
        "guidance_scale": 8.0,
    },
    "high": {
        "num_inference_steps": 50,
        "guidance_scale": 9.0,
    },
}

# Aspect ratio → SDXL dimensions
AI_IMAGE_GENERATOR_DIMENSIONS = {
    "1:1":  (1024, 1024),
    "3:2":  (1216, 832),
    "2:3":  (832, 1216),
}

# ── PDF to Word: Llama 3.1 405B document restructure system prompt ──

PDF_RESTRUCTURE_SYSTEM_PROMPT = (
    "You are a document restoration and formatting expert. "
    "You receive raw OCR-extracted text from a document and must restore it "
    "to clean, well-structured markdown suitable for conversion to Word (.docx).\n\n"
    "Instructions:\n"
    "1. Correct OCR errors: fix common character confusions (rn→m, cl→d, etc.), "
    "punctuation mistakes, and broken words across line breaks.\n"
    "2. Identify heading levels: use # for main titles, ## for section headings, "
    "### for sub-sections. Base this on context and text emphasis, not just size.\n"
    "3. Group sentences into proper paragraphs. Remove artificial line breaks "
    "within paragraphs. Keep intentional paragraph breaks.\n"
    "4. Reconstruct tables using markdown table format (| col1 | col2 |).\n"
    "5. Preserve numbered/bulleted lists with proper markdown formatting.\n"
    "6. Detect block quotes and format with > prefix.\n"
    "7. Remove any OCR artifacts like stray characters, random symbols, or noise.\n"
    "8. Keep ALL original factual content. Do NOT add, summarize, or change meaning.\n"
    "9. Ensure the output language matches the document's language.\n"
    "10. Keep the exact same numbers, dates, names, and technical terms as the original.\n\n"
    "Output ONLY the cleaned, structured markdown. No explanations, no preamble, no "
    "meta-commentary — just the document content."
)

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
