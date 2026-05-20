"""Seed blog posts directly via PostgreSQL connection."""
import os
import uuid
from datetime import datetime, timezone
import psycopg2

DB = os.getenv("DATABASE_PUBLIC_URL")
if not DB:
    raise RuntimeError("DATABASE_PUBLIC_URL environment variable is required")

posts = [
    ('how-to-remove-image-background-without-photoshop', 'How to Remove Image Background Without Photoshop — 3 Free AI Methods', 'Learn three free ways to remove image backgrounds without Photoshop. Compare manual editing, online tools, and AI-powered background removers.', 'Image Editing', 'remove background, background remover, AI background removal, free Photoshop alternative', 'background-remover,object-remover', '<p>Removing the background from an image used to mean spending hours with Photoshop\'s pen tool or paying a designer. Today, AI-powered background removers can do the job in seconds.</p><p>Our <a href="/en/tools/background-remover">free background remover</a> uses BRIA RMBG, a state-of-the-art model that handles complex edges like hair, fur, and transparent objects.</p>'),
    ('best-ai-tools-content-creators-2026', '10 Best AI Tools for Content Creators in 2026', 'Discover the best AI tools for content creators in 2026. Compare free and paid options for writing, image generation, text-to-speech, and more.', 'Content Creation', 'AI content tools, content creator tools, AI writing tools, AI image generator', 'article-generator,text-polish,text-to-speech,ai-image-generator', '<p>Content creators in 2026 have access to an unprecedented range of AI tools that can write, design, and produce content faster than ever.</p><p>Our <a href="/en/tools/article-generator">AI article generator</a> creates complete, well-structured articles from just a topic and keywords.</p>'),
    ('restore-old-photos-ai-guide', 'How to Restore Old Photos with AI — Complete Guide 2026', 'Step-by-step guide to restoring old damaged photos using AI. Learn how to fix scratches, fading, and low resolution, then add color to black and white images.', 'Photo Restoration', 'restore old photos, photo restoration, AI photo repair, colorize photos', 'photo-restorer,colorizer,image-upscaler', '<p>Old photographs are irreplaceable windows into the past, but time takes its toll. AI photo restoration has made it possible for anyone to restore vintage family photos.</p><p>Start with an <a href="/en/tools/photo-restorer">AI photo restorer</a> to fix scratches, dust, and fading.</p>'),
    ('ai-image-generator-prompt-guide', 'How to Write Better AI Image Prompts — Stop Getting Weird Results', 'Getting weird AI images? The problem is your prompt. Learn to write prompts that work.', 'Image Generation', 'AI image prompts, text to image prompts, SDXL prompt guide', 'ai-image-generator,style-transfer', '<p>You type "a cat" into an AI image generator and get a blurry feline blob. A good prompt tells the model exactly what to create: subject, style, lighting, composition, and details.</p><p>Try our <a href="/en/tools/ai-image-generator">AI image generator</a> with your new prompting skills.</p>'),
    ('remove-watermark-from-photo', 'How to Remove a Watermark from a Photo — The Right Way', 'Remove watermarks cleanly without cropping or blurring the whole image.', 'Image Editing', 'remove watermark, watermark remover, erase logo, remove timestamp', 'watermark-remover,object-remover,background-remover', '<p>Cropping is the lazy way to remove a watermark. <a href="/en/tools/watermark-remover">AI watermark removal</a> fills the removed area with content that matches the surrounding image using BRIA Eraser inpainting.</p>'),
    ('text-to-speech-for-content-creators', 'Converting Articles to Audio: A Content Creator\'s Workflow', 'Turn your blog posts into audio versions in minutes. Natural-sounding, not robotic.', 'Content Creation', 'text to speech, convert article to audio, AI voiceover, TTS for bloggers', 'text-to-speech,text-polish,article-generator', '<p>Adding audio versions of your articles is one of the easiest ways to increase engagement. Our <a href="/en/tools/text-to-speech">text to speech tool</a> supports 17 languages with natural voice output.</p>'),
    ('colorize-black-and-white-photos', 'Colorizing Old Family Photos: Before and After Examples', 'See what happens when you run B&W photos through an AI colorizer.', 'Photo Restoration', 'colorize photos, black and white to color, AI colorizer, vintage photos', 'colorizer,photo-restorer,image-upscaler', '<p>My grandmother\'s wedding photo sat in a drawer for 60 years. I ran it through an <a href="/en/tools/colorizer">AI colorizer</a> and saw the blue of her dress for the first time.</p>'),
    ('ai-article-writing-vs-human', 'AI Article Generator: What It Does Well and Where It Falls Short', 'Honest breakdown of AI article writing — what works and what does not.', 'Content Creation', 'AI article generator, AI writer review, AI content creation', 'article-generator,text-polish', '<p>I tested our <a href="/en/tools/article-generator">AI article generator</a> on five topics. It nails the structure and research topics but falls short on original opinions and humor.</p>'),
    ('upscale-images-without-losing-quality', 'How to Upscale an Image Without Making It Look Terrible', 'AI upscaling fills in missing detail intelligently. Learn how it works and when to use it.', 'Image Editing', 'upscale image, AI upscaler, increase image resolution, enlarge photo', 'image-upscaler,ai-image-generator,photo-restorer', '<p>You have a 400-pixel-wide product photo and the printer needs it at 1600. <a href="/en/tools/image-upscaler">AI image upscaling</a> uses Real-ESRGAN to reconstruct high-res images from low-res inputs.</p>'),
    ('blur-faces-in-photos-privacy', 'Blurring Faces in Photos for Privacy: A Quick Guide', 'Protect identities with face blurring — mosaic, gaussian, or fun emoji overlays.', 'Image Editing', 'blur faces, face privacy, pixelate faces, emoji face cover', 'face-blur,background-remover,image-description', '<p>You took a great photo at a school event but there are other people\'s kids in the shot. Our <a href="/en/tools/face-blur">face blur tool</a> handles this in seconds with AI face detection and four overlay styles.</p>'),
]

conn = psycopg2.connect(DB)
cur = conn.cursor()
created = 0
now = datetime.now(timezone.utc).isoformat()

for slug, title, desc, cat, tags, tools, content in posts:
    cur.execute('SELECT id FROM blog_posts WHERE slug = %s', (slug,))
    if cur.fetchone():
        print(f'  SKIP: {slug}')
        continue
    pid = str(uuid.uuid4())
    cur.execute(
        'INSERT INTO blog_posts (id, slug, title, description, content, category, tags, related_tools, published, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s,%s)',
        (pid, slug, title, desc, content, cat, tags, tools, now, now)
    )
    print(f'  OK: {slug}')
    created += 1

conn.commit()
cur.close()
conn.close()
print(f'Done: {created} created')
