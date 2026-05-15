"""Gemini API service for lateral thinking puzzle generation and interaction.

Uses Google Gemini free tier (gemini-2.0-flash): 15 RPM, 1,500 RPD.
All calls are free — no credits deducted from users.
"""

import json as _json
import logging
import httpx

from app.config import settings
from app.services.retry import retry_with_backoff

logger = logging.getLogger(__name__)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

# ── System prompts per action ──

SYSTEM_GENERATE = """You are a master of "Lateral Thinking Puzzles" (also known as "Sea Turtle Soup" mysteries).

Create a lateral thinking puzzle in {language}. Follow these rules:
- The scenario should be SHORT (2-4 sentences), mysterious, and intriguing
- The answer must rely on a clever twist, wordplay, or unexpected assumption
- Category: {category}
- Difficulty: {difficulty}
- Make sure the puzzle is solvable and fair — the twist should make sense once revealed
- The answer should explain the twist clearly in 2-3 sentences

{language_instruction}

Return ONLY valid JSON in this exact format:
{{"scenario": "...", "answer": "...", "twist": "..."}}"""

SYSTEM_HINT = """You are a lateral thinking puzzle host. The player is stuck on this puzzle:

SCENARIO: {scenario}
ANSWER (DO NOT REVEAL): {answer}

Give ONE progressive hint. The hint should nudge them toward the twist WITHOUT giving away the answer.
Be subtle — a good hint makes them think harder, not gives it away.

{language_instruction}

Return ONLY valid JSON: {{"hint": "..."}}"""

SYSTEM_VERIFY = """You are a lateral thinking puzzle judge. The player has submitted a guess for this puzzle:

SCENARIO: {scenario}
CORRECT ANSWER: {answer}
PLAYER'S GUESS: {guess}

Evaluate the guess and respond:
- If they got the key twist exactly right → status: "correct"
- If they're on the right track but not quite → status: "close"
- If they're completely wrong → status: "wrong"
- Never reveal the full answer in your response
- If wrong or close, give a SHORT nudge (one sentence max)

{language_instruction}

Return ONLY valid JSON: {{"status": "correct|close|wrong", "feedback": "..."}}"""

SYSTEM_REVEAL = """You are a lateral thinking puzzle host. The player has given up and wants to know the answer.

SCENARIO: {scenario}
ANSWER: {answer}
TWIST: {twist}

Reveal the answer in an engaging, satisfying way. Explain the twist clearly.
Wrap up the mystery like a good storyteller. Keep it to 3-4 sentences.

{language_instruction}

Return ONLY valid JSON: {{"reveal": "..."}}"""

# ── Language-specific instructions ──

LANG_INSTRUCTIONS = {
    "en": "Write entirely in English.",
    "es": "Escribe completamente en español. Usa lenguaje natural y fluido.",
    "ar": "اكتب بالكامل باللغة العربية. استخدم لغة طبيعية وسلسة.",
}

CATEGORY_LABELS = {
    "classic": {
        "en": "Classic Mystery — logical deduction, clever wordplay, unexpected twists",
        "es": "Misterio Clásico — deducción lógica, juegos de palabras ingeniosos, giros inesperados",
        "ar": "غموض كلاسيكي — استنتاج منطقي، تلاعب ذكي بالكلمات، تحولات غير متوقعة",
    },
    "horror": {
        "en": "Dark Horror — eerie, unsettling, psychological twists, creepy reveals",
        "es": "Terror Oscuro — inquietante, perturbador, giros psicológicos, revelaciones espeluznantes",
        "ar": "رعب مظلم — غريب، مقلق، تحولات نفسية، كشف مخيف",
    },
    "brain-hole": {
        "en": "Brain Hole Fun — absurd, hilarious, mind-bending, creative logic",
        "es": "Agujero Mental Divertido — absurdo, divertido, alucinante, lógica creativa",
        "ar": "ثقب الدماغ الممتع — سخيف، مضحك، محير للعقل، منطق إبداعي",
    },
}

DIFFICULTIES = ["easy", "medium", "hard"]


def _build_prompt(template: str, **kwargs) -> str:
    """Fill a prompt template with variables."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


async def _call_gemini(prompt: str) -> dict:
    """Call Gemini API and return parsed JSON response."""
    async def _call():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={settings.GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.9,
                        "maxOutputTokens": 512,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()

    data = await retry_with_backoff(_call, max_retries=1)
    text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    return _json.loads(text)


async def generate_puzzle(
    category: str = "classic",
    difficulty: str = "medium",
    language: str = "en",
) -> dict:
    """Generate a new lateral thinking puzzle."""
    if category not in CATEGORY_LABELS:
        category = "classic"
    if difficulty not in DIFFICULTIES:
        difficulty = "medium"
    lang = language if language in LANG_INSTRUCTIONS else "en"

    cat_label = CATEGORY_LABELS[category][lang]
    prompt = _build_prompt(
        SYSTEM_GENERATE,
        language=lang,
        category=cat_label,
        difficulty=difficulty,
        language_instruction=LANG_INSTRUCTIONS[lang],
    )

    logger.info("Generating puzzle: category=%s difficulty=%s lang=%s", category, difficulty, lang)
    return await _call_gemini(prompt)


async def get_hint(scenario: str, answer: str, language: str = "en") -> dict:
    """Provide a progressive hint for the puzzle."""
    lang = language if language in LANG_INSTRUCTIONS else "en"
    prompt = _build_prompt(
        SYSTEM_HINT,
        scenario=scenario,
        answer=answer,
        language_instruction=LANG_INSTRUCTIONS[lang],
    )
    return await _call_gemini(prompt)


async def verify_guess(
    scenario: str,
    answer: str,
    guess: str,
    language: str = "en",
) -> dict:
    """Evaluate a player's guess against the correct answer."""
    lang = language if language in LANG_INSTRUCTIONS else "en"
    prompt = _build_prompt(
        SYSTEM_VERIFY,
        scenario=scenario,
        answer=answer,
        guess=guess,
        language_instruction=LANG_INSTRUCTIONS[lang],
    )
    return await _call_gemini(prompt)


async def reveal_answer(
    scenario: str,
    answer: str,
    twist: str,
    language: str = "en",
) -> dict:
    """Reveal the answer with storytelling flair."""
    lang = language if language in LANG_INSTRUCTIONS else "en"
    prompt = _build_prompt(
        SYSTEM_REVEAL,
        scenario=scenario,
        answer=answer,
        twist=twist,
        language_instruction=LANG_INSTRUCTIONS[lang],
    )
    return await _call_gemini(prompt)
