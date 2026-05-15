"""Lateral Thinking Puzzle endpoint — free, no authentication, no credits.

Uses Gemini API free tier for puzzle generation, hint, verify, and reveal.
All actions: generate, hint, verify, reveal.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.gemini_service import (
    generate_puzzle,
    get_hint,
    verify_guess,
    reveal_answer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lateral-thinking", tags=["lateral-thinking"])


class PuzzleRequest(BaseModel):
    action: str = Field(..., description="generate | hint | verify | reveal")
    category: str = Field(default="classic", description="classic | horror | brain-hole")
    difficulty: str = Field(default="medium", description="easy | medium | hard")
    language: str = Field(default="en", description="en | es | ar")
    scenario: str | None = Field(default=None, description="Current scenario (for hint/verify/reveal)")
    answer: str | None = Field(default=None, description="Correct answer (for hint/verify/reveal)")
    twist: str | None = Field(default=None, description="Twist keyword (for reveal)")
    guess: str | None = Field(default=None, description="Player's guess (for verify)")


@router.post("")
async def lateral_thinking(request: PuzzleRequest):
    """Handle lateral thinking puzzle interactions. Free for all users."""
    action = request.action.strip().lower()
    language = request.language.strip().lower()
    if language not in ("en", "es", "ar"):
        language = "en"

    try:
        if action == "generate":
            result = await generate_puzzle(
                category=request.category,
                difficulty=request.difficulty,
                language=language,
            )
            return {"action": "generate", **result}

        elif action == "hint":
            if not request.scenario or not request.answer:
                raise HTTPException(status_code=400, detail="scenario and answer required for hint")
            result = await get_hint(
                scenario=request.scenario,
                answer=request.answer,
                language=language,
            )
            return {"action": "hint", **result}

        elif action == "verify":
            if not request.scenario or not request.answer or not request.guess:
                raise HTTPException(status_code=400, detail="scenario, answer, and guess required for verify")
            result = await verify_guess(
                scenario=request.scenario,
                answer=request.answer,
                guess=request.guess,
                language=language,
            )
            return {"action": "verify", **result}

        elif action == "reveal":
            if not request.scenario or not request.answer:
                raise HTTPException(status_code=400, detail="scenario and answer required for reveal")
            result = await reveal_answer(
                scenario=request.scenario,
                answer=request.answer,
                twist=request.twist or "",
                language=language,
            )
            return {"action": "reveal", **result}

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action: {action}. Use generate, hint, verify, or reveal.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Lateral thinking error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Puzzle generation failed. Please try again.",
        )
