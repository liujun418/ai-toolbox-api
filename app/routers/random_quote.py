"""Random quote API proxy."""

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/random-quote", tags=["random-quote"])

QUOTE_URL = "https://zenquotes.io/api/random"


@router.get("")
async def get_random_quote():
    """Proxy zenquotes API."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(QUOTE_URL)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return {"content": data[0].get("q", ""), "author": data[0].get("a", "")}
            return {"content": "", "author": ""}
    except Exception:
        raise HTTPException(status_code=502, detail="Unable to fetch quote. Please try again.")
