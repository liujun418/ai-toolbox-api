"""Bing Wallpaper API proxy — bypasses CORS restrictions for client-side access."""

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/bing-wallpaper", tags=["bing-wallpaper"])

BING_BASE = "https://www.bing.com/HPImageArchive.aspx?format=js&n=8&mkt=en-US"


@router.get("")
async def get_bing_wallpapers(
    idx: int = Query(default=0, ge=0, le=60, description="Start index (0=today, 8=8days ago...)"),
):
    """Proxy Bing HPImageArchive API. Returns 8 days of wallpapers."""
    url = f"{BING_BASE}&idx={idx}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Unable to fetch wallpapers from Bing. Please try again.")
