"""Bing Wallpaper API proxy — bypasses CORS restrictions for client-side access."""

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/bing-wallpaper", tags=["bing-wallpaper"])

BING_URL = "https://www.bing.com/HPImageArchive.aspx?format=js&idx=0&n=8&mkt=en-US"


@router.get("")
async def get_bing_wallpapers():
    """Proxy Bing HPImageArchive API. Returns 8 days of wallpapers."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(BING_URL)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Unable to fetch wallpapers from Bing. Please try again.")
