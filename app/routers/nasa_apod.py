"""NASA APOD API proxy — bypasses CORS for client-side access."""

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/nasa-apod", tags=["nasa-apod"])

NASA_URL = "https://api.nasa.gov/planetary/apod"
API_KEY = "DEMO_KEY"


@router.get("")
async def get_apod(
    date: str | None = Query(default=None, description="YYYY-MM-DD for specific date"),
    start_date: str | None = Query(default=None, description="Start of date range"),
    end_date: str | None = Query(default=None, description="End of date range"),
    count: int | None = Query(default=None, description="Number of random entries"),
):
    """Proxy NASA APOD API."""
    params: dict = {"api_key": API_KEY}
    if date:
        params["date"] = date
    elif start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    elif count and 1 <= count <= 20:
        params["count"] = str(count)

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(NASA_URL, params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Unable to fetch from NASA. Please try again.")
