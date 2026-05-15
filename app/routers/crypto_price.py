"""CoinGecko price proxy — bypasses network restrictions for client-side access."""

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/crypto-price", tags=["crypto-price"])

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,binancecoin,solana,ripple,cardano,dogecoin,polkadot"
    "&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
)


@router.get("")
async def get_crypto_prices():
    """Proxy CoinGecko simple/price API."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(COINGECKO_URL)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Unable to fetch prices. Please try again.")
