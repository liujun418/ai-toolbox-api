"""Retry logic with exponential backoff for transient API failures."""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    async_func,
    max_retries: int = 2,
    base_delay: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> any:
    """Retry an async function with exponential backoff.

    Args:
        async_func: Async callable to retry.
        max_retries: Number of retries after the initial attempt.
        base_delay: Base delay in seconds (doubles each retry).
        exceptions: Exception types to catch and retry on.

    Returns:
        Result from async_func on success.

    Raises:
        The last exception if all retries fail.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await async_func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Attempt %d failed: %s. Retrying in %.1fs...",
                    attempt + 1, str(e), delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("All %d retries exhausted: %s", max_retries, str(e))
    raise last_exception
