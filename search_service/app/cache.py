"""
Redis cache for search results.

TTL strategy:
  - Queries cached for 12h default
  - Cache key includes normalised dish + cuisine so different
    cuisine variants of the same dish are cached separately.

The SearchResponse is serialised to JSON via model.model_dump_json()
and deserialised back with SearchResponse.model_validate_json().
"""

import hashlib
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from .models import SearchResponse

logger = logging.getLogger(__name__)

DEFAULT_TTL = 12 * 60 * 60  # 12 hours in seconds


class SearchCache:
    PREFIX = "search"

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis

    def make_key(self, dish_name: str, cuisine: str) -> str:
        """
        Stable cache key from normalised dish + cuisine.
        Uses a hash so special characters in dish names don't cause issues.
        """
        raw = f"{dish_name.strip().lower()}::{cuisine.strip().lower()}"
        digest = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{self.PREFIX}:{digest}"

    def _ttl(self, dish_name: str) -> int:
        """Return appropriate TTL based on whether this is a popular query."""
        return DEFAULT_TTL

    async def get(self, key: str) -> Optional[SearchResponse]:
        """Return cached SearchResponse or None on miss."""
        try:
            raw = await self.redis.get(key)
            if raw is None:
                return None
            result = SearchResponse.model_validate_json(raw)
            # Mark that this came from cache so callers can log/track it
            result.from_cache = True
            return result
        except Exception as exc:
            # Cache errors should never break the request — just log and miss
            logger.warning("Cache get failed for %s: %s", key, exc)
            return None

    async def set(self, key: str, response: SearchResponse) -> None:
        """Store SearchResponse. TTL derived from dish name in the response."""
        try:
            ttl = self._ttl(response.dish_name)
            await self.redis.setex(key, ttl, response.model_dump_json())
        except Exception as exc:
            logger.warning("Cache set failed for %s: %s", key, exc)

    async def clear(self, pattern: str = "search:*") -> int:
        """Delete all keys matching pattern. Returns count deleted."""
        keys = await self.redis.keys(pattern)
        if not keys:
            return 0
        return await self.redis.delete(*keys)
