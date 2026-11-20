import httpx
import hashlib
import json
from typing import List, Optional
from redis import Redis


class EmbeddingService:
    def __init__(self, ml_service_url: str, redis_client: Redis):
        self.ml_service_url = ml_service_url
        self.redis = redis_client

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for embedding"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{text_hash}"

    async def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """Get embedding for single text with caching"""

        if use_cache:
            cache_key = self._get_cache_key(text)
            cached = self.redis.get(cache_key)

            if cached:
                return json.loads(cached)

        # Call ML service
        embedding = await self._generate_embedding([text])
        result = embedding[0]

        # Cache for 7 days
        if use_cache:
            self.redis.setex(
                self._get_cache_key(text), 7 * 24 * 3600, json.dumps(result)
            )

        return result

    async def get_embeddings_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently"""

        if not use_cache:
            return await self._generate_embedding(texts)

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached = self.redis.get(cache_key)

            if cached:
                results.append(json.loads(cached))
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self._generate_embedding(uncached_texts)

            # Fill in results and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                cache_key = self._get_cache_key(texts[idx])
                self.redis.setex(cache_key, 7 * 24 * 3600, json.dumps(embedding))

        return results

    async def _generate_embedding(self, texts: List[str]) -> List[List[float]]:
        """Call ML service to generate embeddings"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.ml_service_url}/embed", json={"texts": texts, "normalize": True}
            )
            response.raise_for_status()

            data = response.json()
            return data["embeddings"]
