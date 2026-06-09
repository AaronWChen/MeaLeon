"""
Search Service — wraps Edamam Recipe Search API.

Responsibilities:
  - Accept a user query (dish name + cuisine) from the backend
  - Check Redis cache before hitting Edamam (keyed on normalised query)
  - Call Edamam, parse the response into our internal RecipeSearchResult schema
  - Deduplicate + normalise ingredients across all returned recipes
  - Return structured results to the caller (recommendation service or Flask backend)

Why a separate service:
  - Edamam has rate limits; isolating here lets us add throttling/backoff in one place
  - Cache layer means repeated searches (popular dishes) never hit the API
  - Swapping Edamam for another provider only touches this service
"""

from contextlib import asynccontextmanager
import logging
import os
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis

sys.path.insert(0, "/app")
from shared.secrets import get_redis_url

from .edamam_client import EdamamClient
from .cache import SearchCache
from .models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

# set edamam api access
# app_id = os.environ.get("EDAMAM_API_APPID")
# app_key = os.environ.get("EDAMAM_API_APPKEY")


# with open("/run/secrets/edamam_api_appid") as eda_app_id:
#     app_id = eda_app_id.read().strip()

# with open("/run/secrets/edamam_api_appkey") as eda_key:
#     app_key = eda_key.read().strip()

# ---------------------------------------------------------------------------
# Lifespan — set up shared resources once at startup, tear down on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = get_redis_url()
    app.state.redis = aioredis.from_url(redis_url, decode_responses=True)
    app.state.cache = SearchCache(app.state.redis)
    app.state.edamam = EdamamClient()
    logger.info("Search service ready")
    yield
    await app.state.redis.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MeaLeon Search Service",
    description="Edamam API wrapper with Redis caching",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "service": "search"}


@app.post("/search", response_model=SearchResponse)
async def search_recipes(request: SearchRequest):
    """
    Main search endpoint. Called by the recommendation service.

    Flow:
      1. Normalise query (lowercase, strip extra whitespace)
      2. Check Redis cache — return immediately on hit
      3. Call Edamam API
      4. Normalise + deduplicate ingredients
      5. Cache result for TTL_SECONDS
      6. Return SearchResponse
    """
    cache: SearchCache = app.state.cache
    edamam: EdamamClient = app.state.edamam

    cache_key = cache.make_key(request.dish_name, request.cuisine)
    cached = await cache.get(cache_key)
    if cached:
        logger.info("Cache hit for %s / %s", request.dish_name, request.cuisine)
        return cached

    try:
        results = await edamam.search(
            dish_name=request.dish_name,
            cuisine=request.cuisine,
            max_results=request.max_results,
        )
    except Exception as exc:
        logger.error("Edamam call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Edamam API error: {exc}")

    response = SearchResponse(
        dish_name=request.dish_name,
        cuisine=request.cuisine,
        recipes=results,
        from_cache=False,
    )

    await cache.set(cache_key, response)
    return response


@app.get("/search", response_model=SearchResponse)
async def search_recipes_get(
    dish_name: str = Query(..., description="Dish name, e.g. 'lasagna'"),
    cuisine: str = Query(default="", description="Optional cuisine filter"),
    max_results: int = Query(default=10, le=20),
):
    """GET convenience wrapper — same logic as POST, for browser/curl testing."""
    return await search_recipes(
        SearchRequest(dish_name=dish_name, cuisine=cuisine, max_results=max_results)
    )


@app.delete("/cache")
async def clear_cache(pattern: str = Query(default="search:*")):
    """
    Dev/admin endpoint to clear search cache entries.
    Pattern defaults to all search keys.
    """
    cache: SearchCache = app.state.cache
    count = await cache.clear(pattern)
    return {"deleted": count, "pattern": pattern}
