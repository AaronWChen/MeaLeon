"""
recommend_service/app/main.py

Recommendation service — receives an ingredient list from the Flask backend
and returns ranked recipe candidates using Vespa hybrid search.

The hybrid query does in one round-trip what your old code did in multiple steps:
  - Old: call Edamam → run TF-IDF transform → cosine_similarity scan → filter cuisine
  - New: embed query → fire single Vespa YQL query → get ranked results

Vespa handles:
  - ANN (approximate nearest neighbour) for dense similarity
  - WAND (Weak AND) for sparse/keyword matching
  - RRF fusion of both signals
  - Hard filtering (cuisine exclusion, language)

This service does NOT apply user preference filters — that's Flask's job
(see recommend.py in the backend). This service just scores and ranks.
"""

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

from .vespa_client import VespaHybridClient
from .models import RecommendRequest, RecommendResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vespa = VespaHybridClient(
        vespa_url=os.environ.get("VESPA_URL", "http://vespa:8080"),
        ml_service_url=os.environ.get(
            "ML_SERVICE_URL", "http://embedding_generation:8000"
        ),
    )
    logger.info("Recommend service ready")
    yield


app = FastAPI(
    title="MeaLeon Recommend Service",
    description="Vespa hybrid search for recipe recommendations",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "recommend"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Main recommendation endpoint.

    Receives ingredients (from search service via Flask) and returns
    ranked recipe candidates from Vespa.

    The user_context.preferred_cuisines are used for soft ranking boosts.
    Hard restriction filtering happens in the Flask backend layer.
    """
    client: VespaHybridClient = app.state.vespa

    try:
        results = await client.hybrid_search(
            ingredients=request.ingredients,
            dish_name=request.dish_name,
            exclude_cuisine=request.cuisine,  # filter OUT same cuisine (original logic)
            preferred_cuisines=request.user_context.get("preferred_cuisines", []),
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error("Vespa search failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Search backend error: {exc}")

    return RecommendResponse(
        results=results,
        total=len(results),
        dish_name=request.dish_name,
    )
