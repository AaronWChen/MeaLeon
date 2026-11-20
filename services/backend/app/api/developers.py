from fastapi import APIRouter, Depends, Request, BackgroundTasks
from app.middleware.auth import require_developer
from app.services.recommendation_service import RecommendationService
from app.schemas import IndexRequest, EmbeddingRequest
from typing import List

router = APIRouter()


@router.post("/index")
async def index_items(
    request: Request,
    index_request: IndexRequest,
    background_tasks: BackgroundTasks,
    user=Depends(require_developer),
    rec_service: RecommendationService = Depends(get_recommendation_service),
):
    """Index items for search (developer only)"""

    # Index in background to avoid timeout
    background_tasks.add_task(
        rec_service.index_items,
        items=index_request.items,
        text_field=index_request.text_field,
    )

    return {
        "status": "indexing",
        "items_count": len(index_request.items),
        "message": "Items are being indexed in the background",
    }


@router.post("/embeddings")
async def get_embeddings(
    request: Request,
    embedding_request: EmbeddingRequest,
    user=Depends(require_developer),
):
    """Get raw embeddings for text (developer only)"""

    embedding_service = request.app.state.embedding_service

    embeddings = await embedding_service.get_embeddings_batch(
        texts=embedding_request.texts, use_cache=embedding_request.use_cache
    )

    return {
        "embeddings": embeddings,
        "count": len(embeddings),
        "dimensions": len(embeddings[0]) if embeddings else 0,
    }


@router.delete("/collection/reset")
async def reset_collection(request: Request, user=Depends(require_developer)):
    """Reset vector collection (developer only - dangerous!)"""

    vector_service = request.app.state.vector_service
    # Implementation to delete and recreate collection

    return {"status": "collection reset"}


@router.get("/stats")
async def get_stats(request: Request, user=Depends(require_developer)):
    """Get system statistics"""

    vector_service = request.app.state.vector_service

    # Get collection info
    collection_info = vector_service.client.get_collection(
        collection_name=vector_service.collection_name
    )

    return {
        "total_items": collection_info.points_count,
        "vector_dimensions": collection_info.config.params.vectors.size,
        "cache_stats": {
            # Redis stats
            "cached_embeddings": request.app.state.redis.dbsize()
        },
    }
