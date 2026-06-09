"""
recommend_service/app/models.py

Request/response models for the recommend service.
Kept separate from search_service models so each service
can evolve its schema independently.
"""

from typing import Any
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    dish_name: str
    cuisine: str = ""
    ingredients: list[str] = Field(..., min_length=1)
    user_context: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=50)


class RecipeResult(BaseModel):
    id: str
    title: str
    ingredients: list[str] = []
    ingredient_names: list[str] = []
    cuisine_types: list[str] = []
    description: str = ""
    origin: str = ""
    url: str = ""
    image_url: str | None = None
    similarity_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    diet_labels: list[str] = []
    health_labels: list[str] = []
    restriction_warning: str | None = None


class RecommendResponse(BaseModel):
    results: list[RecipeResult]
    total: int
    dish_name: str
