"""
Internal data models for the search service.

Edamam v2 API reference:
  https://developer.edamam.com/edamam-docs-recipe-api
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    dish_name: str = Field(..., min_length=1, max_length=200)
    cuisine: str = Field(default="", max_length=100)
    max_results: int = Field(default=10, ge=1, le=20)

    @field_validator("dish_name", "cuisine", mode="before")
    @classmethod
    def normalise_text(cls, v: str) -> str:
        """Strip and lowercase so 'Lasagna ' and 'lasagna' hit the same cache key."""
        return v.strip().lower()


# ---------------------------------------------------------------------------
# Per-recipe result
# ---------------------------------------------------------------------------


class RecipeSearchResult(BaseModel):
    """
    One recipe returned from Edamam.
    Fields map to what dish_predictor.py was extracting manually.
    """

    edamam_id: str = Field(..., description="Edamam URI, used as stable ID")
    label: str
    source: str
    url: str
    cuisine_types: List[str] = Field(default_factory=list)

    # Ingredients as full lines ("2 cups flour") and as bare names ("flour")
    ingredient_lines: List[str] = Field(default_factory=list)
    ingredient_names: List[str] = Field(
        default_factory=list,
        description="Normalised bare ingredient names for vectorization",
    )

    # Optional nutritional / metadata fields from Edamam
    calories: Optional[float] = None
    total_time: Optional[int] = None  # minutes
    diet_labels: List[str] = Field(default_factory=list)
    health_labels: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None

    @field_validator("ingredient_names", mode="before")
    @classmethod
    def deduplicate_ingredients(cls, v: List[str]) -> List[str]:
        """Preserve order while removing exact duplicates."""
        seen = set()
        return [x for x in v if not (x in seen or seen.add(x))]


# ---------------------------------------------------------------------------
# Aggregated response
# ---------------------------------------------------------------------------


class SearchResponse(BaseModel):
    dish_name: str
    cuisine: str
    recipes: List[RecipeSearchResult]
    from_cache: bool = False

    # Convenience: all unique ingredients across every returned recipe,
    # deduplicated — this is what gets passed to the recommendation service
    # for vectorisation.
    @property
    def all_ingredients(self) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for recipe in self.recipes:
            for ing in recipe.ingredient_names:
                if ing not in seen:
                    seen.add(ing)
                    out.append(ing)
        return out
