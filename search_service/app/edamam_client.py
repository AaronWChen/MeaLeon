"""
Edamam Recipe Search API v2 client.

Extracted from src/nltk/dish_predictor.py find_similar_dishes() —
that function mixed API call + TF-IDF logic in one place. This class
owns only the HTTP layer.

Key improvements over the original:
  - Async (httpx.AsyncClient) — no blocking the event loop
  - Proper error handling with typed exceptions
  - Ingredient normalisation via _parse_ingredient_name()
  - Type-safe response parsing into RecipeSearchResult
  - Configurable timeout + retry on 429 (rate limit)
"""

import asyncio
import logging
import os
import re
import sys
from typing import List, Optional

import httpx

from .models import RecipeSearchResult

# shared/ is two levels up from search_service/app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.secrets import get_edamam_creds

logger = logging.getLogger(__name__)

EDAMAM_BASE_URL = "https://api.edamam.com/api/recipes/v2"
DEFAULT_TIMEOUT = 15.0  # seconds
RATE_LIMIT_BACKOFF = 2.0  # seconds to wait on 429


class EdamamError(Exception):
    pass


class EdamamRateLimitError(EdamamError):
    pass


class EdamamClient:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        # Read from env var first, fall back to secret file
        self.app_id, self.app_key = get_edamam_creds()
        self.timeout = timeout

    @staticmethod
    def _read_secret(name: str) -> str | None:
        """Read a Docker secret file, trying both upper and lowercase names."""
        for path in [
            f"/run/secrets/{name}",
            f"/run/secrets/{name.lower()}",
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    return f.read().strip()
        return None

    async def search(
        self,
        dish_name: str,
        cuisine: Optional[str] = None,
        max_results: int = 10,
    ) -> List[RecipeSearchResult]:
        """
        Query Edamam for recipes matching dish_name (+ optional cuisine).
        Returns up to max_results parsed RecipeSearchResult objects.

        Retries once on 429 after RATE_LIMIT_BACKOFF seconds.
        """
        params = {
            "type": "public",
            "q": dish_name,
            "app_id": self.app_id,
            "app_key": self.app_key,
        }
        # Edamam v2 cuisine filter — only add if non-empty
        if cuisine:
            params["cuisineType"] = cuisine

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(2):
                resp = await client.get(EDAMAM_BASE_URL, params=params)

                if resp.status_code == 429:
                    if attempt == 0:
                        logger.warning(
                            "Edamam rate limit hit, backing off %ss", RATE_LIMIT_BACKOFF
                        )
                        await asyncio.sleep(RATE_LIMIT_BACKOFF)
                        continue
                    raise EdamamRateLimitError("Edamam rate limit exceeded")

                if resp.status_code != 200:
                    raise EdamamError(
                        f"Edamam returned {resp.status_code}: {resp.text[:200]}"
                    )

                data = resp.json()
                hits = data.get("hits", [])[:max_results]
                return [self._parse_hit(hit) for hit in hits]

        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_hit(self, hit: dict) -> RecipeSearchResult:
        """Map one Edamam 'hit' dict → RecipeSearchResult."""
        recipe = hit.get("recipe", {})

        ingredient_lines = recipe.get("ingredientLines", [])
        ingredient_names = [
            self._parse_ingredient_name(line)
            for line in ingredient_lines
            if line.strip()
        ]
        # Filter out empty strings produced by the parser
        ingredient_names = [n for n in ingredient_names if n]

        return RecipeSearchResult(
            edamam_id=recipe.get("uri", ""),
            label=recipe.get("label", ""),
            source=recipe.get("source", ""),
            url=recipe.get("url", ""),
            cuisine_types=recipe.get("cuisineType", []),
            ingredient_lines=ingredient_lines,
            ingredient_names=ingredient_names,
            calories=recipe.get("calories"),
            total_time=recipe.get("totalTime") or None,
            diet_labels=recipe.get("dietLabels", []),
            health_labels=recipe.get("healthLabels", []),
            image_url=self._best_image(recipe.get("images", {})),
        )

    @staticmethod
    def _parse_ingredient_name(line: str) -> str:
        """
        Extract the bare ingredient name from a full ingredient line.

        Examples:
          "2 cups all-purpose flour"        -> "flour"
          "1/2 tsp freshly ground pepper"   -> "pepper"
          "3 large eggs, beaten"            -> "eggs"
          "olive oil, for drizzling"        -> "olive oil"

        Strategy: strip quantities, units, and common prep words,
        then take what's left. Intentionally simple — the embedding
        model handles semantic similarity, so we don't need perfect
        parsing here.
        """
        # lowercase and strip punctuation notes after comma
        line = line.lower().split(",")[0]

        # Remove quantities: fractions, decimals, integers
        line = re.sub(r"\b\d+[/\d.]*\b", "", line)

        # Remove common units
        units = (
            r"\b(cup|cups|tbsp|tsp|tablespoon|teaspoon|pound|lb|oz|ounce|gram|g|kg|"
            r"ml|liter|litre|quart|pint|gallon|clove|cloves|slice|slices|piece|"
            r"pieces|bunch|pinch|handful|head|can|package|pkg)\b"
        )
        line = re.sub(units, "", line)

        # Remove common prep descriptors
        prep = (
            r"\b(fresh|freshly|dried|ground|chopped|minced|diced|sliced|grated|"
            r"peeled|whole|large|small|medium|extra|finely|coarsely|lightly|"
            r"optional|to taste|room temperature)\b"
        )
        line = re.sub(prep, "", line)

        # Collapse whitespace
        return " ".join(line.split()).strip()

    @staticmethod
    def _best_image(images: dict) -> Optional[str]:
        """Pick highest-quality image URL from Edamam images dict."""
        for size in ("LARGE", "REGULAR", "SMALL", "THUMBNAIL"):
            if size in images:
                return images[size].get("url")
        return None
