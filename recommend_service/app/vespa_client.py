"""
recommend_service/app/vespa_client.py

Vespa hybrid search client.

This is the direct replacement for the TF-IDF logic in dish_predictor.py.

Old approach (dish_predictor.py find_similar_dishes):
    1. Load joblib models (blocking, ~2s on cold start)
    2. TF-IDF transform the query ingredients
    3. Cosine similarity against ALL ~20k recipes
    4. filter_out_cuisine() removes same-cuisine results
    5. Return top 5

New approach:
    1. Call embedding service for dense + sparse vectors (async)
    2. Fire one YQL query to Vespa with ANN + WAND + cuisine filter
    3. Vespa returns pre-ranked results — no Python-side sorting needed
    4. Return top N

Key improvement: Vespa's ANN doesn't scan all 20k recipes — it uses an
HNSW index to find approximate nearest neighbours in O(log n) time.
"""

import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# How many ANN candidates to consider before RRF fusion.
# Higher = better recall, slower query. 100 is a good starting point.
ANN_TARGET_HITS = 100

# Placeholder cuisine values that mean "no real cuisine data" — these
# get excluded from cross-cuisine recommendations the same way empty
# cuisine_str is, since neither represents a genuine cuisine label.
UNKNOWN_CUISINE_VALUES = {"missing cuisine", "unknown", "n/a", "none"}


class VespaHybridClient:
    def __init__(self, vespa_url: str, ml_service_url: str):
        self.vespa_url = vespa_url.rstrip("/")
        self.ml_service_url = ml_service_url.rstrip("/")

    async def hybrid_search(
        self,
        ingredients: list[str],
        dish_name: str,
        exclude_cuisine: str = "",
        preferred_cuisines: list[str] = [],
        top_k: int = 10,
    ) -> list[dict]:
        query_text = f"{dish_name}. Ingredients: {', '.join(ingredients)}"
        query_ingredients_str = " ".join(ingredients)

        try:
            dense_vec, sparse_map = await self._get_query_embeddings(
                query_text, query_ingredients_str
            )
        except Exception as e:
            logger.error("Embedding service unavailable, using zero vector: %s", e)
            dense_vec = [0.0] * 384
            sparse_map = {}

        has_bow = bool(sparse_map)

        rank_profile = "hybrid_cuisine_boost" if preferred_cuisines else "hybrid"

        # YQL now built conditionally based on BoW availability
        yql = self._build_yql(
            exclude_cuisine=exclude_cuisine,
            dish_name=dish_name,
            top_k=top_k,
            has_bow=has_bow,
        )

        body = {
            "yql": yql,
            "ranking.profile": rank_profile,
            "hits": top_k,
            "input.query(q_embedding)": dense_vec,
            **(
                # Only include q_bow in the request when we actually have terms —
                # an empty dict still trips Vespa's "input not set" validation
                # in some versions, so we omit the key entirely rather than
                # sending {}.
                {"input.query(q_bow)": sparse_map}
                if has_bow
                else {}
            ),
            **(
                {"input.query(preferred_cuisine)": preferred_cuisines[0]}
                if preferred_cuisines
                else {}
            ),
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.vespa_url}/search/",
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        return self._parse_results(data, dense_vec)

    def _build_yql(
        self, exclude_cuisine: str, dish_name: str, top_k: int, has_bow: bool = False
    ) -> str:
        """
        Build YQL conditionally based on BoW availability.

        With BoW:    ANN OR weakAnd — full hybrid retrieval
        Without BoW: ANN only — pure dense semantic search

        Once the sklearn TF-IDF model is trained and registered in MLflow,
        has_bow will become True automatically and hybrid retrieval activates
        with no code change needed here.
        """
        if has_bow:
            retrieval = (
                f"("
                f"{{targetHits: {ANN_TARGET_HITS}}}nearestNeighbor(embedding, q_embedding)"
                f" OR "
                f"weakAnd(ingredients_bow contains @q_bow)"
                f")"
            )
        else:
            # Dense-only retrieval — no @q_bow reference in the query at all
            retrieval = f"{{targetHits: {ANN_TARGET_HITS}}}nearestNeighbor(embedding, q_embedding)"

        filters = ""

        if exclude_cuisine:
            placeholder_exclusions = " AND ".join(
                f'!(cuisine_str contains "{val}")' for val in UNKNOWN_CUISINE_VALUES
            )
            filters += (
                f' AND cuisine_str matches ".+" '
                f"AND {placeholder_exclusions} "
                f'AND !(cuisine_str contains "{exclude_cuisine.lower()}")'
            )

        if dish_name:
            # Exclude results whose title contains the queried dish name —
            # prevents "American Lasagna" from showing up when searching
            # for "lasagna, italian". Using the primary word of dish_name
            # (first token) keeps this from being overly strict on
            # multi-word dish names.
            primary_word = dish_name.strip().lower().split()[0]
            filters += f' AND !(title contains "{primary_word}")'

        return (
            f"select id, title, ingredients, cuisine, origin, description, url, image_url "
            f"from recipe "
            f"where {retrieval}{filters} "
            f"limit {top_k}"
        )

    async def _get_query_embeddings(
        self, query_text: str, ingredients_str: str
    ) -> tuple[list[float], dict]:
        """Call the ML service to embed the query."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self.ml_service_url}/embed",
                json={
                    "texts": [query_text],
                    "ingredients": [ingredients_str],
                    "normalize": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        dense_vec = data["embeddings"][0]

        # BoW embedding: your service returns a DataFrame — extract as dict
        # The bow_embeddings response shape depends on your sklearn model output
        bow_raw = data.get("bow_embeddings", {})
        if isinstance(bow_raw, list) and len(bow_raw) > 0:
            sparse_map = bow_raw[0] if isinstance(bow_raw[0], dict) else {}
        elif isinstance(bow_raw, dict):
            sparse_map = bow_raw
        else:
            sparse_map = {}

        # Filter to top 100 terms for query efficiency
        sparse_map = dict(
            sorted(sparse_map.items(), key=lambda x: x[1], reverse=True)[:100]
        )

        return dense_vec, sparse_map

    def _parse_results(
        self, vespa_response: dict, query_vec: list[float]
    ) -> list[dict]:
        """
        Parse Vespa search response into the shape the Flask backend expects.

        Vespa response structure:
        {
            "root": {
                "children": [
                    {
                        "id": "...",
                        "relevance": 0.85,
                        "fields": { ...document fields... },
                        "matchfeatures": {
                            "closeness(field,embedding)": 0.92,
                            "bm25(ingredients)": 12.4
                        }
                    },
                    ...
                ]
            }
        }
        """
        hits = vespa_response.get("root", {}).get("children", [])

        results = []
        for hit in hits:
            fields = hit.get("fields", {})
            match_features = hit.get("matchfeatures", {})

            results.append(
                {
                    "id": fields.get("id", ""),
                    "title": fields.get("title", ""),
                    "ingredients": fields.get("ingredients", []),
                    "ingredient_names": fields.get(
                        "ingredients", []
                    ),  # alias for Flask layer
                    "cuisine_types": fields.get("cuisine", []),
                    "description": fields.get("description", ""),
                    "origin": fields.get("origin", ""),
                    "url": fields.get(
                        "url", ""
                    ),  # Epicurious URL not stored — add if needed
                    "image_url": None,  # Not stored in Vespa — pulled from Edamam
                    # Scores for debugging / display
                    "similarity_score": hit.get("relevance", 0.0),
                    "dense_score": match_features.get(
                        "closeness(field,embedding)", 0.0
                    ),
                    "sparse_score": match_features.get("bm25(ingredients)", 0.0),
                    # No health/diet labels in Epicurious data — comes from Edamam results
                    "diet_labels": [],
                    "health_labels": [],
                }
            )

        return results
