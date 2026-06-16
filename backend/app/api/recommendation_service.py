"""
Flow:
  1. Parse + validate request
  2. Load user preferences if authenticated (anonymous users get unfiltered results)
  3. Call search_service  → get ingredient list from Edamam
  4. Call recommend_service → get ranked recipe candidates from Vespa
  5. Apply user restriction filters
  6. Return merged response
"""

import httpx
import logging
from flask import Blueprint, request, jsonify, current_app
from flask_login import current_user

from app.models import User
from app.api.errors import bad_request, error_response

logger = logging.getLogger(__name__)

bp = Blueprint("recommend", __name__)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@bp.post("/recommend")
def recommend():
    """
    POST /api/recommend
    Body: { "dish_name": str, "cuisine": str (optional) }

    Auth: optional — authenticated users get preference-filtered results,
    anonymous users get unfiltered results.

    Returns:
    {
        "results": [ RecipeResult, ... ],
        "query": { "dish_name": ..., "cuisine": ... },
        "user_filtered": bool,   -- true if user prefs were applied
        "total": int
    }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("dish_name"):
        return bad_request("dish_name is required")

    dish_name = data["dish_name"].strip()
    cuisine = data.get("cuisine", "").strip()

    # ------------------------------------------------------------------
    # Step 1: Load user preferences (only if logged in)
    # ------------------------------------------------------------------
    user_context = _build_user_context(current_user)

    # ------------------------------------------------------------------
    # Step 2: Call search service
    # ------------------------------------------------------------------
    try:
        search_resp = _call_search_service(dish_name, cuisine)

    except httpx.HTTPError as e:
        logger.error("Search service error: %s", e)
        return error_response(502, "Search service unavailable")

    recipes = search_resp.get("recipes", [])

    seen = set()

    # ingredients = search_resp.get("all_ingredients", [])
    ingredients = []
    for recipe in recipes:
        for ingred in recipe.get("ingredient_names", []):
            if ingred not in seen:
                seen.add(ingred)
                ingredients.append(ingred)

    if not ingredients:
        return jsonify(
            {
                "results": [],
                "query": {"dish_name": dish_name, "cuisine": cuisine},
                "user_filtered": False,
                "total": 0,
            }
        )
    source = "vespa"

    # ------------------------------------------------------------------
    # Step 3: Call recommendation service
    # ------------------------------------------------------------------
    try:
        recommend_resp = _call_recommend_service(
            dish_name=dish_name,
            cuisine=cuisine,
            ingredients=ingredients,
            user_context=user_context,
        )
        candidates = recommend_resp.get("results", [])

    except httpx.HTTPError as e:
        logger.error("Recommend service error: %s", e)
        return error_response(502, "Recommendation service unavailable")

    # If Vespa has no results yet (empty index), fall back to Edamam
    # results from the search phase — but apply cuisine exclusion here
    # so we still show cross-cuisine recommendations
    if not candidates:
        source = "edamam_fallback"
        search_recipes = search_resp.get("recipes", [])
        cuisine_lower = cuisine.lower() if cuisine else ""
        cross_cuisine = [
            r
            for r in search_recipes
            if cuisine_lower not in [c.lower() for c in r.get("cuisine_types", [])]
        ]
        # If everything matched the queried cuisine (unlikely but possible),
        # return all results rather than nothing — better UX while corpus is small
        candidates = cross_cuisine if cross_cuisine else recipes

        if not cross_cuisine and recipes:
            logger.info(
                "All Edamam results matched queried cuisine '%s' — "
                "returning unfiltered. Consider expanding the search.",
                cuisine,
            )

    # ------------------------------------------------------------------
    # Step 4: Apply user restriction filters
    # This is intentionally done here in Flask, not in the recommend
    # service, because restriction logic is user data — it belongs with
    # the service that owns users.
    # ------------------------------------------------------------------
    results, user_filtered = _apply_user_filters(candidates, user_context)

    return jsonify(
        {
            "results": results,
            "query": {"dish_name": dish_name, "cuisine": cuisine},
            "user_filtered": user_filtered,
            "total": len(results),
            "source": source,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_context(user) -> dict:
    """
    Build a context dict from the current user's preferences.
    Returns empty dict for anonymous users — all downstream code
    handles an empty context gracefully.
    """
    if not user or not user.is_authenticated:
        return {}

    # user.preferences is the new relationship we'll add to models.py
    # For now, return empty dict if it doesn't exist yet so the endpoint
    # works before the preference system is built out.
    prefs = getattr(user, "preferences", None)
    if not prefs:
        return {"user_id": user.id}

    return {
        "user_id": user.id,
        "diet_labels": prefs.diet_labels or [],  # e.g. ["vegan", "gluten-free"]
        "health_labels": prefs.health_labels or [],  # e.g. ["peanut-free"]
        "excluded_ingredients": prefs.excluded_ingredients or [],  # hard excludes
        "preferred_cuisines": prefs.preferred_cuisines or [],
        "disliked_cuisines": prefs.disliked_cuisines or [],
    }


def _call_search_service(dish_name: str, cuisine: str) -> dict:
    """Synchronous httpx call to the search microservice."""
    url = current_app.config["SEARCH_SERVICE_URL"] + "/search"
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            url,
            json={
                "dish_name": dish_name,
                "cuisine": cuisine,
                "max_results": 10,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _call_recommend_service(
    dish_name: str,
    cuisine: str,
    ingredients: list,
    user_context: dict,
) -> dict:
    """Synchronous httpx call to the recommendation microservice."""
    url = current_app.config["RECOMMEND_SERVICE_URL"] + "/recommend"
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            url,
            json={
                "dish_name": dish_name,
                "cuisine": cuisine,
                "ingredients": ingredients,
                # Pass user context so the recommend service can use
                # preferred_cuisines for ranking boosts (soft signal).
                # Hard restrictions are applied here in Flask, not there.
                "user_context": user_context,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _apply_user_filters(candidates: list, user_context: dict) -> tuple[list, bool]:
    """
    Filter and annotate recipe candidates based on user restrictions.

    Hard filter: remove recipes containing excluded_ingredients entirely.
    Soft filter: add a restriction_warning to recipes that contain
    something from health_labels (shown in the UI as a caution, not hidden).

    Returns (filtered_results, was_filtered_bool).
    """
    if not user_context:
        return candidates, False

    excluded = set(i.lower() for i in user_context.get("excluded_ingredients", []))
    health_labels = set(l.lower() for l in user_context.get("health_labels", []))

    results = []
    filtered = False

    for recipe in candidates:
        recipe_ingredients = {i.lower() for i in recipe.get("ingredient_names", [])}
        recipe_labels = {l.lower() for l in recipe.get("health_labels", [])}

        # Hard exclude — skip recipe entirely
        if excluded and recipe_ingredients & excluded:
            filtered = True
            continue

        # Soft warning — keep but annotate
        conflicts = health_labels - recipe_labels  # labels user wants that recipe lacks
        if conflicts:
            recipe = {
                **recipe,
                "restriction_warning": f"May not suit: {', '.join(conflicts)}",
            }

        results.append(recipe)

    return results, filtered
