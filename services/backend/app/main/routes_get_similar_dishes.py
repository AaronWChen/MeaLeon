"""
Drop-in replacement for the get_similar_dishes() function and
get_results route in backend/app/main/routes.py.

Fixes:
  1. ingred_weights now uses inverse-frequency scoring to approximate
     TF-IDF — distinctive ingredients rank higher than common ones
  2. rec_weights shows "N/A" instead of placeholder 1.0 until Vespa
     provides real cosine similarity scores
  3. Notes on fixing the _('...') Babel rendering issue

For the _('...') issue:
  Add this to create_app() in backend/app/__init__.py:

    from flask_babel import Babel
    babel = Babel(app)

  And confirm flask-babel is in pyproject.toml:
    flask-babel = ">=3.0.0"

  If you don't need i18n right now, replace _('...') in results.html
  with plain strings — but fixing the Babel init is cleaner long-term.
"""

import httpx
from collections import Counter
from flask import current_app, abort, render_template, request


def get_similar_dishes(dish: str, cuisine: str) -> tuple[list, list, list]:
    """
    Phase 1: call search_service to find the queried recipe.
    Phase 2: call recommend_service to find cross-cuisine similar recipes.

    Falls back to Edamam results with cuisine exclusion while Vespa
    index is still being built.

    Returns (results, ingreds, rec_weights) matching the original
    dish_predictor.find_similar_dishes() signature so results.html
    needs no changes.
    """
    try:
        search_resp = _call_search_service(dish, cuisine)
    except Exception as e:
        current_app.logger.error(f"Search service error: {e}")
        return [], [], []

    recipes_from_search = search_resp.get("recipes", [])
    if not recipes_from_search:
        return [], [], []

    # ------------------------------------------------------------------
    # Build ingredient list from search results for Phase 2
    # ------------------------------------------------------------------
    seen: set = set()
    ingredients: list = []
    for recipe in recipes_from_search:
        for ing in recipe.get("ingredient_names", []):
            if ing not in seen:
                seen.add(ing)
                ingredients.append(ing)

    # ------------------------------------------------------------------
    # Phase 2: get cross-cuisine recommendations from Vespa
    # ------------------------------------------------------------------
    candidates = []
    try:
        recommend_resp = _call_recommend_service(dish, cuisine, ingredients)
        candidates = recommend_resp.get("results", [])
    except Exception as e:
        current_app.logger.error(f"Recommend service error: {e}")

    # ------------------------------------------------------------------
    # Edamam fallback while Vespa index is being built
    # ------------------------------------------------------------------
    if not candidates:
        cuisine_lower = cuisine.lower() if cuisine else ""
        cross_cuisine = [
            r
            for r in recipes_from_search
            if cuisine_lower not in [c.lower() for c in r.get("cuisine_types", [])]
        ]
        candidates = cross_cuisine if cross_cuisine else recipes_from_search

    # ------------------------------------------------------------------
    # Compute distinctive ingredients using inverse frequency scoring
    # (approximates TF-IDF until real scores come from embedding service)
    #
    # Logic: ingredients that appear in fewer recipes are more distinctive.
    # e.g. "ricotta" appearing in 1/5 recipes ranks higher than "salt"
    # appearing in 5/5 recipes.
    # ------------------------------------------------------------------
    all_ingredients_across_results = [
        ing for r in candidates for ing in r.get("ingredient_names", [])
    ]
    ingredient_freq = Counter(all_ingredients_across_results)

    results = []
    for r in candidates:
        recipe_ings = r.get("ingredient_names", [])

        # Sort by ascending frequency (rarest = most distinctive first)
        distinctive = sorted(
            recipe_ings,
            key=lambda ing: ingredient_freq.get(ing, 1),
        )

        results.append(
            {
                "hed": r.get("label", ""),
                "title": r.get("label", ""),
                "fixed_url": r.get("url", ""),
                "photo": "",  # Edamam images not in DO Spaces yet
                "imputed_label": ", ".join(r.get("cuisine_types", [])) or "Unknown",
                "ingred_weights": distinctive[:5],
                # Real cosine similarity comes from Vespa similarity_score.
                # Show actual score if available, otherwise None so the
                # template can display "N/A" instead of a misleading 1.0
                "rounded": (
                    round(r.get("similarity_score", 0), 4)
                    if r.get("similarity_score", 0) > 0
                    else None
                ),
                "ingredients": r.get("ingredient_lines", []),
                "source": r.get("source", ""),
            }
        )

    # ------------------------------------------------------------------
    # ingreds — flat deduplicated list across all results
    # ------------------------------------------------------------------
    seen_ingreds: set = set()
    ingreds: list = []
    for r in candidates:
        for ing in r.get("ingredient_names", []):
            if ing not in seen_ingreds:
                seen_ingreds.add(ing)
                ingreds.append(ing)

    # ------------------------------------------------------------------
    # rec_weights — real scores from Vespa, or None as placeholder
    # results.html should check for None and display "N/A"
    # ------------------------------------------------------------------
    rec_weights = [
        (
            round(r.get("similarity_score", 0), 4)
            if r.get("similarity_score", 0) > 0
            else None
        )
        for r in candidates
    ]

    return results, ingreds, rec_weights


def _call_search_service(dish: str, cuisine: str) -> dict:
    url = current_app.config["SEARCH_SERVICE_URL"] + "/search"
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(
            url,
            json={
                "dish_name": dish,
                "cuisine": cuisine,
                "max_results": 10,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _call_recommend_service(dish: str, cuisine: str, ingredients: list) -> dict:
    url = current_app.config["RECOMMEND_SERVICE_URL"] + "/recommend"
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            url,
            json={
                "dish_name": dish,
                "cuisine": cuisine,
                "ingredients": ingredients,
                "user_context": {},
            },
        )
        resp.raise_for_status()
        return resp.json()
