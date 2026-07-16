"""
backend/app/api/preferences.py

User preferences endpoints — lets the React app read and update
dietary restrictions, health labels, and cuisine preferences.

These feed directly into _build_user_context() in recommend_service.py,
which is why they live in the Flask backend rather than in a separate service:
they're tightly coupled to the User model and auth system Flask already owns.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user

from app import db
from app.api.errors import bad_request

bp = Blueprint("preferences", __name__)

ALLOWED_DIET_LABELS = {
    "vegan",
    "vegetarian",
    "pescatarian",
    "paleo",
    "keto",
    "low-carb",
    "low-fat",
    "high-protein",
    "gluten-free",
    "dairy-free",
}

ALLOWED_HEALTH_LABELS = {
    "peanut-free",
    "tree-nut-free",
    "soy-free",
    "egg-free",
    "wheat-free",
    "shellfish-free",
    "sulfite-free",
    "alcohol-free",
    "kosher",
    "halal",
}


@bp.get("/users/me/preferences")
@login_required
def get_preferences():
    """Return the current user's dietary preferences."""
    prefs = getattr(current_user, "preferences", None)

    if not prefs:
        # Return empty defaults — preferences not set yet
        return jsonify(
            {
                "diet_labels": [],
                "health_labels": [],
                "excluded_ingredients": [],
                "preferred_cuisines": [],
                "disliked_cuisines": [],
            }
        )

    return jsonify(
        {
            "diet_labels": prefs.diet_labels or [],
            "health_labels": prefs.health_labels or [],
            "excluded_ingredients": prefs.excluded_ingredients or [],
            "preferred_cuisines": prefs.preferred_cuisines or [],
            "disliked_cuisines": prefs.disliked_cuisines or [],
        }
    )


@bp.put("/users/me/preferences")
@login_required
def update_preferences():
    """
    Replace the current user's preferences entirely.
    Validates labels against known-good sets to prevent junk data.
    """
    data = request.get_json(silent=True)
    if not data:
        return bad_request("Request body required")

    # Validate diet and health labels against allow-lists
    diet_labels = data.get("diet_labels", [])
    invalid_diet = set(diet_labels) - ALLOWED_DIET_LABELS
    if invalid_diet:
        return bad_request(f"Unknown diet labels: {invalid_diet}")

    health_labels = data.get("health_labels", [])
    invalid_health = set(health_labels) - ALLOWED_HEALTH_LABELS
    if invalid_health:
        return bad_request(f"Unknown health labels: {invalid_health}")

    # excluded_ingredients are free-form strings — just limit length
    excluded = data.get("excluded_ingredients", [])
    if any(len(i) > 100 for i in excluded):
        return bad_request("Ingredient names must be under 100 characters")

    # Upsert preferences (create if not exists, update if exists)
    # NOTE: UserPreferences model needs to be added to models.py
    # See models_preferences.py for the model definition
    prefs = getattr(current_user, "preferences", None)
    if prefs is None:
        from app.models import UserPreferences

        prefs = UserPreferences(user_id=current_user.id)
        db.session.add(prefs)

    prefs.diet_labels = diet_labels
    prefs.health_labels = health_labels
    prefs.excluded_ingredients = excluded
    prefs.preferred_cuisines = data.get("preferred_cuisines", [])
    prefs.disliked_cuisines = data.get("disliked_cuisines", [])

    db.session.commit()

    return jsonify({"status": "updated"}), 200
