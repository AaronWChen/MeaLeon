import sqlalchemy as sa
from flask import request, url_for, abort
from app import db
from app.api import bp
from app.api.auth import token_auth
from app.api.errors import bad_request
from app.models import User


@bp.route("/users/<int:id>", methods=["GET"])
@token_auth.login_required
def get_user(id):
    return db.get_or_404(User, id).to_dict()


@bp.route("/users", methods=["GET"])
@token_auth.login_required
def get_users():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 10, type=int), 100)
    return User.to_collection_dict(sa.select(User), page, per_page, "api.get_users")


@bp.route("/users/<int:id>/followers", methods=["GET"])
@token_auth.login_required
def get_followers(id):
    user = db.get_or_404(User, id)
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 10, type=int), 100)
    return User.to_collection_dict(
        user.followers.select(), page, per_page, "api.get_followers", id=id
    )


@bp.route("/users/<int:id>/following", methods=["GET"])
@token_auth.login_required
def get_following(id):
    user = db.get_or_404(User, id)
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 10, type=int), 100)
    return User.to_collection_dict(
        user.following.select(), page, per_page, "api.get_following", id=id
    )


@bp.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    if "username" not in data or "email" not in data or "password" not in data:
        return bad_request("Must include username, email, and password field")

    if db.session.scalar(sa.select(User).where(User.username == data["username"])):
        return bad_request("Please use a different username")

    if db.session.scalar(sa.select(User).where(User.email == data["email"])):
        return bad_request("Please use a different email address")

    user = User()
    user.from_dict(data, new_user=True)
    db.session.add(user)
    db.session.commit()
    return user.to_dict(), 201, {"Location": url_for("api.get_user", id=user.id)}


@bp.route("/users/<int:id>", methods=["PUT"])
@token_auth.login_required
def update_user(id):
    if token_auth.current_user().id != id:
        abort(403)

    user = db.get_or_404(User, id)
    data = request.get_json()
    if (
        "username" in data
        and data["username"] != user.username
        and db.session.scalar(sa.select(User).where(User.username == data["username"]))
    ):
        return bad_request("Please use a different username")

    if (
        "email" in data
        and data["email"] != user.email
        and db.session.scalar(sa.select(User).where(User.email == data["email"]))
    ):
        return bad_request("Please use a different email address")

    user.from_dict(data, new_user=False)
    db.session.commit()
    return user.to_dict()


# from fastapi import APIRouter, Depends, HTTPException, Request
# from app.middleware.auth import get_current_user
# from app.services.recommendation_service import RecommendationService
# from app.schemas import SearchRequest, SearchResponse, RecommendationItem
# from slowapi import Limiter
# from slowapi.util import get_remote_address
# from typing import List, Optional

# router = APIRouter()
# limiter = Limiter(key_func=get_remote_address)

# def get_recommendation_service(request: Request) -> RecommendationService:
#   """Dependency injection for recommendation service"""
#   return request.app.state.recommendation_service

# @router.post("/search", response_model=SearchResponse)
# @limiter.limit("100/hour")
# async def search_items(
#   request: Request,
#   search_request: SearchRequest,
#   user=Depends(get_current_user),
#   rec_service: RecommendationService = Depends(get_recommendation_service)
# ):
#   """Search for items using natural language query"""

#   try:
#       results = await rec_service.recommend_from_query(
#           query=search_request.query,
#           limit=search_request.limit or 10,
#           filters=search_request.filters,
#           min_score=search_request.min_score or 0.5
#       )

#       return SearchResponse(
#           query=search_request.query,
#           results=results,
#           count=len(results)
#       )

#   except Exception as e:
#       raise HTTPException(status_code=500, detail=str(e))

# @router.get("/similar/{item_id}", response_model=List[RecommendationItem])
# @limiter.limit("200/hour")
# async def get_similar_items(
#   request: Request,
#   item_id: str,
#   limit: int = 10,
#   user=Depends(get_current_user),
#   rec_service: RecommendationService = Depends(get_recommendation_service)
# ):
#   """Get items similar to a specific item"""

#   results = await rec_service.recommend_from_item(
#       item_id=item_id,
#       limit=limit
#   )

#   return results
