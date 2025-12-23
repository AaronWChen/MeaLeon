from flask import Blueprint

bp = Blueprint("auth", __name__)

from src.backend.app.auth import routes
