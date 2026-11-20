from flask import Blueprint

bp = Blueprint("auth", __name__)

from services.backend.app.auth import routes
