from flask import Blueprint

bp = Blueprint("main", __name__)

from src.backend.app.main import routes
