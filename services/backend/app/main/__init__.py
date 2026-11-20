from flask import Blueprint

bp = Blueprint("main", __name__)

from services.backend.app.main import routes
