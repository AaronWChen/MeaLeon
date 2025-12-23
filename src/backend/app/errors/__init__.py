from flask import Blueprint

bp = Blueprint("errors", __name__)

from src.backend.app.errors import handlers
