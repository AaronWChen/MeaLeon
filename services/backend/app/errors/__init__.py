from flask import Blueprint

bp = Blueprint("errors", __name__)

from services.backend.app.errors import handlers
