"""
Uses shared.secrets for all credential reads so the same code
works locally (Docker secret files) and on DigitalOcean (env vars).
"""

import os
import sys

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

# shared/ is a sibling of backend/ — add parent to path
sys.path.insert(0, os.path.join(basedir, ".."))
from shared.secrets import get_db_url, get_service_url, get_optional


class Config:
    # ── Database ──────────────────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = get_db_url()

    # ── Microservice URLs ─────────────────────────────────────────────────
    # Local defaults are docker-compose service names.
    # On DigitalOcean, these env vars are set to ${service.PRIVATE_URL}
    # in app.yaml and injected automatically.
    SEARCH_SERVICE_URL = get_service_url(
        "SEARCH_SERVICE_URL", "http://search_service:8001"
    )
    RECOMMEND_SERVICE_URL = get_service_url(
        "RECOMMEND_SERVICE_URL", "http://recommend_service:8002"
    )
    ML_SERVICE_URL = get_service_url(
        "ML_SERVICE_URL", "http://embedding_generation:8000"
    )
    VESPA_URL = get_service_url("VESPA_URL", "http://vespa:8080")
    NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://neo4j:7687")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")

    # ── Mail ──────────────────────────────────────────────────────────────
    MAIL_SERVER = os.environ.get("MAIL_SERVER")
    MAIL_PORT = int(os.environ.get("MAIL_PORT") or 25)
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS") is not None
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = get_optional("MAIL_PASSWORD")
    ADMINS = ["composedandfocused@gmail.com"]

    # ── App ───────────────────────────────────────────────────────────────
    REVIEWS_PER_PAGE = 3
    LANGUAGES = ["en", "es", "zh", "ja", "ko", "pl", "fr", "eo"]
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-change-in-production"
