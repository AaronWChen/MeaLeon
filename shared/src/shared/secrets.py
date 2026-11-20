"""
shared/secrets.py

Unified secret reader that works in:

  Local Docker Compose:
    Secrets are mounted as files at /run/secrets/<NAME>
    e.g. /run/secrets/DB_PASSWORD

  DigitalOcean App Platform:
    Secrets are injected as environment variables
    e.g. DB_PASSWORD=yourpassword

Resolution order for get(name):
  1. /run/secrets/<name>        (Docker, exact case)
  2. /run/secrets/<name.lower>) (Docker, lowercase)
  3. os.environ[name]           (DigitalOcean / any env var)
  4. default if provided, else raise

Usage:
    from shared.secrets import get, get_db_url, get_service_url

    password = get("DB_PASSWORD")
    db_url   = get_db_url()
    search   = get_service_url("SEARCH_SERVICE_URL", "http://search_service:8001")
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------


def get(name: str, default: Optional[str] = None) -> str:
    """
    Read a secret by name. Tries Docker secret file first, then env var.
    Raises ValueError if not found and no default provided.
    """
    # Try Docker secret file (both cases)
    for path in [f"/run/secrets/{name}", f"/run/secrets/{name.lower()}"]:
        if os.path.exists(path):
            with open(path) as f:
                value = f.read().strip()
            if value:
                return value

    # Try environment variable
    value = os.environ.get(name)
    if value:
        return value

    # Fall back to default
    if default is not None:
        return default

    raise ValueError(
        f"Secret '{name}' not found. "
        f"Expected at /run/secrets/{name} or as env var {name}."
    )


def get_optional(name: str) -> Optional[str]:
    """Return None instead of raising if secret is missing."""
    try:
        return get(name)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Typed helpers for common secrets
# ---------------------------------------------------------------------------


def get_db_url() -> str:
    """
    Build the SQLAlchemy database URL from individual components.

    On DigitalOcean, DATABASE_URL is injected directly by the managed
    Postgres add-on — use it if present since it includes the full
    connection string with SSL params.

    Locally, build from individual POSTGRES_* vars + secret file.
    """
    # DigitalOcean managed DB injects this directly
    direct_url = os.environ.get("DATABASE_URL")
    if direct_url:
        # DO injects postgres:// but SQLAlchemy needs postgresql+psycopg://
        return direct_url.replace("postgres://", "postgresql+psycopg://", 1).replace(
            "postgresql://", "postgresql+psycopg://", 1
        )

    user = os.environ.get("POSTGRES_USER", "docker_user")
    host = os.environ.get("POSTGRES_HOST", "db:5432")
    db = os.environ.get("POSTGRES_DB", os.environ.get("PG_DATABASE", "mealeon"))
    password = get("DB_PASSWORD")

    return f"postgresql+psycopg://{user}:{password}@{host}/{db}"


def get_service_url(env_name: str, local_default: str) -> str:
    """
    Get an internal service URL.

    Local:           uses docker-compose service name (local_default)
    DigitalOcean:    injects ${service.PRIVATE_URL} as an env var

    The env_name should match what you configure in app.yaml, e.g.:
      SEARCH_SERVICE_URL: ${search-service.PRIVATE_URL}
    """
    return os.environ.get(env_name, local_default)


def get_edamam_creds() -> tuple[str, str]:
    """Return (app_id, app_key) for the Edamam API."""
    return get("EDAMAM_API_APPID"), get("EDAMAM_API_APPKEY")


def get_redis_url() -> str:
    """
    Redis URL.
    Local: redis://redis:6379
    DigitalOcean: managed Redis injects REDIS_URL with rediss:// (TLS)
    """
    return os.environ.get("REDIS_URL", "redis://redis:6379")
