from dotenv import load_dotenv
import os

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))


def _build_database_uri() -> str:
    # Priority 1: full URL from environment (DigitalOcean, Heroku, etc.)
    if url := os.environ.get("DATABASE_URL"):
        return url

    # Priority 2: individual credential env vars (Docker Compose)
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    host = os.environ.get("POSTGRES_HOST", "db:5432")
    db = os.environ.get("PG_DATABASE", "mealeon")
    if user and password:
        return f"postgresql+psycopg://{user}:{password}@{host}/{db}"

    # Priority 3: local dev fallback (SQLite-style path was wrong before)
    return f"postgresql+psycopg://localhost/mealeon"


class Config:
    # set edamam api access
    EDAMAM_API_APPID = os.environ.get("EDAMAM_API_APPID")
    EDAMAM_API_APPKEY = os.environ.get("EDAMAM_API_APPKEY")

    # set postgres credentials
    # seems like we shouldn't use localhost per https://stackoverflow.com/questions/31249112/allow-docker-container-to-connect-to-a-local-host-postgres-database
    # , but docker docs do
    # https://docs.docker.com/guides/databases/#connect-to-a-containerized-database-from-your-host
    host = os.environ.get("DATABASE_PUBLIC_IP") or "localhost"

    # database
    SQLALCHEMY_DATABASE_URI = _build_database_uri()

    # microservice URLs
    # These are used by the Flask backend to call out to microservices.
    # Default values match the Docker Compose service names.
    SEARCH_SERVICE_URL = os.environ.get(
        "SEARCH_SERVICE_URL", "http://search_service:8001"
    )
    RECOMMEND_SERVICE_URL = os.environ.get(
        "RECOMMEND_SERVICE_URL", "http://recommend_service:8002"
    )
    ML_SERVICE_URL = os.environ.get(
        "ML_SERVICE_URL", "http://embedding_generation:8000"
    )
    VESPA_URL = os.environ.get("VESPA_URL", "http://vespa:8080")
    NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://neo4j:7687")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")

    # email error handling
    MAIL_SERVER = os.environ.get("MAIL_SERVER")
    MAIL_PORT = int(os.environ.get("MAIL_PORT") or 25)
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS") is not None
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
    ADMINS = ["composedandfocused@gmail.com"]

    # app settings
    REVIEWS_PER_PAGE = 3
    LANGUAGES = ["en", "es", "zh", "ja", "ko", "pl", "fr", "eo"]
    # MS_TRANSLATOR_KEY = os.environ.get('MS_TRANSLATOR_KEY') # not using this, no access

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-change-in-production"
