from dotenv import load_dotenv
import os

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))


class Config:
    # set edamam api access
    EDAMAM_API_APPID = os.environ.get("EDAMAM_API_APPID")
    EDAMAM_API_APPKEY = os.environ.get("EDAMAM_API_APPKEY")

    # set postgres credentials
    # seems like we shouldn't use localhost per https://stackoverflow.com/questions/31249112/allow-docker-container-to-connect-to-a-local-host-postgres-database
    # , but docker docs do
    # https://docs.docker.com/guides/databases/#connect-to-a-containerized-database-from-your-host
    host = os.environ.get("DATABASE_PUBLIC_IP") or "localhost"

    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL")
        or "postgresql+psycopg://" + os.path.join(basedir, "app.db")
        or f"postgresql+psycopg://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{host}/{os.environ.get('PG_DATABASE')}"
    )

    # email error handling
    MAIL_SERVER = os.environ.get("MAIL_SERVER")
    MAIL_PORT = int(os.environ.get("MAIL_PORT") or 25)
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS") is not None
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
    ADMINS = ["composedandfocused@gmail.com"]
    REVIEWS_PER_PAGE = 3
    LANGUAGES = ["en", "es", "zh", "ja", "ko", "pl", "fr", "eo"]
    # MS_TRANSLATOR_KEY = os.environ.get('MS_TRANSLATOR_KEY') # not using this, no access
