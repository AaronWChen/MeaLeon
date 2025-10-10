import json
import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # set edamam api access
    # with open("secrets/edamam_api.json", "r") as fo:
    #     edamam_cred = json.loads(fo.read())
    # EDAMAM_API_APPID = (
    #     os.environ.get("EDAMAM_API_APPID") or edamam_cred["EDAMAM_API_APPID"]
    # )
    # EDAMAM_API_APPKEY = (
    #     os.environ.get("EDAMAM_API_APPKEY") or edamam_cred["EDAMAM_API_APPKEY"]
    # )

    SECRET_KEY = os.environ.get("SECRET_KEY") or "playful-passw0rd"

    # import postgres credentials from secrets file
    # postgres_key_path = "secrets/postgres_login.json"
    # with open(postgres_key_path, "r") as fo:
    #     postgres_key = json.loads(fo.read())
    # user = postgres_key["user"]
    # password = postgres_key["password"]
    # host = postgres_key["host"]

    # SQLALCHEMY_DATABASE_URI = (
    #     os.environ.get("DATABASE_URL")
    #     or "postgresql+psycopg://" + os.path.join(basedir, "app.db")
    #     or f"postgresql+psycopg://{user}:{password}@{host}/mealeon"
    # )

    # work with sqlite first (since I don't have access to postgres on desktop)
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL"
    ) or "sqlite:///" + os.path.join(basedir, "app.db")

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
