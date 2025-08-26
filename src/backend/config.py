import json
import os


class Config:
    EDAMAM_API_APPID = os.environ.get("EDAMAM_API_APPID")
    EDAMAM_API_APPKEY = os.environ.get("EDAMAM_API_APPKEY")

    SECRET_KEY = os.environ.get("SECRET_KEY") or "playful-passw0rd"

    # import postgres credentials from secrets file
    postgres_key_path = "../secrets/postgres_login.json"
    with open(postgres_key_path, "r") as fo:
        postgres_key = json.loads(fo.read())
    user = postgres_key["user"]
    password = postgres_key["password"]
    host = postgres_key["host"]

    basedir = os.path.abspath(os.path.dirname(__file__))

    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL")
        or "postgresql+psycopg://" + os.path.join(basedir, "app.db")
        or f"postgresql+psycopg://{user}:{password}@{host}/mealeon"
    )
