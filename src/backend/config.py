import os


class Config:
    EDAMAM_API_APPID = os.environ.get("EDAMAM_API_APPID")
    EDAMAM_API_APPKEY = os.environ.get("EDAMAM_API_APPKEY")

    SECRET_KEY = os.environ.get("SECRET_KEY") or "playful-passw0rd"
