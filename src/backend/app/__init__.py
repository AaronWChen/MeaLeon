from flask import Flask, request
from flask_babel import Babel, lazy_gettext as _l
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy
import logging
from logging.handlers import SMTPHandler, RotatingFileHandler
import os

from src.backend.config import Config


def get_locale():
    return request.accept_languages.best_match(app.config["LANGUAGES"])


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
mail = Mail(app)  # emulated email server,
moment = Moment(app)
babel = Babel(app, locale_selector=get_locale)
# 1. run in terminal `aiosmtpd -n -c aiosmtpd.handlers.Debugging -l localhost:8025`
# 2. terminal `export MAIL_SERVER=localhost`
# 3. terminal `export MAIL_PORT=8025`

# below is for forcing users to login if they reach a protected page
login.login_view = "login"
login.login_message = _l("Please log in to access this page.")

if not app.debug:
    if app.config["MAIL_SERVER"]:
        auth = None
        if app.config["MAIL_USERNAME"] or app.config["MAIL_PASSWORD"]:
            auth - (app.config["MAIL_USERNAME"], app.config["MAIL_PASSWORD"])

        secure = None
        if app.config["MAIL_USE_TLS"]:
            secure = ()

        mail_handler = SMTPHandler(
            mailhost=(app.config["MAIL_SERVER"], app.config["MAIL_PORT"]),
            fromaddr="no-reply@" + app.config["MAIL_SERVER"],
            toaddrs=app.config["ADMINS"],
            subject="MeaLeon Failure",
            credentials=auth,
            secure=secure,
        )
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)

    if not os.path.exists("logs"):
        os.mkdir("logs")
    file_handler = RotatingFileHandler(
        "logs/mealeon.log", maxBytes=10240, backupCount=10
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
        )
    )
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info("MeaLeon startup")

from src.backend.app import routes, models, errors
