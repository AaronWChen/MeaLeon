from src.backend.app import db

from src.backend.app.auth.email import send_password_reset_email
from src.backend.app.auth import bp
from src.backend.app.auth.forms import (
    LoginForm,
    RegistrationForm,
    ResetPasswordRequestForm,
    ResetPasswordForm,
)
from src.backend.app.models import User, Review
from src.backend.app.translate import translate
from src.nltk import dish_predictor as dp  # import find_similar_dishes

from datetime import datetime, timezone
from flask import render_template, request, abort, flash, redirect, url_for, g
from flask_babel import _, get_locale
from flask_login import current_user, login_user, logout_user, login_required
from langdetect import detect, LangDetectException
import sqlalchemy as sa
import sqlalchemy.orm as so
from urllib.parse import urlsplit


@bp.route("/login", methods=["GET", "POST"])
def login():
    # check if user is logged in already, if so, send to index page
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = LoginForm()
    if form.validate_on_submit():
        # get user from database
        user = db.session.scalar(
            sa.select(User).where(User.username == form.username.data)
        )

        # if user does not exist or password is incorrect, ask for relogin
        if not user or not user.check_password(form.password.data):
            flash(_("Invalid username or password"))
            return redirect(url_for("auth.login"))

        # else log the user in and return them to index page
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get("next")
        if not next_page or urlsplit(next_page).netloc != "":
            next_page = url_for("index")
        return redirect(next_page)
    return render_template("auth/login.html", title=_("Sign In"), form=form)


@bp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    # create a place for users to make an account
    if current_user.is_authenticated:
        # if logged in already, just go to homepage
        return redirect(url_for("index"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(_("Registration successful!"))
        return redirect(url_for("auth.login"))
    return render_template("auth/register.html", title="Register", form=form)


@bp.route("/reset_password_request", methods=["GET", "POST"])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = ResetPasswordRequestForm()

    if form.validate_on_submit():
        user = db.session.scalar(sa.select(User).where(User.email == form.email.data))

        if user:
            send_password_reset_email(user)
            print(user)

        flash(_("Check your email for the instructions to reset your password"))

        return redirect(url_for("auth.login"))
    return render_template(
        "auth/reset_password_request.html", title=_("Reset Password"), form=form
    )


@bp.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    user = User.verify_reset_password_token(token)
    if not user:
        return redirect(url_for("index"))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash(_("Your password has been reset."))
        return redirect(url_for("auth.login"))
    return render_template("auth/reset_password.html", form=form)
