from flask_babel import _, lazy_gettext as _l
from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    PasswordField,
    BooleanField,
    SubmitField,
    TextAreaField,
    IntegerField,
)
from wtforms.validators import (
    DataRequired,
    ValidationError,
    Email,
    EqualTo,
    Length,
    NumberRange,
)
import sqlalchemy as sa
from src.backend.app import db
from src.backend.app.models import User


class LoginForm(FlaskForm):
    username = StringField(_l("Username"), validators=[DataRequired()])
    password = PasswordField(_l("Password"), validators=[DataRequired()])
    remember_me = BooleanField(_l("Remember Me"))
    submit = SubmitField(_l("Sign In"))


class RegistrationForm(FlaskForm):
    username = StringField(_l("Username"), validators=[DataRequired()])
    email = StringField(_l("Email"), validators=[DataRequired(), Email()])
    password = PasswordField(_l("Password"), validators=[DataRequired()])
    password2 = PasswordField(
        _l("Repeat Password"), validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField(_l("Register"))

    def validate_username(self, username):
        user = db.session.scalar(sa.select(User).where(User.username == username.data))

        if user is not None:
            raise ValidationError(_("Please use a different username."))

    def validate_email(self, email):
        user = db.session.scalar(sa.select(User).where(User.email == email.data))

        if user is not None:
            raise ValidationError(_("Please use a different email address."))


class EditProfileForm(FlaskForm):
    username = StringField(_l("Username"), validators=[DataRequired()])
    about_me = TextAreaField(_l("About me"), validators=[Length(min=0, max=140)])
    submit = SubmitField(_l("Submit"))

    def __init__(self, original_username, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = db.session.scalar(
                sa.Select(User).where(User.username == username.data)
            )
        if user is not None:
            raise ValidationError(_("Please use a different username."))


class EmptyForm(FlaskForm):
    submit = SubmitField(_l("Submit"))


class ReviewForm(FlaskForm):
    # how to link to a recipe
    review = TextAreaField(_l("Write your review for a recipe"))
    notes = TextAreaField(_l("What notes for your review?"))  # this seems redundant
    modifications = TextAreaField(_l("How did you modify the recipe?"))
    rating = IntegerField(
        _l("Rating"),
        validators=[
            NumberRange(min=1, max=5, message=_("Rating between 1 and 5 please!"))
        ],
    )
    # would like to be able to modify a rating

    make_again = BooleanField(_l("Make again?"))
    submit = SubmitField(_l("Submit"))


class ResetPasswordRequestForm(FlaskForm):
    email = StringField(_l("Email"), validators=[DataRequired(), Email()])
    submit = SubmitField(_l("Request Password Reset"))


class ResetPasswordForm(FlaskForm):
    password = PasswordField(_l("Password"), validators=[DataRequired()])
    password2 = PasswordField(
        _l("Repeat Password"), validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField(_l("Request Password Reset"))


# class ReviewSummaryForm(FlaskForm):
