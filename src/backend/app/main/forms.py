from flask_babel import _, lazy_gettext as _l
from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    BooleanField,
    SubmitField,
    TextAreaField,
    IntegerField,
)
from wtforms.validators import (
    DataRequired,
    ValidationError,
    Length,
    NumberRange,
)
import sqlalchemy as sa
from src.backend.app import db
from src.backend.app.models import User


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


# class ReviewSummaryForm(FlaskForm):
