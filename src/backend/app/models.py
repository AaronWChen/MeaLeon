from datetime import datetime, timezone
from flask_login import UserMixin
import sqlalchemy as sa
import sqlalchemy.orm as so  # refactor out the ORM stuff
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash
from src.backend.app import db, login

followers = sa.Table(
    "followers",
    db.metadata,
    sa.Column("follower_id", sa.Integer, sa.ForeignKey("user.id"), primary_key=True),
    sa.Column("followed_id", sa.Integer, sa.ForeignKey("user.id"), primary_key=True),
)


class User(UserMixin, db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    username: so.Mapped[str] = so.mapped_column(sa.String(64), index=True, unique=True)
    email: so.Mapped[str] = so.mapped_column(sa.String(120), index=True, unique=True)
    password_hash: so.Mapped[Optional[str]] = so.mapped_column(sa.String(256))
    about_me: so.Mapped[Optional[str]] = so.mapped_column(sa.String(140))
    last_seen: so.Mapped[Optional[datetime]] = so.mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )

    reviews: so.WriteOnlyMapped["Review"] = so.relationship(back_populates="author")

    # allergies: so.WriteOnlyMapped["Allergy"] = so.relationship(back_populates="user")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class Review(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    body: so.Mapped[str] = so.mapped_column(sa.String(140))
    timestamp: so.Mapped[datetime] = so.mapped_column(
        index=True, default=lambda: datetime.now(timezone.utc)
    )
    user_id: so.Mapped[int] = so.mapped_column(sa.ForeignKey(User.id), index=True)
    modifications: so.Mapped[str] = so.mapped_column(sa.String(280))
    notes: so.Mapped[str] = so.mapped_column(sa.String(280))
    make_again: so.Mapped[bool] = so.mapped_column(sa.Boolean)
    rating: so.Mapped[int] = so.mapped_column(primary_key=False)

    author: so.Mapped[User] = so.relationship(back_populates="reviews")

    def __repr__(self):
        return f"<Post {self.body}>"


# might want to refactor this to be an ingredient table with an allergy status
# class Allergy(db.Model):
#     id: so.Mapped[int] = so.mapped_column(primary_key=True)
#     ingredient: so.Mapped[str] = so.mapped_column(
#         sa.String(140)
#     )  # expect to replace this with Enum maybe

#     # user_id: so.Mapped[int] = so.mapped_column(sa.ForeignKey(User.id), index=True)

#     # user: so.Mapped[User] = so.relationship(back_populates="allergies")

#     def __repr__(self):
#         return f"<Allergy {self.ingredient}>"


class Recipe(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    title: so.Mapped[str] = so.mapped_column(sa.String(140))

    def __repr__(self):
        return f"<Recipe {self.title}>"


@login.user_loader
def load_user(id):
    return db.session.get(User, int(id))
