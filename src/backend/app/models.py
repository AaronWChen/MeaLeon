from datetime import datetime, timezone
from flask_babel import lazy_gettext as _l
from flask_login import UserMixin
import jwt
import sqlalchemy as sa
import sqlalchemy.orm as so  # refactor out the ORM stuff
from time import time
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash
from src.backend.app import app, db, login

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

    following: so.WriteOnlyMapped["User"] = so.relationship(
        secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        back_populates="followers",
    )

    followers: so.WriteOnlyMapped["User"] = so.relationship(
        secondary=followers,
        primaryjoin=(followers.c.followed_id == id),
        secondaryjoin=(followers.c.follower_id == id),
        back_populates="following",
    )

    # allergies: so.WriteOnlyMapped["Allergy"] = so.relationship(back_populates="user")

    def __repr__(self):
        return "<User {}>".format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def follow(self, user):
        if not self.is_following(user):
            self.following.add(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.following.remove(user)

    def is_following(self, user):
        query = self.following.select().where(User.id == user.id)
        return db.session.scalar(query) is not None

    def followers_count(self):
        query = sa.select(sa.func.count()).select_from(
            self.followers.select().subquery()
        )
        return db.session.scalar(query)

    def following_count(self):
        query = sa.select(sa.func.count()).select_from(
            self.following.select().subquery()
        )
        return db.session.scalar(query)

    def following_reviews(self):
        Author = so.aliased(User)
        Follower = so.aliased(User)
        return (
            sa.select(Review)
            .join(Review.author.of_type(Author))
            .join(Author.followers.of_type(Follower), isouter=True)
            .where(
                sa.or_(
                    Follower.id == self.id,
                    Author.id == self.id,
                )
            )
            .group_by(Review)
            .order_by(Review.timestamp.desc())
        )

    def personal_reviews(self):
        Author = so.aliased(User)

        query = (
            sa.select(sa.func.count())
            .select_from(Review)
            .where(Author.id == self.id)
            .order_by(Review.timestamp.desc())
        )

        return db.session.scalar(query)

    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {"reset_password": self.id, "exp": time() + expires_in},
            app.config["SECRET_KEY"],
            algorithm="HS256",
        )

    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])[
                "reset_password"
            ]

        except:
            return

        return db.session.get(User, id)


class Review(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    body: so.Mapped[str] = so.mapped_column(sa.String(140))
    timestamp: so.Mapped[datetime] = so.mapped_column(
        index=True, default=lambda: datetime.now(timezone.utc)
    )
    user_id: so.Mapped[int] = so.mapped_column(sa.ForeignKey(User.id), index=True)
    modifications: so.Mapped[str] = so.mapped_column(
        sa.String(280), default="No modifications"
    )
    notes: so.Mapped[str] = so.mapped_column(sa.String(280), default="No notes")
    make_again: so.Mapped[bool] = so.mapped_column(sa.Boolean, default=True)
    rating: so.Mapped[int] = so.mapped_column(primary_key=False, default=3)

    author: so.Mapped[User] = so.relationship(back_populates="reviews")
    language: so.Mapped[Optional[str]] = so.mapped_column(sa.String(5))

    def __repr__(self):
        return "<Review {}>".format(self.body)


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
