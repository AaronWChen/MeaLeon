from datetime import datetime, timedelta, timezone
from flask import current_app, url_for
from flask_babel import lazy_gettext as _l
from flask_login import UserMixin
import jwt
import secrets
import sqlalchemy as sa
import sqlalchemy.orm as so  # refactor out the ORM stuff
from sqlalchemy.dialects.postgresql import ARRAY
from time import time
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login

followers = sa.Table(
    "followers",
    db.metadata,
    sa.Column("follower_id", sa.Integer, sa.ForeignKey("user.id"), primary_key=True),
    sa.Column("followed_id", sa.Integer, sa.ForeignKey("user.id"), primary_key=True),
)


class PaginatedAPIMixin(object):
    @staticmethod
    def to_collection_dict(query, page, per_page, endpoint, **kwargs):
        resources = db.paginate(query, page=page, per_page=per_page, error_out=False)

        data = {
            "items": [item.to_dict() for item in resources.items],
            "_meta": {
                "page": page,
                "per_page": per_page,
                "total_pages": resources.pages,
                "total_items": resources.total,
            },
            "_links": {
                "self": url_for(endpoint, page=page, per_page=per_page, **kwargs),
                "next": (
                    url_for(endpoint, page=page + 1, per_page=per_page, **kwargs)
                    if resources.has_next
                    else None
                ),
                "prev": (
                    url_for(endpoint, page=page - 1, per_page=per_page, **kwargs)
                    if resources.has_prev
                    else None
                ),
            },
        }

        return data


class UserPreferences(db.Model):
    """
    Stores a user's dietary preferences and restrictions.
    One-to-one with User.

    Using PostgreSQL ARRAY columns for the list fields — this avoids
    needing a separate join table for something that's always read
    and written as a unit. If you move off Postgres, swap to JSON columns.
    """

    __tablename__ = "user_preferences"

    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    user_id: so.Mapped[int] = so.mapped_column(
        sa.ForeignKey("user.id"), unique=True, index=True
    )

    # e.g. ["vegan", "gluten-free"]
    diet_labels: so.Mapped[list] = so.mapped_column(
        sa.ARRAY(sa.String), server_default="{}", nullable=False
    )
    # e.g. ["peanut-free", "shellfish-free"]
    health_labels: so.Mapped[list] = so.mapped_column(
        sa.ARRAY(sa.String), server_default="{}", nullable=False
    )
    # Hard excludes — individual ingredient names
    excluded_ingredients: so.Mapped[list] = so.mapped_column(
        sa.ARRAY(sa.String), server_default="{}", nullable=False
    )
    # Soft ranking signals — not hard filters
    preferred_cuisines: so.Mapped[list] = so.mapped_column(
        sa.ARRAY(sa.String), server_default="{}", nullable=False
    )
    disliked_cuisines: so.Mapped[list] = so.mapped_column(
        sa.ARRAY(sa.String), server_default="{}", nullable=False
    )

    user: so.Mapped["User"] = so.relationship(back_populates="preferences")

    def __repr__(self):
        return f"<UserPreferences user_id={self.user_id}>"


class User(
    PaginatedAPIMixin,
    UserMixin,
    db.Model,
):
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

    token: so.Mapped[Optional[str]] = so.mapped_column(
        sa.String(32), index=True, unique=True
    )
    token_expiration: so.Mapped[Optional[datetime]]

    preferences: so.Mapped[Optional["UserPreferences"]] = so.relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )

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
            current_app.config["SECRET_KEY"],
            algorithm="HS256",
        )

    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(
                token, current_app.config["SECRET_KEY"], algorithms=["HS256"]
            )["reset_password"]

        except:
            return

        return db.session.get(User, id)

    def reviews_count(self):
        query = sa.select(sa.func.count()).select_from(self.reviews.select().subquery())
        return db.session.scalar(query)

    def to_dict(self, include_email=False):
        data = {
            "id": self.id,
            "username": self.username,
            "last_seen": (
                self.last_seen.replace(tzinfo=timezone.utc).isoformat()
                if self.last_seen
                else None
            ),
            "about_me": self.about_me,
            "review_count": self.reviews_count(),
            "follower_count": self.followers_count(),
            "following_count": self.following_count(),
            "_links": {
                "self": url_for("api.get_user", id=self.id),
                "followers": url_for("api.get_followers", id=self.id),
                "following": url_for("api.get_following", id=self.id),
                # 'avatar': self.avatar(128)
            },
        }

        if include_email:
            data["email"] = self.email

        return data

    def from_dict(self, data, new_user=False):
        for field in ["username", "email", "about_me"]:
            if field in data:
                setattr(self, field, data[field])

        if new_user and "password" in data:
            self.set_password(data["password"])

    def get_token(self, expires_in=3600):
        now = datetime.now(timezone.utc)
        if self.token and self.token_expiration.replace(
            tzinfo=timezone.utc
        ) > now + timedelta(seconds=60):
            return self.token

        self.token = secrets.token_hex(16)
        self.token_expiration = now + timedelta(seconds=expires_in)
        db.session.add(self)
        return self.token

    def revoke_token(self):
        self.token_expiration = datetime.now(timezone.utc) - timedelta(seconds=1)

    @staticmethod
    def check_token(token):
        user = db.session.scalar(sa.select(User).where(User.token == token))
        if user is None or user.token_expiration.replace(
            tzinfo=timezone.utc
        ) < datetime.now(timezone.utc):
            return None

        return user


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
