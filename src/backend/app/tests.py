import os

os.environ["DATABASE_URL"] = "sqlite://"

from config import Config
from datetime import datetime, timezone, timedelta
import unittest
from src.backend.app import create_app, db
from src.backend.app.models import User, Review


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite://"


class UserModelCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_password_hashing(self):
        # make a test for password checking
        u = User(username="susan", email="susan@example.com")
        u.set_password("cat")
        self.assertFalse(u.check_password("dog"))
        self.assertTrue(u.check_password("cat"))

    # def test_avatar(self):
    #     # test avatar matching
    #     u = User(username='john', email='john@example.com')
    #     self.assertEqual(u.avatar(128), ('https://www.gravatar.com/avatar/d4c74594d841139328695756648b2bd6?d=identicon&s=128'))

    def test_follow(self):
        # test follow mechanics

        # create two users and add them to database
        u1 = User(username="john", email="john@example.com")
        u2 = User(username="susan", email="susan@example.com")
        db.session.add(u1)
        db.session.add(u2)
        db.session.commit()

        # following/follower lists should be empty since they have not linked
        following = db.session.scalars(u1.following.select()).all()
        followers = db.session.scalars(u2.followers.select()).all()
        self.assertEqual(following, [])
        self.assertEqual(followers, [])

        # make a user follow another and check functions are working
        u1.follow(u2)
        db.session.commit()
        self.assertTrue(u1.is_following(u2))
        self.assertEqual(u1.following_count(), 1)
        self.assertEqual(u2.followers_count(), 1)
        u1_following = db.session.scalars(u1.following.select()).all()
        u2_followers = db.session.scalars(u2.followers.select()).all()
        self.assertEqual(u1_following[0].username, "susan")
        self.assertEqual(u2_followers[0].username, "john")

        # test unfollow
        u1.unfollow(u2)
        db.session.commit()
        self.assertFalse(u1.is_following(u2))
        self.assertEqual(u1.following_count(), 0)
        self.assertEqual(u2.followers_count(), 0)

    def test_follow_reviews(self):
        # create four users
        u1 = User(username="john", email="john@example.com")
        u2 = User(username="susan", email="susan@example.com")
        u3 = User(username="mary", email="mary@example.com")
        u4 = User(username="david", email="david@example.com")
        db.session.add_all([u1, u2, u3, u4])

        # create four reviews
        now = datetime.now(timezone.utc)
        p1 = Review(
            body="review from john", author=u1, timestamp=now + timedelta(seconds=1)
        )
        p2 = Review(
            body="review from susan", author=u2, timestamp=now + timedelta(seconds=4)
        )
        p3 = Review(
            body="review from mary", author=u3, timestamp=now + timedelta(seconds=3)
        )
        p4 = Review(
            body="review from david", author=u4, timestamp=now + timedelta(seconds=2)
        )
        db.session.add_all([p1, p2, p3, p4])
        db.session.commit()

        # setup the followers
        u1.follow(u2)
        u1.follow(u4)
        u2.follow(u3)
        u3.follow(u4)
        db.session.commit()

        # check the following reviews of each user
        f1 = db.session.scalars(u1.following_reviews()).all()
        f2 = db.session.scalars(u2.following_reviews()).all()
        f3 = db.session.scalars(u3.following_reviews()).all()
        f4 = db.session.scalars(u4.following_reviews()).all()
        self.assertEqual(f1, [p2, p4, p1])
        self.assertEqual(f2, [p2, p3])
        self.assertEqual(f3, [p3, p4])
        self.assertEqual(f4, [p4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
