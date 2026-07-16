-- db/init/01_schema.sql
--
-- Initial schema for MeaLeon.
-- Derived from backend/app/models.py.
--
-- This file is run automatically by the Postgres container on FIRST startup
-- (when the pgdata volume is empty). It is idempotent — safe to re-run.
-- Flask-Migrate handles subsequent schema changes via Alembic migrations.

-- Extension for better UUID support (optional but useful)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Followers join table (before User so FK can reference it) ────────────

CREATE TABLE IF NOT EXISTS followers (
    follower_id INTEGER NOT NULL,
    followed_id INTEGER NOT NULL,
    PRIMARY KEY (follower_id, followed_id)
);

-- ── User ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS "user" (
    id                SERIAL PRIMARY KEY,
    username          VARCHAR(64)  NOT NULL UNIQUE,
    email             VARCHAR(120) NOT NULL UNIQUE,
    password_hash     VARCHAR(256),
    about_me          VARCHAR(140),
    last_seen         TIMESTAMPTZ  DEFAULT NOW(),
    token             VARCHAR(32)  UNIQUE,
    token_expiration  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_user_username ON "user" (username);
CREATE INDEX IF NOT EXISTS ix_user_email    ON "user" (email);
CREATE INDEX IF NOT EXISTS ix_user_token    ON "user" (token);

-- Add FK constraints to followers now that user table exists
ALTER TABLE followers
    ADD CONSTRAINT IF NOT EXISTS fk_followers_follower
        FOREIGN KEY (follower_id) REFERENCES "user" (id) ON DELETE CASCADE,
    ADD CONSTRAINT IF NOT EXISTS fk_followers_followed
        FOREIGN KEY (followed_id) REFERENCES "user" (id) ON DELETE CASCADE;

-- ── Review ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS review (
    id            SERIAL PRIMARY KEY,
    body          VARCHAR(140)  NOT NULL,
    timestamp     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    user_id       INTEGER       NOT NULL REFERENCES "user" (id) ON DELETE CASCADE,
    modifications VARCHAR(280)  NOT NULL DEFAULT 'No modifications',
    notes         VARCHAR(280)  NOT NULL DEFAULT 'No notes',
    make_again    BOOLEAN       NOT NULL DEFAULT TRUE,
    rating        INTEGER       NOT NULL DEFAULT 3,
    language      VARCHAR(5)
);

CREATE INDEX IF NOT EXISTS ix_review_timestamp ON review (timestamp);
CREATE INDEX IF NOT EXISTS ix_review_user_id   ON review (user_id);

-- ── Recipe ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS recipe (
    id    SERIAL PRIMARY KEY,
    title VARCHAR(140) NOT NULL
);

-- ── User preferences (from models_preferences.py) ─────────────────
-- Uncomment when you're ready to add the preference system.
--
-- CREATE TABLE IF NOT EXISTS user_preferences (
--     id                    SERIAL PRIMARY KEY,
--     user_id               INTEGER NOT NULL UNIQUE REFERENCES "user" (id) ON DELETE CASCADE,
--     diet_labels           TEXT[]  NOT NULL DEFAULT '{}',
--     health_labels         TEXT[]  NOT NULL DEFAULT '{}',
--     excluded_ingredients  TEXT[]  NOT NULL DEFAULT '{}',
--     preferred_cuisines    TEXT[]  NOT NULL DEFAULT '{}',
--     disliked_cuisines     TEXT[]  NOT NULL DEFAULT '{}'
-- );
--
-- CREATE INDEX IF NOT EXISTS ix_user_preferences_user_id ON user_preferences (user_id);
