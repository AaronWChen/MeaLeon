#!/bin/sh
# backend/entrypoint.sh
#
# Runs before Flask starts. Handles two things:
#
# 1. Waits for Postgres to be genuinely ready to accept connections.
#    The compose healthcheck covers this via depends_on/service_healthy,
#    but a small retry loop here is cheap insurance against race conditions
#    on slower machines or cold Docker daemon starts.
#
# 2. Runs `flask db upgrade` to apply any pending Alembic migrations.
#    On first boot this creates all tables from scratch using your six
#    existing migration files. On subsequent boots it's a no-op if the
#    schema is already current.
#
# Usage: set as ENTRYPOINT in the backend Dockerfile.

set -e

# Local compose: build DATABASE_URL from the Docker secret.
   # DigitalOcean: DATABASE_URL is injected and no secret file exists.
   if [ -f /run/secrets/db_password ]; then
       export POSTGRES_PASSWORD=$(cat /run/secrets/db_password)
       export DATABASE_URL="postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}/${PG_DATABASE}"
   fi

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    MAX_RETRIES=30
    COUNT=0
    until flask db upgrade 2>&1; do
        COUNT=$((COUNT + 1))
        [ $COUNT -ge $MAX_RETRIES ] && { echo "Max retries reached"; exit 1; }
           echo "  db not ready yet, retrying in 2s... ($COUNT/$MAX_RETRIES)"
           sleep 2
    done
fi

if [ "${USE_GUNICORN:-false}" = "true" ]; then
    exec gunicorn -b 0.0.0.0:5000 -w 2 "app:create_app()"
fi

exec flask run --host=0.0.0.0 --port=5000
