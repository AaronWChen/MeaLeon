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

# Export DB password from secret file so SQLAlchemy can connect
if [ -f /run/secrets/DB_PASSWORD ]; then
    export POSTGRES_PASSWORD=$(cat /run/secrets/DB_PASSWORD)
fi

export DATABASE_URL="postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}/${PG_DATABASE}"

echo "DATABASE_URL host: ${POSTGRES_HOST}"  # log for debugging

echo "Waiting for database..."
MAX_RETRIES=30
COUNT=0
until flask db upgrade 2>&1; do
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Max retries reached, exiting"
        exit 1
    fi

    echo "  db not ready yet, retrying in 2s... ($COUNT / $MAX_RETRIES)"
    sleep 2
done

echo "Migrations applied. Starting Flask..."
exec flask run --host=0.0.0.0 --port=5000
