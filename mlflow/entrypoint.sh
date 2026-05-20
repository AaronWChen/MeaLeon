#!/bin/sh
DB_PASSWORD=$(cat /run/secrets/db_password)
exec mlflow server \
  --backend-store-uri "postgresql+psycopg://docker_user:${DB_PASSWORD}@db:5432/mealeon" \
  --default-artifact-root /mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5001