#!/bin/sh
# --backend-store-uri "postgresql+psycopg://docker_user:${DB_PASSWORD}@db:5432/mealeon" \

DB_PASSWORD=$(cat /run/secrets/db_password)
exec mlflow server \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --artifacts-destination /mlflow/artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5001 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"