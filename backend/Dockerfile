# syntax=docker/dockerfile-upstream:master

# use modified image with uv installed
# using Python 3.11
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

# Set and install project into `/app`
WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Need some dependencies that are not in slim, then removing 
# extra files to reduce container size
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install project's dependencies using `lockfile` and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Add project source code and install
# Installing separately from dependences allows optimal layer caching
COPY backend/ /app
COPY shared/ /app/shared/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the env at the front of the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Update environment variables for flask
ENV FLASK_APP app.py
ENV FLASK_ENV prod
ENV FLASK_RUN_PORT 5000
ENV FLASK_RUN_HOST 0.0.0.0

EXPOSE 5000

# Copy and permission the entrypoint script
COPY backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run migrations then start Flask
ENTRYPOINT ["/entrypoint.sh"]
