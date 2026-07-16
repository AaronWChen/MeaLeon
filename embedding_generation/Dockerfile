# syntax=docker/dockerfile-upstream:master

# use modified image with uv installed
# using Python 3.11
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

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
    --mount=type=bind,source=embedding_generation/uv.lock,target=uv.lock \
    --mount=type=bind,source=embedding_generation/pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Add project source code and install
# Installing separately from dependences allows optimal layer caching
COPY embedding_generation/ /app/
COPY shared/ /app/shared/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the env at the front of the PATH
ENV PATH="/app/.venv/bin:$PATH"

# # Execute python to pre-download Stanza resources
# RUN python -c "import stanza; stanza.download('en')"

# Execute Python script to download models
COPY embedding_generation/download_models.py .
RUN python download_models.py

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]