# recommend_service/Dockerfile_claude
#
# Lightweight FastAPI service — no ML dependencies.
# All model inference happens in embedding_generation service.
# This service only needs HTTP client + Pydantic + FastAPI.

FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

WORKDIR /app

ENV UV_LINK_MODE=copy

# # No C extensions needed — no apt-get build-essential required here.
# # If you add a dependency that needs compilation, add it like the backend Dockerfile.
# COPY requirements.txt .
# RUN uv pip install --system -r requirements.txt

# Install project's dependencies using `lockfile` and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Add project source code and install
# Installing separately from dependences allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the env at the front of the PATH
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
