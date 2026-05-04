
# use modified image with uv installed
# using Python 3.11
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

# Set and install project into `/app`
WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

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


ENV PYTHONUNBUFFERED=1
ENV PORT=8001

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
