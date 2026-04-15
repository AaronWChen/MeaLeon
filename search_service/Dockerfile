# Search service 

FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

WORKDIR /app

ENV UV_LINK_MODE=copy

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8001

EXPOSE 8001

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
