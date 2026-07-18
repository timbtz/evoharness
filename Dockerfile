FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY core core
COPY axes axes
COPY tasks tasks
COPY api api
COPY web web
COPY experiments experiments
COPY prices.json ./
RUN uv run python tasks/tsp/fetch.py
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
