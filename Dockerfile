FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends util-linux gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src src
COPY configs configs
RUN uv sync --frozen --no-dev
ENV EVOHARNESS_HOME=/app/.local
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "evoharness", "serve", "--host", "0.0.0.0", "--port", "8000"]
