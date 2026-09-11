# syntax=docker/dockerfile:1.7

ARG VERSION=dev

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install uv for fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies (no dev)
RUN uv sync --frozen --no-dev --no-install-project

# ---- Runtime stage ----
FROM python:3.12-slim

ARG VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PGRAPH_BACKEND=sqlite \
    PGRAPH_HOME=/data \
    METHODOS_VERSION=${VERSION}

# Create non-root user
RUN groupadd --system --gid 1000 methodos \
    && useradd --system --uid 1000 --gid methodos --create-home methodos \
    && mkdir -p /data \
    && chown -R methodos:methodos /data

# Copy virtual environment from builder
COPY --from=builder --chown=methodos:methodos /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source
COPY --chown=methodos:methodos methodos /app/methodos

WORKDIR /app
USER methodos

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" \
    || exit 1

# `--no-server-header` suppresses uvicorn's default `Server: uvicorn`
# value; the ServerHeaderMiddleware in methodos.service sets a single
# `Server: methodos` at the ASGI layer so the header survives both the
# TestClient (in-process) and uvicorn (production) paths.
CMD ["uvicorn", "methodos.service:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
