# Production image for the catalog API (apps/api).
# Build from the repo root:  docker build -f infra/api.Dockerfile -t twin-api .
#
# `apps/api/` is deliberately the whole build context that ends up in the image, and
# `data/tiles` is deliberately not in it. Until A9 that was a live defect rather than a
# decision: the API's `/api/v1/tiles` StaticFiles mount reads `data/tiles`, which does not
# exist here, so the mount was silently absent and every capture 404'd in the one
# deployment path docs/DEPLOYMENT.md describes. A9 did not fix that by copying 104 MB of
# tiles into the image -- it moved them to object storage, so a deployment's capture URLs
# point at the bucket and this image never needs them. See app/seed/captures.tiles_base_url.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY apps/api/pyproject.toml apps/api/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY apps/api/ ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN useradd --create-home --uid 1000 api
COPY --from=builder --chown=api:api /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER api
EXPOSE 8000
# Migrations run on start so a fresh database is usable immediately.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
