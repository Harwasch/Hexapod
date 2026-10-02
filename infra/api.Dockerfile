# Production image for the catalog API (apps/api) and for the worker that runs beside it.
# Build from the repo root:  docker build -f infra/api.Dockerfile -t twin-api .
#
# One image, two commands. `CMD` below serves the API; `python -m app.worker` is the same
# image run as the worker (docs/DEPLOYMENT.md). That is why `tools/pipeline` and
# `tools/captures` are in here: the worker imports the pipeline as a library
# (app/worker/pipeline_bridge.py) and runs its stages in *this* venv, in a child process
# started with `sys.executable`. Until B1a only `apps/api/` was copied, which was a defect
# rather than a decision -- `pipeline_bridge` calls `ensure_importable()` at module import,
# so `python -m app.worker` could not start at all, and the API answered 404 on
# `/api/v1/recipes` in the one deployment path the docs describe.
#
# The two `tools/` directories are copied whole and kept siblings on purpose:
# `captures_bridge.py` finds the packer at `../captures` relative to itself, and
# `stages.py` reads `tools/pipeline/pyproject.toml` (the manifest's `pipeline` version)
# and hashes `tools/captures/splat_tiles.py` at run time. Modules alone would not do.
# Neither project is installable (`package = false`), so there is nothing to `uv sync`
# here; their runtime dependencies are declared in apps/api/pyproject.toml instead, which
# is also where the note about `imageio-ffmpeg`'s deliberate absence lives.
#
# `--extra modal` is here because the worker is what dispatches to Modal. `ModalAdapter`
# is built by app/worker/cloud.py and `spawn`s from *this* process, with a token of its
# own (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, which the Modal SDK reads straight from the
# environment), so the client has to be in the image the worker runs. B1b left it out on
# the grounds that an image which never dispatches need not carry it; that was right
# about the library and wrong about which machine calls Modal. It is an extra rather than
# a dependency so a build that wants neither can drop the flag -- and
# `Worker.from_settings` refuses at start-up when the providers name Modal and the client
# is absent, rather than failing on the first GPU job.
#
# `data/tiles` is deliberately *not* in the image. Until A9 that was a live defect rather
# than a decision: the API's `/api/v1/tiles` StaticFiles mount reads `data/tiles`, which
# does not exist here, so the mount was silently absent and every capture 404'd in the one
# deployment path docs/DEPLOYMENT.md describes. A9 did not fix that by copying 104 MB of
# tiles into the image -- it moved them to object storage, so a deployment's capture URLs
# point at the bucket and this image never needs them. See app/seed/captures.tiles_base_url.
#
# Ubuntu 24.04 rather than python:3.12-slim-bookworm, for one package: **COLMAP**. Lane 2's
# `pose` stage runs `colmap` on this machine, and everything `tools/pipeline/sfm.py` says
# about it -- `model_aligner`'s `--transform_path` being right while `--output_path` is
# not, the eight numbers it writes -- was measured on 3.9.1, which is what Ubuntu 24.04
# ships (and what CI's `ubuntu-latest` installs). Debian bookworm ships 3.8 and trixie
# 3.10 (sources.debian.org, read 2026-09-23); neither is the version those findings are
# about. Ubuntu 24.04's own Python is 3.12, so the venv is built against /usr/bin/python3.12
# in the builder and runs against the same path here. The closure is 176 packages, about
# 370 MB installed (measured with `apt-cache depends --recurse` on an Ubuntu 24.04 host),
# and it lands on the API machines too because Fly runs one image for both process groups.
FROM ghcr.io/astral-sh/uv:0.8.17 AS uv

FROM ubuntu:24.04 AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_DOWNLOADS=never
COPY apps/api/pyproject.toml apps/api/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra modal
COPY apps/api/ ./
RUN uv sync --frozen --no-dev --extra modal

FROM ubuntu:24.04 AS runtime
# `colmap` for the pose stage; ffmpeg is not here because the pipeline uses the binary in
# the imageio-ffmpeg wheel, never a system one (tools/pipeline/video.py says why).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 ca-certificates colmap \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# uid 1000 is the stock image's `ubuntu` user; the worker's volume was written as uid 1000
# by the previous image's `api`, so the uid is kept and the stock user goes.
RUN userdel --remove ubuntu 2>/dev/null || true; useradd --create-home --uid 1000 api
COPY --from=builder --chown=api:api /app /app
COPY --chown=api:api tools/pipeline /app/tools/pipeline
COPY --chown=api:api tools/captures /app/tools/captures
# PIPELINE_DIR is also what app/config.REPO_ROOT would derive here (it is `/app` in this
# image, so the default is already `/app/tools/pipeline`). Set anyway: it is the one line
# that says out loud where the layout above puts the pipeline, and it is what the worker
# hands its child process. QT_QPA_PLATFORM because COLMAP links Qt: its CLI commands used
# here open no window, and `offscreen` makes sure one that tried would not need a display.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PIPELINE_DIR=/app/tools/pipeline \
    QT_QPA_PLATFORM=offscreen
USER api
EXPOSE 8000
# Migrations run on start so a fresh database is usable immediately.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
