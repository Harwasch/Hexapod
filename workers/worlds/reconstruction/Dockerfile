# syntax=docker/dockerfile:1
# Build explicitly from the repository root. No weights or GPU jobs run at build time.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DO_NOT_TRACK=1 WANDB_MODE=disabled
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
WORKDIR /opt/reconstruction
COPY workers/worlds/reconstruction /opt/reconstruction
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca GIT_SSL_CAINFO=/run/secrets/build_ca; fi; \
    /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock \
    && /opt/venv/bin/pip check \
    && /opt/venv/bin/python bootstrap.py --source /opt/hunyuanworld-mirror \
    && git config --system --add safe.directory /opt/hunyuanworld-mirror \
    && useradd --create-home --uid 10001 worker \
    && mkdir -p /models/hunyuanworld-mirror /data/reconstruction \
    && chown -R worker:worker /data /models \
    && chmod -R a+rX /opt/reconstruction /opt/hunyuanworld-mirror
ENV WORLD_BIND=0.0.0.0 PORT=8790 \
    WORLD_RECONSTRUCTION_SOURCE=/opt/hunyuanworld-mirror \
    WORLD_RECONSTRUCTION_WEIGHTS=/models/hunyuanworld-mirror \
    WORLD_RECONSTRUCTION_DATA_DIR=/data/reconstruction \
    XDG_CACHE_HOME=/tmp/cache HF_HOME=/tmp/huggingface
USER worker
EXPOSE 8790
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD ["/opt/venv/bin/python", "healthcheck.py"]
CMD ["/opt/venv/bin/python", "server.py"]
