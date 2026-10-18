# syntax=docker/dockerfile:1
# Operator-only recipe. No build, model download, GPU test or deployment is run.
# The pinned PyTorch wheels supply CUDA/cuDNN; the host supplies NVIDIA drivers.
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/worlds
COPY workers/worlds /opt/worlds
RUN python -m venv /opt/helixworld-venv && python -m venv /opt/gateway-venv \
    && /opt/helixworld-venv/bin/pip install --no-cache-dir uv==0.12.19
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export SSL_CERT_FILE=/run/secrets/build_ca PIP_CERT=/run/secrets/build_ca GIT_SSL_CAINFO=/run/secrets/build_ca; fi; \
    python helixworld/bootstrap.py --source /opt/helixworld \
    && /opt/helixworld-venv/bin/uv --no-cache pip install --python /opt/gateway-venv/bin/python -r requirements-streaming.lock \
    && /opt/helixworld-venv/bin/uv --no-cache pip install --python /opt/helixworld-venv/bin/python -r helixworld/requirements.lock \
    && /opt/helixworld-venv/bin/pip check && /opt/gateway-venv/bin/pip check \
    && /opt/helixworld-venv/bin/pip uninstall -y uv \
    && rm -rf /opt/helixworld/.git

FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ENV DEBIAN_FRONTEND=noninteractive WORLD_MODEL_ID=helix-world WORLD_BIND=0.0.0.0 PORT=8789 \
    HELIXWORLD_SOURCE=/opt/helixworld HELIXWORLD_PYTHON=/opt/helixworld-venv/bin/python \
    INFERENCE_PYTHON=/opt/helixworld-venv/bin/python HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates gcc libc6-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 worker
COPY --from=build /opt/helixworld-venv /opt/helixworld-venv
COPY --from=build /opt/gateway-venv /opt/gateway-venv
COPY --from=build /opt/helixworld /opt/helixworld
COPY --from=build /opt/worlds /opt/worlds
RUN mkdir -p /opt/helixworld/models && chown worker:worker /opt/helixworld/models \
    && chmod -R a+rX /opt/helixworld /opt/worlds
WORKDIR /opt/worlds
USER worker
EXPOSE 8789
HEALTHCHECK --interval=30s --timeout=5s CMD /opt/gateway-venv/bin/python healthcheck.py
CMD ["/opt/gateway-venv/bin/python", "gateway.py"]
