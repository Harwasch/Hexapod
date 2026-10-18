# syntax=docker/dockerfile:1
# Operator-run recipe, Linux x86_64. No checkpoints or cloud credentials in image.
# CUDA 13.2 user-space libraries come from pinned PyTorch/NVIDIA wheels.
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS build
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/worlds
COPY workers/worlds /opt/worlds
RUN python -m venv /opt/ltx25-venv && python -m venv /opt/gateway-venv \
    && /opt/ltx25-venv/bin/pip install --no-cache-dir uv==0.12.19
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export SSL_CERT_FILE=/run/secrets/build_ca PIP_CERT=/run/secrets/build_ca GIT_SSL_CAINFO=/run/secrets/build_ca; fi; \
    python ltx25/bootstrap.py --source /opt/ltx25 \
    && /opt/ltx25-venv/bin/uv --no-cache pip install --python /opt/gateway-venv/bin/python -r requirements-streaming.lock \
    && /opt/ltx25-venv/bin/uv --no-cache pip install --python /opt/ltx25-venv/bin/python --no-deps -r ltx25/requirements.lock \
    && /opt/ltx25-venv/bin/pip install --no-deps --no-build-isolation /opt/ltx25/packages/ltx-core /opt/ltx25/packages/ltx-pipelines \
    && /opt/gateway-venv/bin/pip check \
    && /opt/ltx25-venv/bin/python -c 'import torch, torchaudio, torchvision; from ltx_pipelines.distilled import DistilledPipeline; from ltx_core.text_encoders.gemma.gemma_assets import GemmaAssets' \
    && /opt/ltx25-venv/bin/pip uninstall -y uv \
    && rm -rf /opt/ltx25/.git
# Full model pip check has one intentional upstream override: torch's cudnn 9.20
# pin is superseded by 9.24.0.43 (upstream pyproject.toml explains missing symbols).

FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ENV DEBIAN_FRONTEND=noninteractive WORLD_MODEL_ID=ltx-2.5 WORLD_BIND=0.0.0.0 PORT=8789 \
    LTX25_SOURCE=/opt/ltx25 LTX25_WEIGHTS=/models/ltx-2.5 LTX25_PYTHON=/opt/ltx25-venv/bin/python \
    INFERENCE_PYTHON=/opt/ltx25-venv/bin/python LTX25_RESIDENCY=gpu \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    WANDB_MODE=disabled XDG_CACHE_HOME=/tmp/cache HF_HOME=/tmp/huggingface \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 worker
COPY --from=build /opt/ltx25-venv /opt/ltx25-venv
COPY --from=build /opt/gateway-venv /opt/gateway-venv
COPY --from=build /opt/ltx25 /opt/ltx25
COPY --from=build /opt/worlds /opt/worlds
RUN mkdir -p /models/ltx-2.5 && chmod -R a+rX /opt/ltx25 /opt/worlds
WORKDIR /opt/worlds
USER worker
EXPOSE 8789
HEALTHCHECK --interval=30s --timeout=5s CMD /opt/gateway-venv/bin/python healthcheck.py
CMD ["/opt/gateway-venv/bin/python", "gateway.py"]
