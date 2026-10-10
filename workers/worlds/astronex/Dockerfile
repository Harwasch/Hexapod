# syntax=docker/dockerfile:1
# No image is built or deployed by the application. Explicit operator action only.
# GPU drivers are supplied by the host (RunPod GPU Pod or local Docker --gpus).
# PyTorch's hash-locked wheels include CUDA/cuDNN; avoid duplicating the full
# runtime image. The host provides the NVIDIA driver through the container runtime.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 AS worker
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /opt/worlds
# Upstream torch.compile/flex_attention uses Triton driver helpers on first GPU
# execution. They require a host compiler even though PyTorch ships CUDA libraries.
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY workers/worlds /opt/worlds
# One final large layer avoids duplicating the CUDA environment on VFS builders.
# Separate interpreters isolate the video transport and research dependencies.
# The optional CA mount supports managed proxies without retaining their trust
# material. Parallel downloads remain hash-verified and use bounded concurrency.
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca SSL_CERT_FILE=/run/secrets/build_ca GIT_SSL_CAINFO=/run/secrets/build_ca; fi; \
    export UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_INSTALLS=2; \
    python astronex/bootstrap.py --source /opt/astronex \
    && pip install --no-cache-dir uv==0.12.19 \
    && python3 -m venv /opt/gateway-venv \
    && uv --no-cache pip install --python /opt/gateway-venv/bin/python -r requirements-streaming.lock \
    && uv --no-cache pip install --python /opt/venv/bin/python --require-hashes -r astronex/requirements.lock \
    && pip check && /opt/gateway-venv/bin/pip check \
    && pip uninstall -y uv \
    && chmod -R a+rX /opt/worlds /opt/astronex \
    && useradd --create-home --uid 10001 worker \
    && mkdir -p /models/astronex && chown worker:worker /models/astronex
USER worker
ENV ASTRONEX_SOURCE=/opt/astronex ASTRONEX_WEIGHTS=/models/astronex WORLD_BIND=0.0.0.0 PORT=8789 \
    INFERENCE_PYTHON=/opt/venv/bin/python NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
EXPOSE 8789
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD /opt/gateway-venv/bin/python healthcheck.py
CMD ["/opt/gateway-venv/bin/python", "gateway.py"]
