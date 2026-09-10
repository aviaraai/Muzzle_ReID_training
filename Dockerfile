# Godhaar muzzle re-id fine-tuning — built for a blank GPU host (nothing
# pre-installed but the NVIDIA driver + Docker + nvidia-container-toolkit,
# which "docker can see the GPU" already implies).
#
# Deliberately NOT based on an nvidia/cuda image: torch/torchvision are
# installed as the cu128 wheels (pyproject.toml's [tool.uv.sources] index),
# which bundle their own CUDA runtime libraries (libcudart, libcublas, ...).
# All that's actually needed from the HOST is the NVIDIA driver, exposed
# into the container by `docker run --gpus all` (or compose's
# `deploy.resources.reservations.devices`, see docker-compose.yaml) — the
# container itself never needs the CUDA toolkit or nvidia/cuda base image.
# Verified locally: `uv sync` + this exact dependency set produces
# torch==2.11.0+cu128 that reports torch.cuda.is_available()==True.
FROM python:3.12-slim-bookworm

# libgl1/libglib2.0-0: opencv-python (a faiss/pandas/pillow transitive
# dependency chain can pull it in) needs libGL at import time even though
# nothing here does any actual windowing/display work.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv itself — pinned install script, not `pip install uv` (keeps the base
# image from needing a resolver just to bootstrap the resolver).
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Dependency files first so `uv sync` is cached across rebuilds that only
# change training code, not dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Now the actual code + dataset.
COPY scripts ./scripts
COPY main.py ./
COPY data ./data

# checkpoints/ and results/ are written at runtime, not baked into the
# image — mount them as a volume (see docker-compose.yaml) so a training
# run's output survives a container restart/removal.
RUN mkdir -p checkpoints results

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/scripts

ENTRYPOINT ["uv", "run", "--frozen", "python", "train_dinov2_arcface.py"]
CMD ["--gpu-preset", "auto"]
