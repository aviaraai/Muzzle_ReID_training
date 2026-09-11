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
#
# build-essential: the a6000 GPU preset turns on torch.compile, whose
# Triton/Inductor backend shells out to a C compiler the FIRST time the
# model is actually called (not when torch.compile() wraps it) -- without
# this, that first forward pass crashes with "Failed to find C compiler"
# on this slim base image, and the try/except around the torch.compile()
# call in train_dinov2_arcface.py can't catch it since the real
# compilation is lazy and happens later, outside that try block.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv itself — pinned install script, not `pip install uv` (keeps the base
# image from needing a resolver just to bootstrap the resolver).
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Dependency files first so `uv sync` is cached across rebuilds that only
# change training code, not dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Code only. The dataset is NOT baked in — it is bind-mounted at run time
# (see docker-compose.yaml). Copying it made sense when `data/` was the
# 120MB Uttarakhand set; the 300-identity pretraining corpus is 17GB of
# 4000x6000 JPEGs, and baking that into the image would mean a 17GB layer
# rebuilt on every code change, pushed and pulled in full each time.
#
# A mount also keeps the image identical across the two training stages —
# pretrain on the 300-corpus, fine-tune on the Uttarakhand split — with only
# the mounted path differing, rather than two divergent images.
COPY scripts ./scripts
COPY main.py ./

# Mount point for the dataset. Left empty in the image; docker-compose binds
# a host directory over it. Created here so a run with no mount fails with a
# clear "no such file" on the manifest rather than a confusing import error.
RUN mkdir -p /workspace/data

# checkpoints/ and results/ are written at runtime, not baked into the
# image — mount them as a volume (see docker-compose.yaml) so a training
# run's output survives a container restart/removal.
RUN mkdir -p checkpoints results

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/scripts

ENTRYPOINT ["uv", "run", "--frozen", "python", "train_dinov2_arcface.py"]
CMD ["--gpu-preset", "auto"]
