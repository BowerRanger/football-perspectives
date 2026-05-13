# syntax=docker/dockerfile:1.7
# AWS Batch handler image for the hmr_world stage.
#
# Two stages:
#   1) builder  — full Python + CUDA + build tools to install the heavy
#                 ML stack into /opt/venv.
#   2) runtime  — CUDA runtime + Python + FFmpeg + the venv. Copies the
#                 GVHMR submodule, shims, src/, and pre-downloaded
#                 checkpoints. Entrypoint: ``python -m src.cloud.handler``.
#
# Build:
#   docker build --platform linux/amd64 -t football-perspectives-hmr-world:dev .
#
# Run locally with --gpus all if you have an NVIDIA host; otherwise
# ``python -m src.cloud.handler`` will detect device=auto → cpu and run
# (slowly) on CPU. Same code path either way.

ARG PYTHON_VERSION=3.11

# ---------- builder ----------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        python3-pip python3-distutils \
        git build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch + torchvision built for CUDA 12.1 explicitly first so
# the resolver doesn't pull a CPU wheel from PyPI under the project's
# torch==2.1.2 pin.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2

# Install the project (hmr extras include smplx, chumpy, hydra, etc.)
# Use the pyproject.toml from the build context — we don't need src/
# yet because pip install -e expects it but we're not really linking
# from this stage; just collecting deps.
WORKDIR /build
COPY pyproject.toml /build/pyproject.toml
# Minimal src/__init__.py so setuptools finds the package layout.
RUN mkdir -p /build/src && echo "" > /build/src/__init__.py
RUN pip install --no-cache-dir -e ".[hmr]"
RUN pip install --no-cache-dir "boto3>=1.34"

# ---------- runtime ----------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python3-distutils \
        ffmpeg libgl1 libglib2.0-0 ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Bring in the venv built above.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Match the local layout — gvhmr_estimator.py resolves the checkpoint
# path relative to repo root: Path(__file__).resolve().parents[2].
# At /app/src/utils/gvhmr_estimator.py, parents[2] is /app — so
# third_party/ must live at /app/third_party/.
ENV PYTHONPATH="/app:/app/third_party/gvhmr_shims:/app/third_party/gvhmr"
WORKDIR /app

# Source (no tests, no docs).
COPY src/ /app/src/
COPY config/ /app/config/
COPY recon.py /app/recon.py
COPY pyproject.toml /app/pyproject.toml

# GVHMR submodule — only the actually-imported subtree.
COPY third_party/gvhmr/hmr4d/ /app/third_party/gvhmr/hmr4d/
COPY third_party/gvhmr/inputs/checkpoints/gvhmr/ /app/third_party/gvhmr/inputs/checkpoints/gvhmr/
COPY third_party/gvhmr/inputs/checkpoints/vitpose/ /app/third_party/gvhmr/inputs/checkpoints/vitpose/
COPY third_party/gvhmr/inputs/checkpoints/hmr2/ /app/third_party/gvhmr/inputs/checkpoints/hmr2/
# body_models: only the SMPLX neutral pickle + npz. The full
# models_smplx_v1_1.zip (~870 MB) is the source archive and is not
# loaded at runtime (verified via grep across hmr4d/).
COPY third_party/gvhmr/inputs/checkpoints/body_models/smpl/ /app/third_party/gvhmr/inputs/checkpoints/body_models/smpl/
COPY third_party/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.pkl /app/third_party/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.pkl
COPY third_party/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz /app/third_party/gvhmr/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
COPY third_party/gvhmr/inputs/checkpoints/body_models/smplx/models/ /app/third_party/gvhmr/inputs/checkpoints/body_models/smplx/models/

# Compatibility shims (pytorch3d stubs, cuda redirects).
COPY third_party/gvhmr_shims/ /app/third_party/gvhmr_shims/

# Sanity: torch can see CUDA. (Doesn't fail the build if no GPU; just
# logs.) The container relies on the host providing --gpus all.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

ENTRYPOINT ["python", "-m", "src.cloud.handler"]
