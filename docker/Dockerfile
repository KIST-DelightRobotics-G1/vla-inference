# kist-vla-inference — GR00T N1.7 VLA inference container (x86_64 GPU box)
#
#   build : ./docker/build.sh          (docker build -f docker/Dockerfile .  — context is the repo root)
#   run   : ./docker/run.sh            (--gpus all --network host, mounts checkpoint + HF cache)
#
# Base image choice (see the Notion "Docker 만들기" notes): torch 2.9.0+cu128
# and the flash-attn cu12/torch2.9/cp312 wheel → CUDA 12.8 + cuDNN; the
# cxx11abiTRUE wheel wants GCC ≥ 5 → Ubuntu 22.04 (GCC 11); uv manages Python
# 3.12 itself, so the base's own Python is irrelevant. `devel` (nvcc) is not
# needed today — every CUDA package arrives as a prebuilt wheel — but is kept
# for parity with gearsonic and for a future Jetson port that builds from source.
#
# What is NOT here any more, and why:
#   - git clone Isaac-GR00T / GR00T_SRC   → the inference path is vendored in thirdparty/gr00t
#   (CycloneDDS 0.10.2 is still built from source — see step 3. The PyPI
#    `cyclonedds` wheel is 11.x and its readers crash the 0.10.2 C++ stack.)
#   - apt ffmpeg                          → torchcodec is gone; `av` bundles its own FFmpeg
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── 1. system packages ────────────────────────────────────────────────────────
# git: checkouts below. cmake/build-essential: CycloneDDS C library and the
# cyclonedds python extension (no cp312 wheel for 0.10.2). No python3 from apt —
# uv brings 3.12 with headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# The CUDA image ships forward-compatibility driver libs (/usr/local/cuda/compat,
# a 570.x libcuda) for data-center GPUs running older drivers. On GeForce they
# are not supported and make every CUDA call fail with error 804 ("forward
# compatibility was attempted on non supported HW") — seen on this box (RTX
# 3090, driver 550). Removing them lets the cu128 wheels fall back to CUDA
# minor-version compatibility with the host driver, exactly as the host venv does.
RUN rm -rf /usr/local/cuda/compat && ldconfig

# ── 2. uv + Python 3.12 venv ──────────────────────────────────────────────────
# Same uv the host was verified with. The venv lives outside /workspace so a
# bind-mounted source tree (run.sh dev mode) never shadows it.
COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --python 3.12 "$VIRTUAL_ENV"

# ── 3. CycloneDDS 0.10.2 — C library + python binding ─────────────────────────
# MUST stay 0.10.x: gearsonic / ext-sensor-io are 0.10.2 and a 0.11 python
# reader on the domain segfaults them (XTypes discovery format). PyPI's default
# is 11.x and has no cp312 wheel for 0.10.2, so build both here — same script
# the host uses. CYCLONEDDS_HOME is where `pip install cyclonedds==0.10.2`
# (pulled in by the lock below) finds libddsc to link against.
ENV CYCLONEDDS_HOME=/opt/cyclonedds \
    LD_LIBRARY_PATH=/opt/cyclonedds/lib:${LD_LIBRARY_PATH}
COPY scripts/install_cyclonedds.sh /tmp/install_cyclonedds.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    PREFIX=$CYCLONEDDS_HOME SUDO= PYTHON=/opt/venv/bin/python bash /tmp/install_cyclonedds.sh \
    && rm /tmp/install_cyclonedds.sh

# ── 4. third-party Python deps (cached layer — changes only with the lock) ────
# requirements.lock.txt is the exact set verified against checkpoint-18000
# (torch/torchvision cu128 wheels via its --extra-index-url line; flash-attn
# from its wheel URL; cyclonedds==0.10.2 already satisfied by step 3; av).
WORKDIR /workspace
COPY requirements.lock.txt .
# The uv download cache lives in a BuildKit cache mount, not in the layer:
# without this the image carried an 8 GB duplicate of every wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-strategy unsafe-best-match -r requirements.lock.txt

# ── 5. unitree_sdk2py (DDS IDL types for the state source) ────────────────────
# Same source the host uses: the copy bundled in GR00T-WholeBodyControl, pinned.
# --no-deps: it pulls opencv-python (we have -headless); its cyclonedds==0.10.2
# pin is exactly what step 3 installed.
# Editable, so the checkout must stay in the image (/opt).
ARG WBC_COMMIT=0a87181c9106d0e49293400714b157676e0ec664
RUN --mount=type=cache,target=/root/.cache/uv \
    git init -q /opt/GR00T-WholeBodyControl \
    && git -C /opt/GR00T-WholeBodyControl remote add origin https://github.com/NVlabs/GR00T-WholeBodyControl.git \
    && git -C /opt/GR00T-WholeBodyControl config core.sparseCheckout true \
    && echo "external_dependencies/unitree_sdk2_python/" > /opt/GR00T-WholeBodyControl/.git/info/sparse-checkout \
    && git -C /opt/GR00T-WholeBodyControl fetch -q --depth 1 --filter=blob:none origin "$WBC_COMMIT" \
    && GIT_LFS_SKIP_SMUDGE=1 git -C /opt/GR00T-WholeBodyControl checkout -q FETCH_HEAD \
    && uv pip install --no-deps -e /opt/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python

# ── 6. this repository (kist_vla + vendored thirdparty/gr00t) ─────────────────
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps -e .

# Fail the build here, not at first run, if anything is missing. No GPU at
# build time, so this is imports only; the GPU smoke test is run.sh's job.
RUN python -c "\
import torch, cyclonedds, av, unitree_sdk2py.core.channel, \
       thirdparty.gr00t.policy.gr00t_policy, thirdparty.gr00t.policy.server_client, \
       kist_vla.runner, kist_vla.io.dds, kist_vla.io.dds_state, kist_vla.io.dds_camera; \
import importlib.metadata as m; v = m.version('cyclonedds'); assert v.startswith('0.10.'), v; \
print('torch', torch.__version__, '| cyclonedds', v, '| av', av.__version__, '| thirdparty.gr00t ok')"

# ── runtime ───────────────────────────────────────────────────────────────────
# The VLM backbone nvidia/Cosmos-Reason2-2B is a gated HF repo: mount a cache
# that already holds it plus a token at $HF_HOME/token (run.sh does this).
ENV HF_HOME=/hf_cache \
    HF_HUB_DISABLE_XET=1 \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

CMD ["/bin/bash"]
