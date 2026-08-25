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
#   (CycloneDDS is still built from source — see step 3. The PyPI `cyclonedds`
#    wheel is 11.x and its readers crash the 0.10.2 C++ stack.)
#   - apt ffmpeg                          → torchcodec is gone; `av` bundles its own FFmpeg
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── 1. system packages ────────────────────────────────────────────────────────
# git: checkouts below. cmake/build-essential: CycloneDDS C library (step 3)
# and the cyclonedds python extension (step 4; no cp312 wheel for 0.10.x). No python3 from apt —
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

# ── 3. CycloneDDS C library ───────────────────────────────────────────────────
# MUST stay 0.10.x: gearsonic / ext-sensor-io are 0.10.2 and a 0.11 python
# reader on the domain segfaults them (XTypes discovery format). PyPI's default
# is 11.x and 0.10.x has no cp312 wheel, so the python binding is compiled from
# source in step 4 — against this library, found via CYCLONEDDS_HOME.
#
# The version is read from the lock so the C library can never diverge from
# the binding step 4 builds: bump `cyclonedds==` in pyproject, regenerate the
# lock, rebuild — nothing here to edit. Same recipe as gearsonic's README.
ENV CYCLONEDDS_HOME=/opt/cyclonedds \
    LD_LIBRARY_PATH=/opt/cyclonedds/lib:${LD_LIBRARY_PATH}
WORKDIR /workspace
COPY requirements.lock.txt .
RUN V="$(grep -oP '^cyclonedds==\K.*' requirements.lock.txt)" && test -n "$V" \
    && git clone -q --depth 1 -b "$V" https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cdds \
    && cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
             -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
             -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF >/dev/null \
    && cmake --build /tmp/cdds/build --parallel "$(nproc)" >/dev/null \
    && cmake --install /tmp/cdds/build >/dev/null \
    && rm -rf /tmp/cdds

# ── 4. third-party Python deps (cached layer — changes only with the lock) ────
# requirements.lock.txt is the exact set verified against checkpoint-18000
# (torch/torchvision cu128 wheels via its --extra-index-url line; flash-attn
# from its wheel URL; cyclonedds compiled here against step 3; av).
# The uv download cache lives in a BuildKit cache mount, not in the layer:
# without this the image carried an 8 GB duplicate of every wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-strategy unsafe-best-match -r requirements.lock.txt

# ── 5. unitree_sdk2py (DDS IDL types for the state source) ────────────────────
# Upstream, pinned. (GR00T-WholeBodyControl bundles an older snapshot whose
# ChannelFactoryInitialize is not idempotent — a second call fails with
# "create domain error" — while upstream added a thread-safe singleton guard.
# The LowState_/HandState_ wire types we read are identical in both.)
# --no-deps: it pulls opencv-python (we have -headless); its cyclonedds==0.10.2
# pin is exactly what step 4 installed. Editable, so the checkout stays in /opt.
ARG UNITREE_SDK2PY_COMMIT=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone -q https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python \
    && git -C /opt/unitree_sdk2_python checkout -q "$UNITREE_SDK2PY_COMMIT" \
    && uv pip install --no-deps -e /opt/unitree_sdk2_python

# ── 6. this repository (kist_vla + vendored thirdparty/gr00t) ─────────────────
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps -e .

# Fail the build here, not at first run, if anything is missing. No GPU at
# build time, so this is imports only; the GPU smoke test is run.sh's job.
RUN python -c "\
import torch, cyclonedds, av, unitree_sdk2py.core.channel, \
       thirdparty.gr00t.policy.gr00t_policy, thirdparty.gr00t.policy.server_client, \
       kist_vla.runner, kist_vla.io.dds, kist_vla.io.dds_state, kist_vla.io.dds_camera; \
import importlib.metadata as m, os; v = m.version('cyclonedds'); assert v.startswith('0.10.'), v; \
assert os.path.exists(f'/opt/cyclonedds/lib/libddsc.so.{v}'), f'C library version != binding {v}'; \
print('torch', torch.__version__, '| cyclonedds', v, '(binding == libddsc)', '| av', av.__version__, '| thirdparty.gr00t ok')"

# ── runtime ───────────────────────────────────────────────────────────────────
# The VLM backbone nvidia/Cosmos-Reason2-2B is a gated HF repo: mount a cache
# that already holds it plus a token at $HF_HOME/token (run.sh does this).
ENV HF_HOME=/hf_cache \
    HF_HUB_DISABLE_XET=1 \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

CMD ["/bin/bash"]
