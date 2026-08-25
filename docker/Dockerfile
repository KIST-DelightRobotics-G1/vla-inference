# x86_64 GPU container — the complete VLA/replay inference environment.
# Plain ubuntu base on purpose: the torch 2.9.0+cu128 wheels vendor the whole
# CUDA 12.8 userland as nvidia-*-cu12 pip packages, so an nvidia/cuda base
# would duplicate ~3 GB — the host contributes only the driver, injected by
# nvidia-container-toolkit (`--gpus all`). No TensorRT either: gr00t's
# tensorrt dependency serves its export scripts, not the Gr00tPolicy
# inference path (TRT lives on the gearsonic side of this stack).
#
# Self-contained (same pattern as kist-gearsonic-inference): dependencies are
# cloned (pinned) and installed INSIDE the image, the repo source is baked in
# via COPY, and the import graph + test suite are verified at image-build
# time — `docker run` gives a ready runner; no host-side vendoring, no bind
# mount, no in-container setup. The only host-side pieces are the NVIDIA
# driver and the HF cache mount (gated Cosmos backbone token, see run.sh).
#
# For iterative dev you can still shadow the baked source with a bind mount
# (see run.sh comments).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# nvidia-container-toolkit injects the host driver into this plain base.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ─── build tools + system deps ────────────────────────────────────────────────
# Deliberately no compilers and no system CycloneDDS: every native piece is a
# prebuilt wheel, and the `cyclonedds` pip wheel bundles libddsc (the
# kist_msgs types are hand-mirrored IdlStructs — no idlc). ffmpeg is the one
# real system library: torchcodec dynamically links the system libav* and
# supports FFmpeg 4-7 only (ubuntu 24.04 ships 6; PyAV bundles its own).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    wget \
    nano \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ─── python 3.12 venv (uv) ────────────────────────────────────────────────────
# gr00t pins >=3.12,<3.13; uv provides the standalone interpreter.
# UV_HTTP_TIMEOUT: the pytorch-index CDN stalls on multi-hundred-MB wheels
# under the default 30s. The uv cache mounts below keep downloaded wheels
# across build retries, so a network hiccup never re-downloads the ~3 GB.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH} UV_HTTP_TIMEOUT=600
RUN uv venv /opt/venv --python 3.12

# ─── torch stack (largest layer, never changes) ───────────────────────────────
# numpy is pinned FIRST from PyPI: otherwise the pytorch index resolves the
# latest numpy for torchvision and the essentials layer downgrades it to the
# verified 1.26.4 — two downloads for one package.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26.4 && \
    uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.0 torchvision==0.24.0

# ─── gr00t inference essentials ───────────────────────────────────────────────
# The curated set the Gr00tPolicy import graph actually reaches (static trace
# of gr00t.policy.gr00t_policy + server_client), pinned to the versions in
# requirements.lock.txt — the set verified with the pinned gr00t commit.
# Training-only packages (deepspeed, wandb, datasets, onnx, tensorrt,
# gymnasium, matplotlib) are deliberately absent; the bake-time import check
# below fails the build if the curation ever goes stale.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    flash-attn@https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    transformers==4.57.3 tokenizers==0.22.2 diffusers==0.35.1 \
    accelerate==1.14.0 peft==0.17.1 safetensors==0.8.0 \
    huggingface-hub==0.36.2 einops==0.8.1 \
    albumentations==1.4.18 pillow==12.3.0 opencv-python-headless==4.11.0.86 \
    numpy==1.26.4 scipy==1.15.3 pandas==2.2.3 torchcodec==0.8.0 \
    dm-tree==0.1.10 tqdm==4.70.0 tyro==0.9.17 omegaconf==2.3.0 pyyaml==6.0.3 \
    msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1

# ─── vendored dep: gr00t (cloned + pinned + --no-deps) ────────────────────────
# The commit is a checkpoint coupling read from gr00t_version.py — the same
# single authority the runtime check and install_gr00t.sh use. Only that file
# is copied here so source edits never invalidate this layer. --no-deps: the
# curated set above replaces gr00t's own pins.
ARG GR00T_REMOTE=https://github.com/foodbanana/Isaac-GR00T.git
COPY src/vla/gr00t_version.py /tmp/gr00t_version.py
RUN COMMIT=$(python -c "import runpy; print(runpy.run_path('/tmp/gr00t_version.py')['EXPECTED_GR00T_COMMIT'])") && \
    git clone --quiet "${GR00T_REMOTE}" /opt/Isaac-GR00T && \
    git -C /opt/Isaac-GR00T checkout --quiet --detach "${COMMIT}" && \
    uv pip install --no-deps /opt/Isaac-GR00T && \
    rm -rf /opt/Isaac-GR00T/.git

# ─── vendored dep: unitree_sdk2py (IDL types for the robot-state source) ──────
# --no-deps: it pins an old cyclonedds and pulls opencv-python; its IDL types
# work with our set (same rationale as the README install step).
ARG UNITREE_SDK2PY_REF=master
RUN git clone --quiet --depth 1 --branch "${UNITREE_SDK2PY_REF}" \
        https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python && \
    uv pip install --no-deps /opt/unitree_sdk2_python

# ─── models: GEAR-SONIC encoder ONNX ──────────────────────────────────────────
# For `--tokens-from joints` (offline joint re-encoding) at its fixed path
# models/ — same convention as kist-gearsonic-inference, whose decoder this
# encoder pairs with. Must be swapped together with that decoder checkpoint.
WORKDIR /workspace/kist-vla-inference
RUN wget -q -P models \
    https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/model_encoder.onnx

# ─── source + install + bake-time verification ────────────────────────────────
# Bake the repo in (build context = repo root minus .venv/.git/models via
# .dockerignore — the wget layer above provides models/). The dep layers stay
# cached; only this re-runs on a source change. The import check proves the
# curated set covers the whole inference graph; the suite runs GPU-less.
COPY . .
RUN uv pip install -e ".[dds,parquet,encode,dev]" && \
    python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy; \
               from gr00t.policy.server_client import PolicyClient; \
               from vla.runner import VLARunner; import replay.cli; \
               print('inference import graph OK')" && \
    python -m pytest tests/ -q

# HF_HOME is a mount (run.sh): the Cosmos-Reason2-2B backbone is a gated HF
# repo fetched at first policy load — token and cache stay on the host.
ENV HF_HOME=/data/huggingface
CMD ["/bin/bash"]
