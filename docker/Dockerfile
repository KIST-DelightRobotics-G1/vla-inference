# x86_64 CPU container — the replay-only environment.
# Publishes recorded sessions/episodes as kist_msgs::LatentActionStep at 50 Hz
# (src/replay/): no policy, no torch, no GPU — the SONIC encoder for
# `--tokens-from joints` runs on onnxruntime's CPU provider. The VLA inference
# environment (torch cu128 + pinned gr00t) lived in this file before
# 2026-08-26; recover it from git history when the vla/ runtime returns.
#
# Self-contained (same pattern as kist-gearsonic-inference): dependencies are
# installed INSIDE the image, the repo source is baked in via COPY, and the
# import graph + replay test suite are verified at image-build time —
# `docker run` gives a ready publisher; no bind mount, no in-container setup.
#
# For iterative dev you can still shadow the baked source with a bind mount
# (see run.sh comments).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ─── system deps ──────────────────────────────────────────────────────────────
# Deliberately no compilers and no system CycloneDDS: every native piece is a
# prebuilt wheel, and the `cyclonedds` pip wheel bundles libddsc (the
# kist_msgs types are hand-mirrored IdlStructs — no idlc). wget fetches the
# encoder model below; nano is in-container convenience.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

# ─── python 3.12 venv (uv) ────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}
RUN uv venv /opt/venv --python 3.12

# ─── replay + vla-io dependencies ─────────────────────────────────────────────
# The third-party set the replay import graph reaches, pinned to
# requirements.lock.txt where recorded: numpy/tyro/pyyaml (direct),
# cyclonedds (publisher), pyarrow (LeRobot episodes), onnxruntime (joint
# re-encoding, CPU provider), pytest (bake-time suite). av + opencv serve
# the vla input sources (H.264 camera decode, tests/view_camera.py).
# Installed before the source COPY so code edits never re-download.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    numpy==2.5.2 tyro==0.9.17 pyyaml==6.0.3 \
    cyclonedds==11.0.1 pyarrow==25.0.1 onnxruntime==1.29.0 \
    av==18.0.0 opencv-python-headless==5.0.0.93 \
    pytest==9.1.1

# ─── models: GEAR-SONIC encoder ONNX ──────────────────────────────────────────
# For `--tokens-from joints` (offline joint re-encoding) at its fixed path
# models/ — same convention as kist-gearsonic-inference, whose decoder this
# encoder pairs with. Must be swapped together with that decoder checkpoint.
WORKDIR /workspace/kist-vla-inference
RUN wget -q -P models \
    https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/model_encoder.onnx

# ─── source + install + bake-time verification ────────────────────────────────
# Bake the repo in (build context = repo root minus .venv/.git/models via
# .dockerignore — the wget layer above provides models/). --no-deps: the
# pinned layer above IS the dependency set; a new import the layer misses
# fails the checks below. The suite is the replay/common files only — vla
# tests need the inference stack this image deliberately lacks.
COPY . .
RUN uv pip install --no-deps -e . && \
    python -c "import replay.cli; \
               from replay.publisher import LatentActionPublisher; \
               print('replay import graph OK')" && \
    python -m pytest -q \
        tests/test_replay.py tests/test_replay_parquet.py \
        tests/test_encoder.py tests/test_latent_action_publisher.py \
        tests/test_dds.py tests/test_dds_config.py tests/test_g1_joints.py

CMD ["/bin/bash"]
