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
# ─── stage 1: CycloneDDS 0.10.2 core + python bindings wheel ──────────────────
# The whole robot bus is CycloneDDS 0.10.x (unitree_sdk2 vendors 0.10.2 into
# gearsonic and ext-sensor-io, and their receivers SEGFAULT on the XTypes
# TypeObject the modern pip wheel's 11.x core advertises in discovery:
# ddsi_xt_type_init_impl "invalid type object"). No 0.10.x wheel exists for
# python 3.12, so the bindings are compiled here against a 0.10.2 core —
# one DDS generation across the bus, and the compilers stay in this stage.
FROM ubuntu:24.04 AS dds-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG CYCLONEDDS_VERSION=0.10.2
RUN git clone --depth 1 --branch "${CYCLONEDDS_VERSION}" \
        https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds && \
    cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds \
        -DBUILD_IDLC=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/cyclonedds/build --target install -j "$(nproc)" && \
    rm -rf /tmp/cyclonedds

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN uv venv /opt/wheelenv --python 3.12 && \
    VIRTUAL_ENV=/opt/wheelenv uv pip install pip && \
    CYCLONEDDS_HOME=/opt/cyclonedds /opt/wheelenv/bin/python -m pip wheel \
        --no-binary cyclonedds --no-deps -w /wheels cyclonedds==0.10.2

# unitree_sdk2py (not on PyPI): the unitree DDS IDL types the state reader
# needs (rt/lowstate LowState_, rt/dex3/* HandState_). Wheel built here so
# the runtime stage needs no git; installed --no-deps there (it would pull
# opencv-python and re-pin cyclonedds — our 0.10.2 matches its pin anyway).
ARG UNITREE_SDK2PY_REF=master
RUN git clone --depth 1 --branch "${UNITREE_SDK2PY_REF}" \
        https://github.com/unitreerobotics/unitree_sdk2_python.git /tmp/unitree_sdk2_python && \
    /opt/wheelenv/bin/python -m pip wheel --no-deps -w /wheels /tmp/unitree_sdk2_python

# ─── stage 2: the runtime image ───────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ─── system deps ──────────────────────────────────────────────────────────────
# Deliberately no compilers: every native piece is a prebuilt wheel (the
# cyclonedds one built in stage 1). wget fetches the encoder model below;
# nano is in-container convenience.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

# The 0.10.2 core the bindings load at runtime (their rpath points here).
COPY --from=dds-builder /opt/cyclonedds /opt/cyclonedds
COPY --from=dds-builder /wheels /wheels
ENV CYCLONEDDS_HOME=/opt/cyclonedds

# ─── python 3.12 venv (uv) ────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}
RUN uv venv /opt/venv --python 3.12

# ─── replay + vla-io dependencies ─────────────────────────────────────────────
# The third-party set the replay import graph reaches, pinned to
# requirements.lock.txt where recorded: numpy/tyro/pyyaml (direct),
# cyclonedds 0.10.2 (publisher — the stage-1 wheel, matching the robot
# bus's DDS generation), pyarrow (LeRobot episodes), onnxruntime (joint
# re-encoding, CPU provider), pytest (bake-time suite). av + opencv serve
# the vla input sources (H.264 camera decode, tests/view_camera.py).
# Installed before the source COPY so code edits never re-download.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    numpy==2.5.2 tyro==0.9.17 pyyaml==6.0.3 \
    /wheels/cyclonedds-0.10.2-*.whl \
    pyarrow==25.0.1 onnxruntime==1.29.0 \
    av==18.0.0 opencv-python-headless==5.0.0.93 \
    pytest==9.1.1 && \
    uv pip install --no-deps /wheels/unitree_sdk2py-*.whl

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
