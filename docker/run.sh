#!/usr/bin/env bash
# Launch (or re-attach to) a persistent named container — same convention as
# kist-gearsonic-inference/docker/run.sh: the image is self-contained, reuse
# the container across sessions until you `docker rm kist-vla-inference`.
#
#   --network host    CycloneDDS discovery/multicast toward gearsonic
#   --gpus all        GR00T inference (harmless no-op for replay work)
#   -e DISPLAY + /tmp/.X11-unix   the pyqtgraph viewer draws on the HOST's X
#                     server (container has none). DISPLAY falls back to :0
#                     for SSH sessions; `xhost +local:docker` grants access.
#                     QT_X11_NO_MITSHM=1 avoids BadAccess from shared-memory
#                     X images across the container boundary.
#
# Mounts:
#   <repo>/shared            -> /workspace/kist-vla-inference/shared
#                               host<->container exchange dir (created here):
#                               collector sessions, LeRobot exports
#   $CHECKPOINT_DIR          -> /workspace/checkpoints/<name>, read-only
#                               (default ~/checkpoint-4500 — the
#                               unitree_g1_sonic_3views finetune)
#
# No HF mount: the Cosmos-Reason2-2B backbone is baked into the image and the
# image runs with HF_HUB_OFFLINE=1 — no network or HF account at runtime.
#
#   <repo>/{src,tests,scripts,config} -> same paths in the container
#                               the working copy shadows the baked source, so a
#                               `git pull` on the host is live in the container
#                               (editable install points at src/). Only these
#                               four: mounting the whole repo would hide the
#                               image's models/ (SONIC encoder ONNX).
#                               PYTHONDONTWRITEBYTECODE keeps root-owned
#                               __pycache__ out of the host tree.
set -euo pipefail

CONTAINER=kist-vla-inference
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR_GIVEN="${CHECKPOINT_DIR+x}"   # user set it (before the default fills in)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HOME}/checkpoint-4500}"

# Mounts only apply at container CREATION: re-attaching ignores
# CHECKPOINT_DIR, so warn instead of silently running the wrong checkpoint.
warn_reuse() {
    if [ -n "${CHECKPOINT_DIR_GIVEN}" ]; then
        echo "WARNING: reusing the existing '${CONTAINER}' container — its mounts" >&2
        echo "         were fixed at creation and CHECKPOINT_DIR is IGNORED now." >&2
        echo "         To mount ${CHECKPOINT_DIR}:" >&2
        echo "         docker rm -f ${CONTAINER}  # then re-run this script" >&2
    fi
}

if [ "$(docker ps -q -f name=^${CONTAINER}$)" ]; then
    warn_reuse
    exec docker exec -it "${CONTAINER}" /bin/bash
elif [ "$(docker ps -aq -f name=^${CONTAINER}$)" ]; then
    warn_reuse
    docker start "${CONTAINER}" >/dev/null
    exec docker exec -it "${CONTAINER}" /bin/bash
fi

mkdir -p "${REPO_ROOT}/shared"
# Let the container's X client reach the host X server (no-op without an X
# session, e.g. plain SSH — the viewer then needs DISPLAY pointed elsewhere).
xhost +local:docker >/dev/null 2>&1 || true
exec docker run -it --name "${CONTAINER}" \
    --network host \
    --gpus all \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${REPO_ROOT}/shared":/workspace/kist-vla-inference/shared \
    -v "${REPO_ROOT}/src":/workspace/kist-vla-inference/src \
    -v "${REPO_ROOT}/tests":/workspace/kist-vla-inference/tests \
    -v "${REPO_ROOT}/scripts":/workspace/kist-vla-inference/scripts \
    -v "${REPO_ROOT}/config":/workspace/kist-vla-inference/config \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${CHECKPOINT_DIR}":"/workspace/checkpoints/$(basename "${CHECKPOINT_DIR}")":ro \
    kist-vla-inference /bin/bash
