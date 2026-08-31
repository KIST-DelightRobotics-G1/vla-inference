#!/usr/bin/env bash
# Launch (or re-attach to) a persistent named container — same convention as
# kist-gearsonic-inference/docker/run.sh: the image is self-contained, reuse
# the container across sessions until you `docker rm kist-vla-inference`.
#
#   --network host    CycloneDDS discovery/multicast toward gearsonic
#   --gpus all        GR00T inference (harmless no-op for replay work)
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
# Iterative dev: add  -v "$(pwd)":/workspace/kist-vla-inference  to shadow the
# baked source with your working copy (editable install picks it up).
set -euo pipefail

CONTAINER=kist-vla-inference
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HOME}/checkpoint-4500}"

if [ "$(docker ps -q -f name=^${CONTAINER}$)" ]; then
    exec docker exec -it "${CONTAINER}" /bin/bash
elif [ "$(docker ps -aq -f name=^${CONTAINER}$)" ]; then
    docker start "${CONTAINER}" >/dev/null
    exec docker exec -it "${CONTAINER}" /bin/bash
fi

mkdir -p "${REPO_ROOT}/shared"
exec docker run -it --name "${CONTAINER}" \
    --network host \
    --gpus all \
    -v "${REPO_ROOT}/shared":/workspace/kist-vla-inference/shared \
    -v "${CHECKPOINT_DIR}":"/workspace/checkpoints/$(basename "${CHECKPOINT_DIR}")":ro \
    kist-vla-inference /bin/bash
