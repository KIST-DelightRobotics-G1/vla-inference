#!/usr/bin/env bash
# Launch (or re-attach to) a persistent named container — same convention as
# kist-gearsonic-inference/docker/run.sh: the image is self-contained, reuse
# the container across sessions until you `docker rm kist-vla-inference`.
#
#   --gpus all        local policy backend (torch cu128; host needs only the
#                     NVIDIA driver + nvidia-container-toolkit)
#   --network host    CycloneDDS discovery/multicast toward gearsonic
#   --ipc host        torch shared-memory convention
#
# Mounts:
#   <repo>/shared        -> /workspace/kist-vla-inference/shared
#                           host<->container exchange dir (created here):
#                           checkpoints, collector sessions, LeRobot exports
#   ~/.cache/huggingface -> /data/huggingface
#                           gated Cosmos backbone token + model cache
#
# Iterative dev: add  -v "$(pwd)":/workspace/kist-vla-inference  to shadow the
# baked source with your working copy (editable install picks it up).
set -euo pipefail

CONTAINER=kist-vla-inference
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(docker ps -q -f name=^${CONTAINER}$)" ]; then
    exec docker exec -it "${CONTAINER}" /bin/bash
elif [ "$(docker ps -aq -f name=^${CONTAINER}$)" ]; then
    docker start "${CONTAINER}" >/dev/null
    exec docker exec -it "${CONTAINER}" /bin/bash
fi

mkdir -p "${REPO_ROOT}/shared" "$HOME/.cache/huggingface"
exec docker run -it --name "${CONTAINER}" \
    --gpus all \
    --network host \
    --ipc host \
    -v "${REPO_ROOT}/shared":/workspace/kist-vla-inference/shared \
    -v "$HOME/.cache/huggingface":/data/huggingface \
    kist-vla-inference /bin/bash
