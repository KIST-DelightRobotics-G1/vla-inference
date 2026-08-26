#!/usr/bin/env bash
# Launch (or re-attach to) a persistent named container — same convention as
# kist-gearsonic-inference/docker/run.sh: the image is self-contained, reuse
# the container across sessions until you `docker rm kist-vla-inference`.
#
#   --network host    CycloneDDS discovery/multicast toward gearsonic
#
# Mounts:
#   <repo>/shared -> /workspace/kist-vla-inference/shared
#                    host<->container exchange dir (created here):
#                    collector sessions, LeRobot exports
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

mkdir -p "${REPO_ROOT}/shared"
exec docker run -it --name "${CONTAINER}" \
    --network host \
    -v "${REPO_ROOT}/shared":/workspace/kist-vla-inference/shared \
    kist-vla-inference /bin/bash
