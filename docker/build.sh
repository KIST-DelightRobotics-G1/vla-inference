#!/usr/bin/env bash
# Build the self-contained inference image (run from anywhere).
#
# Default target is `vla` (torch cu128 + the vendored GR00T core + the baked
# Cosmos-Reason2-2B backbone, ~20 GB). The backbone comes from the host's HF
# cache — see the Dockerfile's "baked backbone" section for how to seed it
# once on a new host. For the slim replay-only image:
#   docker/build.sh --target replay
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HF_HUB_CACHE_DIR="${HF_HUB_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
if [ ! -d "${HF_HUB_CACHE_DIR}/models--nvidia--Cosmos-Reason2-2B" ]; then
    echo "error: ${HF_HUB_CACHE_DIR}/models--nvidia--Cosmos-Reason2-2B not found." >&2
    echo "Seed the backbone into the host HF cache first (see docker/Dockerfile," >&2
    echo "'baked backbone' section), or set HF_HUB_CACHE_DIR." >&2
    exit 1
fi

exec docker build -f docker/Dockerfile \
    --build-context hf-cache="${HF_HUB_CACHE_DIR}" \
    -t kist-vla-inference "$@" .
