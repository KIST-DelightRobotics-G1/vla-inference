#!/usr/bin/env bash
# Shell into the kist-vla-inference container (creates it on first use,
# re-attaches afterwards — same convention as kist-gearsonic-inference).
#
#   --gpus all        model inference
#   --network host    DDS (gearsonic, ext-sensor-io, unitree rt/*) and ZMQ share the host network
#   VLA_DATA          checkpoint directory   (default ~/vla_data)   -> /vla_data (read-only)
#   HF_CACHE          Hugging Face cache     (default ~/hf_cache)   -> /hf_cache (token + gated backbone)
#   DEV=1             bind-mount this repo over /workspace to edit code without rebuilding
#
# Inside:  python scripts/smoke_test_policy.py --model-path /vla_data/checkpoint-18000
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=kist-vla-inference
VLA_DATA="${VLA_DATA:-$HOME/vla_data}"
HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"

if [ "$(docker ps -q -f name=^${CONTAINER}$)" ]; then
    exec docker exec -it "$CONTAINER" /bin/bash
elif [ "$(docker ps -aq -f name=^${CONTAINER}$)" ]; then
    exec docker start -ai "$CONTAINER"
fi

[ -f "$HF_CACHE/token" ] || echo "warning: $HF_CACHE/token missing — the gated backbone will 401" >&2

dev_mount=()
[ "${DEV:-0}" = "1" ] && dev_mount=(-v "$PWD:/workspace")

exec docker run -it \
    --name "$CONTAINER" \
    --gpus all \
    --network host \
    -v "$VLA_DATA:/vla_data:ro" \
    -v "$HF_CACHE:/hf_cache" \
    "${dev_mount[@]}" \
    -w /workspace \
    kist-vla-inference
