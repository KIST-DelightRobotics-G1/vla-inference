#!/usr/bin/env bash
# Build the kist-vla-inference image. Context is the repo root so the
# Dockerfile can COPY the whole source tree (see .dockerignore).
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t kist-vla-inference -f docker/Dockerfile "$@" .
