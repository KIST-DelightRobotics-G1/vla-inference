#!/usr/bin/env bash
# Install CycloneDDS 0.10.2 (C library) and build the matching Python binding.
#
#   bash scripts/install_cyclonedds.sh            # host: needs sudo for /opt, python3.12-dev
#   PREFIX=/opt/cyclonedds SUDO= bash scripts/install_cyclonedds.sh   # inside Docker (root)
#
# WHY 0.10.2 AND NOT THE PyPI WHEEL (reproduced 2026-08-19): pip's default
# `cyclonedds` is 11.x with libddsc 0.11 bundled. kist-gearsonic-inference and
# kist-ext-sensor-io are pinned to CycloneDDS 0.10.2, and the XTypes discovery
# wire format differs between 0.10 and 0.11 — when a 0.11 Python *reader* joins
# the domain, every 0.10.2 C++ participant segfaults in ddsi_xt_type_init_impl.
# One run of our state/camera subscriber kills gearsonic and ext-sensor-io at
# once. (A 0.11 writer alone happens to work, which is why a token-publish test
# does not catch it.) So the Python side must be 0.10.x too; PyPI has no cp312
# wheel for 0.10.2, hence: build the C library, then build the binding against
# it (CYCLONEDDS_HOME) — which also needs the Python headers.
set -euo pipefail

CYCLONEDDS_VERSION="${CYCLONEDDS_VERSION:-0.10.2}"
PREFIX="${PREFIX:-/opt/cyclonedds}"
SUDO="${SUDO-sudo}"                 # SUDO= to run without sudo (Docker)
PYTHON="${PYTHON:-python}"          # interpreter of the target venv

if [ ! -f "$PREFIX/lib/libddsc.so.$CYCLONEDDS_VERSION" ]; then
    echo "[1/2] building CycloneDDS $CYCLONEDDS_VERSION -> $PREFIX"
    SRC="$(mktemp -d)"
    git clone -q --depth 1 -b "$CYCLONEDDS_VERSION" https://github.com/eclipse-cyclonedds/cyclonedds.git "$SRC"
    cmake -S "$SRC" -B "$SRC/build" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF >/dev/null
    cmake --build "$SRC/build" --parallel "$(nproc)" >/dev/null
    $SUDO cmake --install "$SRC/build" >/dev/null
    rm -rf "$SRC"
else
    echo "[1/2] CycloneDDS $CYCLONEDDS_VERSION already at $PREFIX"
fi

echo "[2/2] building the python binding cyclonedds==$CYCLONEDDS_VERSION against $PREFIX"
"$PYTHON" -c 'import sysconfig,os,sys; p=os.path.join(sysconfig.get_paths()["include"],"Python.h"); sys.exit(0 if os.path.exists(p) else print(f"error: {p} missing — install python3.12-dev (apt) or use a uv-managed python", file=sys.stderr) or 1)'
CYCLONEDDS_HOME="$PREFIX" uv pip install --python "$PYTHON" --reinstall --no-cache "cyclonedds==$CYCLONEDDS_VERSION"
"$PYTHON" -c "import importlib.metadata as m; from cyclonedds.domain import DomainParticipant; print('cyclonedds', m.version('cyclonedds'), 'OK')"
