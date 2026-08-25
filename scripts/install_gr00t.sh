#!/usr/bin/env bash
# Install the `gr00t` local policy backend at the commit the checkpoint was
# finetuned with.
#
#   bash scripts/install_gr00t.sh          # clone/fetch + checkout + install
#
# The commit is read from src/vla/gr00t_version.py so there is exactly one
# authority for it. Override the clone location with GR00T_SRC and the fork
# with GR00T_REMOTE.
#
# Skip this entirely on the robot side (`--policy.mode remote`).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_SRC="${GR00T_SRC:-$HOME/Isaac-GR00T}"
GR00T_REMOTE="${GR00T_REMOTE:-https://github.com/foodbanana/Isaac-GR00T.git}"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found — create the venv first:" >&2
    echo "         uv venv --python 3.12 && uv pip install -e \".[dev]\"" >&2
    exit 1
fi

# Single source of truth: the same constant the runtime check compares against.
COMMIT="$("$PYTHON" -c 'from vla.gr00t_version import EXPECTED_GR00T_COMMIT; print(EXPECTED_GR00T_COMMIT)')"
echo "[1/3] target commit $COMMIT"

if [ -d "$GR00T_SRC/.git" ]; then
    echo "[2/3] fetching into existing clone $GR00T_SRC"
    git -C "$GR00T_SRC" fetch --quiet "$GR00T_REMOTE" '+refs/heads/*:refs/remotes/gr00t-install/*' --tags
else
    echo "[2/3] cloning $GR00T_REMOTE -> $GR00T_SRC"
    git clone --quiet "$GR00T_REMOTE" "$GR00T_SRC"
fi

# Detached on purpose. gr00t is installed editable, so leaving the clone on a
# tracking branch means a later `git pull` changes the running model with no
# reinstall and no warning.
git -C "$GR00T_SRC" checkout --quiet --detach "$COMMIT"
echo "      HEAD $(git -C "$GR00T_SRC" rev-parse HEAD)"

echo "[3/3] uv pip install -e $GR00T_SRC"
# gr00t's [tool.uv.sources] supplies the prebuilt flash-attn cp312 wheel and
# the cu128 torch index, so nothing is compiled from source. Python must be
# 3.12 (gr00t pins >=3.12,<3.13).
VIRTUAL_ENV="$REPO_ROOT/.venv" uv pip install -e "$GR00T_SRC"

echo
"$PYTHON" - <<'PY'
import torch, gr00t
from vla.gr00t_version import EXPECTED_GR00T_COMMIT, installed_gr00t_commit
actual = installed_gr00t_commit()
print(f"torch  {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"gr00t  {gr00t.__file__}")
print(f"commit {actual}  {'OK' if actual == EXPECTED_GR00T_COMMIT else 'MISMATCH'}")
PY
