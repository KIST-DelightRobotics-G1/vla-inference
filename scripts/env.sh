# Environment for running the policy locally. Source it, don't execute it:
#
#   source scripts/env.sh
#
# Idempotent, and every path is overridable by exporting it first. No secrets
# live here — the Hugging Face token belongs in $HF_HOME/token.

_KIST_VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# ── Hugging Face ─────────────────────────────────────────────────────────────
# The VLM backbone (nvidia/Cosmos-Reason2-2B) is a GATED repo and is fetched
# from HF even when --policy.model-path is a local directory. The token and the
# cached backbone live under HF_HOME; without it the policy dies with a 401.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# ── Python path hygiene ──────────────────────────────────────────────────────
# A globally sourced ROS 2 Humble puts Python 3.10 site-packages on PYTHONPATH,
# which shadows real imports inside the 3.12 venv. This package does not use
# ROS — the rt/* topic names are unitree's DDS naming convention, not ROS 2.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"

# ── Conda ────────────────────────────────────────────────────────────────────
# An auto-activated conda base can shadow the libstdc++/CUDA libraries torch
# ships with. Leave it before activating the venv — the reverse order leaves
# conda's bin dir ahead on PATH.
if [ -n "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
    while [ -n "${CONDA_PREFIX:-}" ]; do
        conda deactivate || break
    done
fi

# ── Virtualenv ───────────────────────────────────────────────────────────────
if [ -f "$_KIST_VLA_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    [ "${VIRTUAL_ENV:-}" = "$_KIST_VLA_ROOT/.venv" ] || source "$_KIST_VLA_ROOT/.venv/bin/activate"
fi

# ── Report ───────────────────────────────────────────────────────────────────
if [ -f "$HF_HOME/token" ]; then
    echo "HF_HOME=$HF_HOME  (token present)"
else
    echo "HF_HOME=$HF_HOME  (TOKEN MISSING — the gated backbone will 401)"
fi
echo "python=$(command -v python3)  $(python3 --version 2>&1)"
[ -n "${CONDA_PREFIX:-}" ] && echo "warning: conda still active ($CONDA_PREFIX) — deactivate manually"

unset _KIST_VLA_ROOT
