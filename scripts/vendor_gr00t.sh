#!/usr/bin/env bash
# Vendor the Isaac-GR00T *inference* path into thirdparty/gr00t/.
#
#   bash scripts/vendor_gr00t.sh              # re-vendor at the pinned commit
#   bash scripts/vendor_gr00t.sh <commit>     # re-vendor at another commit
#
# What it does, in order:
#   1. fetch exactly one commit of NVIDIA/Isaac-GR00T into a temp dir
#   2. overwrite the FILES below under thirdparty/gr00t/ (and LICENSE)
#   3. rewrite their imports  gr00t.*  ->  thirdparty.gr00t.*
#   4. apply thirdparty/gr00t/patches/*.patch  (our local modifications)
#   5. record the commit in thirdparty/gr00t/VENDORED_FROM.md
#
# Everything else under thirdparty/gr00t/ — every __init__.py, the patches,
# VENDORED_FROM.md — is ours and is never touched by this script. The
# __init__.py files are where the training-only imports are cut (see
# VENDORED_FROM.md), so no upstream file needs editing for that.
#
# The commit is a checkpoint coupling, not a preference: change it only when
# a new checkpoint was finetuned with a different Isaac-GR00T commit, then
# review `git diff thirdparty/gr00t` and run scripts/smoke_test_policy.py.

set -euo pipefail

# NVIDIA/Isaac-GR00T main @ 2026-07-08. The fork point of the KIST fork
# (foodbanana/Isaac-GR00T 5ac4e6b) that finetuned the rab-v2b-20260806
# checkpoints (checkpoint-18000 and friends). The fork's only change to the
# files below is the PolicyClient socket fix, which we carry as a patch.
GR00T_COMMIT_DEFAULT="9c7e746b2cd37a810070a98ef41d290a07e806c2"
GR00T_REMOTE="${GR00T_REMOTE:-https://github.com/NVIDIA/Isaac-GR00T.git}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/thirdparty/gr00t"
COMMIT="${1:-$GR00T_COMMIT_DEFAULT}"

# The inference path: what gr00t.policy.gr00t_policy reaches once the
# training plumbing (gr00t.model.__init__ -> setup.py -> DatasetFactory,
# gr00t.configs.model.__init__ -> base_config -> training_config) is cut.
FILES=(
    policy/gr00t_policy.py
    policy/policy.py
    policy/server_client.py
    model/gr00t_n1d7/gr00t_n1d7.py
    model/gr00t_n1d7/processing_gr00t_n1d7.py
    model/gr00t_n1d7/image_augmentations.py
    model/modules/dit.py
    model/modules/embodiment_conditioned_mlp.py
    model/modules/qwen3_backbone.py
    configs/model/gr00t_n1d7.py
    configs/data/embodiment_configs.py
    data/collator/collators.py
    data/state_action/state_action_processor.py
    data/state_action/action_chunking.py
    data/state_action/pose.py
    data/interfaces.py
    data/types.py
    data/embodiment_tags.py
    data/utils.py
    utils/initial_actions.py
)

echo "[1/5] fetching $GR00T_REMOTE @ $COMMIT"
SRC="$(mktemp -d)"
trap 'rm -rf "$SRC"' EXIT
git -C "$SRC" init -q
git -C "$SRC" fetch -q --depth 1 "$GR00T_REMOTE" "$COMMIT"
# demo_data/ is git-lfs; we only need source files, so never smudge.
GIT_LFS_SKIP_SMUDGE=1 git -C "$SRC" checkout -q FETCH_HEAD
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
COMMIT_DATE="$(git -C "$SRC" log -1 --format=%cs HEAD)"

echo "[2/5] copying ${#FILES[@]} files + LICENSE -> thirdparty/gr00t/"
for f in "${FILES[@]}"; do
    mkdir -p "$DEST/$(dirname "$f")"
    cp "$SRC/gr00t/$f" "$DEST/$f"
done
cp "$SRC/LICENSE" "$DEST/LICENSE"

echo "[3/5] rewriting imports gr00t.* -> thirdparty.gr00t.*"
for f in "${FILES[@]}"; do
    sed -i -E \
        -e 's/^(\s*)from gr00t\./\1from thirdparty.gr00t./' \
        -e 's/^(\s*)from gr00t import/\1from thirdparty.gr00t import/' \
        -e 's/^(\s*)import gr00t\b/\1import thirdparty.gr00t/' \
        "$DEST/$f"
done

echo "[4/5] applying local patches"
shopt -s nullglob
patches=("$DEST"/patches/*.patch)
if [ ${#patches[@]} -eq 0 ]; then
    echo "      (none)"
else
    for p in "${patches[@]}"; do
        echo "      $(basename "$p")"
        git -C "$REPO_ROOT" apply --directory=thirdparty/gr00t "$p"
    done
fi

echo "[5/5] recording commit in VENDORED_FROM.md"
if [ -f "$DEST/VENDORED_FROM.md" ]; then
    sed -i -E "s|^(- \*\*Commit\*\*: ).*|\1\`$COMMIT\` ($COMMIT_DATE)|" "$DEST/VENDORED_FROM.md"
fi

echo
echo "vendored NVIDIA/Isaac-GR00T @ $COMMIT ($COMMIT_DATE)"
echo "next: git diff --stat thirdparty/gr00t && python scripts/smoke_test_policy.py --model-path <checkpoint>"
