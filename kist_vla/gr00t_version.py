"""Pin the Isaac-GR00T commit a checkpoint was finetuned with.

A GR00T checkpoint does not record which Isaac-GR00T produced it. Its
``config.json`` carries ``transformers_version`` and nothing else about the
code — and that config is a *delta*, not a full spec: keys absent from it
(``input_embedding_dim``, ``state_history_length``,
``attend_text_every_n_blocks``, ...) are filled in from defaults that live in
``gr00t/configs/model/gr00t_n1d7.py``. The architecture that gets assembled
is therefore ``config.json`` + *that version of the code*.

Mismatches fail in three ways, and only the first is loud:

- changed tensor **shapes** -> ``from_pretrained`` raises
- renamed **modules** -> ``AutoModel.from_pretrained`` is HF-default
  non-strict, so the renamed weights are randomly initialized behind a
  warning. The policy loads, runs, and emits garbage motion tokens.
- changed **preprocessing** -> everything works, quality just drops

``gr00t`` is installed editable from a local clone, which makes a plain
``git pull`` in that clone enough to change the running model with no
reinstall. Hence this check: compare the clone's HEAD against the commit the
checkpoint was built with, on every policy construction.
"""

import subprocess
from pathlib import Path

EXPECTED_GR00T_COMMIT = "5ac4e6b6ad7467f4ccd441f6d7ec574d4da0a21f"
"""``foodbanana/Isaac-GR00T`` main @ 2026-08-07.

The commit the ``rab-v2b-20260806`` checkpoints (``checkpoint-18000`` and
friends) were finetuned with. It is a KIST fork of NVIDIA's ``9c7e746``
(2026-07-08) plus deployment patches that do not touch model code; the one
that matters here is ``a9c944a`` ("close the abandoned socket when
PolicyClient reconnects") — without it a dead PolicyServer makes this
process unkillable by Ctrl+C, leaving the robot frozen in its last pose.

NVIDIA's ``main`` is *not* a superset: it moved on from the fork point with
commits the fork never took, including ``238ef45`` ("Remove PIL and tensor
round-trips from the inference image path"), which changes the inference
image path. Update this constant only together with the checkpoint.
"""


def installed_gr00t_commit() -> str | None:
    """HEAD of the editable ``gr00t`` clone, or None if it cannot be read.

    Returns None for a non-editable install (no git checkout to inspect) and
    when ``gr00t`` is not installed at all — callers treat that as "unknown",
    not as a mismatch.
    """
    try:
        import gr00t
    except ImportError:
        return None

    if gr00t.__file__ is None:
        return None
    repo = Path(gr00t.__file__).resolve().parent.parent
    if not (repo / ".git").exists():
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def warn_on_gr00t_commit_mismatch(expected: str = EXPECTED_GR00T_COMMIT) -> None:
    """Print a loud warning when the gr00t clone is not at ``expected``.

    Deliberately a warning, not an exception: running a checkpoint from a
    different finetune generation is a legitimate thing to do, it just has to
    be a decision rather than an accident. Silent when the commits match.
    """
    actual = installed_gr00t_commit()
    if actual is None:
        print(
            "[gr00t_version] cannot determine the gr00t commit (not an editable "
            "git checkout) — checkpoint/code compatibility is unverified",
            flush=True,
        )
        return
    if actual == expected:
        return

    print(
        "\n"
        "!! gr00t COMMIT MISMATCH\n"
        f"!!   expected {expected}\n"
        f"!!   installed {actual}\n"
        "!! The checkpoint may load with randomly initialized layers behind a\n"
        "!! warning and emit garbage motion tokens. Either check the clone out\n"
        "!! at the expected commit, or update EXPECTED_GR00T_COMMIT in\n"
        "!! kist_vla/gr00t_version.py to the commit this checkpoint was\n"
        "!! finetuned with.\n",
        flush=True,
    )
