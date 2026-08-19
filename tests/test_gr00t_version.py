"""The gr00t commit pin must work whether or not gr00t is installed.

These run on a machine with no gr00t and no GPU, so they pin the contract
(shape of the return value, no exceptions) rather than a specific commit.
"""

import re
from pathlib import Path

import pytest

from kist_vla.gr00t_version import (
    EXPECTED_GR00T_COMMIT,
    installed_gr00t_commit,
    warn_on_gr00t_commit_mismatch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_expected_commit_is_a_full_sha():
    # A short SHA would silently never match `git rev-parse HEAD` output.
    assert re.fullmatch(r"[0-9a-f]{40}", EXPECTED_GR00T_COMMIT)


def test_installed_commit_is_a_full_sha_or_none():
    commit = installed_gr00t_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


@pytest.mark.parametrize("doc", ["README.md", "pyproject.toml"])
def test_docs_quote_the_same_commit(doc):
    # The commit appears in prose as well as in code; a stale copy would send
    # someone to install the wrong gr00t while the runtime check stays quiet.
    assert EXPECTED_GR00T_COMMIT in (REPO_ROOT / doc).read_text()


def test_warning_never_raises(capsys):
    # Called from create_policy() on the way to loading a model — it must
    # never be the thing that takes the runner down.
    warn_on_gr00t_commit_mismatch()
    warn_on_gr00t_commit_mismatch(expected="0" * 40)
    assert "MISMATCH" in capsys.readouterr().out or installed_gr00t_commit() is None
