"""The gr00t commit pin must work whether or not gr00t is installed.

These run on a machine with no gr00t and no GPU, so they pin the contract
(shape of the return value, no exceptions) rather than a specific commit.
"""

import re

from vla.gr00t_version import (
    EXPECTED_GR00T_COMMIT,
    installed_gr00t_commit,
    warn_on_gr00t_commit_mismatch,
)


def test_expected_commit_is_a_full_sha():
    # A short SHA would silently never match `git rev-parse HEAD` output.
    assert re.fullmatch(r"[0-9a-f]{40}", EXPECTED_GR00T_COMMIT)


def test_installed_commit_is_a_full_sha_or_none():
    commit = installed_gr00t_commit()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_warning_never_raises(capsys):
    # Called from create_policy() on the way to loading a model — it must
    # never be the thing that takes the runner down.
    warn_on_gr00t_commit_mismatch()
    warn_on_gr00t_commit_mismatch(expected="0" * 40)
    assert "MISMATCH" in capsys.readouterr().out or installed_gr00t_commit() is None
