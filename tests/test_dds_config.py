"""DDS network config: config/config.yaml parsing + CYCLONEDDS_URI handling."""

import os

import pytest

from common.cyclonedds.config import DdsConfig, apply_network_interface, load_dds_config


@pytest.fixture
def isolated_uri():
    """Snapshot/restore CYCLONEDDS_URI — apply_network_interface sets it
    directly in os.environ, which monkeypatch would not roll back, and a
    leaked interface breaks every later DomainParticipant in the session."""
    saved = os.environ.pop("CYCLONEDDS_URI", None)
    yield
    if saved is None:
        os.environ.pop("CYCLONEDDS_URI", None)
    else:
        os.environ["CYCLONEDDS_URI"] = saved


def test_missing_default_path_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config/config.yaml here
    cfg = load_dds_config()
    assert cfg == DdsConfig(domain_id=0, network_interface="")


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dds_config(tmp_path / "nope.yaml")


def test_parses_the_dds_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dds:\n  domain_id: 42\n  network_interface: eno2\n")
    cfg = load_dds_config(path)
    assert cfg == DdsConfig(domain_id=42, network_interface="eno2")


def test_partial_section_keeps_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dds:\n  domain_id: 7\n")
    cfg = load_dds_config(path)
    assert cfg == DdsConfig(domain_id=7, network_interface="")


def test_repo_config_file_parses():
    # The checked-in config/config.yaml must stay loadable.
    cfg = load_dds_config("config/config.yaml")
    assert cfg.domain_id == 0


def test_apply_empty_interface_is_a_noop(isolated_uri):
    assert apply_network_interface("") is False
    assert "CYCLONEDDS_URI" not in os.environ


def test_apply_sets_inline_xml(isolated_uri):
    assert apply_network_interface("eno2") is True
    uri = os.environ["CYCLONEDDS_URI"]
    assert uri.startswith("<CycloneDDS>") and 'name="eno2"' in uri


def test_existing_env_wins(isolated_uri):
    os.environ["CYCLONEDDS_URI"] = "file:///etc/cyclonedds.xml"
    assert apply_network_interface("eno2") is False
    assert os.environ["CYCLONEDDS_URI"] == "file:///etc/cyclonedds.xml"
