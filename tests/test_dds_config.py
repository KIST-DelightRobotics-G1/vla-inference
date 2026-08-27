"""DDS config loading + CYCLONEDDS_URI routing (config/cyclonedds.xml)."""

import os

import pytest

from common.cyclonedds.config import DdsConfig, apply_cyclonedds_xml, load_dds_config


@pytest.fixture
def isolated_uri():
    """Snapshot/restore CYCLONEDDS_URI — apply_cyclonedds_xml sets it
    process-wide, and leaking it breaks unrelated tests (and vice versa:
    a URI exported by the environment must not leak in)."""
    had = "CYCLONEDDS_URI" in os.environ
    saved = os.environ.pop("CYCLONEDDS_URI", None)
    yield
    if had:
        os.environ["CYCLONEDDS_URI"] = saved
    else:
        os.environ.pop("CYCLONEDDS_URI", None)


def test_missing_default_config_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config/config.yaml here
    cfg = load_dds_config()
    assert cfg == DdsConfig(domain_id=0, cyclonedds_xml="config/cyclonedds.xml")


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dds_config(tmp_path / "nope.yaml")


def test_loads_yaml_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dds:\n  domain_id: 42\n  cyclonedds_xml: net/dds.xml\n")
    cfg = load_dds_config(path)
    assert cfg == DdsConfig(domain_id=42, cyclonedds_xml="net/dds.xml")


def test_partial_yaml_keeps_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dds:\n  domain_id: 7\n")
    cfg = load_dds_config(path)
    assert cfg == DdsConfig(domain_id=7, cyclonedds_xml="config/cyclonedds.xml")


def test_apply_sets_file_uri(tmp_path, isolated_uri):
    xml = tmp_path / "dds.xml"
    xml.write_text("<CycloneDDS/>")
    assert apply_cyclonedds_xml(xml) is True
    assert os.environ["CYCLONEDDS_URI"] == f"file://{xml.resolve()}"


def test_apply_respects_existing_uri(tmp_path, isolated_uri):
    xml = tmp_path / "dds.xml"
    xml.write_text("<CycloneDDS/>")
    os.environ["CYCLONEDDS_URI"] = "file:///operator/own.xml"
    assert apply_cyclonedds_xml(xml) is False
    assert os.environ["CYCLONEDDS_URI"] == "file:///operator/own.xml"


def test_apply_missing_default_path_is_a_noop(tmp_path, monkeypatch, isolated_uri):
    monkeypatch.chdir(tmp_path)  # no config/cyclonedds.xml here
    assert apply_cyclonedds_xml(DdsConfig.cyclonedds_xml) is False
    assert "CYCLONEDDS_URI" not in os.environ


def test_apply_missing_explicit_path_raises(tmp_path, isolated_uri):
    with pytest.raises(FileNotFoundError):
        apply_cyclonedds_xml(tmp_path / "nope.xml")
