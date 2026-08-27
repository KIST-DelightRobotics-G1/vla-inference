"""DDS network settings: config/config.yaml + config/cyclonedds.xml.

Same convention as kist-ext-sensor-io: the yaml keeps the domain id and
points at a CycloneDDS XML file that carries every transport setting (the
network interface, the socket-buffer tuning). `apply_cyclonedds_xml` routes
that file to CycloneDDS through the CYCLONEDDS_URI environment variable,
which must happen *before* the process creates its first DomainParticipant.

An already-set CYCLONEDDS_URI wins — an operator exporting their own config
must not be silently overridden.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@dataclass(frozen=True)
class DdsConfig:
    """The `dds:` section of config/config.yaml."""

    domain_id: int = 0
    cyclonedds_xml: str = "config/cyclonedds.xml"


def load_dds_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DdsConfig:
    """Read the `dds:` section; missing keys keep their defaults.

    The default path is allowed to be absent (fresh checkout, tests) and
    yields the defaults; an explicitly given path must exist.
    """
    path = Path(path)
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return DdsConfig()
        raise FileNotFoundError(f"{path} not found")

    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    dds = data.get("dds") or {}
    return DdsConfig(
        domain_id=int(dds.get("domain_id", DdsConfig.domain_id)),
        cyclonedds_xml=str(dds.get("cyclonedds_xml", DdsConfig.cyclonedds_xml)),
    )


def apply_cyclonedds_xml(path: str | Path) -> bool:
    """Point CYCLONEDDS_URI at the transport config file.

    Returns True when applied. No-ops (False) when CYCLONEDDS_URI is already
    set (the environment is the operator's explicit choice) or when the
    default-path file is absent (fresh checkout — CycloneDDS then runs on
    its own defaults); an explicitly configured path must exist. Must run
    before the process creates its first DomainParticipant; CycloneDDS
    reads the URI once.
    """
    if os.environ.get("CYCLONEDDS_URI"):
        print(f"CYCLONEDDS_URI already set — ignoring {path}")
        return False
    path = Path(path)
    if not path.exists():
        if str(path) == DdsConfig.cyclonedds_xml:
            return False
        raise FileNotFoundError(f"{path} not found")

    os.environ["CYCLONEDDS_URI"] = f"file://{path.resolve()}"
    return True

