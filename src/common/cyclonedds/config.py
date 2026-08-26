"""DDS network settings from config/config.yaml (gearsonic-style).

The C++ peers configure DDS the same way — gearsonic's config.yaml carries
`domain_id` + `network_interface`, ext-sensor-io routes the interface through
CycloneDDS XML. Python's cyclonedds has no ChannelFactory::Init(domain,
interface), so the interface is applied the ext-sensor-io way: an inline
CycloneDDS XML document in the CYCLONEDDS_URI environment variable, set
*before* the first DomainParticipant is created.

An already-set CYCLONEDDS_URI wins over the yaml — an operator exporting a
full XML config (socket buffers, tracing, ...) must not be silently
overridden by the interface-only snippet.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@dataclass(frozen=True)
class DdsConfig:
    """The `dds:` section of config/config.yaml."""

    domain_id: int = 0
    network_interface: str = ""  # "" = let CycloneDDS pick


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
        network_interface=str(dds.get("network_interface", "") or ""),
    )


def apply_network_interface(interface: str) -> bool:
    """Point CycloneDDS at `interface` via CYCLONEDDS_URI (inline XML).

    Returns True when applied. No-ops (False) when `interface` is empty or
    CYCLONEDDS_URI is already set — the environment is the operator's
    explicit choice and takes precedence. Must run before the process
    creates its first DomainParticipant; CycloneDDS reads the URI once.
    """
    if not interface:
        return False
    if os.environ.get("CYCLONEDDS_URI"):
        print(
            f"CYCLONEDDS_URI already set — ignoring network_interface "
            f"'{interface}' from the yaml"
        )
        return False

    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS><Domain Id=\"any\"><General><Interfaces>"
        f"<NetworkInterface name=\"{interface}\" priority=\"default\" multicast=\"default\"/>"
        "</Interfaces></General></Domain></CycloneDDS>"
    )
    return True
