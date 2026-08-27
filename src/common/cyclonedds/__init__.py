"""CycloneDDS layer: network config, wire-type mirrors, and DDS transports.

Everything that touches CycloneDDS lives here:

    config.py        config/config.yaml `dds:` section (domain_id,
                     config/cyclonedds.xml via CYCLONEDDS_URI) — gearsonic-style
    kist_msgs.py     the wire contract: the `module kist_msgs` mirror of
                     idl/kist_latent_action.idl (topics, IdlStructs, QoS;
                     keep in sync) — counterpart of vla_latent_action.hpp
    kist_msgs_writer.py   KistMsgsWriter (publishes the contract)

Only what BOTH publishers (replay and vla) need lives here — the wire
contract, its writer, and the network config. Single-consumer DDS sources
(camera, robot state) live with their consumer in vla/.

The subpackage name never shadows the pip `cyclonedds` package — absolute
imports inside these modules resolve to site-packages as usual.
"""

from .config import DdsConfig, apply_cyclonedds_xml, load_dds_config

__all__ = [
    "DdsConfig",
    "apply_cyclonedds_xml",
    "load_dds_config",
]
