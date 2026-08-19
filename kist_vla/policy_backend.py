"""Policy backend selection: in-process Gr00tPolicy or remote PolicyClient.

Both backends expose the same surface (``get_action``, ``ping``, ``close``),
so the runner never knows which one it is talking to. ``gr00t`` is imported
lazily — everything else in this package (protocol, joints, tests) works
without it installed.

Keeping the remote path alive is deliberate: it is the escape hatch for
moving the GPU to another machine without touching runner code
(``--policy.mode remote --policy.host <gpu-box>``).
"""

from typing import Any, Protocol

from .config import PolicyConfig
from .gr00t_version import warn_on_gr00t_commit_mismatch


class PolicyBackend(Protocol):
    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict]: ...
    def ping(self) -> bool: ...
    def close(self) -> None: ...


class LocalPolicy:
    """In-process Gr00tPolicy (single-process deployment, needs GPU)."""

    def __init__(self, config: PolicyConfig):
        if config.model_path is None:
            raise ValueError("--policy.model-path is required in local mode")

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        print(f"[LocalPolicy] Loading {config.model_path} on {config.device}...")
        self._policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
        )
        print("[LocalPolicy] Model loaded.")

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict]:
        return self._policy.get_action(observation)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


class RemotePolicy:
    """ZMQ client to a running Isaac-GR00T PolicyServer."""

    def __init__(self, config: PolicyConfig):
        from gr00t.policy.server_client import PolicyClient

        self._client = PolicyClient(host=config.host, port=config.port)
        print(f"[RemotePolicy] Connecting to PolicyServer at {config.host}:{config.port}")

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict]:
        return self._client.get_action(observation)

    def ping(self) -> bool:
        return self._client.ping()

    def close(self) -> None:
        self._client.close()


def create_policy(config: PolicyConfig) -> PolicyBackend:
    # Both modes import gr00t: local for Gr00tPolicy, remote for PolicyClient
    # (whose socket teardown fix is part of the pinned commit).
    warn_on_gr00t_commit_mismatch()
    if config.mode == "local":
        return LocalPolicy(config)
    if config.mode == "remote":
        return RemotePolicy(config)
    raise ValueError(f"Unknown policy mode: {config.mode}")
