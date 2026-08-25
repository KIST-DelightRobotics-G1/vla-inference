"""Model-config registry stub.

LOCAL CUT (vs upstream ``gr00t/configs/model/__init__.py``): upstream globs
every module in this directory and, via ``base_config`` ->
``training_config`` / ``data_config``, drags the whole training
configuration (and tyro subcommand plumbing) into any import of
``Gr00tN1d7Config``. Inference needs only ``register_model_config``, which
``gr00t_n1d7.py`` calls at import time; keep the registry, drop the rest.
"""

MODEL_CONFIG_TYPES: dict[str, type] = {}


def register_model_config(shortname: str, configtype: type) -> None:
    MODEL_CONFIG_TYPES[shortname] = configtype
