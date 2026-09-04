"""Model-config registry (trimmed).

Upstream's __init__ dynamically imports every config module and builds a tyro
CLI union for the training launcher. Inference only needs the registry hook
that `gr00t_n1d7.py` calls at import time, so that is all that remains.
"""

MODEL_CONFIG_TYPES: dict[str, type] = {}


def register_model_config(shortname: str, configtype: type):
    MODEL_CONFIG_TYPES[shortname] = configtype
