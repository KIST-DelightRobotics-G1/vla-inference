"""The observation stage: io snapshots -> one Observation for the policy.

    observation_builder.py  ObservationBuilder — pulls latest*() from every
                            source, judges freshness per stream (one stale
                            or missing stream = no observation), assembles
                            the embodiment's state groups the way the
                            training data was assembled
    observation.py          Observation — the contract with the policy
                            stage: video by view, state by group, prompt
    gravity.py              projected_gravity (pure numpy, verified against
                            the scipy reference)
"""

from .observation import Observation
from .observation_builder import ObservationBuilder

__all__ = ["Observation", "ObservationBuilder"]
