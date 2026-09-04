"""ObservationBuilder — fresh io snapshots -> Observation, or None.

The stage's logic: pull `latest*()` from every source, judge freshness per
stream, and assemble the model's state groups exactly the way the training
data was assembled (the reference pipeline's conventions, hardware-verified
2026-08-28):

- the 43-dim full configuration comes from `common.g1_joints.assemble_full_q`
  (29 body motors + 7+7 hand motors scattered into the model's slot layout),
  split into the embodiment's 7 joint groups;
- the left hand goes through `apply_hand_hardware_coupling` first (the KIST
  gripper's left middle-slot encoders are dead — the live pair's readings
  are copied over, as the reference pipeline does);
- `projected_gravity` is computed from the pelvis quaternion (wxyz).

Staleness is judged HERE, per stream, against the ages the sources stamp at
arrival: one stale or missing stream means no observation (`build` returns
None) — the policy must never act on a frankenstein of old sensors. The
limits are constructor parameters with conservative defaults; the runner
decides how often to try again.
"""

from common.g1_joints import apply_hand_hardware_coupling, assemble_full_q, split_state

from .gravity import compute_projected_gravity
from .observation import Observation

# Default freshness limits (seconds). Cameras stream at 30 fps and a stale
# image mostly costs reaction time; lowstate streams at 500 Hz and feeds the
# balance-critical state, so its limit is tight (gearsonic clears its buffer
# at 60 ms — 0.1 s keeps a little slack for the 2.5 Hz consumer). Hands run
# on their own slower clocks.
CAMERA_MAX_AGE_S = 0.5
STATE_MAX_AGE_S = 0.1
HAND_MAX_AGE_S = 0.5


class ObservationBuilder:
    """Assemble an Observation from the io sources' latest snapshots.

    Args:
        cameras: view name -> ColorSubscriber (every view is REQUIRED — the
            checkpoint's modality config demands all of its views each
            inference).
        state_reader: the UnitreeStateReader carrying lowstate + hands.

    `build(prompt)` returns None when any stream is missing or stale; the
    reason is printed at most once per second so a dead sensor is visible
    without flooding.
    """

    def __init__(
        self,
        cameras: dict,
        state_reader,
        *,
        camera_max_age_s: float = CAMERA_MAX_AGE_S,
        state_max_age_s: float = STATE_MAX_AGE_S,
        hand_max_age_s: float = HAND_MAX_AGE_S,
    ):
        self._cameras = dict(cameras)
        self._state_reader = state_reader
        self._camera_max_age_s = camera_max_age_s
        self._state_max_age_s = state_max_age_s
        self._hand_max_age_s = hand_max_age_s
        self._last_report = 0.0

    def build(self, prompt: str) -> Observation | None:
        """One Observation from the freshest snapshots, or None."""
        video: dict = {}
        for view, camera in self._cameras.items():
            frame, age = camera.latest()
            if frame is None or age > self._camera_max_age_s:
                self._report(f"camera '{view}' {'missing' if frame is None else f'stale ({age:.2f}s)'}")
                return None
            video[view] = frame.rgb

        state, age = self._state_reader.latest_state()
        if state is None or age > self._state_max_age_s:
            self._report(f"lowstate {'missing' if state is None else f'stale ({age:.2f}s)'}")
            return None

        left, left_age = self._state_reader.latest_left_hand()
        right, right_age = self._state_reader.latest_right_hand()
        for name, hand, hand_age in (("left hand", left, left_age), ("right hand", right, right_age)):
            if hand is None or hand_age > self._hand_max_age_s:
                self._report(f"{name} {'missing' if hand is None else f'stale ({hand_age:.2f}s)'}")
                return None

        full_q = assemble_full_q(
            body_q=state.q,
            left_hand_q=apply_hand_hardware_coupling(left.q),
            right_hand_q=right.q,
        )
        groups = {
            group: values.astype("float32") for group, values in split_state(full_q).items()
        }
        groups["projected_gravity"] = compute_projected_gravity(state.imu_pelvis.quaternion)

        return Observation(video=video, state=groups, prompt=prompt)

    def _report(self, reason: str) -> None:
        import time

        now = time.monotonic()
        if now - self._last_report >= 1.0:
            print(f"[ObservationBuilder] no observation: {reason}", flush=True)
            self._last_report = now
