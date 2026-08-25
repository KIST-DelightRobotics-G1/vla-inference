"""VLA inference runner — the kist-vla-inference process.

Single process, two threads (mirroring the reference
``gear_sonic/scripts/run_vla_inference.py``):

- **Main thread**: 50 Hz publish loop. Consumes finished chunks, triggers new
  inferences, plays the cached chunk back one step per tick, and publishes
  each step to gearsonic through the ActionSink. Never blocks.
- **Inference worker** (daemon): waits for a trigger, assembles the
  observation from the latest sensor data, runs the policy (~hundreds of ms),
  and hands the validated chunk back through a maxsize-1 queue.

Everywhere data crosses a boundary, only the latest value is kept (CONFLATE
sockets, maxsize-1 queues) — a slow consumer sees fresh data, never a
backlog.

I/O and the policy are injected via the Protocols in ``io.interfaces``
(defaults built from config), so tests and alternative transports (DDS) swap
in without touching this loop.

Operator keys (via the operator channel):
    p  pause / resume the policy loop
    k  start / stop the gearsonic control loop (PLANNER mode)
    i  blend to initial pose and switch gearsonic to POSE mode
    [  toggle left-hand open/closed for the initial pose
    ]  toggle right-hand open/closed for the initial pose
    prompt:<text>  change the language prompt at runtime
"""

import queue
import threading
import time
from typing import Any

import numpy as np

from .chunking import ActionChunkPlayer, should_trigger_new_inference
from common.config import DEFAULT_INITIAL_MOTION_TOKEN, RunnerConfig
from common.g1_joints import CLOSED_HAND_Q, OPEN_HAND_Q
from common.io import CameraClient, KeyboardSubscriber, StateSubscriber
from common.io.interfaces import ActionSink, CameraSource, OperatorSource, StateSource
from common.io.keyboard import PROMPT_PREFIX
from common.io.zmq_action_sink import ZmqActionSink
from .observation import ObservationBuilder
from .policy_backend import PolicyBackend, create_policy

ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")


def _print_green(x: str) -> None:
    print(f"\033[92m{x}\033[0m")


class VLARunner:
    def __init__(
        self,
        config: RunnerConfig,
        *,
        policy: PolicyBackend | None = None,
        camera: CameraSource | None = None,
        state_source: StateSource | None = None,
        operator: OperatorSource | None = None,
        action_sink: ActionSink | None = None,
    ):
        self.config = config

        self.policy = policy if policy is not None else create_policy(config.policy)
        if self.policy.ping():
            _print_green("Policy backend is ready.")
        else:
            print("WARNING: policy backend not reachable; inference will fail until it is up.")

        self.obs_builder = ObservationBuilder(language_key=config.language_key)

        if state_source is not None:
            self.state_source: StateSource = state_source
        elif config.io.state_transport == "dds":
            from .io.dds_state import DdsStateSource

            self.state_source = DdsStateSource(domain_id=config.io.dds_domain_id)
        else:
            self.state_source = StateSubscriber(
                host=config.io.state_host, port=config.io.state_port
            )

        if camera is not None:
            self.camera: CameraSource = camera
        elif config.io.camera_transport == "dds":
            from .io.dds_camera import DdsCameraSource

            self.camera = DdsCameraSource(
                domain_id=config.io.dds_domain_id, topic=config.io.dds_camera_topic
            )
        else:
            self.camera = CameraClient(
                host=config.io.camera_host, port=config.io.camera_port
            )
        self.operator: OperatorSource = (
            operator
            if operator is not None
            else KeyboardSubscriber(host=config.io.keyboard_host, port=config.io.keyboard_port)
        )
        if action_sink is not None:
            self.action_sink: ActionSink = action_sink
        elif config.io.action_transport == "dds":
            from .io.dds import DdsActionSink

            self.action_sink = DdsActionSink(domain_id=config.io.dds_domain_id)
        else:
            self.action_sink = ZmqActionSink(
                host=config.io.action_host, port=config.io.action_port
            )

        # Chunk playback + inference scheduling
        self.player = ActionChunkPlayer(config.action_horizon)
        self.inference_interval = 1.0 / config.inference_rate
        self.last_inference_time = 0.0

        # Worker plumbing: maxsize-1 queues, latest result wins
        self._trigger_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._busy_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        # Operator / robot-side state
        self.pause_loop = True
        self.cpp_loop_running = False
        self.cpp_mode = "OFF"  # OFF | PLANNER | POSE
        self.initial_pose_left_closed = False
        self.initial_pose_right_closed = False
        self.frame_counter = 0
        self.last_sent_motion_token: np.ndarray | None = None
        self.prompt = config.prompt
        self.initial_motion_token = DEFAULT_INITIAL_MOTION_TOKEN

    # ------------------------------------------------------------------
    # Inference worker
    # ------------------------------------------------------------------

    def _prepare_observation(self) -> dict[str, Any] | None:
        camera_msg = self.camera.read()
        state_msg = self.state_source.get_msg()
        return self.obs_builder.build(camera_msg, state_msg, self.prompt, log_errors=True)

    def _run_inference(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """Run the policy and return a validated, prefix-stripped action chunk."""
        try:
            action, _info = self.policy.get_action(observation)

            # Strip "action." prefixes and drop auxiliary outputs.
            chunk = {
                key.replace("action.", ""): value
                for key, value in action.items()
                if key.replace("action.", "") != "task_progress"
            }

            missing = [k for k in ACTION_KEYS if k not in chunk]
            if missing:
                print(
                    f"[Warning] action chunk missing keys {missing}; "
                    f"got {list(chunk.keys())}. Skipping."
                )
                return None

            token_absmax = float(np.abs(chunk["motion_token"]).max())
            if token_absmax > self.config.action_bound:
                print(
                    f"[Warning] |motion_token| max ({token_absmax:.4f}) exceeds action "
                    f"bound {self.config.action_bound}. Skipping chunk."
                )
                return None

            return chunk
        except Exception as e:
            print(f"Error in inference: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                try:
                    self._trigger_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self._busy_event.set()
                try:
                    observation = self._prepare_observation()
                    if observation is None:
                        continue

                    inference_start = time.monotonic()
                    chunk = self._run_inference(observation)
                    if chunk is None:
                        continue

                    # Latest result wins: drop a stale unconsumed chunk.
                    try:
                        self._result_queue.put_nowait((chunk, inference_start))
                    except queue.Full:
                        try:
                            self._result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._result_queue.put_nowait((chunk, inference_start))
                finally:
                    self._busy_event.clear()
            except Exception as e:
                print(f"Error in inference worker thread: {e}")
                import traceback

                traceback.print_exc()

    # ------------------------------------------------------------------
    # Commands toward the gearsonic control loop
    # ------------------------------------------------------------------

    def _send_cpp_command(self, start: bool, planner: bool = False) -> bool:
        try:
            self.action_sink.send_command(start=start, planner=planner)
            self.cpp_loop_running = start
            self.cpp_mode = ("PLANNER" if planner else "POSE") if start else "OFF"
            _print_green(
                f"Sent command: {'start' if start else 'stop'} control loop "
                f"({'planner' if planner else 'pose'} mode)"
            )
            return True
        except Exception as e:
            print(f"Warning: failed to send control command: {e}")
            return False

    def _initial_pose_hands(self) -> tuple[np.ndarray, np.ndarray]:
        left = CLOSED_HAND_Q["left"] if self.initial_pose_left_closed else OPEN_HAND_Q
        right = CLOSED_HAND_Q["right"] if self.initial_pose_right_closed else OPEN_HAND_Q
        return left, right

    def _send_token(
        self, token: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> None:
        self.action_sink.send_latent_action(
            motion_token=token,
            frame_index=self.frame_counter,
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        self.last_sent_motion_token = np.asarray(token, dtype=np.float32).reshape(-1).copy()

    def _publish_initial_pose(self) -> None:
        print("Moving to initial pose")
        left, right = self._initial_pose_hands()
        self.frame_counter = 0
        self._send_token(self.initial_motion_token, left, right)
        _print_green("Sent initial pose token")
        time.sleep(1.0)

    def _blend_to_initial_pose(self, duration_s: float) -> None:
        """Linearly blend from the last sent token to the initial pose token."""
        if self.last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            self._publish_initial_pose()
            return

        start_token = self.last_sent_motion_token.copy()
        target_token = self.initial_motion_token
        rate = self.config.action_publish_rate
        num_steps = max(1, round(rate * duration_s))
        step_period = 1.0 / rate
        left, right = self._initial_pose_hands()

        print(f"Blending to initial pose over {duration_s:.2f}s ({num_steps} steps at {rate} Hz)")
        self.frame_counter = 0
        for step in range(num_steps):
            t0 = time.monotonic()
            alpha = (step + 1) / num_steps
            blended = ((1.0 - alpha) * start_token + alpha * target_token).astype(np.float32)
            self._send_token(blended, left, right)
            remaining = step_period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
        _print_green("Initial pose blend complete.")

    # ------------------------------------------------------------------
    # Operator handling
    # ------------------------------------------------------------------

    def _handle_operator(self) -> None:
        key = self.operator.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_PREFIX):
            new_prompt = key[len(PROMPT_PREFIX):]
            if new_prompt:
                _print_green(f'Prompt changed: "{self.prompt}" -> "{new_prompt}"')
                self.prompt = new_prompt
            else:
                print("Received empty prompt change — ignoring.")
        elif key == "p":
            self.pause_loop = not self.pause_loop
            print(f"{'Paused' if self.pause_loop else 'Resumed'} policy loop")
        elif key == "k":
            if self.cpp_loop_running:
                print(f"Stopping gearsonic control loop (from {self.cpp_mode} mode)...")
                self._send_cpp_command(start=False, planner=self.cpp_mode == "PLANNER")
            else:
                print("Starting gearsonic control loop in PLANNER mode...")
                if self._send_cpp_command(start=True, planner=True):
                    print("Press 'i' for initial pose, 'p' to resume the policy loop.")
        elif key == "i":
            if self.cpp_loop_running and self.cpp_mode == "PLANNER":
                self._send_cpp_command(start=True, planner=False)
            elif not self.cpp_loop_running:
                print("Note: gearsonic loop not running — press 'k' to start")

            self.pause_loop = True
            if self.config.initial_pose_blend_duration > 0 and self.last_sent_motion_token is not None:
                self._blend_to_initial_pose(self.config.initial_pose_blend_duration)
            else:
                self._publish_initial_pose()

            self.frame_counter = 0
            self.player.clear()
            print("Cleared cached action chunk, reset frame counter")
        elif key == "[":
            self.initial_pose_left_closed = not self.initial_pose_left_closed
            print(f"Initial pose left hand: {'closed' if self.initial_pose_left_closed else 'open'}")
        elif key == "]":
            self.initial_pose_right_closed = not self.initial_pose_right_closed
            print(f"Initial pose right hand: {'closed' if self.initial_pose_right_closed else 'open'}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, stop_event: threading.Event | None = None) -> None:
        """Run the publish loop until KeyboardInterrupt or ``stop_event`` is set."""
        if stop_event is None:
            stop_event = threading.Event()
        loop_period = 1.0 / self.config.action_publish_rate
        _print_green(f'Starting policy loop (prompt: "{self.prompt}", paused — press p)')
        self._worker.start()

        try:
            while not stop_event.is_set():
                t_start = time.monotonic()
                self._handle_operator()

                # Consume a finished chunk first so the trigger check below
                # sees a fresh last_inference_time.
                try:
                    chunk, inference_start = self._result_queue.get_nowait()
                    inference_delay = time.monotonic() - inference_start
                    self.player.update(
                        chunk, inference_delay, self.config.action_publish_rate
                    )
                    self.last_inference_time = time.monotonic()
                    _print_green(
                        f'New action chunk (prompt: "{self.prompt}", '
                        f"latency: {inference_delay:.3f}s)"
                    )
                except queue.Empty:
                    pass

                if should_trigger_new_inference(
                    cached_chunk_exists=self.player.has_chunk,
                    inference_thread_running=self._busy_event.is_set(),
                    time_since_last_inference=time.monotonic() - self.last_inference_time,
                    inference_interval=self.inference_interval,
                ):
                    try:
                        self._trigger_queue.put_nowait(None)
                    except queue.Full:
                        pass

                if self.pause_loop:
                    time.sleep(0.2)
                    continue

                step = self.player.step()
                if step is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                else:
                    self._send_token(
                        step["motion_token"],
                        step["left_hand_joints"],
                        step["right_hand_joints"],
                    )
                    self.frame_counter += 1
                    if self.frame_counter % 50 == 0:
                        _print_green(
                            f"Sent latent action — frame {self.frame_counter}, "
                            f"token shape {step['motion_token'].shape}"
                        )

                elapsed = time.monotonic() - t_start
                if self.config.verbose_timing or elapsed > loop_period:
                    print(f"[timing] loop took {elapsed * 1000:.1f}ms (budget {loop_period * 1000:.0f}ms)")

                remaining = loop_period - (time.monotonic() - t_start)
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            print("VLA inference loop terminated by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self.action_sink.close()
        self.state_source.close()
        self.camera.close()
        self.operator.close()
        self.policy.close()
        print("Shutdown complete.")


def main(config: RunnerConfig) -> None:
    VLARunner(config).run()
