# kist-vla-inference

GR00T N1.7 VLA inference service for the KIST Unitree G1 stack.

Runs an [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7 policy
(UNITREE_G1_SONIC embodiment) and streams 64-dim SONIC motion tokens + hand
joints to [kist-gearsonic-inference](https://github.com/Safety-Node/kist-gearsonic-inference)
(C++ whole-body controller) over ZMQ. The two services are independent peers
with no central supervisor: coordination rides the messages themselves
(liveness via DDS Deadline QoS, session lifecycle via the token stream,
operator commands as plain messages), and process lifecycle is the host's
job (e.g. systemd).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │   kist-vla-inference (this repo, Python)      │
 camera server ──►:5555─┐                                        │
                 │      ▼                                        │
 gearsonic ──────►:5557 Inference worker thread                  │
 (g1_debug state)│      observation → GR00T N1.7 (~0.4s, GPU)    │
                 │      → chunk: motion_token(40,64)+hands(40,7)²│
                 │      ▼  maxsize-1 queue                       │
 keyboard ───────►:5580 Main thread (50 Hz publish loop)         │
                 │      latency-compensated chunk playback       │
                 └──────│───────────────────────────────────────-┘
                        ▼ :5556  latent protocol v4 (PUB)
                 ┌──────────────────────────────────────────────┐
                 │  kist-gearsonic-inference (C++, 50 Hz RT)     │
                 │  token[64] → PolicyDecoder(TRT) → motors      │
                 └──────────────────────────────────────────────┘
```

Single process, two threads:

- **Main thread** — 50 Hz loop: consume finished chunks, trigger inference,
  play the cached chunk back one step per tick, publish each step. Never
  blocks; while inference runs it keeps playing the previous chunk, and a
  stale chunk degrades to holding its last token.
- **Inference worker** — assembles the observation from the latest sensor
  data and runs the policy (~0.4 s per chunk at 2.5 Hz).

Every cross-boundary handoff keeps only the latest value (ZMQ CONFLATE
sockets, maxsize-1 queues) — consumers see fresh data, never a backlog.

## Transports

The runner core talks to the outside world through the Protocols in
`kist_vla/io/interfaces.py`; each channel picks its transport independently
(`--io.action-transport`, `--io.camera-transport`, `--io.state-transport`,
each `zmq` by default):

| Channel | `zmq` (reference/sim compatible) | `dds` (real robot) |
|---|---|---|
| actions → gearsonic | latent protocol v4 PUB :5556 | `kist_msgs::LatentActionStep` / `WbcCommand` (`kist_vla/io/dds.py`) |
| camera | gear_sonic sensor server :5555 | kist-ext-sensor-io `CompressedColorFrame`, H.264 → PyAV decode (`kist_vla/io/dds_camera.py`) |
| robot state | `g1_debug` re-publisher :5557 | unitree `rt/lowstate` + `rt/dex3/{left,right}/state` directly (`kist_vla/io/dds_state.py`) — no re-publisher process needed |

DDS notes:

- **`idl/kist_latent_action.idl` is the shared action contract** — the
  gearsonic C++ side codegens from it (`idlc -l cxx`); the Python side
  mirrors it in `kist_vla/io/dds.py`; keep the two in sync. The camera type
  mirrors kist-ext-sensor-io's `idl/kist_camera_frames.idl`.
- QoS: writers are Reliable + KeepLast(1) — "latest wins" like the ZMQ
  CONFLATE pair, but Reliable so they match both Reliable and BestEffort
  readers (gearsonic subscribes via unitree's ChannelSubscriber, whose
  reader QoS we don't control); commands Reliable + KeepLast(8); our own
  state/camera readers BestEffort + KeepLast(1).
- State wire semantics were verified against the reference deploy
  (`zmq_output_handler.hpp`): `body_q` = 29 absolute joint angles in Unitree
  motor order; `base_quat` = **pelvis** IMU quaternion (w,x,y,z) from
  LowState — not the torso IMU on `rt/secondary_imu`.
- Install: `pip install -e ".[dds]"` plus unitree_sdk2py for the state
  source (see pyproject comment).
- Remaining hardware verification: Dex3 hand motor order and IMU quaternion
  convention against the real robot / final data-collection pipeline.

## Interface contract (latent protocol v4, zmq transport)

Frozen wire format shared with gearsonic; see `kist_vla/protocol.py` and
`tests/test_protocol.py` for the byte-level pin.

| Direction | Port | Content |
|---|---|---|
| → gearsonic | 5556 (PUB, `pose` topic) | `token_state[1,64]` f32, `frame_index[1]` i64, `left/right_hand_joints[1,7]` f32, 50 Hz |
| → gearsonic | 5556 (PUB, `command` topic) | control-loop start/stop/planner flags |
| ← gearsonic | 5557 (SUB, `g1_debug` topic) | msgpack: `body_q[29]`, `left/right_hand_q[7]`, `base_quat[4]` (wxyz) |
| ← camera server | 5555 (SUB) | msgpack: `{timestamps, images}` (JPEG), key `ego_view` (+ optional wrists) |
| ← operator | 5580 (SUB) | keystrokes / `prompt:<text>` |
| ↔ policy server | 5550 (REQ/REP) | remote mode only (Isaac-GR00T PolicyServer) |

Message layout: `[topic][1280-byte NUL-padded JSON header][little-endian binary fields]`.

## Install

Python ≥ 3.10 for the package; **3.12** if you use the local (in-process)
policy backend, because `gr00t` requires it.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ~/Isaac-GR00T        # local policy mode only
```

## Run

**Scope**: this service does inference only — it consumes a finished
checkpoint and streams action tokens. Data collection and finetuning happen
outside this repo; a UNITREE_G1_SONIC-finetuned checkpoint is a hard input
(the base `nvidia/GR00T-N1.7-3B` has no SONIC action head — see the
[finetuning workflow](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/GR00TWholeBodyControl)
for how one is produced).

```bash
# Single process (model in-process, default)
python scripts/run_vla.py --policy.model-path /path/to/checkpoint-XXXX

# Or with the model on another machine:
python scripts/run_policy_server.py --model-path /path/to/checkpoint-XXXX   # GPU box
python scripts/run_vla.py --policy.mode remote --policy.host <gpu-box>      # robot side

# Operator console (separate terminal)
python scripts/keyboard_publisher.py
```

Operator keys: `p` pause/resume · `k` start/stop gearsonic loop · `i` initial
pose · `[` `]` toggle hands · `t` change prompt.

`--help` shows all options (ports, rates, prompt, action bound).

## Tests

All tests run without a GPU, model, or robot:

```bash
pytest
```

This includes the **L1 loopback harness** (`tests/test_loopback.py`): the
real runner over real ZMQ against fake sensors and a fake policy, verifying
the 50 Hz stream, latency compensation, frame ordering, prompt changes, and
control commands. DDS tests run when `cyclonedds` is installed and are
skipped otherwise.

## Provenance

Core loop and utilities are ported from NVIDIA
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
(`gear_sonic/scripts/run_vla_inference.py` and friends, Apache-2.0), with the
pinocchio robot model replaced by static joint tables
(`kist_vla/g1_joints.py`, dumped from the reference model — re-dump, don't
hand-edit). `gr00t` is a dependency; `gear_sonic` is not.

Known checkpoint couplings:

- `DEFAULT_INITIAL_MOTION_TOKEN` (`kist_vla/config.py`) is specific to the
  SONIC checkpoint used in training — must be re-derived if the gearsonic
  SONIC checkpoint changes.
- The normalization statistics used to decode actions live inside the N1.7
  checkpoint's processor and change with every finetune (handled by `gr00t`
  automatically — nothing to copy here).
