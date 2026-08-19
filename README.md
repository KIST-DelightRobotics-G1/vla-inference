# kist-vla-inference

GR00T N1.7 VLA inference service for the KIST Unitree G1 stack. Runs an
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) N1.7 policy
(UNITREE_G1_SONIC embodiment) and streams 64-dim SONIC motion tokens + hand
joints at 50 Hz to
[kist-gearsonic-inference](https://github.com/Safety-Node/kist-gearsonic-inference)
(C++ whole-body controller).

**Scope**: inference only — this service consumes a finished checkpoint and
streams action tokens. Data collection and finetuning happen outside this
repo. The two services are independent peers with no central supervisor:
coordination rides the messages themselves (liveness via DDS Deadline QoS,
session lifecycle via the token stream, operator commands as plain
messages), and process lifecycle is the host's job (e.g. systemd).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │   kist-vla-inference (this repo, Python)      │
 camera ─────────►      Inference worker thread                  │
                 │      observation → GR00T N1.7 (~0.4s, GPU)    │
 robot state ────►      → chunk: motion_token(40,64)+hands(40,7)²│
                 │      ▼  maxsize-1 queue                       │
 keyboard ───────►      Main thread (50 Hz publish loop)         │
                 │      latency-compensated chunk playback       │
                 └──────│───────────────────────────────────────-┘
                        ▼ latent actions, 50 Hz
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

### Transports

The runner core talks to the outside world through the Protocols in
`kist_vla/io/interfaces.py`; each channel picks its transport independently
(`--io.action-transport`, `--io.camera-transport`, `--io.state-transport`,
each `zmq` by default):

| Channel | `zmq` (reference/sim compatible) | `dds` (real robot) |
|---|---|---|
| actions → gearsonic | latent protocol v4 PUB :5556 | `kist_msgs::LatentActionStep` / `WbcCommand` (`kist_vla/io/dds.py`) |
| camera | gear_sonic sensor server :5555 | kist-ext-sensor-io `CompressedColorFrame`, H.264 → PyAV decode (`kist_vla/io/dds_camera.py`) |
| robot state | `g1_debug` re-publisher :5557 | unitree `rt/lowstate` + `rt/dex3/{left,right}/state` directly (`kist_vla/io/dds_state.py`) — no re-publisher process needed |

### Wire contract — DDS (real robot)

- **`idl/kist_latent_action.idl` is the shared action contract** — the
  gearsonic C++ side codegens from it (`idlc -l cxx`); the Python side
  mirrors it in `kist_vla/io/dds.py`; keep the two in sync. The camera type
  mirrors kist-ext-sensor-io's `idl/kist_camera_frames.idl`.
- Topics: `rt/kist/latent_action` (50 Hz stream), `rt/kist/wbc_command`
  (reserved — operator channel, no subscriber yet).
- QoS: writers are Reliable + KeepLast(1) — "latest wins" like the ZMQ
  CONFLATE pair, but Reliable so they match both Reliable and BestEffort
  readers (gearsonic subscribes via unitree's ChannelSubscriber, whose
  reader QoS we don't control); commands Reliable + KeepLast(8); our own
  state/camera readers BestEffort + KeepLast(1).
- State wire semantics were verified against the reference deploy
  (`zmq_output_handler.hpp`): `body_q` = 29 absolute joint angles in Unitree
  motor order; `base_quat` = **pelvis** IMU quaternion (w,x,y,z) from
  LowState — not the torso IMU on `rt/secondary_imu`.
- Remaining hardware verification: Dex3 hand motor order and IMU quaternion
  convention against the real robot / final data-collection pipeline.

### Wire contract — ZMQ (latent protocol v4)

Frozen wire format shared with the reference stack; see
`kist_vla/protocol.py` and `tests/test_protocol.py` for the byte-level pin.

| Direction | Port | Content |
|---|---|---|
| → gearsonic | 5556 (PUB, `pose` topic) | `token_state[1,64]` f32, `frame_index[1]` i64, `left/right_hand_joints[1,7]` f32, 50 Hz |
| → gearsonic | 5556 (PUB, `command` topic) | control-loop start/stop/planner flags |
| ← gearsonic | 5557 (SUB, `g1_debug` topic) | msgpack: `body_q[29]`, `left/right_hand_q[7]`, `base_quat[4]` (wxyz) |
| ← camera server | 5555 (SUB) | msgpack: `{timestamps, images}` (JPEG), key `ego_view` (+ optional wrists) |
| ← operator | 5580 (SUB) | keystrokes / `prompt:<text>` |
| ↔ policy server | 5550 (REQ/REP) | remote mode only (Isaac-GR00T PolicyServer) |

Message layout: `[topic][1280-byte NUL-padded JSON header][little-endian binary fields]`.

### Provenance

Core loop and utilities are ported from NVIDIA
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
(`gear_sonic/scripts/run_vla_inference.py` and friends, Apache-2.0), with the
pinocchio robot model replaced by static joint tables
(`kist_vla/g1_joints.py`, dumped from the reference model — re-dump, don't
hand-edit). `gr00t` is a dependency; `gear_sonic` is not.

## Dependencies

| Component | Version | Role |
|---|---|---|
| Python | 3.12 (package itself runs on ≥ 3.10) | `gr00t` requires 3.12 for the local policy backend |
| numpy, scipy, pyzmq, msgpack(-numpy), opencv-python-headless, tyro | PyPI | core runtime (installed automatically) |
| `gr00t` (Isaac-GR00T) | local clone, pinned commit — see below | N1.7 policy — local mode only; remote mode and tests run without it |
| `cyclonedds`, `av` | PyPI, `[dds]` extra | DDS transports + H.264 camera decode |
| `unitree_sdk2py` | local clone (`--no-deps`) | unitree DDS IDL types for the state source |
| N1.7 checkpoint | UNITREE_G1_SONIC finetune | **hard input** — the base `nvidia/GR00T-N1.7-3B` has no SONIC action head; see the [finetuning workflow](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/GR00TWholeBodyControl) |
| `pytest` | `[dev]` extra | tests |

Checkpoint couplings:

- **The `gr00t` commit is part of the checkpoint contract.** A checkpoint's
  `config.json` is a *delta* against the code's defaults, not a full spec —
  keys it omits (`input_embedding_dim`, `state_history_length`,
  `attend_text_every_n_blocks`, ...) come from
  `gr00t/configs/model/gr00t_n1d7.py`, so the assembled architecture is
  `config.json` + that version of the code. A renamed module makes
  `AutoModel.from_pretrained` (HF-default non-strict) randomly initialize the
  affected weights behind a warning: the policy loads, runs, and emits
  garbage motion tokens. The expected commit lives in
  `EXPECTED_GR00T_COMMIT` (`kist_vla/gr00t_version.py`) and is checked on
  every `create_policy` call; update it only together with the checkpoint.
- `DEFAULT_INITIAL_MOTION_TOKEN` (`kist_vla/config.py`) is specific to the
  SONIC checkpoint used in training — must be re-derived if the gearsonic
  SONIC checkpoint changes.
- The normalization statistics used to decode actions live inside the N1.7
  checkpoint's processor and change with every finetune (handled by `gr00t`
  automatically — nothing to copy here).

## Installation

#### 1. Clone repository

```bash
git clone https://github.com/Safety-Node/kist-vla-inference.git
cd kist-vla-inference
```

#### 2. Create the virtualenv

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

#### 3. DDS transports (real robot)

```bash
uv pip install -e ".[dds]"
# unitree IDL types for the state source (--no-deps: it pins an old
# cyclonedds and pulls opencv-python; its IDL types work fine with current
# cyclonedds and opencv-python-headless)
uv pip install --no-deps -e ~/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python
```

#### 4. Local policy backend (GPU box only)

`gr00t` is not on PyPI — install from a local clone, checked out at the
commit the checkpoint was finetuned with
(`EXPECTED_GR00T_COMMIT` in `kist_vla/gr00t_version.py`):

```bash
git clone https://github.com/foodbanana/Isaac-GR00T.git ~/Isaac-GR00T
# detached on purpose: gr00t is installed editable, so a `git pull` in this
# clone changes the running model with no reinstall
git -C ~/Isaac-GR00T checkout 5ac4e6b6ad7467f4ccd441f6d7ec574d4da0a21f
uv pip install -e ~/Isaac-GR00T
```

That commit is a KIST fork of NVIDIA's `9c7e746`; NVIDIA's `main` is **not** a
superset of it (see `kist_vla/gr00t_version.py`). The install pulls
torch 2.9.0+cu128 and a prebuilt flash-attn 2.8.3 cp312 wheel from the URL the
clone's `[tool.uv.sources]` names — no CUDA source build, but Python must be
3.12.

The VLM backbone (`nvidia/Cosmos-Reason2-2B`) is a **gated** HF repo and is
fetched from Hugging Face even when `--policy.model-path` is a local
directory; without a token in `HF_HOME` the policy dies with a 401.

Skip this whole step on the robot side when using `--policy.mode remote`.

## Build

No build step — pure Python, no codegen (the DDS types in
`kist_vla/io/dds.py` mirror `idl/kist_latent_action.idl` by hand; keep them
in sync when the IDL changes). Verify the install with the test suite,
which runs without a GPU, model, or robot:

```bash
pytest
```

This includes the **L1 loopback harness** (`tests/test_loopback.py`): the
real runner over real ZMQ against fake sensors and a fake policy, verifying
the 50 Hz stream, latency compensation, frame ordering, prompt changes, and
control commands. DDS tests run when `cyclonedds` is installed and are
skipped otherwise.

## Usage

**Tokens drive the robot** — with a live gearsonic on the DDS domain, a
fresh token stream switches it to external-token mode and the robot moves.
Clear the area and keep the e-stop (VR controller) in reach.

```bash
# Single process (model in-process, default)
python scripts/run_vla.py --policy.model-path /path/to/checkpoint-XXXX

# Or with the model on another machine:
python scripts/run_policy_server.py --model-path /path/to/checkpoint-XXXX   # GPU box
python scripts/run_vla.py --policy.mode remote --policy.host <gpu-box>      # robot side

# Operator console (separate terminal)
python scripts/keyboard_publisher.py
```

`--help` shows all options (ports, rates, prompt, action bound).

#### Transport selection

Real robot (DDS for all three channels):

```bash
python scripts/run_vla.py --policy.mode remote --policy.host <gpu-box> \
    --io.action-transport dds --io.camera-transport dds --io.state-transport dds
```

`--io.dds-domain-id` sets the domain (default 0); `--io.dds-camera-topic`
maps a kist-ext-sensor-io color stream to the `ego_view` observation.

#### Operator keys

`p` pause/resume · `k` start/stop gearsonic loop · `i` initial pose ·
`[` `]` toggle hands · `t` change prompt (`prompt:<text>` on the wire).

#### Hardware link check (no checkpoint needed)

Publishes the safe standing token at 50 Hz over DDS — verifies the
token → decoder → motors path against a live gearsonic without a model:

```bash
python scripts/publish_test_tokens.py --duration 15
```
