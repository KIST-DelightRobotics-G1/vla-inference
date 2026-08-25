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
                 │      Tx thread (50 Hz publish loop)           │
                 │      latency-compensated chunk playback       │
                 └──────│───────────────────────────────────────-┘
                        ▼ latent actions, 50 Hz
                 ┌──────────────────────────────────────────────┐
                 │  kist-gearsonic-inference (C++, 50 Hz RT)     │
                 │  token[64] → PolicyDecoder(TRT) → motors      │
                 └──────────────────────────────────────────────┘
```

Single process, three threads:

- **Main thread** — lifecycle only: wires the other two and waits.
- **Tx thread** (owned by the publisher) — 50 Hz loop: consume finished
  chunks, trigger inference, play the cached chunk back one step per tick,
  publish each step. Never blocks; while inference runs it keeps playing
  the previous chunk, and a stale chunk degrades to holding its last token.
- **Inference worker** — assembles the observation from the latest sensor
  data and runs the policy (~0.4 s per chunk at 2.5 Hz).

Every cross-boundary handoff keeps only the latest value (KeepLast(1)
readers, maxsize-1 queues) — consumers see fresh data, never a backlog.

### Transports

All robot-facing I/O rides CycloneDDS (`src/common/cyclonedds/`); the runner
core talks to it through the Protocols in `src/common/interfaces.py`, so
tests inject fakes without touching the loop:

| Channel | DDS implementation |
|---|---|
| actions → gearsonic | `kist_msgs::LatentActionStep` / `WbcCommand` (`src/common/cyclonedds/kist_msgs.py + kist_msgs_writer.py`) |
| camera | kist-ext-sensor-io `CompressedColorFrame`, H.264 → PyAV decode (`src/vla/camera_source.py`) |
| robot state | unitree `rt/lowstate` + `rt/dex3/{left,right}/state` directly (`src/vla/state_source.py`) — no re-publisher process needed |

The only non-DDS endpoint is the remote policy server (port 5550,
Isaac-GR00T's own PolicyServer/PolicyClient pair, `--policy.mode remote`).

### Wire contract — DDS

- **`idl/kist_latent_action.idl` is the shared action contract** — the
  gearsonic C++ side codegens from it (`idlc -l cxx`); the Python side
  mirrors it in `src/common/cyclonedds/kist_msgs.py`; keep the two in sync. The camera type
  mirrors kist-ext-sensor-io's `idl/kist_camera_frames.idl`.
- Topics: `rt/kist/latent_action` (50 Hz stream), `rt/kist/wbc_command`
  (reserved — operator channel, no subscriber yet).
- QoS: writers are Reliable + KeepLast(1) — "latest wins", and Reliable so
  they match both Reliable and BestEffort
  readers (gearsonic subscribes via unitree's ChannelSubscriber, whose
  reader QoS we don't control); commands Reliable + KeepLast(8); our own
  state/camera readers BestEffort + KeepLast(1).
- State wire semantics were verified against the reference deploy
  (`zmq_output_handler.hpp`): `body_q` = 29 absolute joint angles in Unitree
  motor order; `base_quat` = **pelvis** IMU quaternion (w,x,y,z) from
  LowState — not the torso IMU on `rt/secondary_imu`.
- Remaining hardware verification: Dex3 hand motor order and IMU quaternion
  convention against the real robot / final data-collection pipeline.

### Provenance

Core loop and utilities are ported from NVIDIA
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
(`gear_sonic/scripts/run_vla_inference.py` and friends, Apache-2.0), with the
pinocchio robot model replaced by static joint tables
(`src/common/g1_joints.py`, dumped from the reference model — re-dump, don't
hand-edit). `gr00t` is a dependency; `gear_sonic` is not.

## Dependencies

| Component | Version | Role |
|---|---|---|
| Python | 3.12 (package itself runs on ≥ 3.10) | `gr00t` requires 3.12 for the local policy backend |
| numpy, scipy, opencv-python-headless, tyro, pyyaml | PyPI | core runtime (installed automatically) |
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
  `EXPECTED_GR00T_COMMIT` (`src/vla/gr00t_version.py`) and is checked on
  every `create_policy` call; update it only together with the checkpoint.
- `DEFAULT_INITIAL_MOTION_TOKEN` (`src/common/config.py`) is specific to the
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
(`EXPECTED_GR00T_COMMIT` in `src/vla/gr00t_version.py`):

```bash
bash scripts/install_gr00t.sh
```

That clones the fork, checks it out **detached** at the pinned commit, and
installs it. It reads the commit from `src/vla/gr00t_version.py` so there is
one authority for it; `GR00T_SRC` and `GR00T_REMOTE` override the location.
Detached is deliberate — gr00t is installed editable, so leaving the clone on
a tracking branch means a later `git pull` changes the running model with no
reinstall. The equivalent by hand:

```bash
git clone https://github.com/foodbanana/Isaac-GR00T.git ~/Isaac-GR00T
git -C ~/Isaac-GR00T checkout 5ac4e6b6ad7467f4ccd441f6d7ec574d4da0a21f
uv pip install -e ~/Isaac-GR00T
```

That commit is a KIST fork of NVIDIA's `9c7e746`; NVIDIA's `main` is **not** a
superset of it (see `src/vla/gr00t_version.py`). The install pulls
torch 2.9.0+cu128 and a prebuilt flash-attn 2.8.3 cp312 wheel from the URL the
clone's `[tool.uv.sources]` names — no CUDA source build, but Python must be
3.12.

The VLM backbone (`nvidia/Cosmos-Reason2-2B`) is a **gated** HF repo and is
fetched from Hugging Face even when `--policy.model-path` is a local
directory; without a token in `HF_HOME` the policy dies with a 401.

Skip this whole step on the robot side when using `--policy.mode remote`.

#### 5. Runtime environment

```bash
source scripts/env.sh
```

Sets `HF_HOME`, drops out of conda, and clears `PYTHONPATH` — a globally
sourced ROS 2 puts Python 3.10 site-packages there, which shadows imports
inside the 3.12 venv (`pytest` dies on `No module named 'yaml'`). This package
does not use ROS; the `rt/*` topic names are unitree's DDS naming convention.

`requirements.lock.txt` records the exact third-party set this was verified
with. It is a record, not the install path — `install_gr00t.sh` reproduces it,
since gr00t's own pyproject pins nearly everything with `==`.

## Build

No build step — pure Python, no codegen (the DDS types in
`src/common/cyclonedds/kist_msgs.py` mirror `idl/kist_latent_action.idl` by hand; keep them
in sync when the IDL changes). Verify the install with the test suite,
which runs without a GPU, model, or robot:

```bash
pytest
```

DDS tests run when `cyclonedds` is installed and are skipped otherwise;
parquet-replay tests likewise skip without `pyarrow`.

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
```

`--help` shows all options (rates, prompt, action bound).
`--io.dds-domain-id` sets the domain (default 0); `--io.dds-camera-topic`
maps a kist-ext-sensor-io color stream to the `ego_view` observation.

Without an operator source the loop starts unpaused and the session
lifecycle rides the data plane: publishing fresh tokens claims gearsonic,
going stale releases it. (The runner still accepts an injected
OperatorSource — `p` pause · `k` loop start/stop · `i` initial pose ·
`[` `]` hands · `prompt:<text>` — the operator channel over DDS is the
reserved `WbcCommand` topic.)

#### Hardware link check (no checkpoint needed)

Publishes the safe standing token at 50 Hz over DDS — verifies the
token → decoder → motors path against a live gearsonic without a model:

```bash
python scripts/publish_test_tokens.py --duration 15
```

#### Session replay (no checkpoint needed)

Replays a [kist-data-collector](https://github.com/Safety-Node/kist-data-collector)
session on the robot: its recorded `motion_token.csv` is a copy of the token
gearsonic's decoder consumed on each CONTROL tick, so publishing it back on
`rt/kist/latent_action` at 50 Hz drives the whole body through the same latent
trajectory — hand targets come from `hand_cmd_{side}.csv`. The decoder stays
closed-loop on live robot state, so the robot balances itself; this is a latent
replay, not an open-loop joint playback.

```bash
# Against the gearsonic probe (./build/vla_receiver_probe 42)
python scripts/replay_session.py --session <session-dir> --domain 42

# On the real robot — ROBOT MOVES, hang it first
python scripts/replay_session.py --session <session-dir> --domain 0
```

DDS network settings live in `config/config.yaml` (gearsonic-style:
`dds.domain_id` + `dds.network_interface`, the latter applied via
`CYCLONEDDS_URI`); `--domain` overrides the file's domain id.

The published stream is bracketed — standing lead-in, crossfade, replay,
crossfade, standing lead-out — so gearsonic claims VLA from a known pose and
the episode does not end mid-motion. `motion_token.csv` rows exist only for
ticks that decoded a token, so `seq`/`stamp_ns` gaps (INIT ramp, damping,
e-stop) are resampled onto a strict 20 ms grid and blended across; a gap longer
than `--max-gap-ticks` (0.5 s) is reported and aborts the run unless `--force`
is given. `--teleop-only` restricts the replay to `arbiter_mode == 1`, the
segments the training export keeps. The implementation is the
`src/replay/` package (also runnable as `python -m replay`):
session loading is pure data handling with no DDS — `tests/test_replay.py`
pins it against the collector's CSV schemas — and only `cli.py` touches the
wire.

A LeRobot training-export episode replays the same way (needs `pyarrow`,
`uv pip install -e ".[parquet]"`) — its `action.motion_token` /
`teleop.*_hand_joints` columns carry the same quantities the collector CSVs
record:

```bash
# By dataset root + episode index, or by the parquet file directly:
python scripts/replay_session.py --session <dataset-dir> --episode 3 --domain 42
python scripts/replay_session.py --session <dataset-dir>/data/chunk-000/episode_000003.parquet --domain 42
```

**The recorded tokens are latents of the SONIC checkpoint that was running when
the session was collected.** Replaying them against a gearsonic built on a
different SONIC decoder checkpoint produces a different, possibly unsafe motion
— the two latent spaces are not comparable. Same coupling as
`DEFAULT_INITIAL_MOTION_TOKEN`.
