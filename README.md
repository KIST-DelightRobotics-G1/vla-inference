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
hand-edit). `gear_sonic` is not a dependency. The GR00T N1.7 inference path
itself is vendored from NVIDIA [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
into `thirdparty/gr00t/` (Apache-2.0) — see `thirdparty/gr00t/VENDORED_FROM.md`.

## Dependencies

| Component | Version | Role |
|---|---|---|
| Python | 3.12 | the pinned torch/flash-attn wheels are cp312 |
| numpy, scipy, pyzmq, msgpack(-numpy), opencv-python-headless, tyro | PyPI, pinned | core runtime |
| torch 2.9.0+cu128, torchvision, transformers 4.57.3, diffusers, albumentations, pillow, dm-tree, huggingface-hub | PyPI + cu128 index, pinned | what `thirdparty/gr00t` imports — only the inference path, not gr00t's training/export stack |
| `thirdparty/gr00t` | vendored, NVIDIA/Isaac-GR00T `9c7e746` | N1.7 policy (20 files); re-vendor with `scripts/vendor_gr00t.sh` |
| flash-attn 2.8.3 | `[flash]` extra, prebuilt wheel | optional; sdpa fallback without it |
| `cyclonedds` **0.10.2** (C lib + python binding), `av` | `[dds]` extra; C lib built from source (step 3) | DDS transports + H.264 camera decode. **Not the PyPI default 11.x** — see below |
| `unitree_sdk2py` | upstream clone pinned to `65691c8` (`--no-deps`) | unitree DDS IDL types for the state source |
| N1.7 checkpoint | UNITREE_G1_SONIC finetune | **hard input** — the base `nvidia/GR00T-N1.7-3B` has no SONIC action head; see the [finetuning workflow](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/GR00TWholeBodyControl) |
| `pytest` | `[dev]` extra | tests |

Checkpoint couplings:

- **The GR00T code version is part of the checkpoint contract.** A
  checkpoint's `config.json` is a *delta* against the code's defaults, not a
  full spec — keys it omits (`input_embedding_dim`, `state_history_length`,
  `attend_text_every_n_blocks`, ...) come from
  `thirdparty/gr00t/configs/model/gr00t_n1d7.py`, so the assembled
  architecture is `config.json` + that version of the code. A renamed module
  makes `AutoModel.from_pretrained` (HF-default non-strict) randomly
  initialize the affected weights behind a warning: the policy loads, runs,
  and emits garbage motion tokens. The vendored files are therefore taken
  from the Isaac-GR00T commit the checkpoint was finetuned with
  (`thirdparty/gr00t/VENDORED_FROM.md`); re-vendor only together with the
  checkpoint, and run `scripts/smoke_test_policy.py` afterwards — its
  weight-coverage check is what catches this.
- `DEFAULT_INITIAL_MOTION_TOKEN` (`kist_vla/config.py`) is specific to the
  SONIC checkpoint used in training — must be re-derived if the gearsonic
  SONIC checkpoint changes.
- The normalization statistics used to decode actions live inside the N1.7
  checkpoint's processor and change with every finetune (handled by the
  processor code automatically — nothing to copy here).

## Installation

#### 1. Clone repository

```bash
git clone https://github.com/Safety-Node/kist-vla-inference.git
cd kist-vla-inference
```

#### 2. Create the virtualenv and install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[dev,flash]"
```

This installs everything, including the GR00T inference path — it is vendored
in `thirdparty/gr00t/`, so there is no separate clone. The cu128 extra index
supplies the CUDA builds of torch/torchvision; `unsafe-best-match` lets uv
pick versions across both indexes. Drop `flash` on a machine without a
matching wheel (the backbone falls back to sdpa attention). On the robot side
(`--policy.mode remote`) the same install works; torch is only exercised in
local mode.

#### Quick start with Docker

The image bakes in steps 2–3 (venv, all Python deps, CycloneDDS, `unitree_sdk2py`) on
`nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`:

```bash
./docker/build.sh      # docker build -f docker/Dockerfile .  (context = repo root)
./docker/run.sh        # shell in the container; re-attaches if it already exists
```

Host prerequisites the image cannot supply: an x86_64 box with an NVIDIA
driver (verified with 550 on an RTX 3090) and `nvidia-container-toolkit`
(`docker run --gpus all`); ~20 GB for the image plus network access to
GitHub, PyPI, the PyTorch cu128 index and Hugging Face during the build;
and, at run time, the checkpoint under `~/vla_data` and a Hugging Face
token + cached `nvidia/Cosmos-Reason2-2B` under `~/hf_cache` (request
access to the gated repo first). gearsonic / ext-sensor-io run as their own
containers on the same host network and NIC.

`run.sh` wires `--gpus all`, `--network host` (DDS/ZMQ share the host
network with gearsonic and ext-sensor-io), mounts `~/vla_data` at
`/vla_data` and `~/hf_cache` at `/hf_cache` (`HF_HOME` — token + the gated
backbone; override with `VLA_DATA=…`/`HF_CACHE=…`). `DEV=1 ./docker/run.sh`
bind-mounts the repo over `/workspace` to edit code without a rebuild.
Inside, start with `python scripts/smoke_test_policy.py --model-path
/vla_data/checkpoint-18000`. Steps 4–5 below are host-side / manual.

#### 3. DDS transports (real robot)

CycloneDDS must be **0.10.2**, the version gearsonic and ext-sensor-io are
pinned to. PyPI's default `cyclonedds` is 11.x (libddsc 0.11 bundled) and the
XTypes discovery format differs: the moment a 0.11 Python *reader* joins the
domain, every 0.10.2 C++ participant segfaults (`ddsi_xt_type_init_impl`) —
one `run_vla.py` with DDS state/camera kills gearsonic and ext-sensor-io at
once (reproduced 2026-08-19; a 0.11 *writer* alone works, so the token-publish
link check does not reveal it). 0.10.2 has no cp312 wheel, so `pip` compiles
the binding against a C library you build first — same recipe as the
gearsonic README:

```bash
sudo apt install -y cmake build-essential python3.12-dev     # python3.12-dev: headers for the binding
git clone --depth 1 -b 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds
cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DCMAKE_BUILD_TYPE=Release
sudo cmake --build /tmp/cyclonedds/build --target install -j"$(nproc)"

CYCLONEDDS_HOME=/opt/cyclonedds uv pip install -e ".[dds]"   # compiles cyclonedds==0.10.2 (pyproject pin) + av
```

The version is written once, in `pyproject.toml`'s `[dds]` extra; the lock
and the Dockerfile read it from there. `unitree_sdk2py` (DDS IDL types for
the state source) is not on PyPI — clone upstream at the pinned commit:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ~/unitree_sdk2_python
git -C ~/unitree_sdk2_python checkout 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
# --no-deps: it pulls opencv-python (we use -headless); its cyclonedds==0.10.2 pin is what you just built
uv pip install --no-deps -e ~/unitree_sdk2_python
```

#### 4. Hugging Face token (GPU box only)

The VLM backbone (`nvidia/Cosmos-Reason2-2B`) is a **gated** HF repo and is
fetched from Hugging Face even when `--policy.model-path` is a local
directory; without a token in `HF_HOME` the policy dies with a 401. Request
access on Hugging Face, then `hf auth login` with `HF_HOME` set as in
`scripts/env.sh`.

#### 5. Runtime environment

```bash
source scripts/env.sh
```

Sets `HF_HOME`, drops out of conda, and clears `PYTHONPATH` — a globally
sourced ROS 2 puts Python 3.10 site-packages there, which shadows imports
inside the 3.12 venv (`pytest` dies on `No module named 'yaml'`). This package
does not use ROS; the `rt/*` topic names are unitree's DDS naming convention.

`requirements.lock.txt` records the exact third-party set this was verified
with (`uv pip freeze` of a fresh install). It is a record, not the install
path — `pyproject.toml` pins the direct dependencies and reproduces it.

#### Updating the vendored GR00T code

Only when a new checkpoint was finetuned with a different Isaac-GR00T commit:

```bash
bash scripts/vendor_gr00t.sh <nvidia-commit>   # re-copies the 20 files, re-applies patches/
git diff --stat thirdparty/gr00t               # review what upstream changed
python scripts/smoke_test_policy.py --model-path <new-checkpoint>
```

`thirdparty/gr00t/VENDORED_FROM.md` explains which files may be edited
locally (I/O, validation, logging — add a patch) and which must stay
byte-identical to training (model, processor, config defaults).

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
