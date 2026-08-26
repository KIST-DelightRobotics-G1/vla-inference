# kist-vla-inference

Python replay publisher streaming SONIC motion tokens to
[kist-gearsonic-inference](https://github.com/Safety-Node/kist-gearsonic-inference)
on the Unitree G1 humanoid robot.

## Architecture

[![Architecture](docs/kist-vla-inference.svg)](docs/kist-vla-inference.svg)

## Dependencies

| Component | Version | Role |
|---|---|---|
| Python | ≥ 3.10 (docker image: 3.12) | runtime |
| numpy, tyro, pyyaml | PyPI | core runtime (installed automatically) |
| `cyclonedds` | PyPI, `[dds]` extra | DDS publisher (the wheel bundles libddsc — no system install) |
| `pyarrow` | PyPI, `[parquet]` extra | LeRobot training-export episodes |
| `onnxruntime` | PyPI, `[encode]` extra | joint re-encoding (CPU provider — no GPU) |
| `pytest` | `[dev]` extra | tests |
| GEAR-SONIC encoder | HF `nvidia/GEAR-SONIC` | `models/model_encoder.onnx` for `--joints`; pairs with gearsonic's decoder — swap them together |

`requirements.lock.txt` records the exact third-party set this was verified
with (a record, not the install path).

## Installation

#### 1. Clone repository

```bash
git clone https://github.com/Safety-Node/kist-vla-inference.git
cd kist-vla-inference
```

All following steps run from the repository root.

#### Quick Start with Docker

The image bakes in everything below (dependencies, the encoder model, the
repo source) and verifies the import graph + replay test suite at build
time:

```bash
./docker/build.sh      # builds the image (docker build -t kist-vla-inference)
./docker/run.sh        # shell in the container; ready to publish
```

`run.sh` wires `--network host` (CycloneDDS discovery/multicast toward
gearsonic) and mounts `<repo>/shared` for sessions and datasets. The
numbered steps below (2–3) are the manual (non-Docker) alternative.

#### 2. Create the virtualenv

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dds,parquet,encode,dev]"
```

#### 3. Download the encoder model

```bash
wget -P models https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/model_encoder.onnx
```

## Usage

Set up the config once before running:

- `config/config.yaml` — DDS domain (`dds.domain_id`, 0 = real robot) and
  NIC (`dds.network_interface`, must match the gearsonic side).
- All keys and the replay CLI parameters:
  [docs/configuration.md](docs/configuration.md).

**THE ROBOT MOVES** — with a live gearsonic on the DDS domain, a fresh
token stream switches it to external-token mode. Hang the robot or clear
the area, keep the VR controller in reach (A+B+X+Y held 1s = emergency
stop).

#### Session replay

Replays a [kist-data-collector](https://github.com/Safety-Node/kist-data-collector)
session (`motion_token.csv` + `hand_cmd_{side}.csv`) or a LeRobot
training-export episode — both carry the same quantities:

```bash
# Collector session directory:
python scripts/replay_session.py --path <session-dir>

# LeRobot dataset root + episode index, or the parquet file directly:
python scripts/replay_session.py --path <dataset-dir> --episode 3
python scripts/replay_session.py --path <dataset-dir>/data/chunk-000/episode_000003.parquet
```

#### Joint re-encoding

The recorded tokens are only valid against the collection-time SONIC
checkpoint; `--joints` re-encodes the recording's joints instead, so a
session survives a checkpoint change:

```bash
python scripts/replay_session.py --path <dataset-dir> --episode 1 --joints
```
