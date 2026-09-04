# Configuration

All publishers read `config/config.yaml` (`--config` overrides the path; a
missing file at the default path falls back to the built-in defaults below).
Edit it per deployment — on Docker, edit it inside the container, the image
bakes it in. Loaded by `src/common/cyclonedds/config.py`.

## `dds`

| Key | Default | Meaning |
|---|---|---|
| `domain_id` | `0` | DDS domain — must match the gearsonic receiver (`unitree.domain_id` on its side). `--domain` overrides it per run |
| `cyclonedds_xml` | `config/cyclonedds.xml` | CycloneDDS transport config file — the network interface AND the tuning live there |

`config/cyclonedds.xml` (same convention as kist-ext-sensor-io) carries:

- the NIC toward the DDS peers (`NetworkInterface name=...` — must be the
  interface gearsonic/ext-sensor-io use; `lo` for same-machine work, but
  multicast discovery on `lo` has failed before on this stack)
- 16MiB socket buffers (H.264 bursts overflow the kernel defaults — also
  raise `net.core.rmem_max`/`wmem_max` on the HOST, see the file's comment)
- MTU-sized datagrams (`FragmentSize`/`MaxMessageSize`, parity with the
  C++ peers)
- a commented `Tracing` block for diagnosing what actually applied

It is routed to CycloneDDS via `CYCLONEDDS_URI`; an already-set
`CYCLONEDDS_URI` environment variable wins over the config.

## Replay parameters (CLI, not YAML)

The stream-shaping and safety parameters are per-run concerns, so they are
CLI flags on `scripts/replay_session.py` rather than YAML keys
(`src/replay/cli.py` `Config` is the authority — `--help` shows all):

| Flag | Default | Meaning |
|---|---|---|
| `--lead-in-s` | `1.5` | seconds of safe standing token before the replay, so gearsonic's arbiter claims VLA (200 ms freshness) from a known pose |
| `--lead-out-s` | `1.5` | seconds of standing token after the replay, so the stream does not end mid-pose (which would trigger gearsonic's LOST recovery) |
| `--blend-s` | `0.7` | crossfade standing ↔ the recording's first/last token; matches gearsonic's own handoff blend (`ControlArbiter::kHandoffBlendTicks`) |
| `--max-gap-ticks` | `25` (0.5 s) | longest recording gap blended across; a longer gap is compressed and the run refuses to start unless `--force` is given |
| `--force` | off | publish even when a gap had to be compressed (the pose change ramps faster than recorded) |

## Fixed paths (convention, not configuration)

| Path | Meaning |
|---|---|
| `models/model_encoder.onnx` | GEAR-SONIC encoder for `--joints` — same `models/` convention as kist-gearsonic-inference; swap it together with gearsonic's decoder checkpoint |
| `shared/` | host↔container exchange dir (mounted by `docker/run.sh`): collector sessions, LeRobot exports |
