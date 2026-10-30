# Deploying to the target

Target from the brief: a clean **Ubuntu 24.04 LTS** laptop, four cores, 8 GB, no discrete
GPU, integrated or USB camera, mic and speaker through standard Linux audio, wifi.

Development was on Windows. Nothing in the system needs Windows, and the two places that
would have broken on Linux are called out below rather than left to be discovered.

## Install and run

```bash
sudo ./deploy/install.sh          # apt as root, then uv, the virtualenv and every model as you
export ANTHROPIC_API_KEY=...      # optional, see "running without the network"
./deploy/run.sh                   # body with the viewer + brain with the debug window
HEADLESS=1 ./deploy/run.sh        # same, no windows
```

The installer is idempotent. It installs the apt packages as root, then drops back to the
invoking user, so the virtualenv and the models are not owned by root. Every model is
pinned to a version and checked against a sha256 by `tools/fetch_models.py`, and all of
them are fetched at install time into `models/`, so nothing downloads during a demo:

| Model | Licence | Size | Used for |
|---|---|---|---|
| RetinaFace MobileNetV2 (UniFace) | MIT | 12 MB | finding faces, including turned ones |
| MobileGaze MobileOne-S0 (UniFace) | MIT | 5 MB | eye gaze, the "looking at me" test |
| MediaPipe pose landmarker lite | Apache-2.0 | 6 MB | a body, when the face is hidden |
| Silero VAD (onnx, pinned revision) | MIT | 1 MB | speech versus typing, knocks, music |
| NVIDIA Parakeet TDT 0.6B v2 int8 (onnx-asr) | CC-BY-4.0 model, MIT runtime | 631 MB | transcription |
| faster-whisper base.en int8 | MIT | 141 MB | transcription fallback |
| Kokoro-82M (kokoro-onnx) | Apache-2.0 model, MIT runtime | 354 MB | the voice |
| Piper lessac medium | MIT voice, GPL-3.0 runtime | 61 MB | voice fallback |

### Verified on Ubuntu 24.04

A clean copy of the repository (no virtualenv, no models) was installed with
`deploy/install.sh` on Ubuntu 24.04.3 LTS under WSL2. The installer completed, every
checksum matched, every model loaded and ran on Linux, `tools/selftest.py` held 30.1 Hz
telemetry with 0.0003 rad aim error, and `tools/dryrun.py` passed all checks. WSL has no
camera or microphone, so those two were not exercised there; the installer's device check
reports them on real hardware.

Or install the units:

```bash
systemctl --user enable --now lamp-body lamp-brain
journalctl --user -u lamp-brain -f
```

User units, not system units, on purpose: the character needs the session's audio and
video devices, and a user unit inherits the seat's PipeWire socket and device ACLs
without being granted anything.

## What the apt packages are for

| Package | Why |
|---|---|
| `libportaudio2` | sounddevice, which is both the mic and the speaker |
| `libglib2.0-0` | OpenCV, for capture and the debug window |
| `libgl1`, `libglx-mesa0`, `libegl1`, X client libs | MuJoCo's viewer and the debug window. skipped with `WANT_VIEWER=0` |
| `v4l-utils` | `v4l2-ctl --list-devices`, for finding the camera |
| `libgomp1` | OpenMP, used by CTranslate2 under faster-whisper |

The body renders nothing in `--headless` mode, so a deployment that needs no visible
window needs no GL stack at all. That is the mode the units use.

## The two portability points

**Camera backend.** OpenCV exposes `cv2.CAP_DSHOW` as a constant on every platform, so
testing whether the attribute exists picks a Windows-only backend on Linux. `vision.py`
tests `sys.platform` and uses `cv2.CAP_V4L2` on the target.

**Audio device naming.** PortAudio on Ubuntu 24.04 reaches PipeWire through its ALSA
compatibility layer, and the default device is usually right. When it is not:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

then pass the index into `Ears(device=...)` / `AudioOut(device=...)`, or set
`SD_DEFAULT_DEVICE`. The user must be in the `audio` and `video` groups, which the
installer checks.

## Resource budget on four cores and 8 GB

Measured per process on the dev machine, with the target's limits alongside. Re-run
`tools/measure.py` on the target to replace these.

| Process | Steady CPU | Resident | Notes |
|---|---|---|---|
| body | see `run/measurements.txt` | ~200 MB | physics at 500 Hz, control at 50 Hz, telemetry at 30 Hz |
| brain, idle | mostly the camera | ~900 MB-1.1 GB | whisper int8 dominates; the landmarker runs at 15 Hz, not camera rate |

Deliberate choices that buy headroom on four cores:

- The camera stack runs at **10 Hz**, with gaze every second frame and the body model every
  third. Attention does not need camera rate: the 0.4 s notice dwell is still four frames.
  Measured live, the whole stack costs about 48 ms per frame, half of one core.
- Transcription runs **once per utterance**, not streaming. A partial sentence is of no
  use to the director, and streaming would double the transcription work.
- The language and vision model is **remote, one call per spoken exchange** (plus one
  re-check when a plan has to verify a goal), so no VLM or LLM sits in memory. Running both locally alongside whisper and piper is
  what would actually breach 8 GB.
- Audio is one output stream with a software mixer, so there is one PortAudio callback
  rather than three players competing for the device.

A GNOME session itself costs 1.2-1.8 GB. A minimal or headless session is worth about a
gigabyte if the deployment is a fixture rather than a desktop.

## Running without the network

Unset `ANTHROPIC_API_KEY`. Everything that reacts stays: it notices, follows and loses
interest in people, hears and transcribes, speaks, sleeps and wakes. What it says comes
from local intent routing (greetings, light, recall, goals it recognises) instead of the
model, and it cannot name objects it has not been told about, so it asks instead. Nothing
times out waiting for a network that is not there.

## What is sent when the key is set

Only when someone speaks to the lamp: the transcript of what they said, one JPEG of the
camera frame at that moment, the last four exchanges as text, and the short list of labels
already in memory. No audio ever leaves the machine, no continuous video, and nothing is
sent while the character is idling, engaging, following or sleeping.

## Toward the real hardware

The supplied URDF is a fixture, but the body layer is written against the shape of a real
servo arm rather than against the simulator:

- The reference generator is capped at the URDF **velocity** limits, and the servos are
  capped at the URDF **effort** limits, so the two limits a real joint has are both
  enforced and saturation is reported rather than hidden.
- Gravity feedforward is `qfrc_bias / kp` per joint, which is what a real position-servo
  arm does; on hardware the term comes from a model or a lookup rather than from MuJoCo.
- The light is an intensity and a warmth, not a MuJoCo light object. On a real lamp head
  with an LED ring, the same two numbers drive the ring, and the mapping is the only
  thing that changes.
- Posture presets avoid the arm pose measured to saturate shoulder and elbow torque
  (leaning forward with the elbow folded sags 0.2 rad), which is a hardware limit the
  simulator merely happens to reproduce.
