# Deploying to the target

Target from the brief: a clean **Ubuntu 24.04 LTS** laptop, four cores, 8 GB, no discrete
GPU, integrated or USB camera, mic and speaker through standard Linux audio, wifi.

Development was on Windows. Nothing in the system needs Windows, and the two places that
would have broken on Linux are called out below rather than left to be discovered.

## Install

```bash
sudo ./deploy/install.sh      # apt packages, uv, the virtualenv, the models
```

Then either run the two processes by hand, or install the units:

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
| `libgl1`, `libglx-mesa0`, `libegl1` | MuJoCo's viewer. **not needed with `--headless`** |
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

- The face landmarker runs at **15 Hz, not 30**. Gaze does not need camera rate, and this
  halves the perception cost for no felt difference.
- Transcription runs **once per utterance**, not streaming. A partial sentence is of no
  use to the director, and streaming would double the transcription work.
- The vision and planning model is **remote and called twice per goal at most**, so no
  VLM or planner LLM sits in memory. Running both locally alongside whisper and piper is
  what would actually breach 8 GB.
- Audio is one output stream with a software mixer, so there is one PortAudio callback
  rather than three players competing for the device.

A GNOME session itself costs 1.2-1.8 GB. A minimal or headless session is worth about a
gigabyte if the deployment is a fixture rather than a desktop.

## Running without the network

Unset `ANTHROPIC_API_KEY`. The character then engages, converses, remembers and runs the
goals it recognises entirely on the machine; it cannot name objects it has not been told
about, and asks instead. Nothing times out waiting for a network that is not there.

## What is sent when the key is set

One JPEG of the current camera frame, the text of the current request, and the short list
of labels already in memory, at the moment an `observe` or a goal runs. No audio ever
leaves the machine, no continuous video, and nothing at all is sent while the character is
merely idling, engaging or chatting.

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
