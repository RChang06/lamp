# Lamp

A live character built on the supplied five degree-of-freedom lamp. Two processes: a
simulated **body** that owns physics, servos and light, and a **brain** that watches,
listens, remembers and decides. They talk over one websocket in newline delimited JSON.

The body runs on its own with nothing attached, which is how the motion is developed and
judged. The brain runs without the body too, and says so rather than failing.

## Quick start

On Ubuntu 24.04 (the target), see [deploy/README.md](deploy/README.md) for details:

```bash
sudo ./deploy/install.sh            # apt packages, uv, the environment, every model
export ANTHROPIC_API_KEY=...        # optional: conversation and object vision
./deploy/run.sh                     # body + brain, with the viewer and the debug window
```

Anywhere else with [uv](https://docs.astral.sh/uv/):

```bash
uv sync && uv run python tools/fetch_models.py
uv run python -m body.main          # the body, with the viewer
uv run python -m brain.main --debug # the character, in another terminal
```

Everything below runs with no camera, microphone or network.

| What | Command | Time |
|---|---|---|
| Body measurements, over the real socket | `uv run python tools/selftest.py` | 20 s |
| Drive the body by hand, in a browser | open `tools/panel.html` | - |
| Drive the body by hand, in a terminal | `uv run python tools/puppet.py` | - |
| Scripted tour of every posture and gesture | `uv run python tools/puppet.py --showcase` | 33 s |
| The whole character against a scripted person | `uv run python tools/dryrun.py` | 70 s |
| Latency, cpu, memory, real time factor | `uv run python tools/measure.py` | 2 min |

## Layout

```
body/        the simulated body, runs alone
  kinematics.py   closed form aim solve, posture presets
  motion.py       reference generator: gestures, breathing, rate limits
  main.py         mujoco, servos, light, the websocket server
brain/       the character
  vision.py       camera, face attention, engagement dwell
  ears.py         voice activity detection, whisper in a worker
  voice.py        piper, sentence at a time, interruptible
  audio.py        one mixer: voice, effects, music, and the speech envelope
  memory.py       what it saw, and how it answers later
  planner.py      intent routing, local planner, model planner
  cloud.py        the one remote call, with structured output
  director.py     four states, and the edges between them
  link.py         the brain's end of the protocol
common/
  protocol.py     the wire protocol
  actions.py      the action vocabulary a plan may use, and its validator
tools/       tests, measurements, and the hand controls
robot/       the supplied urdf and mesh, unmodified
```

## The two interfaces

**Protocol (brain to body).** The brain sends character intent, never joint angles:
`look(az, el, weight)`, `posture(name)`, `gesture(name, amp)`, `light(intensity, warm)`,
`envelope(level)`. The body answers at 30 Hz with joint positions, commanded and achieved
aim, aim error, servo tracking error, torque saturation and active gestures.

**Action vocabulary (model to body).** A plan is a list of validated actions:
`look_at, observe, scan, light, gesture, say, sound, music, wait, done`. A language model
never emits joint angles or timings. An invalid plan is refused whole rather than run in
part, so a plan cannot lose the `observe` step that verifies it before claiming success.

## Configuration

| Variable | Default | What |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | enables the vision and planning call. unset is a supported mode |
| `LAMP_MODEL` | `claude-opus-5-5` | the model for the one call per turn. `claude-haiku-5-5` answers in about a second, with weaker vision and planning |
| `LAMP_ASR` | `parakeet` | speech recognition: `parakeet` (int8, 0.7 GB), `parakeet-fp32` (more accurate, 2.6 GB) or `whisper` |
| `LAMP_VOICE` | `af_heart` | a Kokoro voice name, or `piper` for the faster, more robotic fallback |
| `LAMP_EFFORT` | `low` | a plan is short, so latency is the thing worth buying |

With no key, the character still engages, converses, remembers and carries out the goals
it recognises. What it loses is naming objects it has not been told about, and there it
looks, says it cannot tell, and asks. The answer is stored from speech, and every later
question about that object works identically.

## Licensing

The supplied URDF and mesh are unmodified. `piper-tts` 1.8.0 is GPL-3.0-or-later (the
MIT `rhasspy/piper` was archived in October 2025 and `pip install piper-tts` now resolves
to `OHF-Voice/piper1-gpl`). MediaPipe is Apache-2.0, faster-whisper and CTranslate2 are
MIT, MuJoCo is Apache-2.0. No Ultralytics YOLO anywhere, since it is AGPL-3.0.
