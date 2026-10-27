# Lamp — technical note

A live character on the supplied five-DOF lamp. Two processes, one protocol. The body owns
physics and time; the brain owns perception, speech, memory and decisions. Measurements
below are taken on the dev machine (i7-10750H, CPU only) by `tools/measure.py`, not
estimated; the few numbers that are budgets rather than measurements say so.

## Architecture and data flow

```
  camera ─┐                      BRAIN  (soft real time, python asyncio)
          │   ┌──────────────────────────────────────────────────────────┐
          └──►│ vision 15 Hz ─ attending? ─┐                            │
              │   face landmarker          │                            │
  mic ───────►│ ears  VAD ─ utterance ─────┤                            │
              │                            ▼                            │
              │                     ┌─────────────┐    ┌──────────────┐  │
              │                     │  director   │───►│ scene memory │  │
              │                     │  4 states   │◄───│ jsonl + find │  │
              │                     └──┬───┬───┬──┘    └──────────────┘  │
              │        whisper ◄───────┘   │   └──────► planner          │
              │        piper ──────────────┘              │  intents     │
  speaker ◄───┤ audio mixer: voice / sfx / music          │  local plan  │
              │    │ voice level                          │  model plan  │
              └────┼──────────────────────────────────────┼──────────────┘
                   │                                      │ 1 jpeg + text
          character intent                                ▼
          look / posture / gesture / light / envelope    Claude  (optional)
                   │                        ▲            vision + plan
                   ▼   ws://127.0.0.1:8765  │ telemetry 30 Hz
              ┌────────────────────────────────────────────┐
              │ BODY  (hard real time)                     │
              │  aim solve ─ posture ─ gestures ─ breathing│
              │  rate limit ─ critically damped filter     │
              │  gravity feedforward ─ mujoco 500 Hz       │
              │  position servos, urdf effort limits       │
              │  lamp light: intensity + warmth            │
              └────────────────────────────────────────────┘
```

**Why two processes.** The body integrates physics and holds servo references on a clock;
the brain is full of models that occasionally take a second. Separating them means a slow
transcription cannot stutter the motion, either half restarts without the other, and the
body is drivable by hand (`tools/panel.html`) with no perception stack in existence. Cost
is one websocket, measured at **32 ms** command-to-acknowledged — which is itself bounded
below by the 33 ms telemetry period, so the transport is under one frame.

**Protocol.** Newline-delimited JSON. The brain sends *character intent* and never joint
angles: `look(az, el, weight)`, `posture(name)`, `gesture(name, amp)`,
`light(intensity, warm)`, `envelope(level)`. The body returns joint positions, commanded
and achieved aim, aim error, servo tracking error, torque saturation and active gestures.
Ownership is strict: all kinematics, rate limits and timing live in the body, so there is
exactly one place where the robot can be made to do something unsafe.

**Model to action.** A plan is a list from a fixed vocabulary — `look_at, observe, scan,
light, gesture, say, sound, music, wait, done` — validated in `common/actions.py` before
anything moves. An invalid plan is refused **whole**, never partially run, because a plan
missing its middle step is worse than no plan: it can lose the `observe` that verifies the
result. If a plan would reach `done` without observing, an `observe` is inserted. A
language model therefore cannot emit a joint angle, a duration, or an unbounded utterance.

## Simulation and motion

MuJoCo loads the supplied URDF directly with `discardvisual="false" fusestatic="false"`
(otherwise it drops the visual meshes and merges the head, light and camera frames). The
supplied head collision cylinder already overlaps the upper arm at the zero pose, which
jams head pitch, so self-collision is excluded between the four upper links and joint
limits bound the motion instead.

The chain **decouples exactly**, measured across the full joint ranges:

```
aim azimuth   = base_yaw + neck_yaw
aim elevation = shoulder_pitch + elbow_pitch − head_pitch
```

So gaze is closed form — no IK, no Jacobian — and the redundancy (one joint in azimuth,
two in elevation) is spent entirely on character: how much of a turn the heavy base takes
versus the light neck, and how tall or folded it stands while looking at the same point.
A glance under `neck_free` (0.20–0.45 rad by posture) moves the base **0%**.

Two layers of physical limit, deliberately: the reference generator is capped at the URDF
*velocity* limits and smoothed by a critically damped filter, then the position servos are
capped at the URDF *effort* limits. What the servos cannot hold is reported as saturation
rather than clipped silently. Gravity feedforward is `qfrc_bias / kp` per joint — what a
real servo arm does — which removed a 0.03–0.05 rad droop with no integrator to wind up.

Gestures, breathing and speech motion are added *after* the filter, since they are already
smooth functions of time; filtering them had been eating 64% of their amplitude.

**Light is a channel, not decoration.** Posture sets a resting level, gestures scale it,
and the speech envelope modulates it, so the bulb pulses with the voice from the same
signal that drives the head bob. There is no roll joint on this body, so a quizzical
head-tilt is physically unavailable and "listening" is expressed with neck yaw and pitch.

## Perception, speech and memory

Engagement gates on **visual attention, not a wake word**: a frontal face held 0.40 s
engages, 2.00 s without one releases. The asymmetry is the point — equal thresholds
flicker at the boundary, and flicker reads as broken rather than alive. A creature with a
face should not need its name said first.

Scene memory is a JSONL append log with token-overlap retrieval and a recency prior. At a
few dozen records a vector index would take longer to load than the data it indexes; the
interface is narrow enough that embeddings swap into one method. A weak match offers what
it does remember instead of denying.

**The remote step is one call.** Everything latency-critical is local. With
`ANTHROPIC_API_KEY` unset the character still engages, converses, remembers and runs the
goals it recognises — it just cannot name an object it was never told about, and there it
looks, admits it, asks, and stores the answer from speech. Sent when the key is set: one
JPEG at the moment of an `observe`, the request text, and the labels already in memory. No
audio, no continuous video, nothing while idling.

## Measurements

| | |
|---|---|
| Face landmarker, 640×480 | **2.3 ms** mean, 2.9 ms p95 — run at 15 Hz, not camera rate |
| Whisper base.en int8 | 1.09 s for 1.79 s of speech, **RTF 0.61**; 1.1 s model load |
| Piper lessac-medium | **99 ms** for the first sentence, RTF 0.04–0.09 |
| Gaze arrival, 0.9 rad swing | **938 ms** in `alert`, 1718 ms in `idle` (bandwidth is the posture) |
| Gaze accuracy | **0.0008 rad** at target; 0.0106 rad at the workspace edge |
| Protocol round trip | **32 ms** mean, 48 ms p95 (≤ one 33 ms telemetry frame) |
| Body CPU | **5.9% of one core** — ~1.5% of the 4-core target |
| Body resident memory | **75 MB** |
| Physics real time factor | **0.991** |
| Telemetry reliability | 823 messages, **0 dropped**, 30.1 Hz over a full scripted run |
| Models on disk | landmarker 3.8 MB, piper 63 MB, whisper ~150 MB |
| Engagement reliability | *run `tools/engagement.py --trials 10` with a person present* |

Spoken round trip is dominated by transcription: an utterance of N seconds costs ≈0.6 N to
transcribe, then ~0.1 s to first sound. For a 2 s question that is roughly **1.3 s** to the
first syllable of the reply.

## Known limitations

- **No acoustic echo cancellation.** Barge-in works by keeping the mic open and raising
  the onset threshold ×3 while speaking. On a laptop at volume it can still hear itself.
  AEC (WebRTC AEC3 or speexdsp) is the correct fix and is not implemented; headphones
  make the demo robust.
- **Transcription is the latency floor**, and it is not streaming. `tiny.en`, or
  Moonshine for short clips, would roughly halve it at some accuracy cost.
- **Camera attention is a proxy for gaze.** Head yaw within 28° counts as looking at the
  lamp; true eye gaze is not estimated, and MediaPipe degrades past ~45° of rotation.
- **One person.** The strongest face wins; there is no person-of-interest arbitration.
- **Object memory is 2D and label-shaped** — a label, a timestamp and the aim the lamp
  held, not a 3D position. "Next to the mug" is not answerable. Scaling that up means a
  depth sensor and a scene graph, which wants a GPU.
- **The remote step is a dependency with no offline equivalent for naming objects.** A
  local VLM (SmolVLM2-500M GGUF) would close that; no published CPU-only per-image
  latency exists for it, and moondream2 is reported at 2–3 images per *minute* on laptop
  CPU, so it was out of scope for the timebox rather than rejected on merit.
- **Deliberately not built:** multi-person tracking, a wake word, streaming STT, emotion
  classification, any learned policy. The brief asked for a coherent small system over a
  wide unfinished one.

## Licensing note

The supplied URDF and mesh are unmodified. `piper-tts` 1.8.0 is **GPL-3.0-or-later**: the
MIT `rhasspy/piper` was archived in October 2025 and `pip install piper-tts` now resolves
to `OHF-Voice/piper1-gpl`. MediaPipe is Apache-2.0, faster-whisper and CTranslate2 MIT,
MuJoCo Apache-2.0. Ultralytics YOLO was avoided on purpose — it is AGPL-3.0.
