#!/usr/bin/env bash
# install on a clean ubuntu 24.04 laptop. idempotent, safe to re-run.
#
# the gl packages are only needed if you want the mujoco viewer. the deployed units run
# the body with --headless, which renders nothing and needs no gl at all.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
WANT_VIEWER="${WANT_VIEWER:-0}"

echo "installing lamp into $HERE for user $USER_NAME"

# ---- apt ---------------------------------------------------------------------------
PKGS=(libportaudio2 libgomp1 v4l-utils ca-certificates curl)
if [ "$WANT_VIEWER" = "1" ]; then
  PKGS+=(libgl1 libglx-mesa0 libegl1 libxinerama1 libxcursor1 libxi6)
fi
if [ "$(id -u)" -eq 0 ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PKGS[@]}"
else
  echo "not root, skipping apt. needed packages: ${PKGS[*]}"
fi

# ---- device access -------------------------------------------------------------------
for grp in audio video; do
  if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$grp"; then
    echo "adding $USER_NAME to $grp (log out and back in for this to take effect)"
    [ "$(id -u)" -eq 0 ] && usermod -aG "$grp" "$USER_NAME" || true
  fi
done

# ---- uv and the environment ----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$HERE"
uv sync --frozen

# opencv only captures and encodes here, it never opens a window, so the headless wheel
# is the right one on a server or a fixture: smaller, and no gtk dependency chain
if [ "$WANT_VIEWER" != "1" ]; then
  echo "swapping opencv for the headless wheel"
  uv pip uninstall opencv-python >/dev/null 2>&1 || true
  uv pip install --quiet opencv-python-headless
fi

# ---- models ---------------------------------------------------------------------------
mkdir -p models/piper
fetch() {  # url, destination
  if [ -s "$2" ]; then echo "  have $(basename "$2")"; return; fi
  echo "  fetching $(basename "$2")"
  curl -fsSL "$1" -o "$2"
}
echo "models:"
fetch "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
      models/face_landmarker.task
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
fetch "$BASE/en_US-lessac-medium.onnx" models/piper/en_US-lessac-medium.onnx
fetch "$BASE/en_US-lessac-medium.onnx.json" models/piper/en_US-lessac-medium.onnx.json

# whisper downloads itself on first use, so warm it here rather than during a demo
echo "warming whisper base.en int8"
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

# ---- checks ---------------------------------------------------------------------------
echo
echo "checking devices"
uv run python - <<'PY'
import sys
try:
    import sounddevice as sd
    print("  input :", sd.query_devices(kind="input")["name"])
    print("  output:", sd.query_devices(kind="output")["name"])
except Exception as exc:
    print("  audio NOT ready:", exc)
try:
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    ok, frame = cap.read()
    print("  camera:", "ok" if ok else "NOT ready", frame.shape if ok else "")
    cap.release()
except Exception as exc:
    print("  camera NOT ready:", exc)
PY

echo
echo "done. to run it:"
echo "  uv run python -m body.main --headless      # or without --headless for the viewer"
echo "  uv run python -m brain.main"
echo
echo "or install the user services:"
echo "  mkdir -p ~/.config/systemd/user"
echo "  sed \"s|%LAMP_DIR%|$HERE|g\" deploy/lamp-body.service > ~/.config/systemd/user/lamp-body.service"
echo "  sed \"s|%LAMP_DIR%|$HERE|g\" deploy/lamp-brain.service > ~/.config/systemd/user/lamp-brain.service"
echo "  systemctl --user daemon-reload && systemctl --user enable --now lamp-body lamp-brain"
