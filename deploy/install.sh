#!/usr/bin/env bash
# usage: [WANT_VIEWER=0] sudo ./deploy/install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
WANT_VIEWER="${WANT_VIEWER:-1}"

echo "installing lamp into $HERE for user $USER_NAME"

PKGS=(libportaudio2 libgomp1 libglib2.0-0 v4l-utils ca-certificates curl)
if [ "$WANT_VIEWER" = "1" ]; then
  PKGS+=(libgl1 libglx-mesa0 libegl1 libxinerama1 libxcursor1 libxi6 libxrandr2 libxkbcommon0)
fi
if [ "$(id -u)" -eq 0 ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PKGS[@]}"
else
  echo "not root, skipping apt. needed packages: ${PKGS[*]}"
fi

for grp in audio video; do
  if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$grp"; then
    echo "adding $USER_NAME to $grp (log out and back in for this to take effect)"
    [ "$(id -u)" -eq 0 ] && usermod -aG "$grp" "$USER_NAME" || true
  fi
done

if [ "$(id -u)" -eq 0 ] && [ "$USER_NAME" != "root" ]; then
  echo "continuing as $USER_NAME"
  exec sudo -u "$USER_NAME" -H env WANT_VIEWER="$WANT_VIEWER" bash "$HERE/deploy/install.sh"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv"
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$HERE"
uv sync --frozen

if [ "$WANT_VIEWER" != "1" ]; then
  echo "swapping opencv for the headless wheel"
  uv pip uninstall opencv-python >/dev/null 2>&1 || true
  uv pip install --quiet opencv-python-headless
fi

uv run python tools/fetch_models.py

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
echo "  ./deploy/run.sh            # body with the viewer, brain with the debug window"
echo "  HEADLESS=1 ./deploy/run.sh # no windows at all"
echo "optional: export ANTHROPIC_API_KEY=... first, for conversation and object vision"
echo
echo "or install the user services:"
echo "  mkdir -p ~/.config/systemd/user"
echo "  sed \"s|%LAMP_DIR%|$HERE|g\" deploy/lamp-body.service > ~/.config/systemd/user/lamp-body.service"
echo "  sed \"s|%LAMP_DIR%|$HERE|g\" deploy/lamp-brain.service > ~/.config/systemd/user/lamp-brain.service"
echo "  systemctl --user daemon-reload && systemctl --user enable --now lamp-body lamp-brain"
