from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

sys.path.insert(0, ".")

MP = "https://storage.googleapis.com/mediapipe-models"
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
SILERO = ("https://raw.githubusercontent.com/snakers4/silero-vad/"
          "7e30209a3e901f9842f81b225f3e93d8199902b1/src/silero_vad/data")

FILES = [
    (f"{MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
     "models/pose_landmarker_lite.task",
     "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    (f"{MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     "models/face_landmarker.task",
     "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    (f"{SILERO}/silero_vad_16k_op15.onnx",
     "models/silero_vad_16k_op15.onnx",
     "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"),
    (f"{PIPER}/en_US-lessac-medium.onnx",
     "models/piper/en_US-lessac-medium.onnx",
     "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f"),
    (f"{PIPER}/en_US-lessac-medium.onnx.json",
     "models/piper/en_US-lessac-medium.onnx.json",
     "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0"),
    (f"{KOKORO}/kokoro-v1.0.onnx", "models/kokoro/kokoro-v1.0.onnx",
     "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    (f"{KOKORO}/voices-v1.0.bin", "models/kokoro/voices-v1.0.bin",
     "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
]
UNIFACE = {
    "models/uniface/retinaface_mnet_v2.onnx":
        "3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757",
    "models/uniface/gaze_mobileone_s0.onnx":
        "8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: str, digest: str) -> None:
    if os.path.isfile(dest) and sha256(dest) == digest:
        print(f"  have  {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  fetch {dest}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if got != digest:
        os.remove(tmp)
        raise SystemExit(f"checksum mismatch for {dest}: got {got}, want {digest}")
    os.replace(tmp, dest)


def main() -> None:
    print("models:")
    for url, dest, digest in FILES:
        fetch(url, dest, digest)

    import uniface
    from brain.vision import DETECTOR, GAZE, UNIFACE_DIR
    uniface.set_cache_dir(UNIFACE_DIR)
    uniface.download_models([DETECTOR, GAZE])
    for path, digest in UNIFACE.items():
        if sha256(path) != digest:
            raise SystemExit(f"checksum mismatch for {path}")
        print(f"  have  {path}")

    import onnx_asr
    from brain.ears import PARAKEET, PARAKEET_DIR
    onnx_asr.load_model(PARAKEET, PARAKEET_DIR, quantization="int8")
    print(f"  have  {PARAKEET_DIR}/ (parakeet tdt 0.6b v2, int8)")

    from brain.ears import WHISPER_DIR
    from faster_whisper import WhisperModel
    WhisperModel("base.en", device="cpu", compute_type="int8", download_root=WHISPER_DIR)
    print(f"  have  {WHISPER_DIR}/ (whisper base.en)")
    print("all models present and verified")


if __name__ == "__main__":
    main()
