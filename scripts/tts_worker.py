"""Supervised Kokoro (mlx-audio) text-to-speech worker for the Grid `speech-v1` lane.

The Rust supervisor (src/tts.rs) owns this process: it writes one JSON frame per line on
stdin and reads one JSON frame per line on stdout. Nothing else is printed to stdout.

Startup:
  1. Re-hash every pinned model file against tts_model_files.json; refuse on any mismatch.
  2. Check the pinned package versions (mlx-audio, mlx, misaki).
  3. Load Kokoro from the operator's local model directory (offline, never downloads).
  4. Speak a fixed smoke sentence and require real, non-silent audio.
  5. Print one `ready` frame.

Requests:  {"type":"speak","request_id":<u64>,"protocol":"speech-v1","text_b64":...,
            "text_sha256":...,"voice":...,"speed":"1.00"}
Replies:   {"type":"result","request_id":...,"duration_seconds":...,"bytes":...,
            "sha256":...,"audio_b64":...}             (m4a, AAC-LC 48 kbps, 24 kHz mono)
           {"type":"request_error","request_id":...,"code":"invalid_input"}

GPL espeak-ng is never imported: `misaki.espeak` is replaced by a lexicon-only stand-in
(same approach as the desktop app's speech engine).
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import wave

PROTOCOL = "speech-v1"
MODEL_ID = "kokoro-82m"
RUNTIME = "mlx-audio"
OUTPUT_FORMAT = "m4a_aac_24k_mono"
SAMPLE_RATE = 24000
MAX_INPUT_BYTES = 1000
MAX_AUDIO_SECONDS = 90
MAX_AUDIO_BYTES = 700_000
VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma", "af_aoede", "af_kore",
          "af_sarah", "am_fenrir", "am_michael", "am_puck"]
PACKAGES = {"mlx-audio": "0.6.1", "mlx": "0.32.4", "misaki": "0.9.4"}
SMOKE_TEXT = "Singularity Layer speech check."
AFCONVERT = "/usr/bin/afconvert"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_out = sys.stdout
sys.stdout = sys.stderr  # libraries that print must never corrupt the frame stream


def emit(frame):
    _out.write(json.dumps(frame, separators=(",", ":")) + "\n")
    _out.flush()


def fail(reason):
    sys.stderr.write(f"tts worker: {reason}\n")
    sys.exit(2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(model_dir, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    lines = []
    for entry in manifest["files"]:
        rel = entry["path"]
        if rel.startswith("/") or ".." in rel.split("/"):
            fail("manifest path invalid")
        path = os.path.join(model_dir, rel)
        if os.path.islink(path) or not os.path.isfile(path):
            fail(f"model file missing: {rel}")
        if os.path.getsize(path) != entry["bytes"]:
            fail(f"model file size mismatch: {rel}")
        if sha256_file(path) != entry["sha256"]:
            fail(f"model file hash mismatch: {rel}")
        lines.append(f"{rel} {entry['bytes']} {entry['sha256']}")
    text = "".join(line + "\n" for line in sorted(lines))
    manifest_sha = hashlib.sha256(text.encode()).hexdigest()
    if manifest_sha != manifest["model_sha256"]:
        fail("model manifest digest mismatch")
    return manifest, manifest_sha


def check_packages():
    from importlib import metadata

    for name, version in PACKAGES.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            fail(f"package missing: {name}")
        if found != version:
            fail(f"package version mismatch: {name} {found} != {version}")
    return ";".join(f"{k}=={v}" for k, v in PACKAGES.items())


def install_espeak_stub():
    stub = types.ModuleType("misaki.espeak")

    class EspeakFallback:  # name kept: the Kokoro pipeline looks it up by name
        def __init__(self, british=False):
            from misaki import en

            self.lex = en.Lexicon(british)

        def _lookup(self, word):
            for key in (word, word.lower(), word.capitalize(), word.upper()):
                value = self.lex.golds.get(key) or self.lex.silvers.get(key)
                if isinstance(value, dict):
                    value = value.get("DEFAULT") or next(iter(value.values()), None)
                if value:
                    return value
            return None

        def __call__(self, token):
            word = re.sub(r"[^A-Za-z]", "", token.text)
            if not word:
                return (None, None)
            out, i = [], 0
            while i < len(word):
                for j in range(len(word), i, -1):
                    piece = self._lookup(word[i:j]) if j - i > 1 else None
                    if piece:
                        out.append(piece)
                        i = j
                        break
                else:
                    out.append(self._lookup(word[i].upper()) or "")
                    i += 1
            return ("".join(out), 1)

    class EspeakG2P:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("language not available")

    stub.EspeakFallback = EspeakFallback
    stub.EspeakG2P = EspeakG2P
    import misaki

    sys.modules["misaki.espeak"] = stub
    misaki.espeak = stub


def synthesize(model, model_dir, text, voice, speed):
    import numpy as np

    # Always the pinned, verified voice file. A bare voice name makes mlx-audio fetch the
    # voice through the Hugging Face cache, which is neither pinned nor available offline.
    voice_path = os.path.join(model_dir, "voices", f"{voice}.safetensors")
    chunks = list(model.generate(text=text, voice=voice_path, speed=speed, lang_code=voice[0]))
    if not chunks:
        raise ValueError("no audio")
    audio = np.concatenate([np.array(c.audio).reshape(-1) for c in chunks])
    rate = chunks[0].sample_rate
    if rate != SAMPLE_RATE:
        raise ValueError("unexpected sample rate")
    if not np.isfinite(audio).all():
        raise ValueError("non-finite audio")
    return audio


def encode_m4a(audio, tmp):
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    src = os.path.join(tmp, "in.wav")
    dst = os.path.join(tmp, "out.m4a")
    with wave.open(src, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    subprocess.run([AFCONVERT, "-f", "m4af", "-d", "aac", "-b", "48000", src, dst],
                   check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=60)
    with open(dst, "rb") as f:
        data = f.read()
    for path in (src, dst):
        os.unlink(path)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if not os.path.isfile(AFCONVERT):
        fail("afconvert unavailable")
    model_dir = os.path.realpath(args.model_dir)
    manifest, manifest_sha = verify_model(model_dir, args.manifest)
    runtime_revision = check_packages()
    install_espeak_stub()
    from mlx_audio.tts.utils import load_model

    model = load_model(model_dir)
    tmp = tempfile.mkdtemp(prefix="sgl-tts-")

    import numpy as np

    smoke = synthesize(model, model_dir, SMOKE_TEXT, "af_heart", 1.0)
    seconds = len(smoke) / SAMPLE_RATE
    if not (0.8 <= seconds <= 10.0) or float(np.sqrt(np.mean(smoke ** 2))) < 0.005:
        fail("smoke audio implausible")
    smoke_m4a = encode_m4a(smoke, tmp)
    if len(smoke_m4a) < 2000:
        fail("smoke encode implausible")

    emit({
        "type": "ready",
        "protocol": PROTOCOL,
        "runtime": RUNTIME,
        "runtime_revision": runtime_revision,
        "model_id": MODEL_ID,
        "model_revision": manifest["model_revision"],
        "model_sha256": manifest_sha,
        "voices": VOICES,
        "output_format": OUTPUT_FORMAT,
        "max_input_bytes": MAX_INPUT_BYTES,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "smoke_seconds": round(seconds, 3),
    })

    for line in sys.stdin:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            fail("bad request frame")
        if not isinstance(frame, dict) or frame.get("type") != "speak":
            fail("unexpected frame")
        request_id = frame.get("request_id")
        if not isinstance(request_id, int):
            fail("request id invalid")

        def reject():
            emit({"type": "request_error", "request_id": request_id, "code": "invalid_input"})

        try:
            text_bytes = base64.b64decode(frame["text_b64"], validate=True)
            text = text_bytes.decode("utf-8")
            speed = float(frame["speed"])
            voice = frame["voice"]
        except Exception:  # noqa: BLE001
            reject()
            continue
        if (frame.get("protocol") != PROTOCOL or voice not in VOICES
                or not (1 <= len(text_bytes) <= MAX_INPUT_BYTES) or "\x00" in text
                or not text.strip() or not (0.5 <= speed <= 2.0)
                or hashlib.sha256(text_bytes).hexdigest() != frame.get("text_sha256")):
            reject()
            continue
        try:
            audio = synthesize(model, model_dir, text, voice, speed)
        except Exception:  # noqa: BLE001
            reject()  # unspeakable text (e.g. only symbols) is the caller's input
            continue
        seconds = len(audio) / SAMPLE_RATE
        if seconds > MAX_AUDIO_SECONDS:
            reject()
            continue
        try:
            data = encode_m4a(audio, tmp)
        except Exception:  # noqa: BLE001
            fail("encode failed")
        if not data or len(data) > MAX_AUDIO_BYTES:
            fail("encoded audio out of bounds")
        emit({
            "type": "result",
            "request_id": request_id,
            "duration_seconds": round(seconds, 3),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "audio_b64": base64.b64encode(data).decode("ascii"),
        })


if __name__ == "__main__":
    main()
