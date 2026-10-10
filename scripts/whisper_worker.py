#!/usr/bin/env python3
"""Pinned whisper.cpp transcription worker for the Singularity Grid node.

Newline-delimited JSON on stdin/stdout, mirroring embeddinggemma_worker.py:
- verify_install: exact model size + SHA-256 against whisper_model_files.json.
- startup smoke: real transcription of a bundled clip; readiness only after it passes.
- requests: bounded mono 16 kHz signed 16-bit PCM inside a JSON frame.
- InputValidationError is the only nonfatal class; everything else is fatal.

The worker never sees network traffic or encryption; the node owns sealed envelopes.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROTOCOL = "transcription-v1"
MODEL_ID = "whisper-1"
MODEL_FILE = "ggml-small.bin"
MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
MODEL_SHA256 = "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"
RUNTIME = "whisper.cpp"
RUNTIME_REVISION = "d1be6fde11ac6e0407606b4e42fe72d34add8037"
RUNTIME_BINARY_SHA256 = "4ac1f78373fa19037ff036c66586db2b775e0b21785c320953a1960bcb425405"
RUNTIME_BINARY_BYTES = 7760160
SAMPLE_RATE = 16000
CHANNELS = 1
BITS = 16
MAX_DURATION_SECONDS = 60
MAX_SAMPLES = MAX_DURATION_SECONDS * SAMPLE_RATE
MAX_INPUT_FRAME = 8 * 1024 * 1024
MAX_OUTPUT_FRAME = 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024
MAX_SEGMENTS = 256
SUPPORTED_LANGUAGES = {
    "en", "fr", "de", "es", "it", "pt", "nl", "hi", "ja", "ko", "zh", "ru", "ar"
}
SEGMENT_LINE = re.compile(
    r"^\[(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3}) --> (?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})\]\s*(.*)$"
)


class InputValidationError(ValueError):
    """Request rejected without losing readiness."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_install(model_path: Path) -> tuple[int, str]:
    manifest_path = Path(__file__).resolve().parent / "whisper_model_files.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = {name: (size, digest) for name, size, digest in manifest}
    expected = entries.get(MODEL_FILE)
    if expected is None:
        raise RuntimeError("model manifest is missing the pinned whisper model")
    size, digest = expected
    if not model_path.is_file():
        raise RuntimeError("whisper model file is missing")
    stat = model_path.stat()
    if stat.st_size != size:
        raise RuntimeError("whisper model file size does not match the pin")
    if file_sha256(model_path) != digest or digest != MODEL_SHA256:
        raise RuntimeError("whisper model file hash does not match the pin")
    return size, digest


def runtime_environment() -> dict[str, str]:
    """Do not let operator environment replace native or Python code."""
    return {
        name: value for name, value in os.environ.items()
        if not name.upper().startswith(("DYLD_", "__XPC_DYLD_", "LD_", "PYTHON"))
        and name.upper() not in {"BASH_ENV", "ENV", "GGML_BACKEND_PATH", "GGML_METAL_PATH_RESOURCES", "GCONV_PATH"}
    }


def verify_runtime(whisper_bin: Path, expected_sha256: str) -> tuple[int, str]:
    if expected_sha256 != RUNTIME_BINARY_SHA256:
        raise RuntimeError("whisper runtime sha256 pin is invalid")
    if not whisper_bin.is_file():
        raise RuntimeError("whisper runtime file is missing")
    if whisper_bin.stat().st_size != RUNTIME_BINARY_BYTES:
        raise RuntimeError("whisper static runtime size does not match the pin")
    actual = file_sha256(whisper_bin)
    if actual != expected_sha256:
        raise RuntimeError("whisper runtime sha256 does not match the pin")
    return whisper_bin.stat().st_size, actual


def verify_runtime_version(whisper_bin: Path) -> None:
    completed = subprocess.run(
        [str(whisper_bin), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        env=runtime_environment(),
    )
    version = completed.stdout.decode("utf-8", errors="replace")
    if completed.returncode != 0 or not re.search(r"whisper\.cpp version: 1\.9\.5(?:-dev)?(?:\s|$)", version):
        raise RuntimeError("whisper runtime does not report pinned version 1.9.5")


def run_whisper(whisper_bin: Path, model_path: Path, wav_path: Path, language):
    args = [
        str(whisper_bin),
        "-m",
        str(model_path),
        "-f",
        str(wav_path),
        "-t",
        "4",
    ]
    if language and language != "auto":
        args.extend(["-l", language])
    else:
        args.extend(["-l", "auto"])
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=MAX_DURATION_SECONDS + 60,
        env=runtime_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("whisper runtime exited with %d" % completed.returncode)
    return completed.stdout.decode("utf-8", errors="replace")


def parse_segments(output: str):
    segments = []
    plain = []
    for line in output.splitlines():
        match = SEGMENT_LINE.match(line.strip())
        if match:
            g = match.groups()
            def to_seconds(hh, mm, ss, ms):
                return (int(hh or 0) * 3600 + int(mm) * 60 + int(ss)) + int(ms) / 1000.0
            start = to_seconds(g[0], g[1], g[2], g[3])
            end = to_seconds(g[4], g[5], g[6], g[7])
            segments.append({"start": start, "end": end, "text": g[8].strip()})
        elif line.strip() and not line.startswith(("whisper_", "ggml_", "system_info", "main:", "read_audio_data", "init_", "[")):
            plain.append(line.strip())
    if segments:
        text = " ".join(seg["text"] for seg in segments if seg["text"])
    else:
        text = " ".join(plain).strip()
    return segments, text.strip()


def wav_bytes(pcm: bytes) -> bytes:
    import struct

    data_size = len(pcm)
    header = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, CHANNELS, SAMPLE_RATE, SAMPLE_RATE * 2, 2, BITS
    )
    header += b"data" + struct.pack("<I", data_size)
    return header + pcm


class Runtime:
    def __init__(
        self,
        model_path: Path,
        whisper_bin: Path,
        whisper_sha256: str,
        smoke_dir: Path,
    ):
        self._model_bytes, self._model_sha256 = verify_install(model_path)
        self._runtime_bytes, self._runtime_sha256 = verify_runtime(
            whisper_bin, whisper_sha256
        )
        verify_runtime_version(whisper_bin)
        smoke_config = json.loads((smoke_dir / "smoke.json").read_text("utf-8"))
        if file_sha256(smoke_dir / "audio.wav") != smoke_config.get("audio_sha256"):
            raise RuntimeError("startup fixture hash mismatch")
        expected_words = [w.lower() for w in smoke_config["expected_contains"]]
        output = run_whisper(whisper_bin, model_path, smoke_dir / "audio.wav", None)
        _, transcript = parse_segments(output)
        lowered = transcript.lower()
        if (
            len(transcript) < int(smoke_config.get("min_chars", 20))
            or any(word not in lowered for word in expected_words)
        ):
            raise RuntimeError("startup smoke transcript mismatch")
        self._smoke_transcript = transcript
        self._model_path = model_path
        self._whisper_bin = whisper_bin
        self._whisper_sha256 = whisper_sha256

    def transcribe(self, request):
        if set(request) != {"type", "request_id", "protocol", "audio"}:
            raise InputValidationError("unknown or missing frame field")
        if request.get("protocol") != PROTOCOL:
            raise InputValidationError("protocol mismatch")
        audio = request.get("audio")
        if not isinstance(audio, dict):
            raise InputValidationError("audio payload missing")
        if set(audio) not in (
            {"encoding", "mime_type", "data", "sha256", "sample_rate", "sample_count", "channels", "bits"},
            {"encoding", "mime_type", "data", "sha256", "sample_rate", "sample_count", "channels", "bits", "language"},
        ):
            raise InputValidationError("unknown or missing audio field")
        if audio.get("encoding") != "base64" or audio.get("mime_type") != "audio/pcm":
            raise InputValidationError("audio must be base64 pcm")
        if audio.get("sample_rate") != SAMPLE_RATE:
            raise InputValidationError("audio must be 16 kHz")
        if audio.get("channels") != CHANNELS:
            raise InputValidationError("audio must be mono")
        if audio.get("bits") != BITS:
            raise InputValidationError("audio must be signed 16-bit")
        sample_count = audio.get("sample_count")
        if type(sample_count) is not int or sample_count <= 0:
            raise InputValidationError("sample count missing")
        if sample_count > MAX_SAMPLES:
            raise InputValidationError("audio exceeds the duration bound")
        data = audio.get("data")
        if not isinstance(data, str) or len(data) > ((MAX_SAMPLES * 2 + 2) // 3) * 4:
            raise InputValidationError("audio data missing")
        try:
            pcm = base64.b64decode(data, validate=True)
        except Exception:
            raise InputValidationError("audio base64 invalid")
        if len(pcm) != sample_count * 2:
            raise InputValidationError("audio length contradicts sample count")
        digest = hashlib.sha256(pcm).hexdigest()
        if not isinstance(audio.get("sha256"), str) or audio["sha256"].lower() != digest:
            raise InputValidationError("audio checksum mismatch")
        language = audio.get("language")
        if language is not None and (
            not isinstance(language, str)
            or len(language) != 2
            or not language.isascii()
            or not language.islower()
            or not language.isalpha()
            or language not in SUPPORTED_LANGUAGES
        ):
            raise InputValidationError("language hint invalid")

        # whisper-cli reloads both artifacts for every request. Re-hash after cheap
        # input validation and immediately before launch so a replaced cache cannot
        # retain the prior readiness claim.
        verify_install(self._model_path)
        verify_runtime(self._whisper_bin, self._whisper_sha256)

        with tempfile.TemporaryDirectory(prefix="sgl-stt-") as temp:
            wav_path = Path(temp) / "request.wav"
            wav_path.write_bytes(wav_bytes(pcm))
            output = run_whisper(
                self._whisper_bin, self._model_path, wav_path, language
            )
        segments, text = parse_segments(output)
        if not text.strip():
            text, segments = "", []
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise RuntimeError("transcript exceeds bound")
        if len(segments) > MAX_SEGMENTS:
            raise RuntimeError("transcript segment count exceeds bound")
        if sum(len(segment["text"].encode("utf-8")) for segment in segments) > MAX_TEXT_BYTES:
            raise RuntimeError("transcript segment text exceeds bound")
        # whisper-cli timestamps refer to its padded analysis window (a 1 s
        # silent clip can end at 2.06 s). Export only times inside the actual
        # recording; duration is always authenticated samples, never model output.
        duration = sample_count / float(SAMPLE_RATE)
        segments = [
            {"start": segment["start"], "end": min(segment["end"], duration), "text": segment["text"]}
            for segment in segments
            if segment["start"] < duration and segment["end"] > segment["start"]
        ]
        result = {
            "type": "result",
            "request_id": request["request_id"],
            "text": text,
            "language": language,
            "duration_seconds": duration,
            "segments": segments,
        }
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > MAX_OUTPUT_FRAME:
            raise RuntimeError("result frame exceeds bound")
        return encoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--whisper-bin", required=True)
    parser.add_argument("--whisper-sha256", required=True)
    parser.add_argument("--smoke-dir", required=True)
    args = parser.parse_args()

    # Route library/native stdout away from the protocol channel.
    try:
        saved = os.dup(1)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        protocol_out = os.fdopen(saved, "w", buffering=1)
    except OSError:
        return 2

    try:
        runtime = Runtime(
            Path(args.model_path),
            Path(args.whisper_bin),
            args.whisper_sha256,
            Path(args.smoke_dir),
        )
    except Exception:
        try:
            sys.stderr.write("whisper worker startup failed\n")
        except Exception:
            pass
        return 2

    protocol_out.write(
        json.dumps(
            {
                "type": "ready",
                "protocol": PROTOCOL,
                "runtime": RUNTIME,
                "runtime_revision": RUNTIME_REVISION,
                "model_revision": MODEL_REVISION,
                "model_sha256": runtime._model_sha256,
                "model_bytes": runtime._model_bytes,
                "runtime_binary_sha256": runtime._runtime_sha256,
                "runtime_binary_bytes": runtime._runtime_bytes,
                "model_id": MODEL_ID,
                "audio_format": "pcm_s16le_16k_mono",
                "max_duration_seconds": MAX_DURATION_SECONDS,
                "smoke_transcript": runtime._smoke_transcript,
            }
        )
        + "\n"
    )

    while True:
        line = sys.stdin.buffer.readline(MAX_INPUT_FRAME + 1)
        if not line:
            return 0
        if len(line) > MAX_INPUT_FRAME or not line.endswith(b"\n"):
            return 2
        try:
            request = json.loads(line)
        except Exception:
            return 2
        try:
            if not isinstance(request, dict) or request.get("type") != "transcribe":
                raise InputValidationError("frame type mismatch")
            encoded = runtime.transcribe(request)
        except InputValidationError:
            encoded = json.dumps(
                {
                    "type": "request_error",
                    "request_id": request.get("request_id", 0)
                    if isinstance(request, dict)
                    else 0,
                    "code": "invalid_input",
                }
            )
        except Exception:
            return 2
        protocol_out.write(encoded + "\n")


if __name__ == "__main__":
    sys.exit(main())
