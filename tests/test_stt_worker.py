import json
import subprocess
import sys
import unittest
import base64
import hashlib
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import whisper_worker as w  # noqa: E402


def make_runtime():
    rt = w.Runtime.__new__(w.Runtime)
    rt._model_path = Path("/unused/model.bin")
    rt._whisper_bin = Path("/unused/whisper-cli")
    rt._whisper_sha256 = "0" * 64
    return rt


def audio_payload(pcm: bytes, sample_count=None, **overrides):
    payload = {
        "encoding": "base64",
        "mime_type": "audio/pcm",
        "data": base64.b64encode(pcm).decode(),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "sample_rate": 16000,
        "sample_count": (len(pcm) // 2) if sample_count is None else sample_count,
        "channels": 1,
        "bits": 16,
    }
    payload.update(overrides)
    return {"type": "transcribe", "request_id": 7, "protocol": "transcription-v1", "audio": payload}


class WhisperWorkerPolicyTests(unittest.TestCase):
    def test_wav_header_is_exact_and_covers_pcm(self):
        pcm = b"\x01\x00" * 100
        blob = w.wav_bytes(pcm)
        self.assertEqual(blob[:4], b"RIFF")
        self.assertEqual(blob[8:12], b"WAVE")
        self.assertEqual(blob[36:40], b"data")
        self.assertEqual(blob[44:], pcm)

    def test_parse_segments_extracts_timestamps_and_text(self):
        output = (
            "whisper_init_from_file_with_params_no_state: loading model\n"
            "[00:00.000 --> 00:02.500]  hello there\n"
            "[00:02.500 --> 00:05.000]  general kenobi\n"
            "system_info: n_threads = 4\n"
        )
        segments, text = w.parse_segments(output)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0]["start"], 0.0)
        self.assertAlmostEqual(segments[0]["end"], 2.5)
        self.assertEqual(segments[1]["text"], "general kenobi")
        self.assertEqual(text, "hello there general kenobi")

    def test_bad_sample_rate_channels_bits_mime_rejected(self):
        rt = make_runtime()
        pcm = b"\x00\x00" * 1600
        for key, value in [
            ("sample_rate", 44100),
            ("channels", 2),
            ("bits", 8),
            ("mime_type", "audio/wav"),
            ("encoding", "base58"),
        ]:
            request = audio_payload(pcm, **{key: value})
            with self.assertRaises(w.InputValidationError, msg=key):
                rt.transcribe(request)

    def test_sample_count_drift_and_checksum_rejected(self):
        rt = make_runtime()
        pcm = b"\x00\x00" * 1600
        with self.assertRaises(w.InputValidationError):
            rt.transcribe(audio_payload(pcm, sample_count=1601))
        with self.assertRaises(w.InputValidationError):
            bad = audio_payload(pcm)
            bad["audio"]["sha256"] = "0" * 64
            rt.transcribe(bad)
        with self.assertRaises(w.InputValidationError):
            rt.transcribe(audio_payload(pcm, sample_count=0))

    def test_duration_bound_enforced(self):
        rt = make_runtime()
        pcm = b"\x00\x00" * (w.MAX_SAMPLES + 1)
        with self.assertRaises(w.InputValidationError):
            rt.transcribe(audio_payload(pcm))

    def test_language_hint_rules(self):
        rt = make_runtime()
        pcm = b"\x00\x00" * 1600
        for language in ["EN", "eng", "e1", 1, "en-US", "xx", ""]:
            with self.assertRaises(w.InputValidationError):
                rt.transcribe(audio_payload(pcm, language=language))
        with patch.object(w, "verify_install"), patch.object(w, "verify_runtime"), patch.object(w, "run_whisper", return_value="[00:00.000 --> 00:00.100] hello"):
            self.assertIn('"text": "hello"', rt.transcribe(audio_payload(b"\x01\x00" * 1600, language="en")))

    def test_artifact_hash_version_and_size_are_verified(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "whisper-cli"
            binary.write_bytes(b"pinned artifact")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            with patch.object(w, "RUNTIME_BINARY_SHA256", digest), patch.object(w, "RUNTIME_BINARY_BYTES", len(b"pinned artifact")):
                self.assertEqual(w.verify_runtime(binary, digest), (len(b"pinned artifact"), digest))
                with self.assertRaises(RuntimeError): w.verify_runtime(binary, "0" * 64)
                binary.write_bytes(b"wrong size")
                with self.assertRaises(RuntimeError): w.verify_runtime(binary, digest)
            with self.assertRaises(RuntimeError): w.verify_runtime(binary, "0" * 64)
            with self.assertRaises(RuntimeError): w.verify_install(binary)
            with patch.object(w.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"whisper.cpp version: 1.9.5-dev")):
                w.verify_runtime_version(binary)
            with patch.object(w.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"whisper.cpp version: 1.9.50")):
                with self.assertRaises(RuntimeError): w.verify_runtime_version(binary)

    def test_unknown_fields_fail_before_inference(self):
        rt = make_runtime()
        request = audio_payload(b"\x00\x00" * 100)
        request["audio"]["url"] = "https://example.invalid/audio"
        with self.assertRaises(w.InputValidationError): rt.transcribe(request)
        request = audio_payload(b"\x00\x00" * 100)
        request["extra"] = True
        with self.assertRaises(w.InputValidationError): rt.transcribe(request)

    def test_silence_is_a_valid_empty_transcript_after_artifact_validation(self):
        rt = make_runtime()
        with patch.object(w, "verify_install") as model, patch.object(w, "verify_runtime") as runtime, patch.object(w, "run_whisper", return_value="") as run:
            result = json.loads(rt.transcribe(audio_payload(b"\x00\x00" * 16000)))
        model.assert_called_once(); runtime.assert_called_once(); run.assert_called_once()
        self.assertEqual(result["text"], "")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["duration_seconds"], 1.0)

    def test_native_padded_timestamps_are_trimmed_to_authenticated_audio(self):
        rt = make_runtime()
        with patch.object(w, "verify_install"), patch.object(w, "verify_runtime"), patch.object(w, "run_whisper", return_value="[00:00:00.000 --> 00:00:02.060] you") as run:
            result = json.loads(rt.transcribe(audio_payload(b"\x00\x00" * 16000)))
        run.assert_called_once()
        self.assertEqual(result["text"], "you")
        self.assertEqual(result["duration_seconds"], 1.0)
        self.assertEqual(result["segments"], [{"start": 0.0, "end": 1.0, "text": "you"}])

    def test_subprocess_loader_environment_is_sanitized(self):
        injected = {"DYLD_LIBRARY_PATH": "/untrusted", "DYLD_INSERT_LIBRARIES": "/bad.dylib", "DYLD_FRAMEWORK_PATH": "/bad", "LD_PRELOAD": "/bad.so", "PYTHONPATH": "/bad", "GGML_BACKEND_PATH": "/bad", "PATH": "/usr/bin", "TMPDIR": "/private/tmp"}
        with patch.dict(w.os.environ, injected, clear=True):
            self.assertEqual(w.runtime_environment(), {"PATH": "/usr/bin", "TMPDIR": "/private/tmp"})
            with patch.object(w.subprocess, "run", return_value=subprocess.CompletedProcess([],0,b"whisper.cpp version: 1.9.5-dev")) as run:
                w.verify_runtime_version(Path("/approved/runtime"))
                self.assertEqual(run.call_args.kwargs["env"], {"PATH": "/usr/bin", "TMPDIR": "/private/tmp"})

    def test_protocol_mismatch_rejected_before_any_work(self):
        rt = make_runtime()
        pcm = b"\x00\x00" * 1600
        request = audio_payload(pcm)
        request["protocol"] = "transcription-v0"
        with self.assertRaises(w.InputValidationError):
            rt.transcribe(request)


if __name__ == "__main__":
    unittest.main()
