#!/usr/bin/env python3
"""Regression checks for release integrity and archive safety boundaries."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest

import verify_whisper_release as release


class WhisperReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = {name: (release.ROOT / path).read_bytes() for name, path in release.FILES.items()}

    def test_committed_approved_artifacts_pass(self):
        release.verify(self.files)

    def test_runtime_and_audio_corruption_fail(self):
        for name in ("whisper-cli", "smoke/audio.wav"):
            with self.subTest(name=name):
                files = dict(self.files)
                files[name] = files[name][:-1] + bytes([files[name][-1] ^ 1])
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    release.verify(files)

    def test_manifest_cannot_extend_approved_runtime_pin(self):
        files = dict(self.files)
        manifest = json.loads(files["whisper_runtime_files.json"])
        manifest["sha256"] = "0" * 64
        files["whisper_runtime_files.json"] = json.dumps(manifest).encode()
        with self.assertRaisesRegex(ValueError, "manifest mismatch: sha256"):
            release.verify(files)

    def test_private_smoke_words_and_wrong_sample_count_fail(self):
        for key, value in (("expected_contains", ["ready", "for", "service"]), ("sample_count", 1)):
            with self.subTest(key=key):
                files = dict(self.files)
                smoke = json.loads(files["smoke/smoke.json"])
                smoke[key] = value
                files["smoke/smoke.json"] = json.dumps(smoke).encode()
                with self.assertRaisesRegex(ValueError, "smoke manifest mismatch"):
                    release.verify(files)

    @staticmethod
    def macho(path=b"/usr/lib/libSystem.B.dylib", extra=b"", cpu=0x0100000C):
        name = path + b"\0"
        size = (24 + len(name) + 7) // 8 * 8
        dylib = struct.pack("<6I", 0xC, size, 24, 0, 0, 0) + name
        dylib += bytes(size - len(dylib))
        commands = dylib + extra
        return struct.pack("<8I", 0xFEEDFACF, cpu, 0, 2, 1 + bool(extra), len(commands), 0, 0) + commands

    def test_non_system_dylibs_and_rpath_fail(self):
        for path in (b"@rpath/libggml.dylib", b"/opt/homebrew/lib/libggml.dylib"):
            with self.assertRaisesRegex(ValueError, "non-system dynamic dependency"):
                release.macho_dependencies(self.macho(path))
        with self.assertRaisesRegex(ValueError, "LC_RPATH"):
            release.macho_dependencies(self.macho(extra=struct.pack("<2I", 0x8000001C, 8)))

    def test_other_architecture_and_truncated_commands_fail(self):
        with self.assertRaisesRegex(ValueError, "arm64 executable"):
            release.macho_dependencies(self.macho(cpu=0x01000007))
        with self.assertRaisesRegex(ValueError, "load-command table"):
            release.macho_dependencies(self.macho()[:-1])

    def test_package_roundtrip_and_deterministic_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            release.package("v1.10.0", output)
            archive = output / "whisper-stt-macos-arm64-v1.10.0.tar.gz"
            first = archive.read_bytes()
            checksum = archive.with_name(archive.name + ".sha256").read_text()
            self.assertEqual(checksum, f"{hashlib.sha256(first).hexdigest()}  {archive.name}\n")
            release.package("v1.10.0", output)
            self.assertEqual(first, archive.read_bytes())
            release.verify_archive(archive, "v1.10.0")
            with self.assertRaisesRegex(ValueError, "prefix"):
                release.verify_archive(archive, "v1.10.1")

    def test_archive_rejects_missing_extra_duplicate_symlink_and_wrong_mode(self):
        for change in ("missing", "extra", "duplicate", "symlink", "mode"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "tampered.tar.gz"
                with tarfile.open(path, "w:gz") as archive:
                    names = list(self.files)
                    if change == "missing":
                        names.remove("smoke/audio.wav")
                    if change == "extra":
                        names.append("../unexpected")
                    if change == "duplicate":
                        names.append("whisper-cli")
                    for name in names:
                        entry = tarfile.TarInfo(f"whisper-stt-macos-arm64-v1.10.0/{name}")
                        data = self.files.get(name, b"unexpected")
                        entry.size = len(data)
                        entry.mode = 0o755 if name == "whisper-cli" else 0o644
                        if name == "whisper-cli" and change == "symlink":
                            entry.type = tarfile.SYMTYPE
                            entry.linkname = "/tmp/whisper-cli"
                            entry.size = 0
                        if name == "whisper-cli" and change == "mode":
                            entry.mode = 0o644
                        archive.addfile(entry, io.BytesIO(data) if entry.isfile() else None)
                with self.assertRaises(ValueError):
                    release.verify_archive(path, "v1.10.0")


if __name__ == "__main__":
    unittest.main()
