#!/usr/bin/env python3
"""Verify and package the exact reviewed STT artifacts; Python stdlib only."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tarfile
import wave

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SHA = "4ac1f78373fa19037ff036c66586db2b775e0b21785c320953a1960bcb425405"
RUNTIME_BYTES = 7760160
SOURCE = "d1be6fde11ac6e0407606b4e42fe72d34add8037"
AUDIO_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
FILES = {
    "whisper-cli": "assets/whisper/runtime/macos-arm64/whisper-cli",
    "whisper_runtime_files.json": "scripts/whisper_runtime_files.json",
    "smoke/audio.wav": "assets/whisper/smoke/audio.wav",
    "smoke/smoke.json": "assets/whisper/smoke/smoke.json",
    "LICENSE": "assets/whisper/runtime/LICENSE",
    "README.md": "assets/whisper/README.md",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def macho_dependencies(data):
    """Read thin little-endian 64-bit Mach-O commands on macOS or Linux."""
    require(len(data) >= 32, "truncated Mach-O header")
    magic, cpu, _subtype, kind, count, size, _flags, _reserved = struct.unpack_from("<8I", data)
    require(magic == 0xFEEDFACF and cpu == 0x0100000C and kind == 2,
            "runtime must be a thin Mach-O arm64 executable")
    require(32 + size <= len(data) and count <= size // 8, "invalid load-command table")
    end = 32 + size
    offset = 32
    dependencies = []
    for _ in range(count):
        require(offset + 8 <= end, "truncated load command")
        command, length = struct.unpack_from("<2I", data, offset)
        require(length >= 8 and length % 8 == 0 and offset + length <= end,
                "invalid load-command length")
        require(command != 0x8000001C, "LC_RPATH is forbidden")
        # LOAD, WEAK, REEXPORT, LAZY_LOAD, and UPWARD_LOAD dylibs.
        if command in (0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023):
            require(length >= 24, "truncated dylib command")
            name_offset = struct.unpack_from("<I", data, offset + 8)[0]
            require(24 <= name_offset < length, "invalid dylib name offset")
            name_bytes = data[offset + name_offset:offset + length]
            require(b"\0" in name_bytes, "unterminated dylib name")
            name = name_bytes.split(b"\0", 1)[0].decode("utf-8")
            require(name.startswith(("/usr/lib/", "/System/Library/")),
                    f"non-system dynamic dependency: {name}")
            dependencies.append(name)
        offset += length
    require(offset == end and dependencies, "invalid or empty dependency table")
    return dependencies


def verify(files):
    data = files["whisper-cli"]
    require(len(data) == RUNTIME_BYTES, "runtime size differs from reviewed pin")
    require(hashlib.sha256(data).hexdigest() == RUNTIME_SHA, "runtime SHA differs from reviewed pin")
    manifest = json.loads(files["whisper_runtime_files.json"])
    for key, expected in {
        "schema": 1, "platform": "macos-aarch64", "filename": "whisper-cli",
        "sha256": RUNTIME_SHA, "bytes": RUNTIME_BYTES, "source_commit": SOURCE,
        "source_repository": "https://github.com/ggml-org/whisper.cpp",
        "runtime_version": "1.9.5-dev", "license": "MIT", "rpath": [],
    }.items():
        require(manifest.get(key) == expected, f"runtime manifest mismatch: {key}")
    require(macho_dependencies(data) == manifest["dynamic_dependencies"],
            "Mach-O dependencies differ from approved manifest")
    smoke = json.loads(files["smoke/smoke.json"])
    require(hashlib.sha256(files["smoke/audio.wav"]).hexdigest() == AUDIO_SHA,
            "smoke audio SHA differs from upstream pin")
    for key, expected in {
        "audio_sha256": AUDIO_SHA, "expected_contains": ["fellow", "americans", "country"],
        "min_chars": 20, "sample_rate": 16000, "channels": 1, "bits": 16,
        "sample_count": 176000, "duration_seconds": 11,
    }.items():
        require(smoke.get(key) == expected, f"smoke manifest mismatch: {key}")
    require(smoke["provenance"]["source_commit"] == SOURCE, "smoke source commit mismatch")
    require(smoke["provenance"]["source_url"] ==
            f"https://github.com/ggml-org/whisper.cpp/blob/{SOURCE}/samples/jfk.wav",
            "smoke upstream attribution mismatch")
    with wave.open(io.BytesIO(files["smoke/audio.wav"]), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                 wav.getnframes(), wav.getcomptype()) == (1, 2, 16000, 176000, "NONE"),
                "smoke WAV must be exactly 11 seconds of 16 kHz mono PCM16")
    require(files["LICENSE"] == (ROOT / FILES["LICENSE"]).read_bytes(), "runtime license mismatch")
    require(files["README.md"] == (ROOT / FILES["README.md"]).read_bytes(), "bundle attribution mismatch")
    require(files["whisper_runtime_files.json"] == (ROOT / FILES["whisper_runtime_files.json"]).read_bytes(),
            "archive build manifest differs from reviewed source manifest")


def source_files():
    files = {name: (ROOT / path).read_bytes() for name, path in FILES.items()}
    verify(files)
    runtime = ROOT / FILES["whisper-cli"]
    require(runtime.stat().st_mode & 0o111 != 0, "source runtime must be executable")
    if sys.platform == "darwin":
        # Native inspection supplements the portable command parser before packaging.
        subprocess.run(["/usr/bin/codesign", "--verify", "--strict", str(runtime)], check=True)
        output = subprocess.check_output(["/usr/bin/otool", "-l", str(runtime)], text=True)
        require("LC_RPATH" not in output, "native inspection found LC_RPATH")
        subprocess.run(["/usr/bin/file", str(runtime)], check=True)
    return files


def bundle_name(version):
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version), "expected version vX.Y.Z")
    return f"whisper-stt-macos-arm64-{version}"


def verify_archive(path, version):
    prefix = bundle_name(version) + "/"
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            require(entry.name.startswith(prefix), "unexpected archive prefix")
            name = entry.name[len(prefix):]
            require(name in FILES and name not in files and entry.isfile(),
                    "archive contains unexpected, duplicate, or non-file member")
            require(entry.size == (ROOT / FILES[name]).stat().st_size, "unexpected archive member size")
            require(entry.mode == (0o755 if name == "whisper-cli" else 0o644), "unexpected archive permissions")
            files[name] = archive.extractfile(entry).read()
    require(set(files) == set(FILES), "archive is missing required runtime/smoke/provenance files")
    verify(files)


def package(version, output):
    files = source_files()
    name = bundle_name(version)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.tar.gz"
    # Stable tar metadata and timestamp-free gzip keep packaging reproducible.
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for filename in sorted(files):
                entry = tarfile.TarInfo(f"{name}/{filename}")
                entry.size = len(files[filename])
                entry.mode = 0o755 if filename == "whisper-cli" else 0o644
                archive.addfile(entry, io.BytesIO(files[filename]))
    verify_archive(path, version)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n", encoding="ascii")
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "package", "archive"))
    parser.add_argument("--version")
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        source_files()
    elif args.command == "package":
        package(args.version or "", args.output)
    else:
        require(args.archive is not None, "--archive is required")
        verify_archive(args.archive, args.version or "")
    print("PASS: reviewed runtime, arm64 Mach-O/system-only dylibs/no RPATH, manifest, and public smoke")


if __name__ == "__main__":
    main()
