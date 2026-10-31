"""Supervised mflux text-to-image worker (FLUX.2 klein 4B, 4-bit) for the Grid `image-v1` lane.

The Rust supervisor (src/image.rs) owns this process: it writes one JSON frame per line on
stdin and reads one JSON frame per line on stdout. Nothing else is printed to stdout.

Startup:
  1. Re-hash every pinned model file against the manifest; refuse on any mismatch.
  2. Check the pinned runtime (mflux 0.24.0).
  3. Load the model ONCE from the operator's local directory (offline, never downloads) and
     keep it resident (hot) for every request that follows.
  4. Generate a small fixed smoke image and require a real, non-blank PNG.
  5. Print one `ready` frame.

Requests:  {"type":"generate","request_id":<u64>,"protocol":"image-v1","prompt_b64":...,
            "prompt_sha256":...,"width":1024,"height":1024,"seed":<u32>,"steps":4}
Replies:   {"type":"result","request_id":...,"width":...,"height":...,"bytes":...,
            "sha256":...,"png_b64":...}
           {"type":"request_error","request_id":...,"code":"invalid_input"}

Generation recipe (the one proven in the desktop app's Create studio, mflux-generate-flux2):
Flux2Klein(model_config=flux2-klein-4b, model_path=<dir>, quantize=None) because the weights are
already 4-bit; generate_image(seed, prompt, width, height, guidance=1.0, num_inference_steps=4,
scheduler="flow_match_euler_discrete"); the PIL image is written as PNG.

FAKE mode (tests only): SGL_IMAGE_WORKER_FAKE=<mode> skips the model and mflux entirely and
answers with a tiny, valid, solid-colour PNG of the requested size. The Rust supervisor removes
this variable from the environment of every production worker and refuses a `ready` frame that
says `fake: true` unless the test harness asked for it. Modes: ok, invalid, lie, crash, hang,
hang_always (alias of hang), hang_first (hangs once per model dir, then serves).
"""

import argparse
import base64
import gc
import hashlib
import io
import json
import os
import struct
import sys
import time
import zlib

PROTOCOL = "image-v1"
MODEL_ID = "flux2-klein-4b-q4"
BASE_MODEL = "flux2-klein-4b"
RUNTIME = "mflux"
RUNTIME_REVISION = "media-2:mflux-0.24.0"
PACKAGES = {"mflux": "0.24.0"}
OUTPUT_FORMAT = "png"
STEPS = 4
GUIDANCE = 1.0
SCHEDULER = "flow_match_euler_discrete"
MAX_PROMPT_BYTES = 2000
MAX_PNG_BYTES = 6_000_000
SIZES = [[1024, 1024], [1152, 768], [768, 1152], [1280, 720], [720, 1280]]
SMOKE_PROMPT = "a red apple on a white table, studio photo"
SMOKE_SIZE = (256, 256)
SMOKE_SEED = 7
FAKE = os.environ.get("SGL_IMAGE_WORKER_FAKE") or None

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_out = sys.stdout
sys.stdout = sys.stderr  # libraries that print (mflux, tqdm) must never corrupt the frame stream


def emit(frame):
    _out.write(json.dumps(frame, separators=(",", ":")) + "\n")
    _out.flush()


def fail(reason):
    sys.stderr.write(f"image worker: {reason}\n")
    sys.stderr.flush()
    sys.exit(2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(files):
    """One line per file sorted by path: `<sha256>  <size>  <path>\\n` (two spaces)."""
    text = "".join(f"{f['sha256']}  {f['size']}  {f['path']}\n"
                   for f in sorted(files, key=lambda f: f["path"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_model(model_dir, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("model") != MODEL_ID:
        fail("manifest is for a different model")
    for entry in manifest["files"]:
        rel = entry["path"]
        parts = rel.split("/")
        if rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
            fail("manifest path invalid")
        # No component may be a symlink: a link could point the hashed bytes and the loaded
        # bytes at different files.
        path = model_dir
        for part in parts:
            path = os.path.join(path, part)
            if os.path.islink(path):
                fail(f"model path is a symlink: {rel}")
        if not os.path.isfile(path):
            fail(f"model file missing: {rel}")
        if os.path.getsize(path) != entry["size"]:
            fail(f"model file size mismatch: {rel}")
        if sha256_file(path) != entry["sha256"]:
            fail(f"model file hash mismatch: {rel}")
    digest = manifest_digest(manifest["files"])
    if digest != manifest["model_sha256"]:
        fail("model manifest digest mismatch")
    return manifest, digest


def check_packages():
    from importlib import metadata

    for name, version in PACKAGES.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            fail(f"package missing: {name}")
        if found != version:
            fail(f"package version mismatch: {name} {found} != {version}")


def png_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def solid_png(width, height, rgb):
    """A valid 8-bit RGB PNG of one colour, built with the standard library only."""
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    return (b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(raw, 9))
            + png_chunk(b"IEND", b""))


def png_size(data):
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


class Engine:
    """The loaded model, kept resident between requests."""

    def __init__(self, model_dir):
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.flux2.variants import Flux2Klein

        # Keep the text encoder and transformer resident (the CLI's memory saver evicts them
        # after every image); this worker registers no callbacks at all.
        try:
            import mlx.core as mx

            mx.set_cache_limit(1 << 30)
        except Exception:  # noqa: BLE001
            pass
        config = ConfigResolution.resolve_restricted(
            model_dir, BASE_MODEL, model_path=model_dir, base_model=BASE_MODEL)
        self.model = Flux2Klein(model_config=config, quantize=None, model_path=model_dir)

    def generate(self, prompt, width, height, seed):
        image = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            width=width,
            height=height,
            guidance=GUIDANCE,
            num_inference_steps=STEPS,
            scheduler=SCHEDULER,
        )
        buf = io.BytesIO()
        # PIL directly: GeneratedImage.save() embeds metadata (the operator's local model path).
        image.image.save(buf, format="PNG")
        self.trim()
        return buf.getvalue()

    @staticmethod
    def trim():
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()


class FakeEngine:
    def __init__(self, mode, model_dir):
        self.mode = mode
        self.marker = os.path.join(model_dir, ".fake-hung")

    def generate(self, prompt, width, height, seed):
        if self.mode in ("hang", "hang_always"):
            time.sleep(3600)
        if self.mode == "hang_first" and not os.path.exists(self.marker):
            open(self.marker, "w").close()
            time.sleep(3600)
        if self.mode == "crash":
            fail("fake crash")
        return solid_png(width, height, (seed % 256, (seed >> 8) % 256, len(prompt) % 256))


def blank(png):
    """True when the smoke image is a single flat colour (a broken pipeline)."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png)).convert("L")
        lo, hi = img.getextrema()
        return hi - lo < 8
    except Exception:  # noqa: BLE001
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    if args.model_id != MODEL_ID:
        fail("unknown model id")

    started = time.monotonic()
    if FAKE:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        digest = manifest_digest(manifest["files"])
        engine = FakeEngine(FAKE, args.model_dir)
        smoke = solid_png(SMOKE_SIZE[0], SMOKE_SIZE[1], (1, 2, 3))
    else:
        model_dir = os.path.realpath(args.model_dir)
        manifest, digest = verify_model(model_dir, args.manifest)
        check_packages()
        engine = Engine(model_dir)
        smoke = engine.generate(SMOKE_PROMPT, SMOKE_SIZE[0], SMOKE_SIZE[1], SMOKE_SEED)
        if png_size(smoke) != SMOKE_SIZE or blank(smoke):
            fail("smoke image implausible")

    emit({
        "type": "ready",
        "protocol": PROTOCOL,
        "runtime": RUNTIME,
        "runtime_revision": RUNTIME_REVISION,
        "model_id": MODEL_ID,
        "model_revision": manifest["revision"],
        "model_sha256": digest,
        "steps": STEPS,
        "sizes": SIZES,
        "output_format": OUTPUT_FORMAT,
        "max_prompt_bytes": MAX_PROMPT_BYTES,
        "smoke_seconds": round(time.monotonic() - started, 3),
        "fake": bool(FAKE),
    })

    for line in sys.stdin:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            fail("bad request frame")
        if not isinstance(frame, dict) or frame.get("type") != "generate":
            fail("unexpected frame")
        request_id = frame.get("request_id")
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            fail("request id invalid")

        def reject():
            emit({"type": "request_error", "request_id": request_id, "code": "invalid_input"})

        try:
            prompt_bytes = base64.b64decode(frame["prompt_b64"], validate=True)
            prompt = prompt_bytes.decode("utf-8")
            width, height, seed = frame["width"], frame["height"], frame["seed"]
        except Exception:  # noqa: BLE001
            reject()
            continue
        if (frame.get("protocol") != PROTOCOL or frame.get("steps") != STEPS
                or [width, height] not in SIZES
                or not isinstance(seed, int) or isinstance(seed, bool)
                or not (0 <= seed <= 0xFFFFFFFF)
                or not (1 <= len(prompt_bytes) <= MAX_PROMPT_BYTES)
                or "\x00" in prompt or not prompt.strip()
                or hashlib.sha256(prompt_bytes).hexdigest() != frame.get("prompt_sha256")):
            reject()
            continue
        if FAKE == "invalid":
            reject()
            continue
        try:
            data = engine.generate(prompt, width, height, seed)
        except Exception:  # noqa: BLE001
            # The prompt was already validated, so a generation failure is a runtime fault
            # (out of memory, Metal error, corrupted state). Exit: the node reports
            # image_runtime_failed and stops advertising the model.
            fail("generation failed")
        if png_size(data) != (width, height) or not (0 < len(data) <= MAX_PNG_BYTES):
            fail("generated image out of bounds")
        digest_hex = hashlib.sha256(data).hexdigest()
        if FAKE == "lie":
            digest_hex = "0" * 64
        emit({
            "type": "result",
            "request_id": request_id,
            "width": width,
            "height": height,
            "bytes": len(data),
            "sha256": digest_hex,
            "png_b64": base64.b64encode(data).decode("ascii"),
        })


if __name__ == "__main__":
    main()
