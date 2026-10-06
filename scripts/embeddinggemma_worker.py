#!/usr/bin/env python3
"""Candidate EG2 JSONL worker. Listing and deployment require a separate release.

The caller owns the interpreter, verified snapshot and smoke fixtures. Only inline
media enters this protocol; private temporary files never enter logs or survive a job.
"""
import argparse
import array
import binascii
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

PROTOCOL = "embedding-multimodal-v1"
OFFICIAL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_REVISION = "1a4ffddb7905d3f63486748deabe091a01fb6201"
RUNTIME_REVISION = "30f177f03cbcb42bc2f65496458de79f51b80c28"
MAX_FRAME = 24 * 1024 * 1024
MAX_AGGREGATE_TEXT_BYTES = 10 * 1024 * 1024
TEXT_TEMPLATE_RESERVE = 12
DIMENSIONS = (768, 512, 256, 128)
PREFIXES = {"query": "task: search result | query: ", "document": "title: none | text: ", "unspecified": ""}
MIME = {"image": {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}, "audio": {"audio/wav": ".wav", "audio/flac": ".flac", "audio/mpeg": ".mp3"}, "video": {"video/mp4": ".mp4"}}


class InputValidationError(ValueError):
    """A known explicit input/media policy failure; no library exception inherits it."""


def verify_install(snapshot):
    for name, version in (("mlx", "0.32.3"), ("transformers", "5.19.0")):
        if importlib.metadata.version(name) != version:
            raise RuntimeError("runtime version mismatch")
    direct = json.loads(importlib.metadata.distribution("mlx-vlm").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != RUNTIME_REVISION:
        raise RuntimeError("runtime revision mismatch")
    manifest = json.loads(Path(__file__).with_name("embeddinggemma_model_files.json").read_text())
    for name, size, sha in manifest:
        file = snapshot / name
        if not file.is_file() or file.stat().st_size != size:
            raise RuntimeError("model asset mismatch")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha:
            raise RuntimeError("model asset digest mismatch")


def check_duration(actual, declared, maximum):
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (actual, declared)):
        raise InputValidationError("invalid duration")
    # Small decoder rounding is tolerated; understated work is rejected.
    if min(actual, declared) <= 0 or max(actual, declared) > maximum or actual > declared + 0.001:
        raise InputValidationError("media duration mismatch")


def verify_video(processor, file, declared):
    """Pinned video processor signature: _decode_video(path, sampler)."""
    sampler = processor.video_processor.sample_frames
    def verify_sample(metadata, **kwargs):
        check_duration(metadata.duration, declared, 32)
        frames = sampler(metadata, **kwargs)
        if len(frames) > 32:
            raise InputValidationError("video frame count exceeds limit")
        return frames
    return processor.video_processor._decode_video(str(file), verify_sample)


def verify_parameter_dtypes(parameters, mx, flatten):
    leaves = flatten(parameters)
    if not leaves or any(value.dtype not in (mx.bfloat16, mx.float32) for _, value in leaves):
        raise RuntimeError("model parameters must be BF16 or FP32")


def processed_usage(item,active_size,image,audio,video):
    row={"text":int(active_size)-image-audio-video,"image":image,"audio":audio,"video":video}
    if active_size<=0 or min(row.values())<0:
        raise RuntimeError("invalid processor usage")
    for kind,tokens in (("image",image),("audio",audio),("video",video)):
        present=any(part["type"]==kind for part in item["content"])
        if present!=(tokens>0):
            raise RuntimeError("processor usage modality mismatch")
    if any(part["type"]=="text" for part in item["content"]) and row["text"]==0:
        raise RuntimeError("processor omitted requested text")
    image_count=sum(part["type"]=="image" for part in item["content"])
    video_bound=sum(math.ceil(part["duration_seconds"])*140 for part in item["content"] if part["type"]=="video")
    if image>image_count*280 or video>video_bound:
        raise RuntimeError("processor media usage exceeds declared bound")
    if active_size>8192:
        raise InputValidationError("processed sample exceeds budget")
    return row


def validate_text_budget(items, input_type):
    """Cheap preflight runs before media decoding, tokenization or model execution."""
    if not isinstance(input_type,str) or input_type not in PREFIXES:
        raise InputValidationError("invalid input type")
    prefix_bytes = len(PREFIXES[input_type].encode("utf-8"))
    aggregate = 0
    for item in items:
        if not isinstance(item,dict) or not isinstance(item.get("content"),list):
            raise InputValidationError("invalid item")
        text_bytes = 0
        for part in item["content"]:
            if not isinstance(part,dict):
                raise InputValidationError("invalid part")
            if part.get("type") == "text":
                text = part.get("text")
                if set(part) != {"type","text"} or not isinstance(text,str) or not text.strip():
                    raise InputValidationError("invalid text")
                try:
                    text_bytes += len(text.encode("utf-8"))
                except UnicodeEncodeError:
                    raise InputValidationError("invalid UTF-8 text") from None
        aggregate += text_bytes
        if aggregate > MAX_AGGREGATE_TEXT_BYTES:
            raise InputValidationError("aggregate text exceeds limit")
        media_bound=0
        for part in item["content"]:
            if part.get("type")=="image": media_bound+=280
            elif part.get("type") in ("audio","video"):
                maximum=30 if part["type"]=="audio" else 32
                duration=part.get("duration_seconds")
                check_duration(duration,duration,maximum)
                media_bound+=math.ceil(duration)*(25 if part["type"]=="audio" else 140)
        if text_bytes+prefix_bytes+TEXT_TEMPLATE_RESERVE+media_bound>8192:
            raise InputValidationError("input exceeds processed sample budget")


def normalize_native_output(vectors, rows, dimensions, np):
    """Broken model output is a runtime failure, never a client validation error."""
    if vectors.shape != (rows,768) or not np.isfinite(vectors).all():
        raise RuntimeError("invalid native output")
    native_norms = np.linalg.norm(vectors,axis=1)
    if not np.isfinite(native_norms).all() or not np.allclose(native_norms,1.0,rtol=0,atol=0.001):
        raise RuntimeError("native output is not normalized")
    vectors = vectors[:,:dimensions]
    norms = np.linalg.norm(vectors,axis=1,keepdims=True)
    if not np.isfinite(norms).all() or (norms<=0).any():
        raise RuntimeError("invalid native norm")
    vectors /= norms
    return vectors


MP4_BRANDS = {b"isom",b"iso2",b"iso3",b"iso4",b"iso5",b"iso6",b"iso7",b"iso8",b"iso9",b"mp41",b"mp42",b"avc1",b"hvc1",b"hev1",b"dash",b"M4V ",b"cmfc",b"cmfs",b"msdh",b"msix"}

def validate_mp4_container(raw):
    if len(raw)<16 or raw[4:8]!=b"ftyp":
        raise InputValidationError("video MIME mismatch")
    size = int.from_bytes(raw[:4],"big")
    start = 8
    if size==1:
        if len(raw)<24: raise InputValidationError("invalid video container")
        size = int.from_bytes(raw[8:16],"big")
        start = 16
    if size<start+8 or size>len(raw) or (size-start-8)%4:
        raise InputValidationError("invalid video container")
    brands = [raw[start:start+4]] + [raw[index:index+4] for index in range(start+8,size,4)]
    if any(brand not in MP4_BRANDS for brand in brands):
        raise InputValidationError("unsupported video container brand")


def dispatch_request(runtime, request):
    """Only authenticated request input failures are nonfatal; framing stays fail-closed."""
    if not isinstance(request,dict) or request.get("type")!="embed" or request.get("protocol")!=PROTOCOL or type(request.get("request_id")) is not int or not 0<=request["request_id"]<2**64:
        raise RuntimeError("worker request protocol mismatch")
    try:
        return runtime.embed(request)
    except InputValidationError:
        return {"type":"request_error","request_id":request["request_id"],"code":"invalid_input"}
    except Exception:
        raise RuntimeError("embedding runtime library failed") from None


def decode_audio_bounded(file,mime,declared,miniaudio_module=None):
    """Verify source metadata before streaming at most 480000 mono 16 kHz samples."""
    if miniaudio_module is None:
        import miniaudio as miniaudio_module
    audio=miniaudio_module
    check_duration(declared,declared,30)
    try:
        info=audio.get_file_info(str(file))
    except (audio.MiniaudioError,OSError,EOFError):
        raise InputValidationError("invalid audio metadata") from None
    formats={"audio/wav":audio.FileFormat.WAV,"audio/flac":audio.FileFormat.FLAC,"audio/mpeg":audio.FileFormat.MP3}
    if info.file_format!=formats[mime] or info.nchannels<=0 or info.sample_rate<=0 or info.num_frames<=0:
        raise InputValidationError("audio metadata or MIME mismatch")
    check_duration(info.duration,declared,30)
    check_duration(info.num_frames/info.sample_rate,declared,30)
    samples=array.array("f")
    stream=None
    try:
        stream=audio.stream_file(str(file),output_format=audio.SampleFormat.FLOAT32,nchannels=1,sample_rate=16000,frames_to_read=4096)
        for chunk in stream:
            if len(chunk)>4096 or len(samples)+len(chunk)>480000:
                raise InputValidationError("decoded audio exceeds sample limit")
            if not chunk: continue
            check_duration((len(samples)+len(chunk))/16000,declared,30)
            if not all(math.isfinite(sample) for sample in chunk):
                raise InputValidationError("invalid audio samples")
            samples.extend(chunk)
    except (audio.MiniaudioError,OSError,EOFError):
        raise InputValidationError("invalid audio stream") from None
    finally:
        if stream is not None: stream.close()
    check_duration(len(samples)/16000,declared,30)
    return samples


class Runtime:
    def __init__(self, model_path, smoke_dir):
        if sys.platform != "darwin" or os.uname().machine != "arm64":
            raise RuntimeError("candidate runtime requires Apple Silicon")
        snapshot = Path(model_path).resolve(strict=True)
        verify_install(snapshot)
        import mlx.core as mx
        from mlx_vlm import load
        self.mx = mx
        self.model, self.processor = load(str(snapshot))
        from mlx.utils import tree_flatten
        verify_parameter_dtypes(self.model.parameters(), mx, tree_flatten)
        self.smoke_dir = Path(smoke_dir).resolve(strict=True)
        # Guard verified video metadata BEFORE the upstream sampler starts decoding.
        original_sample = self.processor.video_processor.sample_frames
        def bounded_sample(metadata, **kwargs):
            duration = metadata.duration
            if duration is None or not math.isfinite(duration) or not 0 < duration <= 32:
                raise InputValidationError("video duration exceeds limit")
            frames = original_sample(metadata, fps=1, max_frames=32)
            if len(frames) > 32:
                raise InputValidationError("video frame count exceeds limit")
            return frames
        self.processor.video_processor.sample_frames = bounded_sample
        self._ready = None

    def _conversations(self, items, input_type, directory):
        from PIL import Image
        import numpy as np
        conversations, total_bytes = [], 0
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != {"content"} or not isinstance(item["content"], list) or not 1 <= len(item["content"]) <= 16:
                raise InputValidationError("invalid item")
            content, counts, image_bytes = [], {"image": 0, "audio": 0, "video": 0}, 0
            for part_index, part in enumerate(item["content"]):
                kind = part.get("type")
                if kind == "text":
                    if set(part) != {"type", "text"} or not isinstance(part["text"], str) or not part["text"].strip():
                        raise InputValidationError("invalid text")
                    content.append({"type": "text", "text": part["text"]})
                    continue
                if kind not in MIME:
                    raise InputValidationError("unsupported part")
                allowed = {"type", "media"} | ({"duration_seconds"} if kind in ("audio", "video") else set())
                if set(part) != allowed:
                    raise InputValidationError("invalid media part")
                media = part["media"]
                if not isinstance(media, dict) or set(media) != {"encoding", "mime_type", "data", "sha256"} or media.get("encoding") != "base64" or media.get("mime_type") not in MIME[kind]:
                    raise InputValidationError("unsupported transport")
                cap = (16 if kind == "video" else 8) * 1024 * 1024
                encoded = media.get("data")
                if not isinstance(encoded, str) or len(encoded) > ((cap + 2) // 3) * 4:
                    raise InputValidationError("encoded media exceeds limit")
                if not encoded.isascii(): raise InputValidationError("invalid media base64")
                try:
                    raw=base64.b64decode(encoded,validate=True)
                except binascii.Error:
                    raise InputValidationError("invalid media base64") from None
                if base64.b64encode(raw).decode() != encoded:
                    raise InputValidationError("noncanonical base64")
                if not raw or len(raw) > cap:
                    raise InputValidationError("decoded media exceeds limit")
                if not isinstance(media["sha256"],str) or hashlib.sha256(raw).hexdigest() != media["sha256"]:
                    raise InputValidationError("media digest mismatch")
                counts[kind] += 1
                total_bytes += len(raw)
                image_bytes += len(raw) if kind == "image" else 0
                if counts["image"] > 8 or counts["audio"] > 1 or counts["video"] > 1 or image_bytes > 8*1024*1024 or total_bytes > 20*1024*1024:
                    raise InputValidationError("media aggregate exceeds limit")
                file = Path(directory) / f"{index}-{part_index}{MIME[kind][media['mime_type']]}"
                file.write_bytes(raw)
                if kind == "image":
                    try:
                        with Image.open(io.BytesIO(raw)) as image:
                            if image.width*image.height>16_000_000 or Image.MIME.get(image.format)!=media["mime_type"]:
                                raise InputValidationError("image shape or MIME mismatch")
                            image.verify()
                    except (OSError,SyntaxError,EOFError,Image.DecompressionBombError):
                        raise InputValidationError("invalid image media") from None
                elif kind == "audio":
                    magic = {"audio/wav": raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "audio/flac": raw[:4] == b"fLaC", "audio/mpeg": raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")}
                    if not magic[media["mime_type"]]:
                        raise InputValidationError("audio MIME mismatch")
                    waveform=np.asarray(decode_audio_bounded(file,media["mime_type"],part["duration_seconds"]),dtype=np.float32)
                else:
                    validate_mp4_container(raw)
                    declared = part["duration_seconds"]
                    check_duration(declared, declared, 32)
                    try:
                        verify_video(self.processor,file,declared)
                    except (OSError,EOFError):
                        raise InputValidationError("invalid video media") from None
                content.append({"type":kind,"url":waveform if kind=="audio" else str(file)})
            conversation = []
            if PREFIXES[input_type]:
                conversation.append({"role": "system", "content": PREFIXES[input_type]})
            conversation.append({"role": "user", "content": content})
            conversations.append(conversation)
        return conversations

    def embed(self, request):
        if request.get("type") != "embed" or request.get("protocol") != PROTOCOL or request.get("input_type") not in PREFIXES:
            raise InputValidationError("request identity mismatch")
        request_id = request.get("request_id")
        if type(request_id) is not int or not 0 <= request_id < 2**64:
            raise InputValidationError("invalid request ID")
        items = request.get("input")
        if not isinstance(items, list) or not 1 <= len(items) <= 16 or request.get("dimensions", 768) not in DIMENSIONS:
            raise InputValidationError("invalid batch or dimensions")
        validate_text_budget(items,request["input_type"])
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="sgl-eg2-") as directory:
            try:
                conversations=self._conversations(items,request["input_type"],directory)
                inputs=self.processor.apply_chat_template(conversations,tokenize=True,return_dict=True,return_tensors="mlx",images_kwargs={"max_soft_tokens":280},videos_kwargs={"fps":1,"max_frames":32,"max_soft_tokens":140},audio_kwargs={"sampling_rate":16000})
            except InputValidationError:
                raise
            except Exception:
                raise RuntimeError("embedding media processor failed") from None
            ids = np.asarray(inputs["input_ids"])
            mask = np.asarray(inputs["attention_mask"]) if "attention_mask" in inputs else np.ones(ids.shape)
            usage = {"text":0,"image":0,"audio":0,"video":0}
            item_usage = []
            for row, row_mask, item in zip(ids, mask, items):
                active = row[row_mask.astype(bool)]
                image = int(np.sum(active == self.processor.image_token_id))
                audio = int(np.sum(active == self.processor.audio_token_id))
                video = int(np.sum(active == self.processor.video_token_id))
                row_usage = processed_usage(item,active.size,image,audio,video)
                item_usage.append(row_usage)
                for kind, tokens in row_usage.items():
                    usage[kind] += tokens
            try:
                vectors = self.model(**inputs).text_embeds.astype(self.mx.float32)
                self.mx.eval(vectors)
                vectors = np.asarray(vectors,dtype=np.float32)
            except Exception:
                raise RuntimeError("embedding model execution failed") from None
            vectors = normalize_native_output(vectors,len(items),request.get("dimensions",768),np)
            return {"type":"result","request_id":request_id,"vectors":vectors.tolist(),"usage":usage,"item_usage":item_usage}

    def ready(self):
        if self._ready is None:
            def fixture(kind, filename, mime, duration=None):
                raw = (self.smoke_dir/filename).read_bytes()
                part = {"type":kind,"media":{"encoding":"base64","mime_type":mime,"data":base64.b64encode(raw).decode(),"sha256":hashlib.sha256(raw).hexdigest()}}
                if duration is not None:
                    part["duration_seconds"] = duration
                return part
            # Fixture manifests are supplied only after real canaries; durations are verified.
            manifest = json.loads((self.smoke_dir/"durations.json").read_text())
            image = fixture("image","image.png","image/png")
            audio = fixture("audio","audio.wav","audio/wav",manifest["audio_seconds"])
            video = fixture("video","video.mp4","video/mp4",manifest["video_seconds"])
            text = {"type":"text","text":"startup smoke"}
            smoke = []
            for content in ([text],[image],[audio],[video],[text,image,audio,video]):
                output = self.embed({"type":"embed","protocol":PROTOCOL,"request_id":0,"input_type":"unspecified","dimensions":768,"input":[{"content":content}]})
                smoke.append(output["vectors"][0])
            self._ready = {"type":"ready","protocol":PROTOCOL,"runtime":"mlx-vlm","processor_revision":RUNTIME_REVISION,"model_revision":MODEL_REVISION,"modalities":["text","image","audio","video"],"dimensions":list(DIMENSIONS),"smoke_vectors":smoke}
        return self._ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--smoke-dir", required=True)
    args = parser.parse_args()
    protocol_out = sys.stdout
    # Suppress native/library stdout and tracebacks: stdout is exclusively bounded frames.
    with open(os.devnull,"w") as null:
        os.dup2(null.fileno(), 2)
        saved = os.dup(1)
        os.dup2(null.fileno(),1)
        protocol_out = os.fdopen(saved,"w", buffering=1)
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            try:
                runtime = Runtime(args.model_path,args.smoke_dir)
                protocol_out.write(json.dumps(runtime.ready(), allow_nan=False)+"\n")
                while True:
                    frame = sys.stdin.buffer.readline(MAX_FRAME+1)
                    if not frame:
                        return 0
                    if len(frame)>MAX_FRAME or not frame.endswith(b"\n"):
                        return 2
                    result = dispatch_request(runtime,json.loads(frame))
                    encoded = json.dumps(result, allow_nan=False)
                    if len(encoded.encode()) > 1024*1024:
                        return 2
                    protocol_out.write(encoded+"\n")
            except Exception:
                return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
