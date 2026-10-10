# Whisper transcription (STT) node

The dedicated `whisper-1` lane uses whisper.cpp without changing the chat,
vision, System One, Laya, or EmbeddingGemma 2 runtimes. It is implemented for
local verification; these changes do not enable the production Grid route.

## Immutable artifacts and platform gate

| Artifact | Pin |
| --- | --- |
| Model repository | `ggerganov/whisper.cpp` |
| Model revision | `5359861c739e955e79d9a303bcbc70fb988958b1` |
| File / size | `ggml-small.bin` / 487,601,967 bytes |
| Model SHA-256 | `1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b` |
| Runtime source | whisper.cpp `v1.9.5`, commit `d1be6fde11ac6e0407606b4e42fe72d34add8037` |
| Approved static macOS arm64 binary SHA-256 | `4ac1f78373fa19037ff036c66586db2b775e0b21785c320953a1960bcb425405` |
| Runtime binary size | 7,760,160 bytes |
| Model / runtime licenses | MIT / MIT |

License provenance: the pinned [converted model card](https://huggingface.co/ggerganov/whisper.cpp/blob/5359861c739e955e79d9a303bcbc70fb988958b1/README.md)
declares MIT; the pinned [runtime license](https://github.com/ggml-org/whisper.cpp/blob/d1be6fde11ac6e0407606b4e42fe72d34add8037/LICENSE)
is MIT. This corrects the earlier labour note that called the weights Apache-2.0.

Production readiness accepts only the compiled approved runtime manifest.
The runtime path is configurable; an operator-provided hash cannot extend the
manifest. Windows, Linux, and Intel Mac STT are gated until reproducible binary
hashes and real runtime evidence are approved. Other node workloads retain
their existing platform support.

`language_hint` in every sealed result echoes the authenticated reservation
(`auto` or the exact requested supported language); detected `language` remains
separate. The static runtime, including embedded Metal source, is hash- and
size-checked before startup and every native request. Only Apple system libraries
and frameworks remain dynamic; the binary has no LC_RPATH. The worker removes
DYLD/LD loader overrides, Python injection variables, and GGML external resource
paths before interpreter/native execution. Python runs in isolated mode.

The operator owns the read-only model and runtime files. The node downloads
neither artifact. Startup verifies exact model size/hash, runtime binary hash,
runtime version, and a real known-audio transcription. The Python worker repeats
artifact hash checks before each inference because whisper-cli reloads the files.

## Runtime build provenance

The exact flags and system dependency list are recorded in
[`scripts/whisper_runtime_files.json`](../scripts/whisper_runtime_files.json).
Source is an unmodified archive of the pinned runtime commit. The reviewed
Release build uses static project libraries (`BUILD_SHARED_LIBS=OFF`), embedded
Metal shaders, Accelerate, disabled dynamic backend loading, disabled OpenMP,
and no RPATH. Compiler: Apple clang 21.0.0 (`clang-2100.3.34.2`); CMake 4.3.4;
SDK 27.0; linker 27037.1; arm64 deployment target macOS 15.0. Actual runtime
canaries run on macOS 27.0.1; deployment targeting does not claim macOS 15 was
tested. C/C++ file-prefix maps remove source/build directory references.

Postprocessing is `strip -S -x`, then a timestamp-free ad-hoc signature with
identifier `cc.x402compute.whisper-cli`. `otool -L` must list only `/usr/lib/`
and `/System/Library/` dependencies, `otool -l` must contain no LC_RPATH,
and `codesign --verify --strict` must pass. Changing build tools, flags,
stripping, or signing requires a fresh hash/size pin and actual runtime canary.
Cross-machine bit-for-bit reproducibility has not been established.

## Running on approved macOS arm64

```sh
sgl start --model-name whisper-1 \
  --model-path /owned/stt/ggml-small.bin \
  --stt-whisper /owned/stt/whisper-cli \
  --stt-whisper-sha256 4ac1f78373fa19037ff036c66586db2b775e0b21785c320953a1960bcb425405 \
  --stt-smoke-dir /owned/stt/approved-smoke
# Optional dedicated stdlib interpreter: --stt-python /owned/stt/bin/python3
```

The versioned `whisper-stt-macos-arm64-v1.10.1.tar.gz` release asset includes
`whisper-cli`, the unchanged approved runtime manifest, runtime MIT license,
public upstream smoke `audio.wav`, `smoke.json`, and attribution. Verify its
GitHub attestation and checksum before extraction. CI packages the reviewed
binary byte-identically; it does not rebuild it or establish cross-machine
reproducibility. Model weights and Python are supplied separately.

The bundled smoke is unchanged whisper.cpp `samples/jfk.wav` at
`d1be6fde11ac6e0407606b4e42fe72d34add8037`, SHA-256
`59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`,
176,000 samples / exactly 11 seconds. Startup expects `fellow`, `americans`,
and `country`. [Bundle attribution](../assets/whisper/README.md) identifies
the public historical speech sample without asserting an additional copyright
or public-domain determination. It replaces the private synthesized fixture
whose redistribution terms were not established. `sgl service install` persists
the same arguments. No STT model is advertised
before its startup smoke passes. REST and WebSocket heartbeats share a nested
`capabilities.transcription` manifest with immutable revisions/hashes, OS/arch,
format/languages, 60-second limit, no streaming, live free slots, and the current
signed encryption-key binding. Worker failure removes readiness and the model.
The accelerator telemetry remains `unknown` unless backend use is measured.

## Private transcription-v1 contract

Dispatch uses `job_type: "transcription"`, model `whisper-1`, and exactly
`{enc, transcription_reservation}`. The reservation contains `protocol`,
`request_id`, `model`, `model_revision`, `model_sha256`, `language`, and
`sample_count`. The request ID must equal the outer job UUID.

`enc` contains exactly `ciphertext`, `client_ephemeral_pubkey`,
`client_response_pubkey`, `algorithm`, and `encoding`. Only base64 ciphertext
and `x25519-xchacha20poly1305-hkdf-v2` are accepted. The response key is bound
by the v2 authenticated data. Plaintext, legacy/base58 ciphertext, and unknown
fields are rejected.

The authenticated inner JSON contains exactly:

```json
{
  "protocol": "transcription-v1",
  "request_id": "00000000-0000-4000-8000-000000000001",
  "model": "whisper-1",
  "model_revision": "5359861c739e955e79d9a303bcbc70fb988958b1",
  "model_sha256": "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
  "language": "auto",
  "audio": {
    "format": "pcm_s16le", "sample_rate": 16000, "channels": 1,
    "bits_per_sample": 16, "sample_count": 1, "data": "AAA="
  }
}
```

Every reservation binding must match the decrypted request. The decoded PCM
length must equal twice its sample count. Language is `auto` or one of `en`,
`fr`, `de`, `es`, `it`, `pt`, `nl`, `hi`, `ja`, `ko`, `zh`, `ru`, `ar`.
Requests are bounded to 960,000 samples (60 seconds / 1,920,000 PCM bytes),
an 8 MiB frame, 64 KiB transcript text, and 256 monotonic segments.

The sealed result contains `protocol`, `request_id`, `job_id`, `model`,
`model_revision`, `model_sha256`, `sample_count`, `text`, `language`, `language_hint`,
`duration_seconds`, `segments`, and `usage.audio_seconds`. Duration and usage
are the exact authenticated sample count divided by 16,000, including fractions;
monetary rounding belongs to the orchestrator. Empty text with no segments is valid. Silent probes still execute the real
runtime after artifact validation; bounded speech-model output may vary. Native
Whisper timestamps from its padded analysis window are trimmed to recording
length before export. The node signs the public
ciphertext together with the outer job ID using the existing result envelope.

Consumed request IDs are rejected within a bounded recent node replay window:
65,536 IDs or 24 hours, whichever limit is reached first. Worker restarts retain
this cache; eviction keeps the node available. The orchestrator must enforce
durable job UUID uniqueness and terminal-job replay protection across node
sessions. Cancellation and timeout terminate the owned process tree and clear
readiness. Local temporary audio is removed when the worker ends.

## Verification

```sh
python3 scripts/test_verify_whisper_release.py
python3 scripts/verify_whisper_release.py verify
cargo test --all-targets
cargo test --features inprocess,metal,vision
python3 tests/test_stt_worker.py
SGL_STT_CANARY_WHISPER=/owned/stt/whisper-cli \
SGL_STT_CANARY_MODEL=/owned/stt/ggml-small.bin \
SGL_STT_CANARY_SMOKE=/owned/stt/approved-smoke \
  cargo test --test stt_supervisor production_worker_real_canary -- --ignored
```

The real canary is opt-in and transcribes bundled known speech locally; it
neither registers a Grid node nor spends credits.
