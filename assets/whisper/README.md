# Approved Whisper STT runtime and startup sample

This bundle contains the reviewed, self-contained macOS arm64 `whisper-cli`,
its immutable build manifest, the upstream runtime MIT license, and a pinned
startup test sample. It contains no model weights or Python interpreter.

The runtime comes from unmodified [whisper.cpp source at
`d1be6fde11ac6e0407606b4e42fe72d34add8037`](https://github.com/ggml-org/whisper.cpp/tree/d1be6fde11ac6e0407606b4e42fe72d34add8037).
The manifest records compiler, SDK, flags, stripping, signing, and exact SHA-256.
CI packages the reviewed source artifact byte-identically; it does not rebuild
this runtime. GitHub attestations prove archive packaging by the canonical
workflow, not an independent CI compilation or cross-machine reproduction.
Only Apple system dylibs/frameworks are dynamic; Metal shaders are embedded.
The binary targets macOS 15.0 and was tested on macOS 27.0.1. macOS 15.0 runtime
compatibility has not been established by this release verification.

`smoke/audio.wav` is the unchanged public JFK inaugural-address sample from
[whisper.cpp `samples/jfk.wav`](https://github.com/ggml-org/whisper.cpp/blob/d1be6fde11ac6e0407606b4e42fe72d34add8037/samples/jfk.wav)
at the same pinned commit. Its [upstream sample README](https://github.com/ggml-org/whisper.cpp/blob/d1be6fde11ac6e0407606b4e42fe72d34add8037/samples/README.md)
describes the test samples as public audio files. This is attribution to the
public upstream test sample, not an independent copyright or public-domain
claim. `LICENSE` applies to the whisper.cpp runtime; no additional audio license
is asserted here. The sample is 16 kHz mono signed 16-bit PCM, 176,000 samples
(11 seconds), with SHA-256
`59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
Startup expects the words `fellow`, `americans`, and `country`.

Extract the versioned archive into an operator-owned directory, keep the model
and runtime read-only, and use its `whisper-cli` and `smoke/` paths with the
arguments documented in `docs/WHISPER.md`. The node verifies the approved model
and runtime before inference. This archive does not enable any Grid route.
