# EmbeddingGemma 2 node runtime

EmbeddingGemma 2 uses a dedicated Python process. It never enters llama.cpp chat,
GGUF pooling, chat vision, or System One paths. The node does not download models
or share either chat runtime's interpreter or model cache.

An operator must explicitly select the model and provide its verified BF16 snapshot
and dedicated interpreter:

```sh
sgl start --model-name embeddinggemma-2 --model-path /owned/eg2/snapshot \
  --embedding-python /owned/eg2/runtime/bin/python
```

The same arguments persist through `sgl service install`. Model discovery and
production listing remain separate release decisions. The factory materializes the
worker, hash manifest and exact founder M4 canary fixtures in a private temporary
directory. It verifies all snapshot assets, runtime pins, and BF16/FP32 parameter
dtypes. Startup runs real text, image, audio, video and mixed embeddings. Only a
matching ready frame with five finite normalized native vectors enables routing.

## Candidate pins

- Official model: `google/embeddinggemma-2`, revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- MLX BF16 model: `mlx-community/embeddinggemma-2-bf16`, revision
  `1a4ffddb7905d3f63486748deabe091a01fb6201`.
- MLX-VLM Git commit: `30f177f03cbcb42bc2f65496458de79f51b80c28`.
- MLX `0.32.3`, Transformers `5.19.0`.
- Portable reference candidate: Sentence Transformers `6.1.0`; this node path uses MLX.

`embeddinggemma_model_files.json` records the size and SHA256 of all 16 model assets.
The candidate runtime requires Apple Silicon macOS. FP16 parameters are refused.
The query prefix is `task: search result | query: `; the document prefix is
`title: none | text: `. Unspecified input has no retrieval prefix.

## Wire and worker

Legacy `input: string | string[]` is retained. EG2 also accepts a batch of
`{content:[...]}` items with ordered text, image, audio and video parts. A top-level
item always produces one vector. Media is inline canonical base64 with an exact
MIME and a required lowercase SHA256; paths and URLs are rejected. Existing GGUF
models retain their own limits and reject structured input.

The worker uses bounded newline-delimited JSON on stdin/stdout. Every embedding
request and result has an integer `request_id`. The request contains the exact
`embedding-multimodal-v1` protocol, typed batch, input type and native dimension
768. The result contains ordered native vectors and integer modality token usage.
The result also carries private `item_usage` rows. Rust validates each row's
8192-token limit, requested media presence and image/video/audio bounds, then
requires that their exact sum equals aggregate usage. Each row is also capped at
its quoted work: UTF-8 text and prefix bytes + 12 + images × 280 + ceiling(video
seconds) × 140 + ceiling(audio seconds) × 25. Private usage rows stay out of the
public OpenAI response. The Rust boundary validates output before returning any
billable result, truncates
MRL dimensions to 768/512/256/128, and normalizes again in FP32.

Images reserve at most 280 processed tokens each. Video uses 1 FPS, at most 32
frames and 140 tokens per frame. Audio is mono 16 kHz and at most 30 seconds.
Each sample shares an 8192-token budget across prefixes and modalities. Batch
usage sums exact processor token counts, excluding padding. Decoded media shape,
MIME and duration are checked before model inference; understated duration fails.
MP4 requires whitelisted major and compatible ISO container brands. Text preflight
counts UTF-8 bytes, the applied retrieval prefix and a 12-token template reserve
before media decoding or tokenization. Declared image, audio and video budgets
also enter this preflight. Empty EG2 text parts are rejected. Audio metadata is
checked before miniaudio streams mono 16 kHz FLOAT32 chunks of at most 4096 samples.
The worker closes and rejects the stream if it exceeds the declared duration or
480000 samples. It passes the bounded waveform array to the processor, avoiding
a second file decode.

Worker stderr and library stdout are suppressed. Request errors contain no input,
media, vectors or Python traceback. Startup has a 180-second deadline; each job
has a 120-second worker write/read deadline. Oversized frames, malformed responses,
EOF, cancellation and timeouts fail the job and remove capability. The supervisor
kills and reaps the process and removes its private media directory. Expected
input errors use a bounded `request_error` frame and return the safe node reason
`embedding_input_invalid`, while preserving worker readiness. Runtime or protocol
failures return `embedding_runtime_failed` and remove readiness. Native output
errors raise Python `RuntimeError`. Only our explicit input/media checks raise
`InputValidationError`, which produces a nonfatal request-error frame. An
unclassified processor or model `ValueError` remains a fatal runtime failure.
Known video-decoder errors at the explicit media-validation boundary instead
report corrupt input and preserve readiness.
Restarts
repeat startup checks and have a lifetime budget of three attempts.

## Confidential transport and capability

Structured media requires the negotiated base64, UTF-8 v2 sealed envelope. The
inner JSON authenticates `embedding_protocol`; the response key remains bound
through the existing HKDF/AAD scheme. Legacy base58 crypto stays unchanged.
Clear input is capped at 24 MiB; encrypted node transport is capped at 36 MiB.

REST and explicit WebSocket heartbeats serialize the same readiness manifest:
protocol, modalities, dimensions, MLX runtime, processor revision, inline media
transport, `embedding_ready:true` and `input_envelope_encodings:["base64"]`.
Worker death removes every embedding capability and the model advertisement.
The existing chat `vision` flag remains separate.

## Verification

Default tests use fake Python workers and require no model download. They exercise
legacy input, all structured modalities, ordering, MRL, output validation, request
IDs, pipe write/read deadlines, cancellation, EOF, oversized frames, restarts,
kill/reap, temporary media cleanup, capability removal and a shared TypeScript-to-
Rust sealed-envelope fixture. Python unit tests check dtype rejection, duration
limits and the exact pinned private video decoder signature.

The founder M4 canary used the exact packaged smoke bytes and pinned assets;
startup plus text/image/audio/video/mixed 128-dimensional vectors were finite and
unit normalized. Full release still requires packaged desktop lifecycle, Local/
Grid parity, platform billing and legacy live canaries. No deployment is implied
by this runtime implementation.
