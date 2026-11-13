# Grid video (text to video) on a node

Status: built dark. Nothing changes for a node unless the operator passes `--video-model`.
Contract: x402Studio `documentation/product/sgl-network/GRID_VIDEO_V1.md` (protocol `video-v1`).
Code: `src/video.rs` (engine supervisor) and `src/video_job.rs` (job glue).

## What it needs

- Linux x86_64 with an NVIDIA GPU. Other platforms refuse to start the video engine with a
  clear message (Windows CUDA comes later).
- GPU memory (from `nvidia-smi`, device 0): `wan2.1-t2v-1.3b` 15360 MiB (so 16 GB cards pass),
  `wan2.2-ti2v-5b` 22528 MiB.
  The 5B runs its VAE decode tiled (`--vae-tiling`) on cards under 34 GB.
- stable-diffusion.cpp `sd-cli`, CUDA build of `master-1043-3be2378`
  (commit `3be23788fc0c9f35a2c24b1211caae11d4a54dbc`). Its shared libraries sit next to it
  (the node sets `LD_LIBRARY_PATH` to that folder, as the desktop app does).
- An ffmpeg executable with `libx264` (default) or `libopenh264`, and `libwebp` for the
  thumbnail (without `libwebp` clips are still served, just without a thumbnail).

## Hash pins (no unpinned program ever runs)

The node checks the sha256 of `sd-cli` and of ffmpeg before it starts them, and again on every
relaunch. A sha256 compiled into the node wins; today none is compiled in
(`PINNED_ENGINE_SHA256` and `PINNED_ENCODER_SHA256` are `None` because no CUDA engine build or
encoder build has been published yet), so the operator must pass both:

```
--video-engine-sha256 <64 hex>   --video-encoder-sha256 <64 hex>
```

When a build is published, its sha256 goes into those constants and any operator value must
then equal it. Note: the engine pin covers the `sd-cli` file only, not the shared libraries
next to it.

## Model files

Put the three files for your model flat in one folder (`--video-model-dir`), under their base
names. Download from Hugging Face at the pinned revision and check the sha256 yourself:

```
https://huggingface.co/<repo>/resolve/<revision>/<path>
```

| model | file | repo @ revision | bytes |
|---|---|---|---|
| both | `umt5-xxl-encoder-Q8_0.gguf` | city96/umt5-xxl-encoder-gguf @ b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7 | 6043068256 |
| 1.3B | `Wan2.1-T2V-1.3B-Q8_0.gguf` | samuelchristlie/Wan2.1-T2V-1.3B-GGUF @ 5a512b15fc35d1b67a074cfe55a591be9e9ef9b5 | 1535768800 |
| 1.3B | `wan_2.1_vae.safetensors` (path `split_files/vae/wan_2.1_vae.safetensors`) | Comfy-Org/Wan_2.1_ComfyUI_repackaged @ 123acf1cc74bccbb9bfff8ac1ee72edc08c2341d | 253815318 |
| 5B | `Wan2.2-TI2V-5B-Q8_0.gguf` | QuantStack/Wan2.2-TI2V-5B-GGUF @ 57437632ddd08bdcbd1508c866aa22e126ed51d2 | 5400179040 |
| 5B | `wan2.2_vae.safetensors` (path `split_files/vae/wan2.2_vae.safetensors`) | Comfy-Org/Wan_2.2_ComfyUI_Repackaged @ ee6f4a40737a995bf5818954cfce6d59443b0f04 | 1409400960 |

The sha256 of every file is pinned in `src/video.rs` (`MODELS`); `sgl video doctor` prints the
exact URL of any file that is missing or wrong. Symlinks are refused. After a file verifies
once, its size and modified time are remembered in `.sgl-video-verified.json` in the same
folder so a restart does not re-hash about 13 GB.

## Check the machine

```
sgl video doctor --video-model wan2.1-t2v-1.3b \
  --video-engine /opt/sdcpp/sd-cli --video-engine-sha256 <hex> \
  --video-encoder /usr/bin/ffmpeg --video-encoder-sha256 <hex> \
  --video-model-dir /srv/models/wan
```

It prints the GPU, memory, driver and CUDA version, both executables with their sha256, each
model file, and then makes the smoke clip (256x160, 9 frames, 4 steps, encoded and checked
like a real job). It exits non-zero when the machine is not ready.

## Serve

Same flags on `sgl start` (or `sgl service install`); `--video-model` sets the advertised model
and cannot be combined with another `--model-name` or `--model-path`. Optional:
`--video-encoder-codec libopenh264`.

The node advertises `capabilities.video` and the model id only after the smoke clip passed,
runs one job at a time, and stops advertising after a crash or a timeout until a relaunch
passes the smoke clip again (3 crash relaunches per hour; a node that times out on 3 jobs in a
row is parked until it is restarted).

## What a job does

1. Checks the cleartext reservation against the model (size, seconds, fps, frames, steps, cfg,
   pins, upload caps, deadline), unseals the prompt (v2 only) and binds it to the job id, the
   model, its byte length and its sha256. Any mismatch: `video_input_invalid`.
2. Refuses only prompts that pair sexual content with minors: `video_content_refused`. A prompt
   containing sd.cpp LoRA syntax (`<lora:`) is refused as `video_input_invalid`.
3. Runs `sd-cli` in a private folder (0700) under the node's config directory, with the prompt
   as a single argument (no shell), reporting progress at most every 5 s. A `continue:false`
   answer kills the engine's whole process group at once; the node then reports nothing.
4. Encodes MP4 (H.264, yuv420p, faststart) with the label `AI-generated video. Made with
   <model> on the Singularity Layer Grid. Job <job id>.` as `comment` and `description`, makes
   a WebP thumbnail of the middle frame, and checks the MP4 itself (brand, duration within
   0.5 s, exact size, label) before anything is uploaded.
5. Uploads with retries on 5xx and network errors (never on other 4xx), signs
   `sgl-result-v1\n<job id>\nvideo\n<sha256 of the MP4>` and completes over REST with retries
   until the deadline. The job folder is deleted on every path.
