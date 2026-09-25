# Run CLM as a System One node

`sgl` can serve typed System One models through a local loopback sidecar. This
guide runs Contrastive-LM CLM on a GPU host and advertises it to Singularity Grid
as `Contrastive-LM/CLM-v0.1-8B`.

This is an operator/CLI path. The desktop app does not yet manage the CLM runtime.

## Requirements

- `sgl` v1.9.16 or newer.
- A GPU host that can run `vllm` with `Qwen/Qwen3-8B` in pooling mode.
- Python with `contrastive-lm`, `vllm`, and `huggingface_hub`.

## Start the CLM sidecar

Install the runtime:

```bash
python -m pip install contrastive-lm vllm huggingface_hub
```

Download the pinned CLM head:

```bash
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt \
  --revision e939398d4556fcd9400c76fa8c5a513202f42b0a \
  --local-dir ~/.cache/sgl/clm-v0.1-8b
```

Start the Qwen3-8B pooling encoder:

```bash
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --runner pooling \
  --max-model-len 2048 \
  --port 8090
```

In another shell, start CLM under the exact Grid model id:

```bash
clm-serve \
  --no-ui \
  --port 8700 \
  --emb-url http://127.0.0.1:8090/v1/embeddings \
  --emb-model qwen3-8b \
  --max-tokens 2048 \
  --model Contrastive-LM/CLM-v0.1-8B=$HOME/.cache/sgl/clm-v0.1-8b/CLM_v0.1-8B.pt
```

Check health:

```bash
curl http://127.0.0.1:8700/health
curl http://127.0.0.1:8700/v1/models
```

## Start the node

```bash
sgl start \
  --model-name Contrastive-LM/CLM-v0.1-8B \
  --systemone-sidecar-url http://127.0.0.1:8700 \
  --max-jobs 1
```

`--systemone-sidecar-url` is loopback-only. The node forwards System One jobs to
the sidecar, signs the result, and reports usage for settlement.

## Smoke test

After the node heartbeat is active, this model should appear in:

```bash
curl 'https://grid.x402compute.cc/v1/models?type=systemone&tier=standard'
```

Then send a small `/v1/systemone` request using an API key or x402 payment.

## Privacy

Plain `/v1/systemone` requests are visible to the orchestrator. Private sealed
System One requests remain client-to-node encrypted, but they require an
attested node. If the CLM host is not attested, advertise it as standard and do
not claim confidential execution for that node.
