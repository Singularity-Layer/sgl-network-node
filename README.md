<div align="center">

<img src="assets/logo.png" alt="Singularity Layer" width="120" />

# SGL Node

**Turn your Mac into a compute node on the Singularity grid — earn $SGL by serving confidential AI inference.**

[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/macOS-Apple%20Silicon-black.svg)](https://www.apple.com/mac/)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org)

</div>

---

`sgl` is the node agent for the **Singularity Layer** decentralized compute grid. It
runs a local [llama.cpp](https://github.com/ggerganov/llama.cpp) inference server on
your machine, pulls jobs from the grid orchestrator, decrypts end-to-end-sealed user
prompts, runs inference, signs the result with your node key, and reports usage back
for settlement. Operators stake **$SGL** to join the grid and earn a share of every
compute payment.

- **Grid:** [grid.x402compute.cc](https://grid.x402compute.cc) · **Cloud:** [cloud.x402layer.cc](https://cloud.x402layer.cc)
- **Staking:** [staking.x402layer.cc](https://staking.x402layer.cc) — stake $SGL to register a node
- **Staking program:** [`Singularity-Layer/sgl-staking`](https://github.com/Singularity-Layer/sgl-staking)

## Requirements

- **macOS on Apple Silicon** (M-series). Intel is unsupported; Linux/Windows land in Phase 2.
- **16 GB RAM minimum** (more for larger models).
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — `brew install llama.cpp`.
- A **GGUF model** file (e.g. Llama 3.2 3B Instruct, Q4_K_M).
- A **Solana wallet with staked $SGL** (see [staking.x402layer.cc](https://staking.x402layer.cc)).

## Install

```bash
curl -sSf https://grid.x402compute.cc/install.sh | sh
```

Or build from source (see [Build](#build-from-source)).

## Quickstart

```bash
sgl detect                          # check your hardware (chip, RAM, GPU, Secure Enclave)
sgl login                           # browser device-auth → registers this node to your wallet
sgl start \
  --model-path ~/models/llama-3.2-3b-instruct-q4_k_m.gguf \
  --model-name llama-3.2-3b \
  --resource-percent 50             # dedicate ~50% of the machine to the grid
```

`sgl login` opens a browser device-authorization flow and binds the node to your staked
wallet. `sgl init --wallet <ADDRESS>` is the non-interactive alternative.

## Commands

| Command | What it does |
|---------|--------------|
| `sgl detect` | Show hardware capabilities (chip, cores, RAM, GPU, Secure Enclave, SIP). |
| `sgl login` | Browser device-auth login + register this node (recommended). |
| `sgl init --wallet <ADDR>` | Non-interactive: generate keys + register under a wallet. |
| `sgl start --model-path <gguf> --model-name <name>` | Start serving: heartbeat + process jobs. |
| `sgl attest` | Sign the orchestrator's challenge + send the hardware report. |
| `sgl status` | Node status, reputation, jobs completed/failed. |
| `sgl service install \| status \| uninstall` | Run as a background OS service across reboots. |

`sgl start` flags include `--resource-percent` (a preset that scales threads / GPU
layers / concurrent jobs), or fine-grained `--threads`, `--gpu-layers`, `--context-size`,
`--max-jobs`, `--batch-size`, `--inference-port`, `--heartbeat-interval`. The global
`--orchestrator-url` defaults to `https://grid.x402compute.cc`.

## Run as a service (production)

```bash
sgl service install --model-path <gguf> --model-name <name> --resource-percent 50
sgl service status
sgl service uninstall
```

macOS uses **launchd** (wrapped in `caffeinate -i` to block idle sleep); Linux uses a
**systemd --user** unit. Restarting reloads the model (~30–60 s) and briefly drops the
node from the grid.

## How confidentiality works

Prompts can be **end-to-end encrypted** between the caller and your node — the
orchestrator relays ciphertext it cannot read:

1. Each node has an **ed25519** identity key (`crypto.rs`). An **X25519** key is
   deterministically derived from it (`encryption.rs`) and published on every heartbeat.
2. A caller seals the prompt to the node's X25519 key using **X25519 + XChaCha20-Poly1305**
   (ephemeral sender key per message), and includes a response key.
3. The node decrypts in memory, runs inference locally, and **seals the result back** to
   the caller's response key. Only token-usage counts are sent in cleartext, so the grid
   can bill without seeing prompt or response content.
4. The node **signs every result** with its ed25519 key. A forged or altered result is
   detectable by the orchestrator and is grounds for slashing.

### Integrity & anti-tamper

- **Result signatures** — every output is signed by the node key; the orchestrator
  verifies them and slashes nodes that return forged results.
- **Binary allowlist** — the orchestrator only accepts results from `sgl` builds whose
  sha256 is on a published allowlist, so a forked/tampered binary can't quietly serve.
- **Self-reported hardware attestation** (`tee.rs`) — the node reports SIP status, a
  hash of its own binary, and machine fingerprints on attest. **This is a defense-in-depth
  deterrent, not a hardware root of trust:** the report is produced by the node itself, so
  it raises the bar against casual tampering but does not cryptographically prove the
  execution environment the way a hardware-signed TEE quote (e.g. SGX/SEV/Apple DCAppAttest)
  would. Hardware-backed attestation is on the roadmap.
- **Runtime hardening** (`runtime_hardening.rs`) — anti-debugger (`PT_DENY_ATTACH` /
  `prctl`) to discourage live process tampering.
- **Keys at rest** — the node keypair is stored `0600` in a `0700` directory; the node
  warns if permissions are looser.

> **Confidentiality scope.** Your prompts are protected in transit and are not persisted
> by the grid, but inference runs in your node's normal process memory. Treat this as
> strong transport + operational confidentiality, **not** hardware-enclave-isolated
> execution.

## Build from source

Requires [Rust](https://rustup.rs) (2021 edition) and llama.cpp.

```bash
git clone https://github.com/Singularity-Layer/sgl-network-node.git
cd sgl-network-node
cargo build --release
sudo cp target/release/sgl /usr/local/bin/
sgl detect
```

## Security

Found a vulnerability? Please report it privately — open a GitHub security advisory on
this repo rather than a public issue. The source is public for independent review; issues
and PRs are welcome.

## License

[MIT](LICENSE) © 2026 Singularity Layer
