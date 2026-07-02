# Headless deployment (one-click GPU machines, #172)

How a machine bought through "Deploy a Machine" boots into a serving grid node
with zero human interaction — and why it's safe.

## Flow

```
buyer (cloud UI)                deploy API (orchestrator)            GPU VM (cloud-init)
────────────────                ─────────────────────────            ───────────────────
pick SKU + model + period
pay USDC/credits  ──────────▶   verify payment + 50k compute stake
                                mint SINGLE-USE registration code
                                (compute_grid_registration_codes,
                                 bound to buyer wallet, short expiry)
                                render cloud-init.yaml.tmpl
                                provision VM at provider  ─────────▶ install llama-server (pinned tag, CUDA)
                                                                     install sgl (pinned release, sha256 fail-closed)
                                                                     download model GGUF (sha256 fail-closed)
                                                                     sgl login --code … --wallet …   ← generates
                                                                       keypair ON the machine, registers via the
                                                                       normal /grid/nodes/register path
                                                                     systemd: sgl start … (serving)
                                poll: node heartbeating? ◀────────── heartbeats (binary hash re-gated
                                mark deployment active               against the allowlist every beat)
```

## Security properties

- **No long-lived secrets in user-data.** The only sensitive value in cloud-init
  is the provision code: single-use, wallet-bound, short-lived, dead after
  registration. The node's ed25519 keypair is generated on the machine; the
  auth token is issued directly by the orchestrator at register time.
- **Same trust gate as every node.** Headless registration goes through the
  same `/grid/nodes/register` validation (code + wallet + stake) as the
  browser flow, and every heartbeat re-checks the binary hash against
  `ALLOWED_NODE_BINARY_HASHES`.
- **Fail-closed supply chain.** sgl binary and model GGUF are pinned by sha256;
  a mismatch aborts boot and the deployment reconciler marks it failed.
- **Tier honesty.** `--tee-type linux_se` machines are the STANDARD tier (E2E
  encrypted, operator-trust — same claim as Mac nodes). Only machines
  provisioned on confidential hardware (Azure NCC / TDX) get `tdx_cc` and the
  Confidential badge, and that claim must be backed by attestation (#99).

## Operator notes

- Node config lands at `/root/.config/sgl-node/node.json`; keypair beside it.
- `sgl update` works on Linux (release asset `sgl-linux-x86_64`), but a new
  binary only serves after its hash is allowlisted — same as macOS.
- To re-provision by hand: mint a code via the deploy API (admin), then
  `sgl login --code <code> --wallet <wallet> --tee-type linux_se --models <id>`.
