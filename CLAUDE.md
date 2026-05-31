# sgl-node — operator agent (Rust CLI)

Rust CLI that turns a machine into an SGL grid compute node. Own repo:
`Singularity-Layer/sgl-network-node`. Binary name: `sgl` (built to
`target/release/sgl`).

## ⚠️ Binary-hash allowlist — READ THIS BEFORE REBUILDING THE NODE

The orchestrator enforces a **binary allowlist**: only `sgl` builds whose
sha256 is in the orchestrator env var `ALLOWED_NODE_BINARY_HASHES` are allowed
to attest and serve on the grid. This stops a tampered/forked node binary from
serving user prompts.

**What this means in practice:** every time you run `cargo build --release` and
the binary changes, its sha256 changes too — so the **new** binary will FAIL
attestation ("Node binary is not a recognized SGL build") until you add the new
hash to the allowlist.

### After ANY node rebuild that will run in production, do this:

1. Get the new hash:
   ```
   shasum -a 256 target/release/sgl | awk '{print $1}'
   ```
   ⚠️ NOTE: the hash the **running node reports** (via its hardware report) is
   what the orchestrator checks. Confirm it with:
   ```sql
   select metadata->>'binary_hash' from compute_grid_nodes
   where wallet_address = '<node wallet>';
   ```
   (after restarting the node + re-attesting). Use THAT value, not necessarily
   the on-disk `shasum` — they can differ if a different build is running.

2. Add it to `ALLOWED_NODE_BINARY_HASHES` in
   `SGLNetwork_Orchestrator/wrangler.toml` `[vars]` (comma-separated to keep
   multiple released versions valid during a rollout), then
   `npx wrangler deploy` from the orchestrator dir.

3. Restart the node service and re-attest:
   ```
   launchctl kickstart -k gui/$(id -u)/cc.x402compute.sglnode
   sgl attest
   ```

**Versioning recommendation:** treat the allowlist as a list of *released*
build hashes. Keep the **previous** hash in the list during a rollout so old
nodes don't drop off the instant you deploy a new build; remove it once all
operators have upgraded. For a real release process, bump `Cargo.toml`
`version`, tag the commit, and record `version → sha256` somewhere durable
(e.g. a RELEASES.md) so you know which hash corresponds to which version.

To temporarily DISABLE the allowlist (dev/debug): unset
`ALLOWED_NODE_BINARY_HASHES` (SIP enforcement still applies).

## Run as a service (production)
```
sgl service install --model-path <gguf> --model-name <name> --resource-percent 50
sgl service status | sgl service uninstall
```
macOS → launchd (wraps `caffeinate -i` to block idle sleep). Linux → systemd
--user. Restarting reloads the ~2GB model (~30-60s) and briefly drops the grid.

## Attestation / security model
- ed25519 node keypair (`crypto.rs`); X25519 enc key derived from it
  (`encryption.rs`) for E2E-sealed prompts — published every REST heartbeat.
- `tee.rs generate_attestation_report()` reports SIP status + binary self-hash.
- Orchestrator gates on: SIP must be enabled + binary hash on allowlist.
- See orchestrator `lib/attestation.ts` (gate) and `lib/tamper.ts` (slash on
  forged result signature).
