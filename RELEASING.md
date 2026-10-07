# Releasing the `sgl` node agent

## Cross-platform tag flow (mac + linux + windows in one draft)

Pushing a `vX.Y.Z` tag from the exact `origin/main` commit runs
`.github/workflows/release.yml`. It builds every platform on GitHub's own runners
and stages **one draft GitHub release**. A draft does not change
`/releases/latest`, so installed nodes stay on the prior release while the new
hashes are reviewed and allowlisted.

| Asset | Runner | Notes |
|-------|--------|-------|
| `sgl-darwin-arm64` | `macos-14` | in-process + Metal inference |
| `sgl-linux-x86_64` | `ubuntu-24.04` | future confidential TDX/SEV tier |
| `sgl-linux-arm64` | `ubuntu-24.04-arm` | |
| `sgl-linux-x86_64-gpu` | `ubuntu-24.04` | Vulkan runtime required |
| `sgl-linux-arm64-gpu` | `ubuntu-24.04-arm` | Vulkan runtime required |
| `sgl-windows-x86_64.exe` | `windows-latest` | MSVC + NASM; non-confidential node |

Jobs: a matrix `build` (mac/linux) + a dedicated `build-windows` (MSVC toolchain +
`ilammy/setup-nasm@v1` for `ring`) → a single `release` job that attests the
downloaded artifacts with GitHub Artifact Attestations, verifies all six binary /
checksum pairs, writes a combined `sgl-node-vX.Y.Z.sha256`, then runs one
`gh release create --draft`. macOS, Linux, and Windows therefore share one immutable
candidate set.

### What is automated vs owner-manual

- **Automated on tag push:** verify that the tag matches `Cargo.toml` and points at
  exact `origin/main`; build, checksum, and attest all platforms; create one draft.
- **Build-only on manual dispatch:** build and upload Actions artifacts. It does not
  create or edit a GitHub release.
- **Owner-manual staging:** download the draft with authenticated `gh`, verify the
  attestations and checksums, run package canaries, then append every new binary
  hash to `ALLOWED_NODE_BINARY_HASHES` while keeping the previous release hashes.
- **Owner-manual promotion:** after the allowlist is live and both old and new nodes
  are accepted, publish the existing draft. Only then can it become
  `/releases/latest`; public R2 sync remains a separate later action.

For the **Windows** binary specifically, see `WINDOWS.md` → *"Windows release +
allowlist"* for exactly where the sha256 comes from and which env var to set.

## The allowlist (why step 3 is non-optional)

Every node reports the sha256 of the `sgl` binary it's running. The orchestrator
checks it against `ALLOWED_NODE_BINARY_HASHES`. A hash that isn't listed can't
serve traffic. On every release:

- Take each `*.sha256` (release asset, or the release run's *"Show checksums"*
  step) — the 64-char lowercase hex before the filename.
- Verify provenance for any downloaded asset before syncing it publicly:
  `gh attestation verify <asset> --repo Singularity-Layer/sgl-network-node`.
- Append the new hashes to `ALLOWED_NODE_BINARY_HASHES` on the orchestrator
  (comma-separated). **Add before removing** old hashes so nodes mid-upgrade keep
  serving.
- The canonical v1.9.16 hashes are preserved in
  `docs/releases/v1.9.16.sha256`. Keep all six while v1.9.17 rolls out.

For a staged candidate:

```sh
gh release download v1.9.17 --repo Singularity-Layer/sgl-network-node --dir /tmp/sgl-v1.9.17
cd /tmp/sgl-v1.9.17
sha256sum --check sgl-node-v1.9.17.sha256
for asset in sgl-darwin-arm64 sgl-linux-* sgl-windows-x86_64.exe; do
  gh attestation verify "$asset" --repo Singularity-Layer/sgl-network-node
done
```

After the orchestrator contains both the v1.9.16 and v1.9.17 hashes and the
candidate canaries pass, promote the same immutable draft:

```sh
gh release edit v1.9.17 --repo Singularity-Layer/sgl-network-node --draft=false --latest
```

Do not delete and recreate a draft to promote it. That would make the reviewed
asset set ambiguous.
