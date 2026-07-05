# Releasing the `sgl` node agent

## Cross-platform tag flow (mac + linux + windows on ONE prerelease)

Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`. It builds every
platform on GitHub's own runners (verifiable supply chain — no dev laptop) and
publishes **one GitHub prerelease** carrying one binary + `.sha256` per platform:

| Asset | Runner | Notes |
|-------|--------|-------|
| `sgl-darwin-arm64` | `macos-14` | in-process + Metal inference |
| `sgl-linux-x86_64` | `ubuntu-24.04` | future confidential TDX/SEV tier |
| `sgl-linux-arm64` | `ubuntu-24.04-arm` | |
| `sgl-windows-x86_64.exe` | `windows-latest` | MSVC + NASM; non-confidential node |

Jobs: a matrix `build` (mac/linux) + a dedicated `build-windows` (MSVC toolchain +
`ilammy/setup-nasm@v1` for `ring`) → a single `release` job that runs one
`gh release create --prerelease` with ALL platform assets. So mac + linux + windows
land on ONE release, not several.

The release is a **`--prerelease`** and this repo is **private**, so assets are not
anonymously downloadable.

### What is automated vs owner-manual

- **Automated (on tag push):** build all platforms → one private prerelease with
  all binaries + `.sha256` files. Nothing else.
- **Owner-manual (to go live):** nothing serves publicly until the owner:
  1. **reviews** the prerelease assets,
  2. **syncs** the chosen binaries to the public endpoint
     `https://cloud.x402compute.cc/downloads/node/` (CI never touches it), and
  3. **allowlists** each new binary's sha256 into the orchestrator's
     `ALLOWED_NODE_BINARY_HASHES` (add-before-remove) — otherwise a node running
     that build is **rejected from serving**.

For the **Windows** binary specifically, see `WINDOWS.md` → *"Windows release +
allowlist"* for exactly where the sha256 comes from and which env var to set.

## The allowlist (why step 3 is non-optional)

Every node reports the sha256 of the `sgl` binary it's running. The orchestrator
checks it against `ALLOWED_NODE_BINARY_HASHES`. A hash that isn't listed can't
serve traffic. On every release:

- Take each `*.sha256` (release asset, or the release run's *"Show checksums"*
  step) — the 64-char lowercase hex before the filename.
- Append the new hashes to `ALLOWED_NODE_BINARY_HASHES` on the orchestrator
  (comma-separated). **Add before removing** old hashes so nodes mid-upgrade keep
  serving.

Nothing here is automated — CI only produces the private prerelease. Publishing +
allowlisting are deliberate owner actions.
