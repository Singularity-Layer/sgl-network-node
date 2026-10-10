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
| `whisper-stt-macos-arm64-vX.Y.Z.tar.gz` | `macos-14` | byte-identical approved runtime, manifest, MIT license, public smoke + attribution |

Jobs: a matrix `build` (mac/linux) + a dedicated `build-windows` (MSVC toolchain +
`ilammy/setup-nasm@v1` for `ring`) + `package-whisper` → a single `release` job
that verifies all six node binary/checksum pairs and the versioned Whisper
archive/checksum pair, then attests the complete downloaded set with GitHub
Artifact Attestations, writes a combined `sgl-node-vX.Y.Z.sha256`, and runs one
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
- The historical v1.9.16 hashes are preserved in
  `docs/releases/v1.9.16.sha256`. Preserve every currently accepted production
  hash, including v1.9.17, while v1.10.0 rolls out. The Whisper archive/runtime
  hash is a separate STT artifact pin; do not add it to the node executable allowlist.

For a staged candidate:

```sh
gh release download v1.10.0 --repo Singularity-Layer/sgl-network-node --dir /tmp/sgl-v1.10.0
cd /tmp/sgl-v1.10.0
sha256sum --check sgl-node-v1.10.0.sha256
for asset in sgl-darwin-arm64 sgl-linux-* sgl-windows-x86_64.exe whisper-stt-macos-arm64-v1.10.0.tar.gz; do
  gh attestation verify "$asset" --repo Singularity-Layer/sgl-network-node
done
```

After the orchestrator preserves the prior accepted hashes, adds all six
v1.10.0 node hashes, and the candidate canaries pass, promote the same immutable draft:

```sh
gh release edit v1.10.0 --repo Singularity-Layer/sgl-network-node --draft=false --latest
```

Do not delete and recreate a draft to promote it. That would make the reviewed
asset set ambiguous.

## Reviewed Whisper runtime packaging

`assets/whisper/runtime/macos-arm64/whisper-cli` is an explicitly reviewed
7,760,160-byte source artifact, SHA-256
`4ac1f78373fa19037ff036c66586db2b775e0b21785c320953a1960bcb425405`.
The existing `scripts/whisper_runtime_files.json` records its pinned unmodified
whisper.cpp source and exact local build provenance. GitHub CI packages these
bytes unchanged, together with the runtime MIT license and the unchanged
public upstream JFK smoke sample; it does not rebuild this binary. The archive
attestation proves canonical-workflow packaging, not independent CI compilation.
No cross-machine bit-for-bit runtime reproduction is claimed.

`python3 scripts/verify_whisper_release.py verify` checks SHA-256, size, thin
arm64 Mach-O executable format, exact manifest dependencies, system-only dylibs,
absence of LC_RPATH, WAV hash/format/sample count, and smoke metadata. macOS also
checks the ad-hoc signature with `codesign --verify --strict`. The same portable
Mach-O parser rechecks the extracted archive members on the Ubuntu release job;
missing, duplicate, non-file, extra, or incorrectly permissioned members fail.
The archive includes no model weights or interpreter. Its macOS deployment
target is 15.0; actual runtime evidence is macOS 27.0.1, not a macOS 15 canary.

The public JFK test recording is attributed in `assets/whisper/README.md` and
`smoke.json` to pinned whisper.cpp `samples/jfk.wav`; no additional copyright
or public-domain assertion is made. Its startup words are `fellow`, `americans`,
and `country`. A release candidate must also pass the real node startup canary
with this bundled fixture before promotion.
