# Windows support for the `sgl` node agent

The `sgl` node agent compiles and packages on Windows
(`x86_64-pc-windows-msvc`), verified green on GitHub Actions Windows runners.
This is the runtime half of the Windows node app: the Singularity Node desktop
shell already builds on Windows (separate repo); this makes the actual node
binary it drives — `sgl.exe` — compile + package on Windows too.

> **Status:** compile + packaging are proven in CI. A *functional* node run on
> Windows (connect, infer, heartbeat end-to-end) has **not** been validated on a
> real Windows box yet — that is an owner follow-up (see below). The owner has no
> Windows machine, so CI is the proof of the compile/package milestone only.

## What works on Windows

- **The binary compiles + packages.** `cargo build --release --target
  x86_64-pc-windows-msvc` produces `sgl.exe`, uploaded by CI as an artifact.
- **Foreground node commands** (compile + logic are cross-platform):
  - `sgl login` / `sgl login --code ... --wallet ...` (browser + headless device flow)
  - `sgl init`
  - `sgl start ...` — the minimum viable Windows node: connects to the
    orchestrator over WebSocket/HTTPS, runs inference via `llama-server.exe`,
    heartbeats, streams sealed tokens. (Needs a local `llama-server.exe`; see below.)
  - `sgl status`, `sgl version`, `sgl detect`, `sgl off-grid`, `sgl on-grid`,
    `sgl attest`, `sgl price ...`
- **Secure key storage** — on Windows the ed25519 identity is written with the
  default ACL (the Unix `0600`/`0700` `chmod` path is `#[cfg(unix)]`-gated and a
  no-op on Windows; NTFS inherits the user profile ACL). Config/keys live under
  the standard per-user config dir (`%APPDATA%`) via the `dirs` crate.
- **Crypto / E2E encryption** — pure Rust (`ed25519-dalek`, `x25519-dalek`,
  `chacha20poly1305`, `rustls`), fully cross-platform.

### llama-server on Windows

The default engine (`--engine=server`) shells out to `llama-server`. On Windows
the node looks for **`llama-server.exe`**:

1. on `PATH` (`llama-server.exe`, `llama-cli.exe`),
2. `%LOCALAPPDATA%\sgl-node\bin\llama-server.exe` (where the desktop shell /
   one-click installer is expected to drop the bundled build), and
3. `%ProgramFiles%\llama.cpp\llama-server.exe`.

Grab a Windows build of llama.cpp from
<https://github.com/ggerganov/llama.cpp/releases> and put `llama-server.exe` in
one of those locations.

## What is stubbed on Windows (and why)

These return a clear, honest error on Windows so the crate **compiles** and the
foreground node path works. None of them block running a node in the foreground.

| Area | Windows behavior | Why |
|------|------------------|-----|
| **Service install** (`sgl service install/uninstall/status`) | Returns "not wired up yet — run `sgl start ...` in the foreground (the desktop app supervises it), or `sc.exe create` yourself." | launchd/systemd have no Windows equivalent. A real Windows service (Service Control Manager / `sc.exe` or the `windows-service` crate) is deeper than a compile milestone and needs a real box to validate. The desktop shell can supervise the foreground process in the meantime. |
| **Self-update** (`sgl update`) | Returns "not supported on Windows — a running .exe can't self-replace. Update via the installer / desktop app." | Windows can't `rename()` over a running `.exe` the way Unix can. The release pipeline also doesn't publish a Windows asset on the allowlist yet. |
| **Attestation / TEE** (`sgl detect`, `sgl attest`) | Reports no Secure Enclave / no TEE (`tee_type = "none"`); the macOS App-Attest / `system_profiler` / `ioreg` probes are `#[cfg(target_os = "macos")]`-gated and return empty on Windows. | Windows nodes are **non-confidential** — no App Attest, no SEV/TDX path here. That matches the product framing (confidential tier is macOS Secure Enclave + Linux TEE). |
| **Runtime hardening** (anti-debug) | No-op with a debug log ("no debugger protection available on this platform"). | `PT_DENY_ATTACH` (macOS) / `prctl` (Linux) have no direct equivalent wired up. |

## Platform-gating summary (every site touched)

- `src/inference.rs` — `find_llama_server()` gains a `#[cfg(windows)]` branch that
  looks for `llama-server.exe` in the locations above; the Unix branch is
  unchanged under `#[cfg(not(windows))]`.
- `src/update.rs` — added a `#[cfg(windows)]` early return in `platform_asset()`
  (no Windows self-update). The `set_permissions(0o755)` staging step is already
  `#[cfg(unix)]`-gated.
- `src/service.rs` — the `#[cfg(not(any(macos, linux)))]` fallbacks for
  install/uninstall/status now give Windows-specific guidance; the macOS
  (launchd/`libc::getuid`) and Linux (systemd) impls stay `#[cfg]`-gated and are
  compiled out on Windows. Unused-on-Windows helpers are `allow(dead_code)`-annotated.
- `src/crypto.rs` — **already** had `#[cfg(not(unix))]` fallbacks for
  `write_secure_file` / `create_secure_dir` / `check_file_permissions` (plain
  write / `create_dir_all` / no-op). Unchanged.
- `src/tee.rs` — **already** had `#[cfg(not(target_os = "macos"))]` fallbacks for
  every hardware probe (Secure Enclave, Metal, SIP, UUIDs, firmware, serial).
  Unchanged.
- `src/runtime_hardening.rs` — **already** had a `#[cfg(not(any(macos, linux)))]`
  no-op branch. Unchanged.
- `libc` stays a dependency but every `libc::*` call site is inside a
  `#[cfg(target_os = "macos")]` or `"linux"` block, so nothing Unix-only is
  referenced on Windows (the crate has Windows stubs and compiles fine).

## Owner follow-ups (need a real Windows box)

1. **Functional smoke test** — install `llama-server.exe`, run `sgl login` +
   `sgl start ...` against the real orchestrator, confirm heartbeat, a job
   round-trips, and sealed streaming works.
2. **Native Windows service** — implement `sgl service install` via the Service
   Control Manager (`sc.exe create` shell-out or the `windows-service` crate) so
   the node survives reboot/logout, replacing the current stub.
3. **Code signing** — sign `sgl.exe` (Authenticode / EV cert) so SmartScreen and
   the desktop installer don't flag it. CI currently produces an unsigned binary.
4. **Updater feed** — decide the Windows update story (installer-driven or a
   sidecar updater that swaps the `.exe` while stopped), then publish a
   `sgl-windows-x86_64.exe` release asset and add its sha256 to the
   orchestrator's `ALLOWED_NODE_BINARY_HASHES` so a Windows node can serve.
5. **Confidential tier** — decide whether Windows ever gets an attestation path
   (likely stays non-confidential).

## Windows release + allowlist (owner-manual go-live)

On a `v*` tag, `.github/workflows/release.yml` now builds Windows too (its
`build-windows` job, MSVC + NASM) and attaches **`sgl-windows-x86_64.exe`** +
**`sgl-windows-x86_64.exe.sha256`** to the **same GitHub prerelease** as the
macOS/Linux binaries. The release is a **`--prerelease`** and this repo is
**private**, so the binary is not anonymously downloadable and no Windows node can
serve until the owner does BOTH of these:

1. **Sync** `sgl-windows-x86_64.exe` (rename to your download convention) to the
   public endpoint `https://cloud.x402compute.cc/downloads/node/` alongside the
   mac/linux binaries. (Owner-manual; CI never touches it.)
2. **Allowlist the sha256.** A node's binary hash is checked against the
   orchestrator's `ALLOWED_NODE_BINARY_HASHES` env var. If the Windows binary's
   sha256 is not in that list, a Windows node is **rejected from serving**
   (masquerades as a stake/binary error at the orchestrator).

   - **Where the sha256 comes from:** the published **`sgl-windows-x86_64.exe.sha256`**
     release asset (also printed by the release run's *"Show checksums"* step, and
     by the `build-windows` job's checksum step). It's the lowercase hex before the
     filename, e.g.:

     ```text
     3f8a…<64 hex chars>…c1  sgl-windows-x86_64.exe
     ```

   - **What to set:** append that 64-char hex to `ALLOWED_NODE_BINARY_HASHES`
     (comma-separated, add-before-remove — keep the existing mac/linux hashes) on
     the orchestrator (`sgl-network-orchestrator` Cloudflare Worker env / secret).
     Do NOT remove old hashes until every node is on the new build.

Nothing above is automated. The tag build only produces the private prerelease.

## Fetching the build-only CI artifact

The `windows` workflow (`.github/workflows/windows.yml`) is **build-only** and runs
on `workflow_dispatch` and every PR (no `v*` tag trigger — the release build lives
in `release.yml`). It uploads an artifact named **`sgl-windows-x86_64`** containing
`sgl-windows-x86_64.exe` and its `.sha256`.

- **Web:** open the run under the repo's *Actions → windows*, scroll to
  *Artifacts*, download `sgl-windows-x86_64`.
- **CLI:**
  ```sh
  gh run list --workflow=windows.yml
  gh run download <run-id> -n sgl-windows-x86_64
  ```

This workflow compiles and packages `sgl.exe` but does not run it and does not
publish a GitHub Release. Release publishing (mac + linux + windows) is owned by
`release.yml`; the macOS/Linux publish there is unchanged in behavior.
