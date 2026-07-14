//! `sgl setup` — install the llama.cpp inference backend (llama-server) for this node.
//!
//! This is what makes a WINDOWS node able to serve (#225): `find_llama_server()` already looks in
//! `%LOCALAPPDATA%\sgl-node\bin`, but nothing put a binary there. `sgl setup` downloads the pinned
//! llama.cpp release for this platform, sha256-verifies it (fail-closed — same discipline as
//! `sgl update`), and extracts it into that dir. Cross-platform: Windows (.zip) + Linux (.tar.gz).
//!
//! Variant policy: **Vulkan by default** — one self-contained download that runs on NVIDIA, AMD,
//! and Intel GPUs with no CUDA-runtime version matching. `--cpu` forces the CPU build (no GPU).
//! CUDA (faster on NVIDIA, but 250MB + a separate cudart zip + version hell) is intentionally not
//! offered yet.
//!
//! Trust model: WE pin the exact sha256 of a vetted llama.cpp asset (upstream doesn't publish
//! per-asset checksums), so a tampered download fails the hash and aborts with nothing installed.

use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Pinned llama.cpp release. Bump the tag + all three hashes together after vetting a new build.
/// Hashes verified against ggml-org/llama.cpp release assets on 2026-07-10.
const LLAMA_TAG: &str = "b9949";

struct Asset {
    /// Release asset filename.
    name: &'static str,
    /// sha256 of the asset (lowercase hex).
    sha256: &'static str,
    /// true = zip (Windows), false = tar.gz (Linux).
    is_zip: bool,
}

/// The asset to install for (os, variant). Returns None on unsupported platforms (e.g. macOS,
/// which installs llama.cpp via Homebrew).
fn asset_for(cpu_only: bool) -> Option<Asset> {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Some(if cpu_only {
            Asset {
                name: "llama-b9949-bin-win-cpu-x64.zip",
                sha256: "68d98d746236099f8ef08588a42d54a965631919a277a659b83914fb67802c24",
                is_zip: true,
            }
        } else {
            Asset {
                name: "llama-b9949-bin-win-vulkan-x64.zip",
                sha256: "4f03b4646d8537798445399a679eb7b30b0a32da8158433fa80a09ad336fc04d",
                is_zip: true,
            }
        });
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // Linux ships CPU inside the vulkan tarball too; one artifact covers both.
        let _ = cpu_only;
        return Some(Asset {
            name: "llama-b9949-bin-ubuntu-vulkan-x64.tar.gz",
            sha256: "619620947b2c6253573ac3d026c3b377d8d1a3631fe43be3e06daf0d8322e46d",
            is_zip: false,
        });
    }
    #[allow(unreachable_code)]
    {
        let _ = cpu_only;
        None
    }
}

/// Per-user install dir for the bundled llama.cpp — must match `find_llama_server()`.
fn install_dir() -> Result<PathBuf, String> {
    let base = dirs::data_local_dir().ok_or("Cannot resolve local data directory")?;
    Ok(base.join("sgl-node").join("bin"))
}

/// The llama-server executable name for this platform.
fn server_exe() -> &'static str {
    if cfg!(windows) { "llama-server.exe" } else { "llama-server" }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// Promote the staging dir to the live bin dir, retrying the rename with backoff.
/// Windows antivirus scans freshly-extracted executables and can hold them for
/// seconds, failing the rename with "Access is denied (os error 5)".
fn promote_staging(staging: &Path, bin_dir: &Path) -> Result<(), String> {
    // Never delete a live install before its replacement is in place (Codex HIGH):
    // park it at bin.prev, promote staging, then drop the parked copy. If promotion
    // fails the parked install is restored, so a blocked rename can't strand the
    // node with no engine at all.
    let prev = bin_dir.with_extension("prev");
    let _ = std::fs::remove_dir_all(&prev);
    let had_prev = bin_dir.exists();
    if had_prev {
        std::fs::rename(bin_dir, &prev)
            .map_err(|e| format!("couldn't move the existing install aside: {e}"))?;
    }
    let mut last_err = String::new();
    for attempt in 0..6u32 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(500 * u64::from(attempt)));
            // The target may exist again (partial recreate by a racing process); clear it.
            let _ = std::fs::remove_dir_all(bin_dir);
        }
        match std::fs::rename(staging, bin_dir) {
            Ok(_) => {
                let _ = std::fs::remove_dir_all(&prev);
                return Ok(());
            }
            Err(e) => last_err = e.to_string(),
        }
    }
    // Promotion failed — put the previous install back so the node keeps serving.
    if had_prev {
        let _ = std::fs::rename(&prev, bin_dir);
    }
    Err(last_err)
}

pub async fn run(cpu_only: bool) -> Result<(), String> {
    let asset = asset_for(cpu_only).ok_or_else(|| {
        "`sgl setup` has no llama.cpp package for this platform.\n\
         macOS: `brew install llama.cpp`. Other: build from https://github.com/ggml-org/llama.cpp"
            .to_string()
    })?;

    let bin_dir = install_dir()?;
    let server_path = bin_dir.join(server_exe());
    if server_path.exists() {
        println!("llama-server already installed at {}", server_path.display());
        println!("Re-run to reinstall, or delete that file first.");
        // Not an error — idempotent. Verify it launches so a corrupt install is still caught below.
    }

    // Adopt a valid leftover staging dir (a prior run that downloaded + verified but lost
    // the final swap to an AV race) instead of re-downloading ~100MB. Only when there's no
    // live install to protect and the staged server actually launches.
    if !server_path.exists() {
        if let Some(parent) = bin_dir.parent() {
            let staging = parent.join("bin.staging");
            let staged_server = staging.join(server_exe());
            // `output().is_ok()` only proves the process SPAWNED — an exe with a missing
            // DLL spawns and dies nonzero (0xC0000135), so require a successful exit
            // (Codex HIGH: never adopt a broken staging dir).
            if staged_server.exists()
                && std::process::Command::new(&staged_server)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            {
                println!("Found a verified install from a previous run — finishing it…");
                match promote_staging(&staging, &bin_dir) {
                    Ok(_) => {
                        println!("✅ llama.cpp installed (recovered previous download)");
                        println!("   Server: {}", server_path.display());
                        return Ok(());
                    }
                    Err(e) => {
                        // Couldn't promote; fall through to a fresh download attempt.
                        println!("  (couldn't finish the previous install: {e} — reinstalling)");
                        let _ = std::fs::remove_dir_all(&staging);
                    }
                }
            }
        }
    }

    let url = format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{}",
        asset.name
    );
    println!("Downloading llama.cpp {LLAMA_TAG} ({})…", asset.name);
    let client = reqwest::Client::builder()
        .https_only(true)
        .timeout(std::time::Duration::from_secs(600))
        .user_agent(concat!("sgl-node/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;
    let bytes = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Download failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Download failed: {e}"))?
        .bytes()
        .await
        .map_err(|e| format!("Download failed: {e}"))?;

    // ── Fail-closed integrity check ──────────────────────────────────────────
    let actual = sha256_hex(&bytes);
    if actual != asset.sha256 {
        return Err(format!(
            "Checksum mismatch — refusing to install a tampered/corrupt archive.\n  \
             expected: {}\n  got:      {}",
            asset.sha256, actual
        ));
    }
    println!("  Checksum verified ✓ ({})", &actual[..12]);

    // ── Extract into a STAGING dir, verify, then atomically swap (Codex HIGH) ──
    // Never write into the live bin dir directly: a disk-full / crash / Ctrl-C mid-extract
    // would leave a half-written install that find_llama_server() then picks up. Instead we
    // extract to a sibling staging dir, smoke-test the staged server, and only then replace
    // the real bin dir — so a failed setup leaves any previous good install untouched.
    let parent = bin_dir
        .parent()
        .ok_or("Cannot resolve install parent dir")?;
    std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create {}: {e}", parent.display()))?;
    let staging = parent.join("bin.staging");
    let _ = std::fs::remove_dir_all(&staging); // clear any leftover from a prior aborted run
    std::fs::create_dir_all(&staging)
        .map_err(|e| format!("Cannot create staging dir {}: {e}", staging.display()))?;
    println!("Installing to {}…", bin_dir.display());
    let extracted = if asset.is_zip {
        extract_zip(&bytes, &staging)
    } else {
        extract_tar_gz(&bytes, &staging)
    };
    let extracted = match extracted {
        Ok(n) => n,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };
    let staged_server = staging.join(server_exe());
    let fail = |staging: &Path, msg: String| -> String {
        let _ = std::fs::remove_dir_all(staging);
        msg
    };
    if extracted == 0 || !staged_server.exists() {
        return Err(fail(&staging, format!(
            "Archive extracted but {} is missing — layout changed; nothing installed.",
            server_exe()
        )));
    }

    // ── Windows: drop the MSVC runtime trio beside llama-server.exe ───────────────
    // Upstream llama.cpp builds dynamically link the VC++ runtime; clean Windows
    // installs lack it ("VCRUNTIME140.dll was not found" — hit live in the beta). We
    // bundle the officially redistributable DLLs (packaged by our CI from the build
    // runner), hash-pinned and hosted on our origin. Windows loads DLLs from the exe's
    // own directory first, so this removes the VC++ Redistributable install entirely —
    // and the smoke test below then validates EXACTLY what a clean machine runs.
    #[cfg(windows)]
    {
        const VCCRT_URL: &str =
            "https://cloud.x402compute.cc/downloads/node/vccrt-x64-20f7ee2e8e81.zip";
        const VCCRT_SHA256: &str =
            "20f7ee2e8e81db01cb11d7f343114302bfd9a4f817e00a51e99fd24530adb358";
        println!("Downloading MSVC runtime bundle…");
        let crt_bytes = client
            .get(VCCRT_URL)
            .send()
            .await
            .map_err(|e| fail(&staging, format!("Runtime bundle download failed: {e}")))?
            .error_for_status()
            .map_err(|e| fail(&staging, format!("Runtime bundle download failed: {e}")))?
            .bytes()
            .await
            .map_err(|e| fail(&staging, format!("Runtime bundle download failed: {e}")))?;
        let actual = sha256_hex(&crt_bytes);
        if actual != VCCRT_SHA256 {
            return Err(fail(&staging, format!(
                "Runtime bundle checksum mismatch — refusing to install.\n  expected: {VCCRT_SHA256}\n  got:      {actual}"
            )));
        }
        println!("  Runtime checksum verified ✓ ({})", &actual[..12]);
        if let Err(e) = extract_zip(&crt_bytes, &staging) {
            return Err(fail(&staging, format!("Runtime bundle extract failed: {e}")));
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&staged_server, std::fs::Permissions::from_mode(0o755));
    }
    // Smoke-check the STAGED binary before it replaces anything. A nonzero exit is a
    // failure too (missing-DLL exes spawn then die 0xC0000135 — `is_ok()` would pass).
    match std::process::Command::new(&staged_server).arg("--version").output() {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
            return Err(fail(&staging, format!(
                "Downloaded llama-server exited with {} during the launch check{}\n\
                 (Missing runtime libs? On Windows a GPU driver may be needed.) Nothing installed.",
                o.status,
                if stderr.is_empty() { String::new() } else { format!(": {stderr}") },
            )));
        }
        Err(e) => {
            return Err(fail(&staging, format!(
                "Downloaded llama-server failed to launch: {e}\n\
                 (Missing runtime libs? On Windows a GPU driver may be needed.) Nothing installed.",
            )));
        }
    }

    // Atomic-ish swap: drop the old dir, promote staging (retry inside — AV scan race).
    // On final failure KEEP staging: the adopt-staging path above finishes the install on
    // the next run without re-downloading (previously this stranded testers with no bin).
    if let Err(e) = promote_staging(&staging, &bin_dir) {
        return Err(format!(
            "Verified the download but couldn't install into {}: {e}\n\
             (The verified files were kept at {} — close anything using that folder and \
             re-run `sgl setup` to finish without re-downloading.)",
            bin_dir.display(),
            staging.display()
        ));
    }

    println!();
    println!("✅ llama.cpp installed ({LLAMA_TAG}, {} files)", extracted);
    println!("   Server: {}", server_path.display());
    println!("The node will find it automatically. Next: `sgl login` then `sgl service install`.");
    Ok(())
}

/// Extract every entry of a zip (flat layout: exe + DLLs at root) into `dir`. Returns file count.
/// Guards against zip-slip (entries escaping `dir` via `..`).
fn extract_zip(bytes: &[u8], dir: &Path) -> Result<usize, String> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| format!("Bad zip: {e}"))?;
    let mut count = 0usize;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| format!("Zip entry {i}: {e}"))?;
        let Some(rel) = entry.enclosed_name() else {
            // `enclosed_name()` returns None for path-traversal entries — skip them (zip-slip guard).
            continue;
        };
        // Flatten: install only the file's basename into bin/ (llama.cpp win zips are already flat,
        // but be defensive against nested dirs).
        let Some(fname) = rel.file_name() else { continue };
        if entry.is_dir() {
            continue;
        }
        let out = dir.join(fname);
        let mut f =
            std::fs::File::create(&out).map_err(|e| format!("Write {}: {e}", out.display()))?;
        std::io::copy(&mut entry, &mut f).map_err(|e| format!("Extract {}: {e}", out.display()))?;
        count += 1;
    }
    Ok(count)
}

/// Extract a .tar.gz, flattening every regular file into `dir`. Returns file count. Zip-slip safe
/// (we take only the basename).
fn extract_tar_gz(bytes: &[u8], dir: &Path) -> Result<usize, String> {
    let gz = flate2::read::GzDecoder::new(Cursor::new(bytes));
    let mut tar = tar::Archive::new(gz);
    let mut count = 0usize;
    for entry in tar.entries().map_err(|e| format!("Bad tar: {e}"))? {
        let mut entry = entry.map_err(|e| format!("Tar entry: {e}"))?;
        // Regular files only — skip dirs, symlinks, hardlinks, and special entries (Codex LOW:
        // a symlink/hardlink could otherwise collide with an expected basename).
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path().map_err(|e| format!("Tar path: {e}"))?.into_owned();
        let Some(fname) = path.file_name() else { continue };
        let out = dir.join(fname);
        let mut f =
            std::fs::File::create(&out).map_err(|e| format!("Write {}: {e}", out.display()))?;
        std::io::copy(&mut entry, &mut f).map_err(|e| format!("Extract {}: {e}", out.display()))?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod setup_tests {
    use super::*;

    #[test]
    fn server_exe_matches_platform() {
        if cfg!(windows) {
            assert_eq!(server_exe(), "llama-server.exe");
        } else {
            assert_eq!(server_exe(), "llama-server");
        }
    }

    #[test]
    fn asset_hashes_are_64_hex() {
        // Whatever platform we compile on, if an asset is defined its hash must be a real sha256.
        if let Some(a) = asset_for(false) {
            assert_eq!(a.sha256.len(), 64, "sha256 must be 64 hex chars");
            assert!(a.sha256.chars().all(|c| c.is_ascii_hexdigit()));
            assert!(a.name.contains(LLAMA_TAG));
        }
    }

    #[test]
    fn extract_zip_flattens_and_counts() {
        // Build a tiny in-memory zip with a nested path; expect it flattened to basename.
        let mut buf = Vec::new();
        {
            let mut w = zip::ZipWriter::new(Cursor::new(&mut buf));
            let opts: zip::write::FileOptions<()> =
                zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            use std::io::Write;
            w.start_file("nested/dir/hello.txt", opts).unwrap();
            w.write_all(b"hi").unwrap();
            w.finish().unwrap();
        }
        let tmp = std::env::temp_dir().join(format!("sgl-setup-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let n = extract_zip(&buf, &tmp).unwrap();
        assert_eq!(n, 1);
        assert!(tmp.join("hello.txt").exists(), "flattened to basename");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
