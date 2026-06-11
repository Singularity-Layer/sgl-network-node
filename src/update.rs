//! `sgl update` — self-update to the latest official release, fail-closed.
//!
//! Trust model (no single point of compromise):
//!   1. The release binary + its `.sha256` come from GitHub Releases over HTTPS.
//!   2. The downloaded bytes must hash to the published `.sha256`. (Integrity of
//!      the download channel.)
//!   3. INDEPENDENTLY, the hash must be on the orchestrator's binary allowlist
//!      (`GET /grid/allowed-binaries` — the same allowlist that gates serving).
//!      An attacker must compromise BOTH the GitHub release assets AND the
//!      orchestrator's configuration to ship a malicious update.
//!   4. Every check fails CLOSED: any unreachable endpoint or mismatch aborts
//!      with the binary untouched.
//!
//! The swap is atomic (write temp file in the same directory, then rename over
//! the running executable). If the install dir needs root, we say so instead of
//! half-installing.

use sha2::{Digest, Sha256};
use std::path::PathBuf;

const REPO: &str = "Singularity-Layer/sgl-network-node";
const USER_AGENT: &str = concat!("sgl-node/", env!("CARGO_PKG_VERSION"));

/// Release asset for the running platform. Must match the names produced by
/// .github/workflows/release.yml.
fn platform_asset() -> Result<&'static str, String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Ok("sgl-darwin-arm64");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Ok("sgl-linux-x86_64");
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return Ok("sgl-linux-arm64");
    #[allow(unreachable_code)]
    Err("sgl update has no prebuilt binary for this platform.\n\
         Update from source: git pull && cargo build --release"
        .to_string())
}

fn https_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .https_only(true)
        .timeout(std::time::Duration::from_secs(120))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn current_exe_hash() -> Result<(PathBuf, String), String> {
    let exe = std::env::current_exe().map_err(|e| format!("Cannot resolve current executable: {e}"))?;
    let bytes = std::fs::read(&exe).map_err(|e| format!("Cannot read current executable: {e}"))?;
    let hash = sha256_hex(&bytes);
    Ok((exe, hash))
}

pub async fn run(orchestrator_url: &str) -> Result<(), String> {
    {
        let asset = platform_asset()?;
        let client = https_client()?;

        // ── 1. Resolve the latest release ────────────────────────────────────
        println!("Checking latest release…");
        let rel: serde_json::Value = client
            .get(format!("https://api.github.com/repos/{REPO}/releases/latest"))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|e| format!("Failed to reach GitHub releases: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GitHub releases lookup failed: {e}"))?
            .json()
            .await
            .map_err(|e| format!("Bad release metadata: {e}"))?;

        let tag = rel["tag_name"].as_str().unwrap_or("unknown").to_string();
        let assets = rel["assets"].as_array().cloned().unwrap_or_default();
        let asset_url = assets
            .iter()
            .find(|a| a["name"].as_str() == Some(asset))
            .and_then(|a| a["browser_download_url"].as_str())
            .ok_or_else(|| format!("Release {tag} has no `{asset}` asset."))?
            .to_string();
        let sum_url = assets
            .iter()
            .find(|a| a["name"].as_str() == Some(format!("{asset}.sha256").as_str()))
            .and_then(|a| a["browser_download_url"].as_str())
            .ok_or_else(|| {
                format!("Release {tag} has no `{asset}.sha256` checksum — refusing unverified update.")
            })?
            .to_string();
        println!("  Latest: {tag}");

        // ── 2. Download binary + published checksum ──────────────────────────
        println!("Downloading {asset}…");
        let bin = client
            .get(&asset_url)
            .send()
            .await
            .map_err(|e| format!("Download failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("Download failed: {e}"))?
            .bytes()
            .await
            .map_err(|e| format!("Download failed: {e}"))?;

        let sum_body = client
            .get(&sum_url)
            .send()
            .await
            .map_err(|e| format!("Checksum download failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("Checksum download failed: {e}"))?
            .text()
            .await
            .map_err(|e| format!("Checksum download failed: {e}"))?;

        let expected = sum_body
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        let actual = sha256_hex(&bin);
        if expected.len() != 64 || actual != expected {
            return Err(format!(
                "Checksum mismatch — refusing to install.\n  published: {expected}\n  downloaded: {actual}"
            ));
        }
        println!("  Checksum verified ✓ ({})", &actual[..12]);

        // ── 3. Independent allowlist check against the orchestrator ─────────
        println!("Verifying against the grid allowlist…");
        let allow: serde_json::Value = client
            .get(format!("{orchestrator_url}/grid/allowed-binaries"))
            .send()
            .await
            .map_err(|e| format!("Cannot reach the orchestrator allowlist (failing closed): {e}"))?
            .error_for_status()
            .map_err(|e| format!("Allowlist lookup failed (failing closed): {e}"))?
            .json()
            .await
            .map_err(|e| format!("Bad allowlist response (failing closed): {e}"))?;

        let allowed = allow["hashes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|h| h.as_str())
                    .map(|s| s.to_lowercase())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !allowed.contains(&actual) {
            return Err(format!(
                "This build ({}) is not on the grid's approved-binary allowlist — refusing to install.\n\
                 If a release was just published, the allowlist may not be updated yet; try later.",
                &actual[..12]
            ));
        }
        println!("  Allowlisted ✓");

        // ── 4. Already current? ──────────────────────────────────────────────
        let (exe, current) = current_exe_hash()?;
        if current == actual {
            println!("Already up to date ({tag}, {}).", &actual[..12]);
            return Ok(());
        }

        // ── 5. Atomic self-replace ───────────────────────────────────────────
        let dir = exe
            .parent()
            .ok_or("Cannot resolve install directory")?
            .to_path_buf();
        let tmp = dir.join(".sgl-update.tmp");

        if let Err(e) = std::fs::write(&tmp, &bin) {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(format!(
                    "No write permission for {} — re-run with sudo:\n  sudo sgl update",
                    dir.display()
                ));
            }
            return Err(format!("Failed to stage update: {e}"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| format!("Failed to set permissions: {e}"))?;
        }
        if let Err(e) = std::fs::rename(&tmp, &exe) {
            let _ = std::fs::remove_file(&tmp);
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(format!(
                    "No permission to replace {} — re-run with sudo:\n  sudo sgl update",
                    exe.display()
                ));
            }
            return Err(format!("Failed to install update: {e}"));
        }

        println!();
        println!("✅ Updated to {tag} ({})", &actual[..12]);
        println!("   Installed at: {}", exe.display());
        println!();
        println!("If the node runs as a background service, restart it onto the new build:");
        #[cfg(target_os = "macos")]
        println!("  launchctl kickstart -k gui/$(id -u)/cc.x402compute.sglnode");
        #[cfg(target_os = "linux")]
        println!("  systemctl --user restart cc.x402compute.sglnode.service");
        println!("Then re-attest so the grid records the new build:");
        println!("  sgl attest");
        Ok(())
    }
}
