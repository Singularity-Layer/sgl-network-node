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
use std::time::Duration;

const REPO: &str = "Singularity-Layer/sgl-network-node";
const USER_AGENT: &str = concat!("sgl-node/", env!("CARGO_PKG_VERSION"));
const RELEASE_BINARY_MAX_BYTES: usize = 128 * 1024 * 1024;
const CHECKSUM_MAX_BYTES: usize = 4 * 1024;
const DOWNLOAD_ATTEMPTS: usize = 3;
const DOWNLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_RETRY_DELAY: Duration = Duration::from_millis(500);

/// Can this machine actually RUN the Vulkan (GPU) build?
///
/// Deliberately conservative, because getting it wrong is not a slowdown - it is a node that
/// will not start. The GPU asset hard-links libvulkan.so.1, so picking it on a host without
/// the Vulkan runtime produces "error while loading shared libraries" and the node is dead.
/// We therefore require BOTH the loader library and a real GPU device node, and fall back to
/// the CPU build on any doubt. `SGL_GPU=0|1` overrides for operators who know better.
#[cfg(target_os = "linux")]
fn linux_gpu_capable() -> bool {
    match std::env::var("SGL_GPU").ok().as_deref() {
        Some("1") | Some("true") => return true,
        Some("0") | Some("false") => return false,
        _ => {}
    }
    let have_loader = [
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists());
    // A render node (or an NVIDIA device) means there is actually something to accelerate on.
    let have_device = std::path::Path::new("/dev/dri").exists()
        || std::path::Path::new("/dev/nvidiactl").exists();
    have_loader && have_device
}

/// Release asset for the running platform. Must match the names produced by
/// .github/workflows/release.yml.
///
/// Linux ships TWO builds per arch: the plain asset (CPU, runs anywhere) and `-gpu` (Vulkan,
/// needs the Vulkan runtime). We pick by detection rather than shipping one binary, because a
/// Vulkan-linked binary cannot start at all on a CPU-only host.
fn platform_asset() -> Result<&'static str, String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Ok("sgl-darwin-arm64");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Ok(if linux_gpu_capable() {
        "sgl-linux-x86_64-gpu"
    } else {
        "sgl-linux-x86_64"
    });
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return Ok(if linux_gpu_capable() {
        "sgl-linux-arm64-gpu"
    } else {
        "sgl-linux-arm64"
    });
    // Windows self-update is intentionally not supported: an .exe can't be renamed
    // over while it's running, and the release pipeline doesn't publish a Windows
    // asset on the allowlist yet. Update via the installer / desktop shell instead.
    #[cfg(windows)]
    return Err(
        "sgl update is not supported on Windows — a running .exe can't self-replace.\n\
         Update by re-running the installer (or the Singularity Node desktop app)."
            .to_string(),
    );
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

#[derive(Clone, Copy)]
struct DownloadPolicy {
    max_bytes: usize,
    attempts: usize,
    stall_timeout: Duration,
    retry_delay: Duration,
}

impl DownloadPolicy {
    fn release_binary() -> Self {
        Self {
            max_bytes: RELEASE_BINARY_MAX_BYTES,
            attempts: DOWNLOAD_ATTEMPTS,
            stall_timeout: DOWNLOAD_STALL_TIMEOUT,
            retry_delay: DOWNLOAD_RETRY_DELAY,
        }
    }

    fn checksum() -> Self {
        Self {
            max_bytes: CHECKSUM_MAX_BYTES,
            ..Self::release_binary()
        }
    }
}

enum DownloadFailure {
    Retryable(String),
    Permanent(String),
}

async fn download_once(
    client: &reqwest::Client,
    url: &str,
    policy: DownloadPolicy,
) -> Result<Vec<u8>, DownloadFailure> {
    let mut response = client
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "identity")
        .send()
        .await
        .map_err(|e| DownloadFailure::Retryable(format!("request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let message = format!("HTTP {status}");
        return Err(
            if status == reqwest::StatusCode::REQUEST_TIMEOUT
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status.is_server_error()
            {
                DownloadFailure::Retryable(message)
            } else {
                DownloadFailure::Permanent(message)
            },
        );
    }

    if let Some(encoding) = response.headers().get(reqwest::header::CONTENT_ENCODING) {
        let encoding = encoding.to_str().map_err(|_| {
            DownloadFailure::Permanent("invalid Content-Encoding header".to_string())
        })?;
        if !encoding.trim().eq_ignore_ascii_case("identity") {
            return Err(DownloadFailure::Permanent(format!(
                "server ignored identity encoding request (Content-Encoding: {encoding})"
            )));
        }
    }

    let advertised_len = response.content_length();
    if let Some(len) = advertised_len {
        if len > policy.max_bytes as u64 {
            return Err(DownloadFailure::Permanent(format!(
                "Content-Length {len} exceeds {}-byte limit",
                policy.max_bytes
            )));
        }
    }

    let capacity = advertised_len
        .and_then(|len| usize::try_from(len).ok())
        .unwrap_or(0);
    let mut body = Vec::with_capacity(capacity);
    loop {
        let chunk = tokio::time::timeout(policy.stall_timeout, response.chunk())
            .await
            .map_err(|_| {
                DownloadFailure::Retryable(format!(
                    "response body stalled for {} seconds",
                    policy.stall_timeout.as_secs_f32()
                ))
            })?
            .map_err(|e| DownloadFailure::Retryable(format!("response body read failed: {e}")))?;
        let Some(chunk) = chunk else {
            break;
        };
        let new_len = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| DownloadFailure::Permanent("response size overflow".to_string()))?;
        if new_len > policy.max_bytes {
            return Err(DownloadFailure::Permanent(format!(
                "response body exceeds {}-byte limit",
                policy.max_bytes
            )));
        }
        body.extend_from_slice(&chunk);
    }

    if let Some(len) = advertised_len {
        if body.len() as u64 != len {
            return Err(DownloadFailure::Retryable(format!(
                "incomplete response body (expected {len} bytes, received {})",
                body.len()
            )));
        }
    }
    if body.is_empty() {
        return Err(DownloadFailure::Retryable(
            "empty response body".to_string(),
        ));
    }

    Ok(body)
}

async fn download_bytes_with_retry(
    client: &reqwest::Client,
    url: &str,
    label: &str,
    policy: DownloadPolicy,
) -> Result<Vec<u8>, String> {
    if policy.attempts == 0 {
        return Err(format!("{label} failed: retry policy has no attempts"));
    }

    for attempt in 1..=policy.attempts {
        match download_once(client, url, policy).await {
            Ok(body) => return Ok(body),
            Err(DownloadFailure::Permanent(reason)) => {
                return Err(format!("{label} failed: {reason}"));
            }
            Err(DownloadFailure::Retryable(reason)) if attempt == policy.attempts => {
                return Err(format!(
                    "{label} failed after {} attempts: {reason}",
                    policy.attempts
                ));
            }
            Err(DownloadFailure::Retryable(reason)) => {
                eprintln!(
                    "  {label} attempt {attempt}/{} failed ({reason}); retrying…",
                    policy.attempts
                );
                tokio::time::sleep(policy.retry_delay.saturating_mul(attempt as u32)).await;
            }
        }
    }

    unreachable!("non-zero retry policy must return from the attempt loop")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn current_exe_hash() -> Result<(PathBuf, String), String> {
    let exe =
        std::env::current_exe().map_err(|e| format!("Cannot resolve current executable: {e}"))?;
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
            .get(format!(
                "https://api.github.com/repos/{REPO}/releases/latest"
            ))
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
                format!(
                    "Release {tag} has no `{asset}.sha256` checksum — refusing unverified update."
                )
            })?
            .to_string();
        println!("  Latest: {tag}");

        // ── 2. Download binary + published checksum ──────────────────────────
        println!("Downloading {asset}…");
        let bin = download_bytes_with_retry(
            &client,
            &asset_url,
            "Download",
            DownloadPolicy::release_binary(),
        )
        .await?;

        let sum_body = download_bytes_with_retry(
            &client,
            &sum_url,
            "Checksum download",
            DownloadPolicy::checksum(),
        )
        .await?;
        let sum_body = std::str::from_utf8(&sum_body)
            .map_err(|_| "Checksum download failed: response was not valid UTF-8".to_string())?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn test_policy(max_bytes: usize, attempts: usize, stall_timeout: Duration) -> DownloadPolicy {
        DownloadPolicy {
            max_bytes,
            attempts,
            stall_timeout,
            retry_delay: Duration::ZERO,
        }
    }

    fn test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap()
    }

    async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8(request).unwrap()
    }

    async fn spawn_scripted_server(
        responses: Vec<Vec<u8>>,
    ) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let handle = tokio::spawn(async move {
            for response in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_request(&mut socket).await;
                captured.lock().unwrap().push(request);
                socket.write_all(&response).await.unwrap();
                socket.shutdown().await.unwrap();
            }
        });
        (format!("http://{address}/asset"), requests, handle)
    }

    #[tokio::test]
    async fn retries_truncated_body_and_requests_identity_encoding() {
        let (url, requests, server) = spawn_scripted_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabc".to_vec(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef".to_vec(),
        ])
        .await;

        let body = download_bytes_with_retry(
            &test_client(),
            &url,
            "Download",
            test_policy(16, 2, Duration::from_secs(1)),
        )
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(body, b"abcdef");
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|request| request
            .to_ascii_lowercase()
            .contains("accept-encoding: identity\r\n")));
    }

    #[tokio::test]
    async fn rejects_oversized_content_length_without_retrying() {
        let (url, requests, server) = spawn_scripted_server(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\n".to_vec(),
        ])
        .await;

        let error = download_bytes_with_retry(
            &test_client(),
            &url,
            "Checksum download",
            test_policy(5, 3, Duration::from_secs(1)),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.contains("Content-Length 6 exceeds 5-byte limit"));
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn caps_chunked_bodies_without_content_length() {
        let response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n";
        let (url, requests, server) = spawn_scripted_server(vec![response.to_vec()]).await;

        let error = download_bytes_with_retry(
            &test_client(),
            &url,
            "Download",
            test_policy(5, 3, Duration::from_secs(1)),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.contains("response body exceeds 5-byte limit"));
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn chunk_stalls_retry_only_up_to_the_attempt_limit() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let _ = read_request(&mut socket).await;
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n")
                    .await
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(60)).await;
                let _ = socket.write_all(b"late").await;
            }
        });

        let error = download_bytes_with_retry(
            &test_client(),
            &format!("http://{address}/asset"),
            "Download",
            test_policy(16, 2, Duration::from_millis(20)),
        )
        .await
        .unwrap_err();
        server.await.unwrap();

        assert!(error.contains("Download failed after 2 attempts"));
        assert!(error.contains("response body stalled"));
    }
}
