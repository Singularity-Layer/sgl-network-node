//! `sgl update` — self-update to the latest official release, fail-closed.
//!
//! Trust model (no single point of compromise):
//!   1. The release binary + its `.sha256` come from GitHub Releases over HTTPS, or, when
//!      GitHub fails (5xx/429/network after retries, 404, bad checksum), from our R2 mirror
//!      (`cloud.x402compute.cc/downloads/node/sgl-<platform>` + `.sha256`, version in
//!      `sgl-latest.json`). The mirror never downgrades, and if GitHub's checksum was
//!      readable the mirror copy must match it.
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

use crate::mirror::{self, Failure};
use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;

const REPO: &str = "Singularity-Layer/sgl-network-node";
const USER_AGENT: &str = concat!("sgl-node/", env!("CARGO_PKG_VERSION"));
const RELEASE_BINARY_MAX_BYTES: usize = 128 * 1024 * 1024;
const CHECKSUM_MAX_BYTES: usize = 4 * 1024;
const METADATA_MAX_BYTES: usize = 1024 * 1024;

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

/// The release we are about to verify against the allowlist.
#[derive(Debug)]
struct Fetched {
    /// Release tag, e.g. `v1.12.3`.
    tag: String,
    /// The binary, already verified against `sha256`.
    bin: Vec<u8>,
    sha256: String,
    /// "GitHub" or "the Singularity Layer mirror".
    source: &'static str,
}

/// Where `sgl update` looks. Tests point these at scripted fakes.
struct Sources {
    github_latest_api: String,
    mirror_base: String,
}

impl Sources {
    fn live() -> Self {
        Self {
            github_latest_api: format!("https://api.github.com/repos/{REPO}/releases/latest"),
            mirror_base: mirror::MIRROR_BASE.to_string(),
        }
    }
}

/// First 64 hex chars of a `<sha256>  <name>` checksum file, lowercased; fails closed.
fn parse_checksum(body: &[u8], label: &str) -> Result<String, String> {
    let text = std::str::from_utf8(body)
        .map_err(|_| format!("{label} failed: response was not valid UTF-8"))?;
    let sha = text.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
    if !mirror::valid_sha256(&sha) {
        return Err(format!("{label} failed: not a sha256 checksum file"));
    }
    Ok(sha)
}

/// `1.12.3` / `v1.12.3` -> (1, 12, 3). Anything else (pre-release suffixes included) is None.
fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let mut it = v.strip_prefix('v').unwrap_or(v).split('.');
    let parsed = (
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
    );
    it.next().is_none().then_some(parsed)
}

/// What the GitHub attempt learned before it failed. A known checksum binds the mirror copy to
/// GitHub's latest release.
#[derive(Default)]
struct GithubPartial {
    sha256: Option<String>,
}

async fn fetch_from_github<F, Fut>(
    src: &Sources,
    asset: &str,
    delays: &[Duration],
    get: &mut F,
    partial: &mut GithubPartial,
) -> Result<Fetched, String>
where
    F: FnMut(String, usize) -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Failure>>,
{
    let meta = mirror::get_with_retry(&src.github_latest_api, METADATA_MAX_BYTES, "GitHub release lookup", delays, get).await?;
    let rel: serde_json::Value =
        serde_json::from_slice(&meta).map_err(|e| format!("Bad release metadata: {e}"))?;
    let tag = rel["tag_name"].as_str().unwrap_or("unknown").to_string();
    let assets = rel["assets"].as_array().cloned().unwrap_or_default();
    let url_of = |name: &str| {
        assets
            .iter()
            .find(|a| a["name"].as_str() == Some(name))
            .and_then(|a| a["browser_download_url"].as_str())
            .map(str::to_string)
    };
    let asset_url = url_of(asset).ok_or_else(|| format!("Release {tag} has no `{asset}` asset."))?;
    let sum_url = url_of(&format!("{asset}.sha256")).ok_or_else(|| {
        format!("Release {tag} has no `{asset}.sha256` checksum — refusing unverified update.")
    })?;
    println!("  Latest: {tag}");

    let sum = mirror::get_with_retry(&sum_url, CHECKSUM_MAX_BYTES, "Checksum download", delays, get).await?;
    let expected = parse_checksum(&sum, "Checksum download")?;
    partial.sha256 = Some(expected.clone());

    println!("Downloading {asset}…");
    let bin = mirror::get_with_retry(&asset_url, RELEASE_BINARY_MAX_BYTES, "Download", delays, get).await?;
    let actual = mirror::sha256_hex(&bin);
    if actual != expected {
        return Err(format!(
            "Checksum mismatch on the GitHub download (published {expected}, downloaded {actual})"
        ));
    }
    Ok(Fetched { tag, bin, sha256: actual, source: "GitHub" })
}

/// The mirror keeps the current release's binaries at fixed names (`sgl-<platform>` +
/// `.sha256`, refreshed on every promote) and `sgl-latest.json` = `{"version":"X.Y.Z"}`.
async fn fetch_from_mirror<F, Fut>(
    src: &Sources,
    asset: &str,
    current_version: &str,
    github_sha256: Option<&str>,
    delays: &[Duration],
    get: &mut F,
) -> Result<Fetched, String>
where
    F: FnMut(String, usize) -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Failure>>,
{
    let base = src.mirror_base.trim_end_matches('/');
    let latest = mirror::get_with_retry(
        &format!("{base}/sgl-latest.json"),
        CHECKSUM_MAX_BYTES,
        "Mirror version check",
        delays,
        get,
    )
    .await?;
    let latest: serde_json::Value = serde_json::from_slice(&latest)
        .map_err(|e| format!("Mirror version check failed: bad sgl-latest.json ({e})"))?;
    let version = latest["version"].as_str().unwrap_or("");
    let mirror_v = parse_version(version)
        .ok_or_else(|| format!("Mirror version check failed: bad version {version:?}"))?;
    // Never let a stale mirror walk a node backwards (older builds stay allowlisted for a while).
    if let Some(cur) = parse_version(current_version) {
        if mirror_v < cur {
            return Err(format!(
                "the mirror is on v{version}, older than this build (v{current_version}); refusing to downgrade"
            ));
        }
    }
    let tag = format!("v{version}");
    println!("  Latest (mirror): {tag}");

    let sum = mirror::get_with_retry(
        &format!("{base}/{asset}.sha256"),
        CHECKSUM_MAX_BYTES,
        "Mirror checksum download",
        delays,
        get,
    )
    .await?;
    let expected = parse_checksum(&sum, "Mirror checksum download")?;
    if let Some(gh) = github_sha256 {
        if gh != expected {
            return Err(format!(
                "the mirror is not on GitHub's latest build yet (GitHub {}, mirror {}); refusing",
                &gh[..12],
                &expected[..12]
            ));
        }
    }

    println!("Downloading {asset} from the mirror…");
    let bin = mirror::get_with_retry(
        &format!("{base}/{asset}"),
        RELEASE_BINARY_MAX_BYTES,
        "Mirror download",
        delays,
        get,
    )
    .await?;
    let actual = mirror::sha256_hex(&bin);
    if actual != expected {
        return Err(format!(
            "Checksum mismatch on the mirror download (published {expected}, downloaded {actual})"
        ));
    }
    Ok(Fetched { tag, bin, sha256: actual, source: "the Singularity Layer mirror" })
}

/// GitHub first (each request retried on 5xx/429/network), then our mirror. Either way the
/// binary is verified against its published sha256 here, and against the orchestrator
/// allowlist by the caller. Fails closed: both failing returns both reasons.
async fn fetch_release<F, Fut>(
    src: &Sources,
    asset: &str,
    current_version: &str,
    delays: &[Duration],
    mut get: F,
) -> Result<Fetched, String>
where
    F: FnMut(String, usize) -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Failure>>,
{
    let mut partial = GithubPartial::default();
    let gh_err = match fetch_from_github(src, asset, delays, &mut get, &mut partial).await {
        Ok(f) => return Ok(f),
        Err(e) => e,
    };
    eprintln!("  GitHub unavailable ({gh_err}).");
    eprintln!("  Trying the Singularity Layer download mirror…");
    fetch_from_mirror(src, asset, current_version, partial.sha256.as_deref(), delays, &mut get)
        .await
        .map_err(|m| format!("Could not fetch a verified update.\n  GitHub: {gh_err}\n  Mirror: {m}"))
}

fn current_exe_hash() -> Result<(PathBuf, String), String> {
    let exe =
        std::env::current_exe().map_err(|e| format!("Cannot resolve current executable: {e}"))?;
    let bytes = std::fs::read(&exe).map_err(|e| format!("Cannot read current executable: {e}"))?;
    let hash = mirror::sha256_hex(&bytes);
    Ok((exe, hash))
}

pub async fn run(orchestrator_url: &str) -> Result<(), String> {
    {
        let asset = platform_asset()?;
        let client = https_client()?;

        // ── 1+2. Resolve the latest release, download + verify its checksum ──
        // GitHub first; on failure, our R2 mirror (same checksum discipline).
        println!("Checking latest release…");
        let Fetched { tag, bin, sha256: actual, source } = fetch_release(
            &Sources::live(),
            asset,
            env!("CARGO_PKG_VERSION"),
            &mirror::RETRY_DELAYS,
            |url, max| {
                let client = client.clone();
                async move { mirror::get_once(&client, &url, max, mirror::STALL_TIMEOUT).await }
            },
        )
        .await?;
        println!("  Checksum verified ✓ ({}, from {source})", &actual[..12]);

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

    struct DownloadPolicy {
        max_bytes: usize,
        attempts: usize,
        stall_timeout: Duration,
    }

    fn test_policy(max_bytes: usize, attempts: usize, stall_timeout: Duration) -> DownloadPolicy {
        DownloadPolicy { max_bytes, attempts, stall_timeout }
    }

    /// Real HTTP through the same retry loop `sgl update` uses, with zero back-off.
    async fn download_bytes_with_retry(
        client: &reqwest::Client,
        url: &str,
        label: &str,
        policy: DownloadPolicy,
    ) -> Result<Vec<u8>, String> {
        let delays = vec![Duration::ZERO; policy.attempts - 1];
        let stall = policy.stall_timeout;
        mirror::get_with_retry(url, policy.max_bytes, label, &delays, &mut |u: String, m| {
            let c = client.clone();
            async move { mirror::get_once(&c, &u, m, stall).await }
        })
        .await
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

    // ── GitHub -> mirror fallback ────────────────────────────────────────────────

    use crate::mirror::tests::{Fake, NO_WAIT};

    const API: &str = "gh-api";
    const GH_BIN: &str = "gh/sgl-test";
    const GH_SUM: &str = "gh/sgl-test.sha256";
    const M_LATEST: &str = "mirror/sgl-latest.json";
    const M_BIN: &str = "mirror/sgl-test";
    const M_SUM: &str = "mirror/sgl-test.sha256";

    fn sources() -> Sources {
        Sources { github_latest_api: API.into(), mirror_base: "mirror".into() }
    }

    fn release_json() -> Vec<u8> {
        serde_json::json!({
            "tag_name": "v1.13.0",
            "assets": [
                {"name": "sgl-test", "browser_download_url": GH_BIN},
                {"name": "sgl-test.sha256", "browser_download_url": GH_SUM},
            ]
        })
        .to_string()
        .into_bytes()
    }

    fn sum_of(b: &[u8]) -> Vec<u8> {
        format!("{}  sgl-test\n", mirror::sha256_hex(b)).into_bytes()
    }

    fn leak(v: Vec<u8>) -> &'static [u8] {
        Box::leak(v.into_boxed_slice())
    }

    async fn run_fetch(f: &Fake, current: &str) -> Result<Fetched, String> {
        fetch_release(&sources(), "sgl-test", current, &NO_WAIT, f.get()).await
    }

    #[tokio::test]
    async fn github_ok_never_touches_the_mirror() {
        let f = Fake::new(vec![
            (API, vec![Ok(leak(release_json()))]),
            (GH_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (GH_BIN, vec![Ok(b"new-bin")]),
        ]);
        let got = run_fetch(&f, "1.12.3").await.unwrap();
        assert_eq!((got.tag.as_str(), got.bin.as_slice(), got.source), ("v1.13.0", &b"new-bin"[..], "GitHub"));
        assert!(f.calls().iter().all(|c| !c.starts_with("mirror")), "{:?}", f.calls());
    }

    #[tokio::test]
    async fn github_5xx_retried_then_mirror_serves_the_update() {
        let f = Fake::new(vec![
            (API, vec![Err("503")]),
            (M_LATEST, vec![Ok(br#"{"version":"1.13.0"}"#)]),
            (M_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (M_BIN, vec![Ok(b"new-bin")]),
        ]);
        let got = run_fetch(&f, "1.12.3").await.unwrap();
        assert_eq!(got.tag, "v1.13.0");
        assert_eq!(got.bin, b"new-bin");
        assert_eq!(got.sha256, mirror::sha256_hex(b"new-bin"));
        assert_eq!(got.source, "the Singularity Layer mirror");
        // 1 try + 3 retries at GitHub, then the mirror.
        assert_eq!(f.calls(), vec![API, API, API, API, M_LATEST, M_SUM, M_BIN]);
    }

    #[tokio::test]
    async fn checksum_mismatch_on_github_and_mirror_fails_closed() {
        let f = Fake::new(vec![
            (API, vec![Ok(leak(release_json()))]),
            (GH_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (GH_BIN, vec![Ok(b"evil")]),
            (M_LATEST, vec![Ok(br#"{"version":"1.13.0"}"#)]),
            (M_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (M_BIN, vec![Ok(b"also evil")]),
        ]);
        let err = run_fetch(&f, "1.12.3").await.unwrap_err();
        assert!(err.contains("GitHub: Checksum mismatch") && err.contains("Mirror: Checksum mismatch"), "{err}");
        // A mismatch is never retried against the same source.
        assert_eq!(f.calls().iter().filter(|c| *c == GH_BIN).count(), 1);
        assert_eq!(f.calls().iter().filter(|c| *c == M_BIN).count(), 1);
    }

    #[tokio::test]
    async fn tampered_github_binary_falls_back_to_matching_mirror_copy() {
        let f = Fake::new(vec![
            (API, vec![Ok(leak(release_json()))]),
            (GH_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (GH_BIN, vec![Ok(b"evil")]),
            (M_LATEST, vec![Ok(br#"{"version":"1.13.0"}"#)]),
            (M_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (M_BIN, vec![Ok(b"new-bin")]),
        ]);
        assert_eq!(run_fetch(&f, "1.12.3").await.unwrap().bin, b"new-bin");
    }

    #[tokio::test]
    async fn mirror_must_match_githubs_checksum_when_github_published_one() {
        // GitHub's binary download is down, but its checksum was readable: the mirror copy
        // must be that exact build, not merely self-consistent.
        let f = Fake::new(vec![
            (API, vec![Ok(leak(release_json()))]),
            (GH_SUM, vec![Ok(leak(sum_of(b"new-bin")))]),
            (GH_BIN, vec![Err("502")]),
            (M_LATEST, vec![Ok(br#"{"version":"1.13.0"}"#)]),
            (M_SUM, vec![Ok(leak(sum_of(b"other-bin")))]),
            (M_BIN, vec![Ok(b"other-bin")]),
        ]);
        let err = run_fetch(&f, "1.12.3").await.unwrap_err();
        assert!(err.contains("not on GitHub's latest build"), "{err}");
        assert!(!f.calls().contains(&M_BIN.to_string()));
    }

    #[tokio::test]
    async fn stale_mirror_never_downgrades() {
        let f = Fake::new(vec![
            (API, vec![Err("net")]),
            (M_LATEST, vec![Ok(br#"{"version":"1.12.2"}"#)]),
            (M_SUM, vec![Ok(leak(sum_of(b"old-bin")))]),
            (M_BIN, vec![Ok(b"old-bin")]),
        ]);
        let err = run_fetch(&f, "1.12.3").await.unwrap_err();
        assert!(err.contains("refusing to downgrade"), "{err}");
        assert!(!f.calls().contains(&M_BIN.to_string()));
    }

    #[tokio::test]
    async fn missing_mirror_version_file_fails_closed() {
        let f = Fake::new(vec![(API, vec![Err("503")]), (M_LATEST, vec![Err("404")])]);
        let err = run_fetch(&f, "1.12.3").await.unwrap_err();
        assert!(err.contains("GitHub:") && err.contains("Mirror: Mirror version check failed"), "{err}");
    }

    #[test]
    fn versions_parse_strictly() {
        assert_eq!(parse_version("v1.12.3"), Some((1, 12, 3)));
        assert_eq!(parse_version("1.12.3"), Some((1, 12, 3)));
        assert!(parse_version("1.12").is_none());
        assert!(parse_version("1.12.3-rc1").is_none());
        assert!(parse_version("1.12.3.4").is_none());
        assert!(parse_version("1.9.99") < parse_version("1.10.0"));
    }
}
