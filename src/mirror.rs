//! Downloads that survive a GitHub outage.
//!
//! Every file `sgl` fetches from GitHub has a copy on our own R2 bucket, served at
//! `MIRROR_BASE`. Two kinds:
//!
//! * **Pinned artifacts** (llama.cpp builds for `sgl setup`): we know their sha256 in advance.
//!   The copy lives at `MIRROR_BASE/mirror-<first 16 hex of sha256>-<file name>`, the same
//!   content-addressed scheme the desktop app uses (sgl-node-desktop `src-tauri/src/mirror.rs`).
//! * **Node release binaries** (`sgl update`): `MIRROR_BASE/sgl-<platform>` + `.sha256`, refreshed
//!   on every promote, plus `sgl-latest.json` naming the version (see `update.rs`).
//!
//! Rules (same as the desktop app):
//! * A source is retried only on 5xx, 429 and network errors, 2 s / 5 s / 10 s apart.
//! * ANY primary failure, including a 404 or a checksum mismatch, then tries the mirror. That is
//!   safe because the mirror copy is held to the same pinned sha256.
//! * Nothing is ever used unless it matches the pin, whichever source it came from. A checksum
//!   mismatch is never retried against the same source.
//!
//! When a pinned GitHub artifact changes, upload its mirror copy and add it to `ARTIFACTS`
//! (RELEASING.md, "GitHub outage mirror").

use sha2::{Digest, Sha256};
use std::future::Future;
use std::time::Duration;

pub(crate) const MIRROR_BASE: &str = "https://cloud.x402compute.cc/downloads/node";

/// Back-off between attempts at one source. Tests pass their own (zero) delays.
pub(crate) const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_secs(2),
    Duration::from_secs(5),
    Duration::from_secs(10),
];

/// A response body that stops arriving for this long counts as a network error (retried).
pub(crate) const STALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Every pinned artifact `sgl` can fetch from GitHub, on every platform: (primary URL, sha256,
/// file name used for the mirror copy). Tests check each call site's pin is listed here, so a
/// new pin cannot ship without a mirror name.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const ARTIFACTS: &[(&str, &str, &str)] = &[
    // llama.cpp b11719, one per platform/variant. setup.rs asset_for().
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-macos-arm64.tar.gz", "076a2b0d15253078b90bd520a82dc2edc6c17657968878876ba3def3e80f54ba", "llama-b11719-bin-macos-arm64.tar.gz"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-macos-x64.tar.gz", "739708dca398960462725dd885010e10e6b9fed79e812a29f18d260af23f9929", "llama-b11719-bin-macos-x64.tar.gz"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-win-cpu-x64.zip", "1b1234d208a8b851cc27cd3d96b850d1547a4e5cfc5cd75f0a9fdd1d4c5bea20", "llama-b11719-bin-win-cpu-x64.zip"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-win-vulkan-x64.zip", "08dde7f4133443ac629431c35f0435e267ca73251fc002cb95e10625424e2963", "llama-b11719-bin-win-vulkan-x64.zip"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-ubuntu-x64.tar.gz", "2c50edd1422cdc686a1620923a7dda9a33634509566d2821155fb3d4879510a3", "llama-b11719-bin-ubuntu-x64.tar.gz"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-ubuntu-vulkan-x64.tar.gz", "9a09cce245acf8823feb99fb7ba8ddd3e878e3827564888131e0185044aa4429", "llama-b11719-bin-ubuntu-vulkan-x64.tar.gz"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-ubuntu-arm64.tar.gz", "290aa4432ad8e28b638285be6a75d0dd6f327d9b2d95f7048a8cf5e69704bee9", "llama-b11719-bin-ubuntu-arm64.tar.gz"),
    ("https://github.com/ggml-org/llama.cpp/releases/download/b11719/llama-b11719-bin-ubuntu-vulkan-arm64.tar.gz", "ec2b1e886192c9e1e686e4a5a4d95079f75b98966274264e7bf6e2e13a063185", "llama-b11719-bin-ubuntu-vulkan-arm64.tar.gz"),
];

pub(crate) fn valid_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// The mirror object name for a pinned file: `mirror-<sha256 prefix 16>-<sanitized name>`.
/// Content-addressed, so a changed file always gets a new name. Must match the cloud route's
/// allowlist `^mirror-[0-9a-f]{16}-[A-Za-z0-9._-]{1,128}$`.
pub(crate) fn mirror_name(sha256: &str, file_name: &str) -> String {
    let mut clean: String = file_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '-' })
        .take(128)
        .collect();
    if clean.is_empty() {
        clean.push_str("file");
    }
    let prefix: String = sha256.to_ascii_lowercase().chars().take(16).collect();
    format!("mirror-{prefix}-{clean}")
}

pub(crate) fn mirror_url(sha256: &str, file_name: &str) -> String {
    format!("{MIRROR_BASE}/{}", mirror_name(sha256, file_name))
}

/// One attempt at one source failed. `Retry` = worth another try (5xx, 429, 408, network,
/// truncated body); `NextSource` = this source will not work (4xx, oversized), go to the next.
#[derive(Debug)]
pub(crate) enum Failure {
    Retry(String),
    NextSource(String),
}

/// One HTTP GET into memory, capped at `max_bytes`, refusing compressed bodies (so the bytes we
/// hash are the bytes the server stored).
pub(crate) async fn get_once(
    client: &reqwest::Client,
    url: &str,
    max_bytes: usize,
    stall_timeout: Duration,
) -> Result<Vec<u8>, Failure> {
    let mut response = client
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "identity")
        .send()
        .await
        .map_err(|e| Failure::Retry(format!("request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let message = format!("HTTP {status}");
        return Err(
            if status == reqwest::StatusCode::REQUEST_TIMEOUT
                || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status.is_server_error()
            {
                Failure::Retry(message)
            } else {
                Failure::NextSource(message)
            },
        );
    }

    if let Some(encoding) = response.headers().get(reqwest::header::CONTENT_ENCODING) {
        let encoding = encoding
            .to_str()
            .map_err(|_| Failure::NextSource("invalid Content-Encoding header".to_string()))?;
        if !encoding.trim().eq_ignore_ascii_case("identity") {
            return Err(Failure::NextSource(format!(
                "server ignored identity encoding request (Content-Encoding: {encoding})"
            )));
        }
    }

    let advertised_len = response.content_length();
    if let Some(len) = advertised_len {
        if len > max_bytes as u64 {
            return Err(Failure::NextSource(format!(
                "Content-Length {len} exceeds {max_bytes}-byte limit"
            )));
        }
    }

    let capacity = advertised_len
        .and_then(|len| usize::try_from(len).ok())
        .unwrap_or(0);
    let mut body = Vec::with_capacity(capacity);
    loop {
        let chunk = tokio::time::timeout(stall_timeout, response.chunk())
            .await
            .map_err(|_| {
                Failure::Retry(format!(
                    "response body stalled for {} seconds",
                    stall_timeout.as_secs_f32()
                ))
            })?
            .map_err(|e| Failure::Retry(format!("response body read failed: {e}")))?;
        let Some(chunk) = chunk else {
            break;
        };
        let new_len = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| Failure::NextSource("response size overflow".to_string()))?;
        if new_len > max_bytes {
            return Err(Failure::NextSource(format!(
                "response body exceeds {max_bytes}-byte limit"
            )));
        }
        body.extend_from_slice(&chunk);
    }

    if let Some(len) = advertised_len {
        if body.len() as u64 != len {
            return Err(Failure::Retry(format!(
                "incomplete response body (expected {len} bytes, received {})",
                body.len()
            )));
        }
    }
    if body.is_empty() {
        return Err(Failure::Retry("empty response body".to_string()));
    }

    Ok(body)
}

/// GET one URL, retrying `Retry` failures after each of `delays`. `get(url, max_bytes)` does
/// one attempt (real HTTP in production, scripted in tests).
pub(crate) async fn get_with_retry<F, Fut>(
    url: &str,
    max_bytes: usize,
    label: &str,
    delays: &[Duration],
    get: &mut F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(String, usize) -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Failure>>,
{
    let attempts = delays.len() + 1;
    for attempt in 0..attempts {
        if attempt > 0 {
            tokio::time::sleep(delays[attempt - 1]).await;
        }
        match get(url.to_string(), max_bytes).await {
            Ok(body) => return Ok(body),
            Err(Failure::NextSource(reason)) => return Err(format!("{label} failed: {reason}")),
            Err(Failure::Retry(reason)) if attempt + 1 == attempts => {
                return Err(format!("{label} failed after {attempts} attempts: {reason}"));
            }
            Err(Failure::Retry(reason)) => {
                eprintln!(
                    "  {label} attempt {}/{attempts} failed ({reason}); retrying…",
                    attempt + 1
                );
            }
        }
    }
    unreachable!("the attempt loop always returns")
}

/// Fetch a pinned file from `sources` in order (each with retries) and return the first body
/// whose sha256 matches the pin. Fails closed: a body that does not match is discarded and the
/// next source is tried; if none matches, nothing is returned.
pub(crate) async fn fetch_verified_with<F, Fut>(
    sources: &[String],
    sha256: &str,
    max_bytes: usize,
    delays: &[Duration],
    mut get: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(String, usize) -> Fut,
    Fut: Future<Output = Result<Vec<u8>, Failure>>,
{
    let want = sha256.to_ascii_lowercase();
    if !valid_sha256(&want) {
        return Err("Internal error: this download has no valid checksum pin.".into());
    }
    let mut errors: Vec<String> = Vec::new();
    for (i, url) in sources.iter().enumerate() {
        let label = if i == 0 { "download" } else { "backup download" };
        if i > 0 {
            eprintln!("  Trying the Singularity Layer download mirror…");
        }
        match get_with_retry(url, max_bytes, label, delays, &mut get).await {
            Ok(body) => {
                if sha256_hex(&body) == want {
                    return Ok(body);
                }
                errors.push(format!(
                    "{label}: the file from {url} did not match its expected checksum, so it was discarded"
                ));
            }
            Err(e) => errors.push(format!("{e} ({url})")),
        }
    }
    Err(errors.join("; "))
}

/// Fetch a pinned file: `primary` first, then our mirror copy of `file_name`. The returned bytes
/// always hash to `sha256`.
pub(crate) async fn fetch_pinned(
    client: &reqwest::Client,
    primary: &str,
    sha256: &str,
    file_name: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, String> {
    let sources = [primary.to_string(), mirror_url(sha256, file_name)];
    fetch_verified_with(&sources, sha256, max_bytes, &RETRY_DELAYS, |url, max| {
        let client = client.clone();
        async move { get_once(&client, &url, max, STALL_TIMEOUT).await }
    })
    .await
}

/// The mirror copy for a pinned GitHub URL, if it is one we mirror.
#[cfg(test)]
pub(crate) fn mirror_for(primary: &str, sha256: &str) -> Option<String> {
    ARTIFACTS
        .iter()
        .find(|(u, s, _)| *u == primary && s.eq_ignore_ascii_case(sha256))
        .map(|(_, s, n)| mirror_url(s, n))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    pub(crate) const NO_WAIT: [Duration; 3] = [Duration::ZERO; 3];

    /// A scripted source: `script[url]` is a queue of responses (the last one repeats).
    /// `Err("5..")`/`Err("net")` are retryable, any other `Err` moves to the next source.
    pub(crate) struct Fake {
        pub script: RefCell<HashMap<String, Vec<Result<Vec<u8>, String>>>>,
        pub calls: RefCell<Vec<String>>,
    }

    impl Fake {
        pub fn new(entries: Vec<(&str, Vec<Result<&[u8], &str>>)>) -> Self {
            let script = entries
                .into_iter()
                .map(|(u, q)| {
                    (
                        u.to_string(),
                        q.into_iter()
                            .map(|r| r.map(|b| b.to_vec()).map_err(|e| e.to_string()))
                            .collect(),
                    )
                })
                .collect();
            Self { script: RefCell::new(script), calls: RefCell::new(vec![]) }
        }

        pub fn get(&self) -> impl FnMut(String, usize) -> std::future::Ready<Result<Vec<u8>, Failure>> + '_ {
            move |url, _max| {
                self.calls.borrow_mut().push(url.clone());
                let mut s = self.script.borrow_mut();
                let r = match s.get_mut(&url) {
                    None => Err("404".to_string()),
                    Some(q) if q.len() > 1 => q.remove(0),
                    Some(q) => q[0].clone(),
                };
                std::future::ready(match r {
                    Ok(body) => Ok(body),
                    Err(e) if e.starts_with('5') || e == "net" || e == "429" => Err(Failure::Retry(e)),
                    Err(e) => Err(Failure::NextSource(e)),
                })
            }
        }

        pub fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }

    fn srcs() -> Vec<String> {
        vec!["primary".into(), "mirror".into()]
    }

    #[tokio::test]
    async fn primary_ok_never_touches_mirror() {
        let f = Fake::new(vec![("primary", vec![Ok(b"good")]), ("mirror", vec![Err("404")])]);
        let got = fetch_verified_with(&srcs(), &sha256_hex(b"good"), 64, &NO_WAIT, f.get()).await.unwrap();
        assert_eq!(got, b"good");
        assert_eq!(f.calls(), vec!["primary"]);
    }

    #[tokio::test]
    async fn primary_5xx_retried_then_mirror_ok() {
        let f = Fake::new(vec![("primary", vec![Err("503")]), ("mirror", vec![Ok(b"good")])]);
        let got = fetch_verified_with(&srcs(), &sha256_hex(b"good"), 64, &NO_WAIT, f.get()).await.unwrap();
        assert_eq!(got, b"good");
        // 1 try + 3 retries on the primary, then the mirror.
        assert_eq!(f.calls(), vec!["primary", "primary", "primary", "primary", "mirror"]);
    }

    #[tokio::test]
    async fn primary_429_and_network_errors_are_retried() {
        let f = Fake::new(vec![("primary", vec![Err("429"), Err("net"), Ok(b"good")]), ("mirror", vec![Err("404")])]);
        fetch_verified_with(&srcs(), &sha256_hex(b"good"), 64, &NO_WAIT, f.get()).await.unwrap();
        assert_eq!(f.calls(), vec!["primary", "primary", "primary"]);
    }

    #[tokio::test]
    async fn primary_404_goes_straight_to_mirror_without_retries() {
        let f = Fake::new(vec![("primary", vec![Err("404")]), ("mirror", vec![Ok(b"good")])]);
        fetch_verified_with(&srcs(), &sha256_hex(b"good"), 64, &NO_WAIT, f.get()).await.unwrap();
        assert_eq!(f.calls(), vec!["primary", "mirror"]);
    }

    #[tokio::test]
    async fn checksum_mismatch_on_both_fails_closed() {
        let f = Fake::new(vec![("primary", vec![Ok(b"evil")]), ("mirror", vec![Ok(b"also evil")])]);
        let err = fetch_verified_with(&srcs(), &sha256_hex(b"good"), 64, &NO_WAIT, f.get()).await.unwrap_err();
        assert!(err.contains("checksum"), "{err}");
        // A mismatch is never retried on the same source.
        assert_eq!(f.calls(), vec!["primary", "mirror"]);
    }

    #[tokio::test]
    async fn tampered_primary_falls_back_to_good_mirror() {
        let f = Fake::new(vec![("primary", vec![Ok(b"evil")]), ("mirror", vec![Ok(b"good")])]);
        let got = fetch_verified_with(&srcs(), &sha256_hex(b"good"), 64, &NO_WAIT, f.get()).await.unwrap();
        assert_eq!(got, b"good");
    }

    #[tokio::test]
    async fn missing_or_malformed_pin_refuses_to_download() {
        let f = Fake::new(vec![("primary", vec![Ok(b"x")]), ("mirror", vec![Ok(b"x")])]);
        assert!(fetch_verified_with(&srcs(), "", 64, &NO_WAIT, f.get()).await.is_err());
        assert!(fetch_verified_with(&srcs(), "abc", 64, &NO_WAIT, f.get()).await.is_err());
        assert!(f.calls().is_empty());
    }

    /// Minimal HTTP server: answers each connection with the next scripted (status, body).
    async fn serve(responses: Vec<(u16, &'static [u8])>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for (status, body) in responses {
                let (mut s, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf).await;
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes()).await;
                let _ = s.write_all(body).await;
                let _ = s.shutdown().await;
            }
        });
        format!("http://{addr}/file")
    }

    async fn real_fetch(sources: &[String], sha: &str) -> Result<Vec<u8>, String> {
        let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build().unwrap();
        fetch_verified_with(sources, sha, 1024, &NO_WAIT, |url, max| {
            let client = client.clone();
            async move { get_once(&client, &url, max, Duration::from_secs(2)).await }
        })
        .await
    }

    #[tokio::test]
    async fn real_http_5xx_then_mirror() {
        let primary = serve(vec![(502, b"x"), (503, b"x"), (500, b"x"), (504, b"x")]).await;
        let mirror = serve(vec![(200, b"payload")]).await;
        let got = real_fetch(&[primary, mirror], &sha256_hex(b"payload")).await.unwrap();
        assert_eq!(got, b"payload");
    }

    #[tokio::test]
    async fn real_http_404_then_bad_mirror_fails_closed() {
        let primary = serve(vec![(404, b"x")]).await;
        let mirror = serve(vec![(200, b"wrong")]).await;
        let err = real_fetch(&[primary, mirror], &sha256_hex(b"payload")).await.unwrap_err();
        assert!(err.contains("404") && err.contains("checksum"), "{err}");
    }

    /// Same pattern as the cloud route allowlist (sgl-network-cloud src/lib/node-downloads.ts).
    fn route_allows(name: &str) -> bool {
        let Some(rest) = name.strip_prefix("mirror-") else { return false };
        let (prefix, tail) = rest.split_at(16.min(rest.len()));
        prefix.len() == 16
            && prefix.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            && tail.starts_with('-')
            && (2..=129).contains(&tail.len())
            && tail[1..].bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    }

    #[test]
    fn every_pinned_artifact_has_a_valid_unique_mirror_name() {
        let mut names = std::collections::HashSet::new();
        for (url, sha, file) in ARTIFACTS {
            assert!(valid_sha256(sha), "{url}: bad sha");
            assert!(url.starts_with("https://github.com/") || url.starts_with("https://raw.githubusercontent.com/"), "{url}");
            assert!(url.ends_with(&format!("/{file}")), "{url} vs {file}");
            let name = mirror_name(sha, file);
            assert!(route_allows(&name), "{name} would be refused by the cloud route");
            assert!(name.starts_with(&format!("mirror-{}-", &sha[..16])));
            assert!(names.insert(name.clone()), "duplicate mirror name {name}");
            assert_eq!(mirror_for(url, sha), Some(format!("{MIRROR_BASE}/{name}")));
        }
    }

    #[test]
    fn mirror_name_sanitizes() {
        assert_eq!(mirror_name(&"a".repeat(64), "a b/c?.zip"), format!("mirror-{}-a-b-c-.zip", "a".repeat(16)));
        assert!(route_allows(&mirror_name(&"b".repeat(64), "")));
        assert!(route_allows(&mirror_name(&"c".repeat(64), &"x".repeat(500))));
        assert!(!route_allows("mirror-zzzz-x"));
    }

    /// Live proof against the real mirror: a dead GitHub URL falls back to our R2 copy.
    /// `cargo test --bin sgl mirror::tests::live_mirror_serves_every_artifact -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn live_mirror_serves_every_artifact() {
        let client = reqwest::Client::builder().https_only(true).build().unwrap();
        for (_, sha, file) in ARTIFACTS {
            let dead = format!("https://github.com/Singularity-Layer/does-not-exist/{file}");
            let got = fetch_pinned(&client, &dead, sha, file, 512 * 1024 * 1024).await.unwrap();
            assert_eq!(sha256_hex(&got), *sha, "{file}");
        }
    }

    /// Prints the upload table: `cargo test --bin sgl mirror::tests::print_mirror_table -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn print_mirror_table() {
        for (url, sha, file) in ARTIFACTS {
            println!("{url}\t{sha}\t{}", mirror_name(sha, file));
        }
    }
}
