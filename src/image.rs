//! Dedicated mflux text-to-image worker supervisor for the Grid `image-v1` lane
//! (registry: FLUX.2 klein 4B, 4-bit, one model per node).
//! Mirrors `tts.rs`: one owned, serialized, framed worker that loads the model ONCE, keeps it
//! hot, and is timed out, killed and reaped on failure. Prompt in (UTF-8, byte count bound in
//! the reservation), PNG out. Spec: x402Studio GRID_IMAGE_V2.md ("GRID_IMAGE_V1 spec v2").
//!
//! Self-contained on purpose (no orchestrator/node imports) so the lib target and its
//! examples/tests can drive the engine directly. The job glue (sealing, signing, REST
//! completion with retry) lives in `image_job.rs`.
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

pub const MODEL_ID: &str = "flux2-klein-4b-q4";
pub const MODEL_REPOSITORY: &str = "mflux-community/flux2-klein-4b-mflux-q4";
pub const MODEL_REVISION: &str = "794cd159538149ad9830848508c31f0ea7088e58";
/// sha256 of the canonical file manifest (one `<sha256>  <size>  <path>\n` line per file,
/// sorted by path), not of one file.
pub const MODEL_SHA256: &str = "c8e510f8bf50ee177528b1cbe4780fbaa55417b676b95b0aa85567c22c456435";
pub const MODEL_LICENSE: &str = "Apache-2.0";
pub const PROTOCOL: &str = "image-v1";
pub const RUNTIME: &str = "mflux";
pub const RUNTIME_REVISION: &str = "media-2:mflux-0.24.0";
/// `mode_label()` of the engine and `capabilities.image.engine`.
pub const ENGINE_LABEL: &str = "image-worker";
pub const STEPS: u32 = 4;
pub const OUTPUT_FORMAT: &str = "png";
pub const OUTPUT_MIME: &str = "image/png";
pub const MAX_PROMPT_BYTES: usize = 2000;
pub const SIZES: [(u32, u32); 5] = [
    (1024, 1024),
    (1152, 768),
    (768, 1152),
    (1280, 720),
    (720, 1280),
];
pub const MAX_SEED: u64 = 4_294_967_295;
pub const MAX_PNG_BYTES: usize = 6_000_000;
/// Cap on the base64 ciphertext of the sealed result the orchestrator accepts.
pub const MAX_SEALED_RESULT_B64: usize = 8_400_000;
/// Sealed request: canonical JSON with a base64 prompt; 2000 prompt bytes is ~2.7 KB of base64
/// plus the fixed keys, well inside this.
pub const MAX_SEALED_INPUT: usize = 16 * 1024;
const MAX_WORKER_INPUT_FRAME: usize = 8 * 1024;
/// A 6,000,000-byte PNG as base64 (8,000,000 chars) plus a small JSON frame.
const MAX_OUTPUT_FRAME: u64 = 8 * 1024 * 1024 + 64 * 1024;
const REPLAY_CACHE_SIZE: usize = 65_536;
const REPLAY_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);
/// Crash-loop guard: at most this many non-timeout restarts in any rolling hour. A worker
/// killed because one request ran past REQUEST_TIMEOUT is relaunched outside this budget, so a
/// few slow jobs can never leave the node permanently de-advertised.
const MAX_RESTARTS_PER_WINDOW: usize = 3;
const RESTART_WINDOW: Duration = Duration::from_secs(60 * 60);
const TIMED_OUT: &str = "image worker timed out";
const INPUT_TIMED_OUT: &str = "image worker input timed out";

/// One generation, worker side.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(420);
/// Covers the worker's own re-hash of ~4.6 GB of weights, the model load and the smoke image.
pub const STARTUP_TIMEOUT: Duration = Duration::from_secs(420);
/// Orchestrator default `IMAGE_JOB_TIMEOUT_SECONDS`: the reaper fails a job still dispatched
/// after this long.
pub const ORCHESTRATOR_JOB_TIMEOUT: Duration = Duration::from_secs(600);
/// The node gives up on a job (generation + every completion retry) after this long from
/// receipt. Must stay below the reaper timeout minus 60 s so a late completion can never race
/// the reaper's hold release.
pub const NODE_DEADLINE: Duration = Duration::from_secs(480);
/// No completion attempt starts with less than this left before the node deadline: an 8 MB
/// upload that cannot finish in time only adds load.
pub const UPLOAD_MARGIN: Duration = Duration::from_secs(20);
/// Per-attempt HTTP timeout for an image completion (the client default is 30 s, too short
/// for an 8 MB body on a slow uplink). Always capped by the time left before the deadline.
pub const COMPLETE_HTTP_TIMEOUT: Duration = Duration::from_secs(60);
const RETRY_BASE: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(16);
/// Busy/no-progress watchdog threshold for an image node. Must exceed the node deadline, or the
/// watchdog would restart a worker that is legitimately generating or uploading.
pub const WEDGE_MS: u64 = 720_000;

const _: () = assert!(NODE_DEADLINE.as_secs() + 60 < ORCHESTRATOR_JOB_TIMEOUT.as_secs());
const _: () =
    assert!(REQUEST_TIMEOUT.as_secs() + UPLOAD_MARGIN.as_secs() < NODE_DEADLINE.as_secs());
const _: () = assert!(WEDGE_MS > NODE_DEADLINE.as_secs() * 1000);
const _: () = assert!(MAX_OUTPUT_FRAME >= 8 * 1024 * 1024);
const _: () = assert!(MAX_OUTPUT_FRAME as usize > MAX_PNG_BYTES.div_ceil(3) * 4 + 512);

const MODEL_MANIFEST: &str = include_str!("../scripts/image_model_files_flux2_klein_4b_q4.json");
const WORKER_SCRIPT: &[u8] = include_bytes!("../scripts/image_worker.py");
const WORKER_FILE: &str = "image_worker.py";
const MANIFEST_FILE: &str = "image_model_files.json";
/// Test-only switch understood by `image_worker.py`. Always stripped from production workers.
pub const FAKE_ENV: &str = "SGL_IMAGE_WORKER_FAKE";

/// Operator-facing lookup for `--model-name` (case-insensitive, like the rest of the CLI).
pub fn is_image_model(name: &str) -> bool {
    name.eq_ignore_ascii_case(MODEL_ID)
}

pub fn size_allowed(width: u32, height: u32) -> bool {
    SIZES.contains(&(width, height))
}

fn pins_match(model: &str, revision: &str, sha256: &str) -> bool {
    model == MODEL_ID && revision == MODEL_REVISION && sha256 == MODEL_SHA256
}

fn platform_approved() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

#[derive(Clone)]
pub struct WorkerConfig {
    pub program: String,
    pub args: Vec<String>,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    /// Set after the environment is scrubbed. Production passes nothing.
    pub extra_env: Vec<(String, String)>,
    /// Accept a worker that reports `fake: true`. Production is always false.
    pub allow_fake: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct KeyBinding {
    pub encryption_public_key: String,
    pub encryption_public_key_signature: String,
    pub key_version: u32,
}

/// Advertised under heartbeat `capabilities.image`. Field order is the wire order.
#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub kind: String,
    pub engine: String,
    pub image_ready: bool,
    pub image_protocol: String,
    pub image_models: Vec<String>,
    pub image_runtime: String,
    pub runtime_revision: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub model_repository: String,
    pub model_license: String,
    pub os: String,
    pub architecture: String,
    pub max_prompt_bytes: usize,
    pub sizes: Vec<[u32; 2]>,
    pub steps: u32,
    pub output_format: String,
    pub free_slots: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_binding: Option<KeyBinding>,
}

impl Capabilities {
    pub fn bind_heartbeat(
        mut self,
        free_slots: u32,
        encryption_public_key: &str,
        encryption_public_key_signature: Option<&str>,
        key_version: Option<u32>,
    ) -> Self {
        self.free_slots = free_slots.min(1);
        self.key_binding =
            encryption_public_key_signature
                .zip(key_version)
                .map(|(signature, version)| KeyBinding {
                    encryption_public_key: encryption_public_key.to_string(),
                    encryption_public_key_signature: signature.to_string(),
                    key_version: version,
                });
        if self.key_binding.is_none() {
            self.image_ready = false;
            self.free_slots = 0;
        }
        self
    }
}

pub struct ReadinessSnapshot {
    pub available_models: Vec<String>,
    pub capabilities: Option<Capabilities>,
}

/// Validated text-to-image request (from the decrypted canonical plaintext).
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    pub request_id: String,
    pub model: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub width: u32,
    pub height: u32,
    /// None = the node picks one (reported back in the result header).
    pub seed: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalPrompt {
    encoding: String,
    bytes: u64,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalRequest {
    protocol: String,
    request_id: String,
    model: String,
    model_revision: String,
    model_sha256: String,
    width: u32,
    height: u32,
    seed: Option<u64>,
    prompt: CanonicalPrompt,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReservationBinding {
    pub protocol: String,
    pub request_id: String,
    pub model: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub width: u32,
    pub height: u32,
    pub seed: Option<u64>,
    pub prompt_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageDispatch {
    enc: serde_json::Value,
    image_reservation: serde_json::Value,
}

/// `seed` is nullable but never optional: an absent key is a different (non-canonical) message.
fn has_seed_key(value: &serde_json::Value) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.contains_key("seed"))
}

fn seed_ok(seed: Option<u64>) -> bool {
    seed.is_none_or(|s| s <= MAX_SEED)
}

impl ReservationBinding {
    pub fn parse_dispatch(value: &serde_json::Value) -> Result<Self, String> {
        let dispatch: ImageDispatch = serde_json::from_value(value.clone())
            .map_err(|_| "image dispatch shape invalid".to_string())?;
        if !dispatch.enc.is_object() {
            return Err("image encrypted envelope missing".into());
        }
        if !has_seed_key(&dispatch.image_reservation) {
            return Err("image reservation binding invalid".into());
        }
        let binding: ReservationBinding = serde_json::from_value(dispatch.image_reservation)
            .map_err(|_| "image reservation binding invalid".to_string())?;
        if binding.protocol != PROTOCOL
            || !canonical_uuid_v4(&binding.request_id)
            || !pins_match(
                &binding.model,
                &binding.model_revision,
                &binding.model_sha256,
            )
            || !size_allowed(binding.width, binding.height)
            || !seed_ok(binding.seed)
            || binding.prompt_bytes == 0
            || binding.prompt_bytes > MAX_PROMPT_BYTES as u64
        {
            return Err("image reservation binding invalid".into());
        }
        Ok(binding)
    }

    pub fn matches(&self, request: &ImageRequest) -> bool {
        self.protocol == PROTOCOL
            && self.request_id == request.request_id
            && self.model == request.model
            && self.model_revision == request.model_revision
            && self.model_sha256 == request.model_sha256
            && self.width == request.width
            && self.height == request.height
            && self.seed == request.seed.map(u64::from)
            && self.prompt_bytes == request.prompt.len() as u64
    }
}

impl ImageRequest {
    pub fn parse(value: &serde_json::Value) -> Result<Self, String> {
        if !has_seed_key(value) {
            return Err("image request shape invalid".into());
        }
        let request: CanonicalRequest = serde_json::from_value(value.clone())
            .map_err(|_| "image request shape invalid".to_string())?;
        if request.protocol != PROTOCOL {
            return Err("image protocol mismatch".into());
        }
        if !canonical_uuid_v4(&request.request_id) {
            return Err("image request id invalid".into());
        }
        if !pins_match(
            &request.model,
            &request.model_revision,
            &request.model_sha256,
        ) {
            return Err("image model binding mismatch".into());
        }
        if !size_allowed(request.width, request.height) {
            return Err("image size not allowed".into());
        }
        if !seed_ok(request.seed) {
            return Err("image seed out of range".into());
        }
        // Base64 like speech's text input, so JSON escaping can never change the sealed length
        // the orchestrator checks against `prompt_bytes`.
        let prompt = request.prompt;
        if prompt.encoding != "base64"
            || prompt.bytes == 0
            || prompt.bytes > MAX_PROMPT_BYTES as u64
            || prompt.data.len() > MAX_PROMPT_BYTES.div_ceil(3) * 4
        {
            return Err("image prompt out of bounds".into());
        }
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&prompt.data)
            .map_err(|_| "image prompt base64 invalid".to_string())?;
        if decoded.len() as u64 != prompt.bytes {
            return Err("image prompt length contradicts byte count".into());
        }
        let text = String::from_utf8(decoded).map_err(|_| "image prompt is not UTF-8")?;
        if text.trim().is_empty() || text.contains('\0') {
            return Err("image prompt empty".into());
        }
        Ok(Self {
            prompt: text,
            request_id: request.request_id,
            model: request.model,
            model_revision: request.model_revision,
            model_sha256: request.model_sha256,
            width: request.width,
            height: request.height,
            seed: request.seed.map(|s| s as u32),
        })
    }
}

/// Lowercase canonical UUIDv4 (the orchestrator mints request ids with `crypto.randomUUID`).
pub fn canonical_uuid_v4(value: &str) -> bool {
    let b = value.as_bytes();
    if b.len() != 36 || b[14] != b'4' || !matches!(b[19], b'8' | b'9' | b'a' | b'b') {
        return false;
    }
    b.iter().enumerate().all(|(index, byte)| match index {
        8 | 13 | 18 | 23 => *byte == b'-',
        _ => byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase(),
    })
}

#[derive(Debug, Clone)]
pub struct ImageOutput {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    pub sha256: String,
}

#[derive(Serialize)]
struct ResultImage<'a> {
    format: &'a str,
    mime_type: &'a str,
    width: u32,
    height: u32,
    bytes: usize,
    sha256: &'a str,
}

/// Field order is the wire order (serde serializes struct fields in declaration order).
#[derive(Serialize)]
struct ResultHeader<'a> {
    object: &'a str,
    protocol: &'a str,
    request_id: &'a str,
    job_id: &'a str,
    model: &'a str,
    model_revision: &'a str,
    model_sha256: &'a str,
    width: u32,
    height: u32,
    seed: u32,
    steps: u32,
    prompt_bytes: usize,
    image: ResultImage<'a>,
}

/// `image-result-v1` plaintext: one canonical JSON header line, `\n`, then the raw PNG bytes.
pub fn result_plaintext(job_id: &str, request: &ImageRequest, output: &ImageOutput) -> Vec<u8> {
    let header = ResultHeader {
        object: "image",
        protocol: PROTOCOL,
        request_id: &request.request_id,
        job_id,
        model: &request.model,
        model_revision: &request.model_revision,
        model_sha256: &request.model_sha256,
        width: output.width,
        height: output.height,
        seed: output.seed,
        steps: STEPS,
        prompt_bytes: request.prompt.len(),
        image: ResultImage {
            format: OUTPUT_FORMAT,
            mime_type: OUTPUT_MIME,
            width: output.width,
            height: output.height,
            bytes: output.png.len(),
            sha256: &output.sha256,
        },
    };
    let mut bytes = serde_json::to_vec(&header).expect("header serializes");
    bytes.push(b'\n');
    bytes.extend_from_slice(&output.png);
    bytes
}

/// Cleartext usage reported with the completion. Billing is computed server-side from the
/// registry contract; this is informational and never trusted for money.
pub fn usage(output: &ImageOutput) -> serde_json::Value {
    serde_json::json!({
        "pixels": u64::from(output.width) * u64::from(output.height),
        "steps": STEPS,
    })
}

// ── Completion retry planning (pure, so it is testable without a network) ─────────────────

/// Exponential backoff before retry number `attempt` (1-based): 1, 2, 4, 8, 16, 16, ... s.
pub fn retry_backoff(attempt: u32) -> Duration {
    let factor = 1u32
        .checked_shl(attempt.saturating_sub(1))
        .unwrap_or(u32::MAX);
    RETRY_BASE.saturating_mul(factor).min(RETRY_MAX)
}

/// Whether an HTTP status from `/complete` is worth retrying. 5xx (the billing transaction
/// rolled back), 408 and 429 are transient; any other 4xx means the orchestrator refused this
/// completion for good (wrong node, reaped, tampered), so retrying only adds load.
pub fn completion_status_retryable(status: u16) -> bool {
    status >= 500 || status == 408 || status == 429
}

/// Plan the next completion attempt. `elapsed` is the time since the node received the job and
/// `attempt` the 0-based number of the attempt about to be made. Returns `(wait, http_timeout)`
/// or None when the deadline (minus the upload margin) leaves no room for another attempt.
pub fn plan_completion_attempt(elapsed: Duration, attempt: u32) -> Option<(Duration, Duration)> {
    let wait = if attempt == 0 {
        Duration::ZERO
    } else {
        retry_backoff(attempt)
    };
    let start = elapsed.checked_add(wait)?;
    let remaining = NODE_DEADLINE.checked_sub(start)?;
    if remaining < UPLOAD_MARGIN {
        return None;
    }
    Some((wait, remaining.min(COMPLETE_HTTP_TIMEOUT)))
}

// ── Worker protocol ───────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    #[serde(rename = "type")]
    frame_type: String,
    protocol: String,
    runtime: String,
    runtime_revision: String,
    model_id: String,
    model_revision: String,
    model_sha256: String,
    steps: u32,
    sizes: Vec<[u32; 2]>,
    output_format: String,
    max_prompt_bytes: usize,
    smoke_seconds: f64,
    fake: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    #[serde(rename = "type")]
    frame_type: String,
    request_id: u64,
    width: u32,
    height: u32,
    bytes: usize,
    sha256: String,
    png_b64: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestError {
    #[serde(rename = "type")]
    frame_type: String,
    request_id: u64,
    code: String,
}

enum WorkerFailure {
    InvalidInput,
    Fatal(String),
}
impl From<String> for WorkerFailure {
    fn from(reason: String) -> Self {
        Self::Fatal(reason)
    }
}
impl From<&str> for WorkerFailure {
    fn from(reason: &str) -> Self {
        Self::Fatal(reason.into())
    }
}

struct Job {
    id: u64,
    request: ImageRequest,
    seed: u32,
    reply: tokio::sync::oneshot::Sender<Result<ImageOutput, String>>,
}

/// Request ids this node has already SERVED (24 h window) plus the ones in flight. An id is
/// recorded as served only after a successful generation, so a runtime failure or a timeout
/// never turns the client's retry into a false replay.
struct SeenRequestIds {
    set: std::collections::HashSet<String>,
    order: std::collections::VecDeque<(String, Instant)>,
    pending: std::collections::HashSet<String>,
}

impl SeenRequestIds {
    fn new() -> Self {
        Self {
            set: std::collections::HashSet::new(),
            order: std::collections::VecDeque::new(),
            pending: std::collections::HashSet::new(),
        }
    }

    fn expire(&mut self, now: Instant) {
        while self
            .order
            .front()
            .is_some_and(|(_, inserted)| now.duration_since(*inserted) >= REPLAY_WINDOW)
        {
            if let Some((expired, _)) = self.order.pop_front() {
                self.set.remove(&expired);
            }
        }
    }

    /// Claim an id for one generation. False when it was already served or is in flight.
    fn begin(&mut self, request_id: &str) -> bool {
        self.expire(Instant::now());
        if self.set.contains(request_id) || self.pending.contains(request_id) {
            return false;
        }
        self.pending.insert(request_id.to_string());
        true
    }

    /// Release the claim; record the id as served only when the generation succeeded.
    fn finish(&mut self, request_id: &str, served: bool) {
        self.pending.remove(request_id);
        if !served {
            return;
        }
        let now = Instant::now();
        self.expire(now);
        if self.order.len() >= REPLAY_CACHE_SIZE {
            if let Some((expired, _)) = self.order.pop_front() {
                self.set.remove(&expired);
            }
        }
        if self.set.insert(request_id.to_string()) {
            self.order.push_back((request_id.to_string(), now));
        }
    }
}

/// Releases a replay claim even when the generating future is dropped mid-flight.
struct ReplayClaim<'a> {
    seen: &'a Mutex<SeenRequestIds>,
    request_id: String,
    served: bool,
}

impl Drop for ReplayClaim<'_> {
    fn drop(&mut self) {
        self.seen
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .finish(&self.request_id, self.served);
    }
}

/// Rolling-window crash-loop budget. Records the restart when it is allowed.
fn restart_allowed(history: &mut std::collections::VecDeque<Instant>, now: Instant) -> bool {
    while history
        .front()
        .is_some_and(|at| now.duration_since(*at) >= RESTART_WINDOW)
    {
        history.pop_front();
    }
    if history.len() >= MAX_RESTARTS_PER_WINDOW {
        return false;
    }
    history.push_back(now);
    true
}

pub struct ImageEngine {
    tx: Mutex<Option<mpsc::SyncSender<Job>>>,
    config: WorkerConfig,
    restart_lock: tokio::sync::Mutex<()>,
    restart_history: Mutex<std::collections::VecDeque<Instant>>,
    /// Set by the supervisor when the last fault was a request timeout.
    timed_out: Arc<AtomicBool>,
    assets: Option<PrivateWorkerTemp>,
    healthy: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    next_id: AtomicU64,
    capabilities: Capabilities,
    seen_request_ids: Mutex<SeenRequestIds>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl ImageEngine {
    /// Start one explicitly configured worker; no capability exists until smoke validation.
    pub async fn start(config: WorkerConfig) -> Result<Self, String> {
        let saved_config = config.clone();
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let healthy = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let timed_out = Arc::new(AtomicBool::new(false));
        let (h, stop, late) = (healthy.clone(), stopped.clone(), timed_out.clone());
        let handle = std::thread::Builder::new()
            .name("image-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, h, stop, late))
            .map_err(|_| "could not start image supervisor")?;
        let capabilities = match ready_rx.await {
            Ok(Ok(ready)) => ready,
            _ => {
                stopped.store(true, Ordering::Release);
                let _ = handle.join();
                return Err("image worker startup failed".into());
            }
        };
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            config: saved_config,
            restart_lock: tokio::sync::Mutex::new(()),
            restart_history: Mutex::new(std::collections::VecDeque::new()),
            timed_out,
            assets: None,
            healthy,
            stopped,
            next_id: AtomicU64::new(1),
            capabilities,
            seen_request_ids: Mutex::new(SeenRequestIds::new()),
            worker: Mutex::new(Some(handle)),
        })
    }

    /// Operator supplies the interpreter (the desktop app's managed media venv with the pinned
    /// mflux) and the local model directory. The node verifies every pinned model file before
    /// starting the worker, and the worker verifies them again before it loads them.
    pub async fn production(
        model_dir: &std::path::Path,
        python: &std::path::Path,
    ) -> Result<Self, String> {
        if !platform_approved() {
            return Err(
                "image runtime is not approved on this platform (Apple Silicon macOS only)".into(),
            );
        }
        if !model_dir.is_dir() {
            return Err(
                "image model directory unavailable; --image-model-dir must point at the pinned flux2-klein-4b-mflux-q4 folder"
                    .into(),
            );
        }
        if !python.is_file() {
            return Err(
                "image interpreter unavailable; --image-python must point at the mflux venv python"
                    .into(),
            );
        }
        verify_model_dir(model_dir)?;
        let assets = materialize_assets()?;
        let path = |p: &std::path::Path, what: &str| -> Result<String, String> {
            p.to_str()
                .map(str::to_string)
                .ok_or_else(|| format!("image {what} path is not UTF-8"))
        };
        let config = WorkerConfig {
            program: path(python, "interpreter")?,
            args: vec![
                "-I".into(),
                "-u".into(),
                path(&assets.0.join(WORKER_FILE), "runtime")?,
                "--model-dir".into(),
                path(model_dir, "model")?,
                "--manifest".into(),
                path(&assets.0.join(MANIFEST_FILE), "runtime")?,
                "--model-id".into(),
                MODEL_ID.into(),
            ],
            startup_timeout: STARTUP_TIMEOUT,
            request_timeout: REQUEST_TIMEOUT,
            extra_env: vec![],
            allow_fake: false,
        };
        let mut engine = Self::start(config).await?;
        engine.assets = Some(assets);
        Ok(engine)
    }

    /// Relaunch the worker. A worker killed for running past REQUEST_TIMEOUT is always
    /// relaunched; any other fault spends the rolling crash-loop budget (3 per hour). A
    /// successful relaunch repeats all startup checks.
    pub async fn restart(&self) -> Result<(), String> {
        let _guard = self.restart_lock.lock().await;
        let after_timeout = self.timed_out.swap(false, Ordering::AcqRel);
        if !after_timeout
            && !restart_allowed(
                &mut self
                    .restart_history
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()),
                Instant::now(),
            )
        {
            return Err("image worker restart budget exhausted for this hour".into());
        }
        self.stop();
        self.stopped.store(false, Ordering::Release);
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (healthy, stopped) = (self.healthy.clone(), self.stopped.clone());
        let late = self.timed_out.clone();
        let config = self.config.clone();
        let handle = std::thread::Builder::new()
            .name("image-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, healthy, stopped, late))
            .map_err(|_| "could not restart image supervisor")?;
        *self.worker.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
        match ready_rx.await {
            Ok(Ok(_)) if self.is_healthy() => {
                *self.tx.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
                Ok(())
            }
            _ => {
                self.stop();
                Err("image worker restart failed".into())
            }
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && !self.stopped.load(Ordering::Acquire)
    }

    pub fn capabilities(&self) -> Option<Capabilities> {
        if self.is_healthy() {
            Some(self.capabilities.clone())
        } else {
            None
        }
    }

    /// `available_models` carries the model id only while the worker is healthy.
    pub fn readiness_snapshot(&self, models: &[String]) -> ReadinessSnapshot {
        let capabilities = self.capabilities();
        let available_models = if capabilities.is_some() {
            models.to_vec()
        } else {
            vec![]
        };
        ReadinessSnapshot {
            available_models,
            capabilities,
        }
    }

    /// One image at a time (the worker channel holds one job; a second concurrent call fails
    /// fast instead of queueing past its deadline).
    pub async fn generate(&self, request: ImageRequest) -> Result<ImageOutput, String> {
        if !self.is_healthy() {
            return Err("image_runtime_failed".into());
        }
        if request.prompt.is_empty()
            || request.prompt.len() > MAX_PROMPT_BYTES
            || request.model != MODEL_ID
            || !size_allowed(request.width, request.height)
        {
            return Err("image_input_invalid".into());
        }
        if !self
            .seen_request_ids
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .begin(&request.request_id)
        {
            return Err("image_input_invalid".into());
        }
        let mut claim = ReplayClaim {
            seen: &self.seen_request_ids,
            request_id: request.request_id.clone(),
            served: false,
        };
        let result = self.dispatch(request).await;
        claim.served = result.is_ok();
        result
    }

    async fn dispatch(&self, request: ImageRequest) -> Result<ImageOutput, String> {
        let seed = request.seed.unwrap_or_else(rand::random::<u32>);
        let (reply, receive) = tokio::sync::oneshot::channel();
        let job = Job {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            request,
            seed,
            reply,
        };
        self.tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .ok_or("image_runtime_failed")?
            .try_send(job)
            .map_err(|_| "image_runtime_failed")?;
        receive.await.map_err(|_| "image_runtime_failed")?
    }

    pub fn stop(&self) {
        self.healthy.store(false, Ordering::Release);
        self.stopped.store(true, Ordering::Release);
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(handle) = self.worker.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ImageEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Model manifest ────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    model: String,
    repository: String,
    revision: String,
    model_sha256: String,
    files: Vec<ManifestFile>,
}

/// The canonical manifest digest: one `<sha256>  <size>  <path>\n` line per file, sorted by path.
fn manifest_digest(files: &[ManifestFile]) -> String {
    let mut sorted: Vec<&ManifestFile> = files.iter().collect();
    sorted.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
    let text: String = sorted
        .iter()
        .map(|f| format!("{}  {}  {}\n", f.sha256, f.size, f.path))
        .collect();
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(text.as_bytes()))
}

fn embedded_manifest() -> Result<Manifest, String> {
    let manifest: Manifest =
        serde_json::from_str(MODEL_MANIFEST).map_err(|_| "image model manifest invalid")?;
    if manifest.model != MODEL_ID
        || manifest.repository != MODEL_REPOSITORY
        || manifest.revision != MODEL_REVISION
        || manifest.model_sha256 != MODEL_SHA256
        || manifest_digest(&manifest.files) != MODEL_SHA256
    {
        return Err("image model manifest does not match its pins".into());
    }
    Ok(manifest)
}

/// A manifest path must be relative, `/`-separated, and free of empty, `.` and `..` parts.
fn manifest_path_ok(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

pub fn verify_model_dir(model_dir: &std::path::Path) -> Result<(), String> {
    verify_files(model_dir, &embedded_manifest()?.files)
}

fn verify_files(model_dir: &std::path::Path, files: &[ManifestFile]) -> Result<(), String> {
    for file in files {
        if !manifest_path_ok(&file.path) {
            return Err("image model manifest path invalid".into());
        }
        // No component may be a symlink: the bytes hashed here must be the bytes the worker
        // loads, and a link can be re-pointed between the two.
        let mut path = model_dir.to_path_buf();
        for part in file.path.split('/') {
            path.push(part);
            if std::fs::symlink_metadata(&path)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(true)
            {
                return Err(format!("image model file missing: {}", file.path));
            }
        }
        if !std::fs::symlink_metadata(&path).is_ok_and(|m| m.is_file()) {
            return Err(format!("image model file missing: {}", file.path));
        }
        verify_file(&path, file.size, &file.sha256, &file.path)?;
    }
    Ok(())
}

fn verify_file(
    path: &std::path::Path,
    expected_bytes: u64,
    expected_sha256: &str,
    label: &str,
) -> Result<(), String> {
    if std::fs::metadata(path).map(|m| m.len()).ok() != Some(expected_bytes) {
        return Err(format!("{label} size does not match its pin"));
    }
    let mut file = std::fs::File::open(path).map_err(|_| format!("{label} unavailable"))?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    use sha2::Digest;
    let mut total = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| format!("{label} could not be hashed"))?;
        if read == 0 {
            break;
        }
        total += read as u64;
        hasher.update(&buffer[..read]);
    }
    if total != expected_bytes || hex::encode(hasher.finalize()) != expected_sha256 {
        return Err(format!("{label} sha256 does not match its pin"));
    }
    Ok(())
}

// ── Supervisor ────────────────────────────────────────────────────────────────────────────

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            // The Python worker leads its own process group.
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct PrivateWorkerTemp(std::path::PathBuf);
impl PrivateWorkerTemp {
    fn create() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "sgl-image-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).map_err(|_| "image temporary storage unavailable")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).is_err() {
                let _ = std::fs::remove_dir(&path);
                return Err("image temporary storage permissions failed".into());
            }
        }
        Ok(Self(path))
    }
}
impl Drop for PrivateWorkerTemp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn materialize_assets() -> Result<PrivateWorkerTemp, String> {
    let assets = PrivateWorkerTemp::create()?;
    for (name, bytes) in [
        (WORKER_FILE, WORKER_SCRIPT),
        (MANIFEST_FILE, MODEL_MANIFEST.as_bytes()),
    ] {
        std::fs::write(assets.0.join(name), bytes)
            .map_err(|_| "image runtime materialization failed")?;
    }
    Ok(assets)
}

fn blocked_worker_env(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("DYLD_")
        || name.starts_with("__XPC_DYLD_")
        || name.starts_with("LD_")
        || name.starts_with("PYTHON")
        || name.starts_with("MLX_")
        || name.starts_with("MFLUX_")
        || name.starts_with(FAKE_ENV)
        || matches!(name.as_str(), "BASH_ENV" | "ENV" | "GCONV_PATH")
}

/// The worker process: scrubbed environment (no loader, interpreter, MLX/mflux or test-switch
/// variables inherited), offline Hugging Face, a private TMPDIR, its own process group.
fn worker_command(config: &WorkerConfig, temporary: &std::path::Path) -> Command {
    let mut command = Command::new(&config.program);
    command
        .args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    for (name, _) in std::env::vars_os() {
        if blocked_worker_env(&name.to_string_lossy()) {
            command.env_remove(name);
        }
    }
    command
        .env("TMPDIR", temporary)
        .env("TEMP", temporary)
        .env("TMP", temporary)
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1");
    for (name, value) in &config.extra_env {
        command.env(name, value);
    }
    command
}

fn supervise(
    config: WorkerConfig,
    jobs: mpsc::Receiver<Job>,
    ready: tokio::sync::oneshot::Sender<Result<Capabilities, String>>,
    healthy: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
) {
    let temporary = match PrivateWorkerTemp::create() {
        Ok(tmp) => tmp,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut command = worker_command(&config, &temporary.0);
    let mut child = match command.spawn() {
        Ok(child) => OwnedChild(child),
        Err(_) => {
            let _ = ready.send(Err("image worker spawn failed".into()));
            return;
        }
    };
    let mut stdin = child.0.stdin.take().unwrap();
    let (writes_tx, writes_rx) = mpsc::sync_channel::<(Vec<u8>, mpsc::SyncSender<bool>)>(1);
    let writer = std::thread::spawn(move || {
        while let Ok((bytes, ack)) = writes_rx.recv() {
            let success = stdin.write_all(&bytes).and_then(|_| stdin.flush()).is_ok();
            let _ = ack.try_send(success);
            if !success {
                break;
            }
        }
    });
    let stdout = child.0.stdout.take().unwrap();
    let (frames_tx, frames_rx) = mpsc::sync_channel(2);
    let h = healthy.clone();
    let output_open = Arc::new(AtomicBool::new(true));
    let open = output_open.clone();
    let reader = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let delivered = read_frame(&mut reader, MAX_OUTPUT_FRAME)
                .is_some_and(|frame| frames_tx.try_send(frame).is_ok());
            if !delivered {
                h.store(false, Ordering::Release);
                break;
            }
        }
        open.store(false, Ordering::Release);
    });
    let startup = receive_frame(&frames_rx, config.startup_timeout, &stopped, None)
        .and_then(|frame| validate_ready(&frame, config.allow_fake));
    match startup {
        Ok(cap)
            if output_open.load(Ordering::Acquire)
                && !stopped.load(Ordering::Acquire)
                && child.0.try_wait().ok().flatten().is_none() =>
        {
            healthy.store(true, Ordering::Release);
            if ready.send(Ok(cap)).is_err() {
                stopped.store(true, Ordering::Release);
            }
        }
        _ => {
            let _ = ready.send(Err("invalid image readiness".into()));
            stopped.store(true, Ordering::Release);
        }
    }
    while output_open.load(Ordering::Acquire)
        && healthy.load(Ordering::Acquire)
        && !stopped.load(Ordering::Acquire)
    {
        let job = match jobs.recv_timeout(Duration::from_millis(25)) {
            Ok(job) => job,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // An unsolicited frame or a dead worker while idle: de-advertise.
                if frames_rx.try_recv().is_ok() || !matches!(child.0.try_wait(), Ok(None)) {
                    healthy.store(false, Ordering::Release);
                }
                continue;
            }
            Err(_) => break,
        };
        if job.reply.is_closed() {
            continue;
        }
        let response = match run_job(
            &job,
            &writes_tx,
            &frames_rx,
            &stopped,
            config.request_timeout,
        ) {
            Ok(output) => Ok(output),
            Err(WorkerFailure::InvalidInput) => Err("image_input_invalid".into()),
            Err(WorkerFailure::Fatal(reason)) => {
                if reason == TIMED_OUT || reason == INPUT_TIMED_OUT {
                    timed_out.store(true, Ordering::Release);
                }
                healthy.store(false, Ordering::Release);
                Err("image_runtime_failed".into())
            }
        };
        let _ = job.reply.send(response);
    }
    healthy.store(false, Ordering::Release);
    drop(writes_tx);
    drop(child);
    let _ = reader.join();
    let _ = writer.join();
}

/// One `\n`-terminated JSON frame of a known type, at most `max` bytes. None on EOF, an
/// oversized or unterminated frame, invalid JSON or an unknown frame type.
fn read_frame(reader: &mut impl BufRead, max: u64) -> Option<Vec<u8>> {
    let mut frame = Vec::new();
    let result = reader.take(max + 1).read_until(b'\n', &mut frame);
    if !matches!(result, Ok(n) if n > 0 && n as u64 <= max) || frame.last() != Some(&b'\n') {
        return None;
    }
    let known = serde_json::from_slice::<serde_json::Value>(&frame)
        .ok()
        .and_then(|v| {
            v.get("type")
                .and_then(|t| t.as_str())
                .map(|t| matches!(t, "ready" | "result" | "request_error"))
        })
        .unwrap_or(false);
    known.then_some(frame)
}

fn receive_frame(
    frames: &mpsc::Receiver<Vec<u8>>,
    timeout: Duration,
    stopped: &AtomicBool,
    reply: Option<&tokio::sync::oneshot::Sender<Result<ImageOutput, String>>>,
) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if stopped.load(Ordering::Acquire) || reply.is_some_and(|reply| reply.is_closed()) {
            return Err("image request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(TIMED_OUT.into());
        }
        match frames.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(frame) => return Ok(frame),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => return Err("image worker closed output".into()),
        }
    }
}

pub(crate) fn base_capabilities(runtime_revision: String) -> Capabilities {
    Capabilities {
        kind: "image".into(),
        engine: ENGINE_LABEL.into(),
        image_ready: true,
        image_protocol: PROTOCOL.into(),
        image_models: vec![MODEL_ID.into()],
        image_runtime: RUNTIME.into(),
        runtime_revision,
        model_revision: MODEL_REVISION.into(),
        model_sha256: MODEL_SHA256.into(),
        model_repository: MODEL_REPOSITORY.into(),
        model_license: MODEL_LICENSE.into(),
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        max_prompt_bytes: MAX_PROMPT_BYTES,
        sizes: SIZES.iter().map(|&(w, h)| [w, h]).collect(),
        steps: STEPS,
        output_format: OUTPUT_FORMAT.into(),
        free_slots: 1,
        key_binding: None,
    }
}

fn validate_ready(frame: &[u8], allow_fake: bool) -> Result<Capabilities, String> {
    let ready: Ready = serde_json::from_slice(frame).map_err(|_| "invalid image readiness")?;
    let sizes_match = ready.sizes.len() == SIZES.len()
        && ready
            .sizes
            .iter()
            .zip(SIZES.iter())
            .all(|(a, &(w, h))| *a == [w, h]);
    if ready.frame_type != "ready"
        || ready.protocol != PROTOCOL
        || ready.runtime != RUNTIME
        || ready.runtime_revision != RUNTIME_REVISION
        || !pins_match(&ready.model_id, &ready.model_revision, &ready.model_sha256)
        || ready.steps != STEPS
        || !sizes_match
        || ready.output_format != OUTPUT_FORMAT
        || ready.max_prompt_bytes != MAX_PROMPT_BYTES
        || !ready.smoke_seconds.is_finite()
        || ready.smoke_seconds < 0.0
        || (ready.fake && !allow_fake)
    {
        return Err("image worker identity or smoke mismatch".into());
    }
    Ok(base_capabilities(ready.runtime_revision))
}

fn run_job(
    job: &Job,
    writes: &mpsc::SyncSender<(Vec<u8>, mpsc::SyncSender<bool>)>,
    frames: &mpsc::Receiver<Vec<u8>>,
    stopped: &AtomicBool,
    timeout: Duration,
) -> Result<ImageOutput, WorkerFailure> {
    let bytes = request_frame(job.id, &job.request, job.seed)?;
    let deadline = Instant::now() + timeout;
    let (ack_tx, ack_rx) = mpsc::sync_channel(1);
    writes
        .try_send((bytes, ack_tx))
        .map_err(|_| "image worker input unavailable")?;
    loop {
        if stopped.load(Ordering::Acquire) || job.reply.is_closed() {
            return Err("image request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(INPUT_TIMED_OUT.into());
        }
        match ack_rx.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(true) => break,
            Ok(false) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("image worker input closed".into())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
        }
    }
    let frame = receive_frame(
        frames,
        deadline.saturating_duration_since(Instant::now()),
        stopped,
        Some(&job.reply),
    )?;
    parse_reply(&frame, job.id, &job.request, job.seed)
}

fn request_frame(id: u64, request: &ImageRequest, seed: u32) -> Result<Vec<u8>, WorkerFailure> {
    use base64::Engine;
    let prompt_sha256 = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(request.prompt.as_bytes()))
    };
    let frame = serde_json::json!({
        "type": "generate",
        "request_id": id,
        "protocol": PROTOCOL,
        "prompt_b64": base64::engine::general_purpose::STANDARD.encode(request.prompt.as_bytes()),
        "prompt_sha256": prompt_sha256,
        "width": request.width,
        "height": request.height,
        "seed": seed,
        "steps": STEPS,
    });
    let mut bytes = serde_json::to_vec(&frame).map_err(|_| "invalid image request")?;
    if bytes.len() > MAX_WORKER_INPUT_FRAME {
        return Err("image request exceeds frame limit".into());
    }
    bytes.push(b'\n');
    Ok(bytes)
}

fn parse_reply(
    frame: &[u8],
    id: u64,
    request: &ImageRequest,
    seed: u32,
) -> Result<ImageOutput, WorkerFailure> {
    let value: serde_json::Value =
        serde_json::from_slice(frame).map_err(|_| "invalid image worker frame")?;
    if value.get("type").and_then(|v| v.as_str()) == Some("request_error") {
        let error: RequestError =
            serde_json::from_value(value).map_err(|_| "invalid image request-error frame")?;
        if error.frame_type != "request_error"
            || error.request_id != id
            || error.code != "invalid_input"
        {
            return Err("image request-error mismatch".into());
        }
        return Err(WorkerFailure::InvalidInput);
    }
    let result: Response =
        serde_json::from_value(value).map_err(|_| "invalid image worker output")?;
    if result.frame_type != "result" || result.request_id != id {
        return Err("image worker response mismatch".into());
    }
    validate_output(result, request.width, request.height, seed).map_err(WorkerFailure::Fatal)
}

/// Width and height from a PNG's IHDR chunk (the first chunk, by the PNG spec).
pub fn png_dimensions(png: &[u8]) -> Option<(u32, u32)> {
    if png.len() < 33 || &png[..8] != b"\x89PNG\r\n\x1a\n" || &png[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(png[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(png[20..24].try_into().ok()?);
    Some((width, height))
}

fn validate_output(
    result: Response,
    width: u32,
    height: u32,
    seed: u32,
) -> Result<ImageOutput, String> {
    if result.width != width || result.height != height {
        return Err("image size contradicts the request".into());
    }
    if result.bytes == 0 || result.bytes > MAX_PNG_BYTES {
        return Err("image size out of bounds".into());
    }
    if result.png_b64.len() > MAX_PNG_BYTES.div_ceil(3) * 4 {
        return Err("image payload out of bounds".into());
    }
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD
        .decode(&result.png_b64)
        .map_err(|_| "image base64 invalid")?;
    let digest = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(&png))
    };
    if png.len() != result.bytes || digest != result.sha256 {
        return Err("image contradicts its digest".into());
    }
    if png_dimensions(&png) != Some((width, height)) {
        return Err("image is not a PNG of the requested size".into());
    }
    Ok(ImageOutput {
        png,
        width,
        height,
        seed,
        sha256: digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    const RID: &str = "3f1c2a54-9b7e-4d21-8c3a-0f6e5d4c3b2a";

    pub(crate) fn canonical(
        prompt: &str,
        width: u32,
        height: u32,
        seed: Option<u64>,
    ) -> serde_json::Value {
        serde_json::json!({
            "protocol": PROTOCOL, "request_id": RID, "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "width": width, "height": height, "seed": seed,
            "prompt": {"encoding": "base64", "bytes": prompt.len(),
                       "data": base64::engine::general_purpose::STANDARD.encode(prompt)},
        })
    }

    fn reservation(prompt_bytes: u64, seed: Option<u64>) -> serde_json::Value {
        serde_json::json!({"enc": {}, "image_reservation": {
            "protocol": PROTOCOL, "request_id": RID, "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "width": 1024, "height": 1024, "seed": seed, "prompt_bytes": prompt_bytes}})
    }

    #[test]
    fn embedded_manifest_matches_pins_and_spec_formula() {
        let manifest = embedded_manifest().expect("manifest matches its pins");
        assert_eq!(manifest.files.len(), 13);
        assert!(manifest
            .files
            .iter()
            .any(|f| f.path == "transformer/0.safetensors"));
        assert!(manifest.files.iter().all(|f| manifest_path_ok(&f.path)));
        // The digest is order-independent (sorted by path) and changes with any field.
        let mut files = serde_json::from_str::<Manifest>(MODEL_MANIFEST)
            .unwrap()
            .files;
        files.reverse();
        assert_eq!(manifest_digest(&files), MODEL_SHA256);
        files[0].size += 1;
        assert_ne!(manifest_digest(&files), MODEL_SHA256);
        assert!(is_image_model("FLUX2-klein-4b-q4") && !is_image_model("kokoro-82m"));
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sgl-image-test-{tag}-{}-{:x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(path: &str, bytes: &[u8]) -> ManifestFile {
        use sha2::{Digest, Sha256};
        ManifestFile {
            path: path.into(),
            size: bytes.len() as u64,
            sha256: hex::encode(Sha256::digest(bytes)),
        }
    }

    #[test]
    fn model_dir_verification_good_bad_symlink_and_traversal() {
        let dir = tmpdir("verify");
        std::fs::create_dir_all(dir.join("vae")).unwrap();
        std::fs::write(dir.join("README.md"), b"readme").unwrap();
        std::fs::write(dir.join("vae/0.safetensors"), b"weights").unwrap();
        let good = vec![
            entry("README.md", b"readme"),
            entry("vae/0.safetensors", b"weights"),
        ];
        verify_files(&dir, &good).expect("good dir verifies");

        // Wrong content, same size.
        let bad = vec![entry("vae/0.safetensors", b"WEIGHTS")];
        assert!(verify_files(&dir, &bad).unwrap_err().contains("sha256"));
        // Wrong size.
        let short = vec![entry("vae/0.safetensors", b"weight")];
        assert!(verify_files(&dir, &short).unwrap_err().contains("size"));
        // Missing file.
        let missing = vec![entry("vae/1.safetensors", b"weights")];
        assert!(verify_files(&dir, &missing).is_err());
        // Traversal and absolute paths never touch the filesystem.
        for path in [
            "../README.md",
            "vae/../README.md",
            "/etc/passwd",
            "./README.md",
            "vae//0.safetensors",
            "vae\\0.safetensors",
            "",
        ] {
            let evil = vec![entry(path, b"readme")];
            assert_eq!(
                verify_files(&dir, &evil).unwrap_err(),
                "image model manifest path invalid",
                "{path}"
            );
        }
        #[cfg(unix)]
        {
            // A symlinked file, and a symlinked parent directory, are both refused even when
            // the bytes behind them are correct.
            let other = tmpdir("verify-other");
            std::fs::write(other.join("real"), b"weights").unwrap();
            std::fs::create_dir_all(dir.join("linked")).unwrap();
            std::os::unix::fs::symlink(other.join("real"), dir.join("linked/0.safetensors"))
                .unwrap();
            let link = vec![entry("linked/0.safetensors", b"weights")];
            assert!(verify_files(&dir, &link).is_err());
            std::fs::create_dir_all(other.join("sub")).unwrap();
            std::fs::write(other.join("sub/0.safetensors"), b"weights").unwrap();
            std::os::unix::fs::symlink(other.join("sub"), dir.join("subdir")).unwrap();
            let parent = vec![entry("subdir/0.safetensors", b"weights")];
            assert!(verify_files(&dir, &parent).is_err());
            let _ = std::fs::remove_dir_all(&other);
        }
        // The real manifest against a directory without the model fails closed.
        assert!(verify_model_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn request_parse_enforces_limits_and_binding() {
        let request = ImageRequest::parse(&canonical("a red fox", 1024, 1024, Some(42))).unwrap();
        assert_eq!(request.seed, Some(42));
        let binding = ReservationBinding::parse_dispatch(&reservation(9, Some(42))).unwrap();
        assert!(binding.matches(&request));
        assert!(
            !ReservationBinding::parse_dispatch(&reservation(8, Some(42)))
                .unwrap()
                .matches(&request)
        );
        assert!(!ReservationBinding::parse_dispatch(&reservation(9, None))
            .unwrap()
            .matches(&request));
        assert!(
            !ReservationBinding::parse_dispatch(&reservation(9, Some(43)))
                .unwrap()
                .matches(&request)
        );

        // Prompt byte limits count UTF-8 bytes, not characters.
        let max = "a".repeat(MAX_PROMPT_BYTES);
        assert!(ImageRequest::parse(&canonical(&max, 1024, 1024, None)).is_ok());
        let over = "a".repeat(MAX_PROMPT_BYTES + 1);
        assert!(ImageRequest::parse(&canonical(&over, 1024, 1024, None)).is_err());
        let accented = ImageRequest::parse(&canonical("café", 1024, 1024, None)).unwrap();
        assert_eq!(accented.prompt.len(), 5);
        let mut lying = canonical("café", 1024, 1024, None);
        lying["prompt"]["bytes"] = serde_json::json!(4);
        assert!(
            ImageRequest::parse(&lying).is_err(),
            "declared bytes = decoded UTF-8 bytes"
        );
        assert!(ImageRequest::parse(&canonical("   ", 1024, 1024, None)).is_err());
        assert!(ImageRequest::parse(&canonical("a\0b", 1024, 1024, None)).is_err());
        // Only base64: the raw-string form (and anything else) is refused.
        let mut utf8 = canonical("hi", 1024, 1024, None);
        utf8["prompt"] = serde_json::json!({"encoding": "utf8", "bytes": 2, "data": "hi"});
        assert!(ImageRequest::parse(&utf8).is_err());
        let mut not_b64 = canonical("hi", 1024, 1024, None);
        not_b64["prompt"]["data"] = serde_json::json!("h!i=");
        assert!(ImageRequest::parse(&not_b64).is_err());
        let mut not_utf8 = canonical("hi", 1024, 1024, None);
        not_utf8["prompt"] = serde_json::json!({"encoding": "base64", "bytes": 2,
            "data": base64::engine::general_purpose::STANDARD.encode([0xffu8, 0xfe])});
        assert!(ImageRequest::parse(&not_utf8).is_err());

        // Size allow-list only.
        for (w, h) in SIZES {
            assert!(
                ImageRequest::parse(&canonical("x", w, h, None)).is_ok(),
                "{w}x{h}"
            );
        }
        for (w, h) in [(512, 512), (1024, 768), (1280, 1280), (0, 0), (720, 720)] {
            assert!(
                ImageRequest::parse(&canonical("x", w, h, None)).is_err(),
                "{w}x{h}"
            );
        }

        // Seed range 0..=u32::MAX, nullable but never absent, never negative.
        assert!(ImageRequest::parse(&canonical("x", 1024, 1024, Some(0))).is_ok());
        assert!(ImageRequest::parse(&canonical("x", 1024, 1024, Some(MAX_SEED))).is_ok());
        assert!(ImageRequest::parse(&canonical("x", 1024, 1024, Some(MAX_SEED + 1))).is_err());
        let mut negative = canonical("x", 1024, 1024, None);
        negative["seed"] = serde_json::json!(-1);
        assert!(ImageRequest::parse(&negative).is_err());
        let mut fractional = canonical("x", 1024, 1024, None);
        fractional["seed"] = serde_json::json!(1.5);
        assert!(ImageRequest::parse(&fractional).is_err());
        let mut absent = canonical("x", 1024, 1024, None);
        absent.as_object_mut().unwrap().remove("seed");
        assert!(ImageRequest::parse(&absent).is_err());

        // Wrong model pins, unknown fields, other protocols, non-canonical ids.
        let mut wrong_model = canonical("x", 1024, 1024, None);
        wrong_model["model_sha256"] = serde_json::json!("0".repeat(64));
        assert!(ImageRequest::parse(&wrong_model).is_err());
        let mut other = canonical("x", 1024, 1024, None);
        other["model"] = serde_json::json!("flux2-klein-4b-q8");
        assert!(ImageRequest::parse(&other).is_err());
        let mut extra = canonical("x", 1024, 1024, None);
        extra["steps"] = serde_json::json!(8);
        assert!(
            ImageRequest::parse(&extra).is_err(),
            "unknown fields rejected"
        );
        let mut protocol = canonical("x", 1024, 1024, None);
        protocol["protocol"] = serde_json::json!("speech-v1");
        assert!(ImageRequest::parse(&protocol).is_err());
        for id in [
            RID.to_ascii_uppercase(),
            "3f1c2a54-9b7e-1d21-8c3a-0f6e5d4c3b2a".into(), // v1
            "3f1c2a54-9b7e-4d21-cc3a-0f6e5d4c3b2a".into(), // bad variant
            "3f1c2a549b7e4d218c3a0f6e5d4c3b2a".into(),
        ] {
            let mut bad = canonical("x", 1024, 1024, None);
            bad["request_id"] = serde_json::json!(id);
            assert!(ImageRequest::parse(&bad).is_err(), "{id}");
        }
    }

    #[test]
    fn dispatch_requires_exact_shape() {
        assert!(ReservationBinding::parse_dispatch(&reservation(12, None)).is_ok());
        let mut extra = reservation(12, None);
        extra["speech_reservation"] = serde_json::json!({});
        assert!(ReservationBinding::parse_dispatch(&extra).is_err());
        let mut no_enc = reservation(12, None);
        no_enc["enc"] = serde_json::json!("x");
        assert!(ReservationBinding::parse_dispatch(&no_enc).is_err());
        let mut no_seed = reservation(12, None);
        no_seed["image_reservation"]
            .as_object_mut()
            .unwrap()
            .remove("seed");
        assert!(ReservationBinding::parse_dispatch(&no_seed).is_err());
        let mut steps = reservation(12, None);
        steps["image_reservation"]["steps"] = serde_json::json!(4);
        assert!(ReservationBinding::parse_dispatch(&steps).is_err());
        for bytes in [0, MAX_PROMPT_BYTES as u64 + 1] {
            assert!(ReservationBinding::parse_dispatch(&reservation(bytes, None)).is_err());
        }
        assert!(ReservationBinding::parse_dispatch(&reservation(1, Some(MAX_SEED + 1))).is_err());
        let mut size = reservation(12, None);
        size["image_reservation"]["width"] = serde_json::json!(512);
        assert!(ReservationBinding::parse_dispatch(&size).is_err());
        let mut upper = reservation(12, None);
        upper["image_reservation"]["request_id"] = serde_json::json!(RID.to_ascii_uppercase());
        assert!(ReservationBinding::parse_dispatch(&upper).is_err());
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut out = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        out.extend_from_slice(&width.to_be_bytes());
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(&[8, 2, 0, 0, 0, 0, 0, 0, 0]);
        out.extend_from_slice(b"rest-of-file");
        out
    }

    fn sha(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(bytes))
    }

    #[test]
    fn result_plaintext_is_header_line_then_png() {
        let request = ImageRequest::parse(&canonical("a red fox", 1152, 768, None)).unwrap();
        let data = png(1152, 768);
        let output = ImageOutput {
            sha256: sha(&data),
            png: data.clone(),
            width: 1152,
            height: 768,
            seed: 99,
        };
        let bytes = result_plaintext("job-1", &request, &output);
        let split = bytes.iter().position(|b| *b == b'\n').unwrap();
        assert_eq!(&bytes[split + 1..], data.as_slice());
        let text = std::str::from_utf8(&bytes[..split]).unwrap();
        assert!(text.starts_with(r#"{"object":"image","protocol":"image-v1","request_id":"#));
        let header: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(header["job_id"], "job-1");
        assert_eq!(header["model"], MODEL_ID);
        assert_eq!(header["model_sha256"], MODEL_SHA256);
        assert_eq!(header["seed"], 99, "the seed actually used is reported");
        assert_eq!(header["steps"], STEPS);
        assert_eq!(header["prompt_bytes"], 9);
        assert_eq!(header["image"]["bytes"], data.len());
        assert_eq!(header["image"]["mime_type"], "image/png");
        assert_eq!(usage(&output)["pixels"], 1152 * 768);
    }

    #[test]
    fn output_validation_rejects_lies() {
        use base64::Engine;
        let data = png(1024, 1024);
        let make = |w: u32, h: u32, bytes: usize, digest: &str, payload: &[u8]| Response {
            frame_type: "result".into(),
            request_id: 1,
            width: w,
            height: h,
            bytes,
            sha256: digest.into(),
            png_b64: base64::engine::general_purpose::STANDARD.encode(payload),
        };
        let ok = make(1024, 1024, data.len(), &sha(&data), &data);
        assert_eq!(validate_output(ok, 1024, 1024, 5).unwrap().seed, 5);
        let bad_sha = make(1024, 1024, data.len(), &"0".repeat(64), &data);
        assert!(validate_output(bad_sha, 1024, 1024, 5).is_err());
        let bad_len = make(1024, 1024, data.len() + 1, &sha(&data), &data);
        assert!(validate_output(bad_len, 1024, 1024, 5).is_err());
        let wrong_claim = make(768, 1152, data.len(), &sha(&data), &data);
        assert!(validate_output(wrong_claim, 1024, 1024, 5).is_err());
        // The worker claims the right size but the PNG itself is another size.
        let other = png(768, 1152);
        let lie = make(1024, 1024, other.len(), &sha(&other), &other);
        assert!(validate_output(lie, 1024, 1024, 5).is_err());
        let jpeg = b"\xff\xd8\xff\xe0 not a png at all, really not".to_vec();
        let not_png = make(1024, 1024, jpeg.len(), &sha(&jpeg), &jpeg);
        assert!(validate_output(not_png, 1024, 1024, 5).is_err());
        let huge = make(1024, 1024, MAX_PNG_BYTES + 1, &sha(&data), &data);
        assert!(validate_output(huge, 1024, 1024, 5).is_err());
    }

    #[test]
    fn worker_frames_are_bounded_and_typed() {
        let mut ok = std::io::Cursor::new(b"{\"type\":\"result\"}\n".to_vec());
        assert!(read_frame(&mut ok, 64).is_some());
        let mut unknown = std::io::Cursor::new(b"{\"type\":\"log\"}\n".to_vec());
        assert!(read_frame(&mut unknown, 64).is_none());
        let mut garbage = std::io::Cursor::new(b"progress 50%\n".to_vec());
        assert!(read_frame(&mut garbage, 64).is_none());
        let mut unterminated = std::io::Cursor::new(b"{\"type\":\"result\"}".to_vec());
        assert!(read_frame(&mut unterminated, 64).is_none());
        let long = format!("{{\"type\":\"result\",\"x\":\"{}\"}}\n", "a".repeat(100));
        let mut oversized = std::io::Cursor::new(long.into_bytes());
        assert!(read_frame(&mut oversized, 64).is_none());

        let request = ImageRequest::parse(&canonical(&"é".repeat(1000), 1280, 720, None)).unwrap();
        let frame = request_frame(7, &request, 123).ok().unwrap();
        assert_eq!(frame.last(), Some(&b'\n'));
        let value: serde_json::Value = serde_json::from_slice(&frame).unwrap();
        assert_eq!(value["type"], "generate");
        assert_eq!(value["seed"], 123);
        assert_eq!(value["steps"], 4);
        assert_eq!(value["width"], 1280);
        assert_eq!(value["prompt_sha256"], sha("é".repeat(1000).as_bytes()));
    }

    #[test]
    fn capability_json_shape_is_exact() {
        let cap = base_capabilities(RUNTIME_REVISION.into());
        let bound = cap.clone().bind_heartbeat(5, "pk", Some("sig"), Some(2));
        let wire = serde_json::to_value(&bound).unwrap();
        let keys: Vec<&str> = wire
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let mut expected = vec![
            "kind",
            "engine",
            "image_ready",
            "image_protocol",
            "image_models",
            "image_runtime",
            "runtime_revision",
            "model_revision",
            "model_sha256",
            "model_repository",
            "model_license",
            "os",
            "architecture",
            "max_prompt_bytes",
            "sizes",
            "steps",
            "output_format",
            "free_slots",
            "key_binding",
        ];
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        assert_eq!(wire["kind"], "image");
        assert_eq!(wire["engine"], "image-worker");
        assert_eq!(wire["image_ready"], true);
        assert_eq!(wire["image_protocol"], "image-v1");
        assert_eq!(
            wire["image_models"],
            serde_json::json!(["flux2-klein-4b-q4"])
        );
        assert_eq!(wire["image_runtime"], "mflux");
        assert_eq!(wire["runtime_revision"], "media-2:mflux-0.24.0");
        assert_eq!(wire["model_revision"], MODEL_REVISION);
        assert_eq!(wire["model_sha256"], MODEL_SHA256);
        assert_eq!(wire["model_repository"], MODEL_REPOSITORY);
        assert_eq!(wire["model_license"], "Apache-2.0");
        assert_eq!(wire["max_prompt_bytes"], 2000);
        assert_eq!(
            wire["sizes"],
            serde_json::json!([
                [1024, 1024],
                [1152, 768],
                [768, 1152],
                [1280, 720],
                [720, 1280]
            ])
        );
        assert_eq!(wire["steps"], 4);
        assert_eq!(wire["output_format"], "png");
        assert_eq!(wire["free_slots"], 1, "concurrency 1");
        assert_eq!(wire["key_binding"]["key_version"], 2);
        let unbound = cap.bind_heartbeat(1, "pk", None, None);
        assert!(
            !unbound.image_ready && unbound.free_slots == 0,
            "no key signature, no routing"
        );
    }

    #[test]
    fn retry_plan_backs_off_and_stops_before_the_deadline() {
        assert_eq!(retry_backoff(1), Duration::from_secs(1));
        assert_eq!(retry_backoff(2), Duration::from_secs(2));
        assert_eq!(retry_backoff(3), Duration::from_secs(4));
        assert_eq!(retry_backoff(5), Duration::from_secs(16));
        assert_eq!(
            retry_backoff(40),
            Duration::from_secs(16),
            "capped, no overflow"
        );
        // First attempt is immediate with the full per-attempt timeout.
        assert_eq!(
            plan_completion_attempt(Duration::from_secs(30), 0),
            Some((Duration::ZERO, COMPLETE_HTTP_TIMEOUT))
        );
        // Near the deadline the attempt timeout shrinks to what is left.
        let late = NODE_DEADLINE - Duration::from_secs(50);
        let (wait, timeout) = plan_completion_attempt(late, 2).unwrap();
        assert_eq!(wait, Duration::from_secs(2));
        assert_eq!(timeout, Duration::from_secs(48));
        // No attempt starts inside the upload margin, or after the deadline.
        let edge = NODE_DEADLINE - UPLOAD_MARGIN;
        assert!(plan_completion_attempt(edge, 0).is_some());
        assert!(plan_completion_attempt(edge + Duration::from_millis(1), 0).is_none());
        assert!(plan_completion_attempt(edge - Duration::from_millis(500), 1).is_none());
        assert!(plan_completion_attempt(NODE_DEADLINE * 2, 0).is_none());
        // Simulated worst case: every attempt times out. The loop ends before the deadline.
        let mut elapsed = Duration::from_secs(200);
        let mut attempts = 0;
        while let Some((wait, timeout)) = plan_completion_attempt(elapsed, attempts) {
            elapsed += wait + timeout;
            attempts += 1;
            assert!(elapsed <= NODE_DEADLINE);
        }
        assert!(attempts >= 3, "{attempts}");
        // Statuses.
        for s in [500, 502, 503, 504, 408, 429] {
            assert!(completion_status_retryable(s), "{s}");
        }
        for s in [400, 401, 403, 404, 409, 413, 422] {
            assert!(!completion_status_retryable(s), "{s}");
        }
    }

    #[test]
    fn replay_claims_only_stick_after_success() {
        let mut seen = SeenRequestIds::new();
        assert!(seen.begin(RID));
        assert!(
            !seen.begin(RID),
            "in flight: a concurrent duplicate is refused"
        );
        seen.finish(RID, false);
        assert!(
            seen.begin(RID),
            "a failed generation leaves the id retryable"
        );
        seen.finish(RID, true);
        assert!(!seen.begin(RID), "a served id is a replay");
    }

    #[test]
    fn restart_budget_is_per_rolling_hour() {
        let mut history = std::collections::VecDeque::new();
        let t0 = Instant::now();
        for i in 0..MAX_RESTARTS_PER_WINDOW as u64 {
            assert!(restart_allowed(&mut history, t0 + Duration::from_secs(i)));
        }
        assert!(!restart_allowed(&mut history, t0 + Duration::from_secs(60)));
        assert!(
            restart_allowed(&mut history, t0 + RESTART_WINDOW + Duration::from_secs(1)),
            "the budget refills after an hour"
        );
    }

    #[tokio::test]
    async fn timeouts_relaunch_outside_the_budget_and_keep_the_id_retryable() {
        // hang_always: every request times out. Four timeouts in a row (more than the budget
        // of 3) must all relaunch, so slow jobs never strand the node de-advertised.
        let (mut config, _assets) = fake_worker("hang_always");
        config.request_timeout = Duration::from_millis(1500);
        let engine = ImageEngine::start(config).await.unwrap();
        let request = ImageRequest::parse(&canonical("a red fox", 1024, 1024, Some(7))).unwrap();
        for round in 0..4 {
            assert_eq!(
                engine.generate(request.clone()).await.unwrap_err(),
                "image_runtime_failed",
                "round {round}: a timeout is a runtime failure, never a replay"
            );
            assert!(!engine.is_healthy());
            engine
                .restart()
                .await
                .expect("timeout relaunch is outside the budget");
            assert!(engine.is_healthy());
        }
        // Non-timeout faults still spend the rolling budget.
        for _ in 0..MAX_RESTARTS_PER_WINDOW {
            engine.restart().await.unwrap();
        }
        assert!(engine.restart().await.is_err());
    }

    #[tokio::test]
    async fn retry_after_a_timeout_is_served_not_called_a_replay() {
        let (mut config, assets) = fake_worker("hang_first");
        config.request_timeout = Duration::from_millis(1500);
        let engine = ImageEngine::start(config).await.unwrap();
        let request = ImageRequest::parse(&canonical("a red fox", 1024, 1024, Some(7))).unwrap();
        assert_eq!(
            engine.generate(request.clone()).await.unwrap_err(),
            "image_runtime_failed"
        );
        engine.restart().await.unwrap();
        assert!(assets.0.join(".fake-hung").exists());
        let out = engine.generate(request.clone()).await.unwrap();
        assert_eq!(png_dimensions(&out.png), Some((1024, 1024)));
        assert_eq!(
            engine.generate(request).await.unwrap_err(),
            "image_input_invalid",
            "once served, the same id is a replay"
        );
    }

    #[test]
    fn deadlines_are_ordered() {
        assert!(NODE_DEADLINE.as_secs() < ORCHESTRATOR_JOB_TIMEOUT.as_secs() - 60);
        assert!(REQUEST_TIMEOUT < NODE_DEADLINE);
        assert!(WEDGE_MS > NODE_DEADLINE.as_millis() as u64);
        assert_eq!(STARTUP_TIMEOUT, Duration::from_secs(420));
        assert_eq!(REQUEST_TIMEOUT, Duration::from_secs(420));
        assert_eq!(NODE_DEADLINE, Duration::from_secs(480));
        assert_eq!(ORCHESTRATOR_JOB_TIMEOUT, Duration::from_secs(600));
    }

    #[test]
    fn worker_env_is_scrubbed() {
        for name in [
            "DYLD_INSERT_LIBRARIES",
            "PYTHONPATH",
            "LD_PRELOAD",
            "MLX_METAL_PATH",
            "MFLUX_CACHE_DIR",
            FAKE_ENV,
        ] {
            assert!(blocked_worker_env(name), "{name}");
        }
        assert!(!blocked_worker_env("HOME"));
    }

    /// The real `image_worker.py`, in FAKE mode (no mflux, no GPU).
    pub(crate) fn fake_worker(mode: &str) -> (WorkerConfig, PrivateWorkerTemp) {
        let assets = materialize_assets().unwrap();
        let config = WorkerConfig {
            program: "python3".into(),
            args: vec![
                "-I".into(),
                "-u".into(),
                assets.0.join(WORKER_FILE).to_str().unwrap().into(),
                "--model-dir".into(),
                assets.0.to_str().unwrap().into(),
                "--manifest".into(),
                assets.0.join(MANIFEST_FILE).to_str().unwrap().into(),
            ],
            startup_timeout: Duration::from_secs(20),
            request_timeout: Duration::from_secs(20),
            extra_env: vec![(FAKE_ENV.into(), mode.into())],
            allow_fake: true,
        };
        (config, assets)
    }

    #[tokio::test]
    async fn fake_worker_generates_rejects_and_fails_closed() {
        let request = ImageRequest::parse(&canonical("a red fox", 1024, 1024, Some(7))).unwrap();

        let (config, _assets) = fake_worker("ok");
        let engine = ImageEngine::start(config).await.unwrap();
        assert!(engine.capabilities().is_some());
        let models = vec![MODEL_ID.to_string()];
        assert_eq!(engine.readiness_snapshot(&models).available_models, models);
        let out = engine.generate(request.clone()).await.unwrap();
        assert_eq!(png_dimensions(&out.png), Some((1024, 1024)));
        assert_eq!(out.seed, 7);
        assert!(out.png.len() < 100_000, "fake PNG stays tiny");
        assert_eq!(
            engine.generate(request.clone()).await.unwrap_err(),
            "image_input_invalid",
            "same request id again is a replay"
        );
        // A null seed is filled in by the node and reported.
        let mut unseeded = ImageRequest::parse(&canonical("a fox", 720, 1280, None)).unwrap();
        unseeded.request_id = "4f1c2a54-9b7e-4d21-8c3a-0f6e5d4c3b2a".into();
        let out = engine.generate(unseeded).await.unwrap();
        assert_eq!(png_dimensions(&out.png), Some((720, 1280)));
        engine.stop();
        assert!(engine
            .readiness_snapshot(&models)
            .available_models
            .is_empty());

        let (config, _assets) = fake_worker("invalid");
        let engine = ImageEngine::start(config).await.unwrap();
        assert_eq!(
            engine.generate(request.clone()).await.unwrap_err(),
            "image_input_invalid"
        );
        assert!(engine.is_healthy(), "client mistakes keep the node ready");
        engine.stop();

        for mode in ["lie", "crash"] {
            let (config, _assets) = fake_worker(mode);
            let engine = ImageEngine::start(config).await.unwrap();
            assert_eq!(
                engine.generate(request.clone()).await.unwrap_err(),
                "image_runtime_failed",
                "{mode}"
            );
            assert!(
                !engine.is_healthy(),
                "{mode}: a broken worker is de-advertised"
            );
            assert!(engine.capabilities().is_none());
            assert!(engine
                .readiness_snapshot(&models)
                .available_models
                .is_empty());
        }

        let (mut config, _assets) = fake_worker("hang");
        config.request_timeout = Duration::from_secs(1);
        let engine = ImageEngine::start(config).await.unwrap();
        assert_eq!(
            engine.generate(request).await.unwrap_err(),
            "image_runtime_failed",
            "a hung generation times out"
        );
        assert!(!engine.is_healthy());
    }

    #[tokio::test]
    async fn fake_worker_is_refused_outside_tests_and_bad_readiness_never_starts() {
        let (mut config, _assets) = fake_worker("ok");
        config.allow_fake = false;
        assert!(ImageEngine::start(config).await.is_err());
        // Without the FAKE switch the worker verifies the real model and fails closed here.
        let (mut config, _assets) = fake_worker("ok");
        config.extra_env.clear();
        assert!(ImageEngine::start(config).await.is_err());
        let engine = ImageEngine::start(WorkerConfig {
            program: "python3".into(),
            args: vec![
                "-I".into(),
                "-c".into(),
                "import json; print(json.dumps({'type':'ready'}), flush=True)".into(),
            ],
            startup_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(10),
            extra_env: vec![],
            allow_fake: true,
        })
        .await;
        assert!(engine.is_err());
    }

    #[test]
    fn fake_env_never_reaches_a_production_worker() {
        // The supervisor scrubs the switch from the inherited environment; only extra_env (never
        // set by production()) can turn it on.
        std::env::set_var(FAKE_ENV, "ok");
        let (mut config, assets) = fake_worker("ok");
        config.extra_env.clear();
        let command = worker_command(&config, &assets.0);
        std::env::remove_var(FAKE_ENV);
        let envs: Vec<_> = command.get_envs().collect();
        assert!(envs
            .iter()
            .any(|(k, v)| *k == std::ffi::OsStr::new(FAKE_ENV) && v.is_none()));
        assert!(envs
            .iter()
            .any(|(k, v)| *k == std::ffi::OsStr::new("HF_HUB_OFFLINE") && v.is_some()));
        let (config, assets) = fake_worker("ok");
        let command = worker_command(&config, &assets.0);
        assert!(command
            .get_envs()
            .any(|(k, v)| k == std::ffi::OsStr::new(FAKE_ENV)
                && v == Some(std::ffi::OsStr::new("ok"))));
    }

    /// Real model on this Mac. Run with:
    /// SGL_IMAGE_CANARY_PYTHON=... SGL_IMAGE_CANARY_MODEL_DIR=... [SGL_IMAGE_CANARY_OUT=x.png]
    /// cargo test --lib image::tests::real_canary -- --ignored
    #[tokio::test]
    #[ignore]
    async fn real_canary() {
        let python = std::env::var("SGL_IMAGE_CANARY_PYTHON").expect("SGL_IMAGE_CANARY_PYTHON");
        let model =
            std::env::var("SGL_IMAGE_CANARY_MODEL_DIR").expect("SGL_IMAGE_CANARY_MODEL_DIR");
        let started = Instant::now();
        let engine =
            ImageEngine::production(std::path::Path::new(&model), std::path::Path::new(&python))
                .await
                .expect("production worker starts");
        eprintln!("startup {:?}", started.elapsed());
        let request = ImageRequest::parse(&canonical(
            "a lighthouse on a cliff at sunset, photograph",
            1024,
            1024,
            Some(1234),
        ))
        .unwrap();
        let started = Instant::now();
        let out = engine.generate(request.clone()).await.unwrap();
        eprintln!("generate {:?} ({} bytes)", started.elapsed(), out.png.len());
        assert_eq!(png_dimensions(&out.png), Some((1024, 1024)));
        assert!(out.png.len() > 50_000 && out.png.len() <= MAX_PNG_BYTES);
        if let Ok(path) = std::env::var("SGL_IMAGE_CANARY_OUT") {
            std::fs::write(path, &out.png).unwrap();
        }
        engine.stop();
    }
}
