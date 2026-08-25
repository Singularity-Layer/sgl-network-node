use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Unix-ms now (for the inference-progress watchdog).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Empty-completion self-heal state (the one zombie mode the /health + wedge watchdogs miss:
/// a llama-server that stays up and answers /health but returns a FAST 0-token reply, so it
/// keeps advertising and serving empties). The two chat job-outcome sites advance `consecutive`
/// on an empty reply to a request that SHOULD have produced output; any real (or legitimately
/// empty/exempt) reply clears it. The heartbeat loop reads `consecutive`, and past a threshold
/// CANARY-CONFIRMS with its own probe before restarting — so crafted empties from one client
/// can't be weaponized into a node kill switch. `restarts`/`last_restart_ms` cap + cooldown the
/// self-heal independently of the crash/wedge restart budgets (which reset on /health OK and so
/// give this mode zero protection). All engine-agnostic: recorded at the node.rs layer, so it
/// covers the in-process (Mac) engine and the subprocess (Linux/Windows) engine alike.
struct EmptyHealth {
    consecutive: AtomicU32,
    restarts: AtomicU32,
    last_restart_ms: AtomicU64,
    /// Sticky "empty-suspect" flag. Set when a streak trips, held while we canary-probe and while a
    /// canary-confirmed-dead engine is parked, cleared only when a canary produces real output.
    /// Consulted by both the advertise decision AND `maybe_spawn_job` (refuse new work while set),
    /// so a suspect engine is truly taken out of rotation — not just left off the next heartbeat.
    quarantined: std::sync::atomic::AtomicBool,
    /// Unix-ms of the first entry into the current quarantine (0 when not quarantined). Bounds the
    /// park: if only INCONCLUSIVE probes come back for too long (an alive-but-canary-slow engine
    /// that /health can't restart), release quarantine so real traffic re-tests it instead of the
    /// node going dark forever.
    quarantined_since_ms: AtomicU64,
    /// On-disk mirror of `restarts`/`last_restart_ms`. The in-process (Mac) restart is a whole
    /// `std::process::abort()`, which would wipe an in-memory budget and let a broken model
    /// abort-loop forever; persisting makes the cap + cooldown bind across relaunches.
    state_path: PathBuf,
}

impl EmptyHealth {
    /// Load the persisted restart budget from disk (so it survives the in-process abort recovery).
    fn load(config_dir: &Path) -> Self {
        let state_path = config_dir.join("empty_restart_state.json");
        let (restarts, last_restart_ms) = std::fs::read_to_string(&state_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| {
                (
                    v.get("restarts").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                    v.get("last_restart_ms").and_then(|x| x.as_u64()).unwrap_or(0),
                )
            })
            .unwrap_or((0, 0));
        Self {
            consecutive: AtomicU32::new(0),
            restarts: AtomicU32::new(restarts),
            last_restart_ms: AtomicU64::new(last_restart_ms),
            quarantined: std::sync::atomic::AtomicBool::new(false),
            quarantined_since_ms: AtomicU64::new(0),
            state_path,
        }
    }
    /// A chat completion produced output — or was a legitimate/exempt empty. Clears the streak.
    fn record_ok(&self) {
        self.consecutive.store(0, Ordering::Relaxed);
    }
    /// A chat completion returned empty content when it should have produced output. Advances the streak.
    fn record_empty(&self) {
        let n = self.consecutive.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!("empty chat completion ({n} consecutive) — engine may be producing nothing");
    }
    #[cfg(test)]
    fn consecutive(&self) -> u32 {
        self.consecutive.load(Ordering::Relaxed)
    }
    /// Atomically consume a trip: reset the streak to 0 ONLY if it's still exactly the observed
    /// value, so an increment that lands between the read and the reset is preserved (a concurrent
    /// job's empty is not silently lost — it re-accumulates and re-trips next cycle).
    fn take_trip(&self, threshold: u32) -> bool {
        let v = self.consecutive.load(Ordering::Relaxed);
        if v < threshold {
            return false;
        }
        self.consecutive
            .compare_exchange(v, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }
    fn is_quarantined(&self) -> bool {
        self.quarantined.load(Ordering::Relaxed)
    }
    /// Enter (or stay in) quarantine, stamping the entry time only on the first transition so the
    /// max-age escape measures from when parking actually began — not from each re-probe cycle.
    fn enter_quarantine(&self, now: u64) {
        let was = self.quarantined.swap(true, Ordering::Relaxed);
        if !was {
            self.quarantined_since_ms.store(now, Ordering::Relaxed);
        }
    }
    fn clear_quarantine(&self) {
        self.quarantined.store(false, Ordering::Relaxed);
        self.quarantined_since_ms.store(0, Ordering::Relaxed);
    }
    fn quarantined_since(&self) -> u64 {
        self.quarantined_since_ms.load(Ordering::Relaxed)
    }
    /// Empty-triggered restarts within `window_ms`. Older restarts don't count (a node that was
    /// flaky an hour ago isn't capped forever); when the window lapses the persisted count is
    /// decayed to 0 so the budget genuinely resets.
    fn effective_restarts(&self, now: u64, window_ms: u64) -> u32 {
        let last = self.last_restart_ms.load(Ordering::Relaxed);
        if last != 0 && now.saturating_sub(last) > window_ms {
            self.restarts.store(0, Ordering::Relaxed);
            self.persist();
            0
        } else {
            self.restarts.load(Ordering::Relaxed)
        }
    }
    fn last_restart_ms(&self) -> u64 {
        self.last_restart_ms.load(Ordering::Relaxed)
    }
    /// Record an empty-triggered restart and persist BEFORE the caller restarts — so the budget
    /// survives an in-process abort() that never returns.
    fn note_restart(&self, now: u64) {
        self.restarts.fetch_add(1, Ordering::Relaxed);
        self.last_restart_ms.store(now, Ordering::Relaxed);
        self.persist();
    }
    #[cfg(test)]
    fn restarts(&self) -> u32 {
        self.restarts.load(Ordering::Relaxed)
    }
    fn persist(&self) {
        let body = serde_json::json!({
            "restarts": self.restarts.load(Ordering::Relaxed),
            "last_restart_ms": self.last_restart_ms.load(Ordering::Relaxed),
        })
        .to_string();
        // Atomic write (tmp + rename) so a torn/partial file can never be read back as (0,0),
        // which would silently reset BOTH the cap and the cooldown.
        let tmp = self.state_path.with_extension("json.tmp");
        if std::fs::write(&tmp, &body).is_ok() {
            let _ = std::fs::rename(&tmp, &self.state_path);
        }
    }
    #[cfg(test)]
    fn new_for_test() -> Self {
        Self {
            consecutive: AtomicU32::new(0),
            restarts: AtomicU32::new(0),
            last_restart_ms: AtomicU64::new(0),
            quarantined: std::sync::atomic::AtomicBool::new(false),
            quarantined_since_ms: AtomicU64::new(0),
            state_path: std::env::temp_dir().join("sgl_empty_health_test_state.json"),
        }
    }
}

/// True iff a message's `content` carries non-whitespace text. Handles both the string form and
/// OpenAI's array-of-parts form (`[{"type":"text","text":"..."}]`); anything else (null, tool
/// scaffolding) is treated as no-text.
fn message_has_text(content: Option<&serde_json::Value>) -> bool {
    match content {
        Some(serde_json::Value::String(s)) => !s.trim().is_empty(),
        Some(serde_json::Value::Array(parts)) => parts.iter().any(|p| {
            p.get("text")
                .and_then(|t| t.as_str())
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)
        }),
        _ => false,
    }
}

/// A request "should produce output" iff it allows more than one token AND at least one message
/// carries non-whitespace text. A whitespace/empty prompt or `max_tokens<=1` can legitimately
/// yield an empty reply, so those are exempt from the empty-completion signal (no false trip).
fn request_expects_output(messages: &serde_json::Value, max_tokens: i32) -> bool {
    if max_tokens <= 1 {
        return false;
    }
    messages
        .as_array()
        .map(|arr| arr.iter().any(|m| message_has_text(m.get("content"))))
        .unwrap_or(false)
}

fn is_homura_model(job: &PendingJob) -> bool {
    job.model
        .as_deref()
        .map(|m| {
            let id = m.to_ascii_lowercase();
            id == "homura-30b" || id.contains("homura")
        })
        .unwrap_or(false)
}

fn homura_envelope_prefix_is_valid(prefix: &str) -> bool {
    let lower = prefix.to_ascii_lowercase();
    let mut s = lower.trim();
    if let Some(rest) = s.strip_prefix("<|start|>") {
        s = rest.trim_start();
    }
    if let Some(rest) = s.strip_prefix("assistant") {
        s = rest.trim_start();
    }
    let Some(rest) = s.strip_prefix("to=") else {
        return false;
    };
    let role_len = rest
        .char_indices()
        .find_map(|(i, ch)| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                None
            } else {
                Some(i)
            }
        })
        .unwrap_or(rest.len());
    if role_len == 0 {
        return false;
    }
    let tail = rest[role_len..].trim();
    tail.is_empty()
        || tail.starts_with("<|channel|>")
        || tail.split_whitespace().all(|part| {
            part.starts_with("<|channel|>")
                || part
                    .chars()
                    .all(|c| c.is_ascii_alphabetic() || c == '_')
        })
}

fn homura_prefix_might_be_envelope(s: &str) -> bool {
    let lower = s.trim_start().to_ascii_lowercase();
    if lower.is_empty() {
        return true;
    }
    ["to=", "<|start|>", "assistant to="]
        .iter()
        .any(|start| start.starts_with(&lower) || lower.starts_with(start))
}

fn find_homura_terminal(s: &str) -> Option<usize> {
    ["<|eot|>", "<|end|>"]
        .iter()
        .filter_map(|marker| s.find(marker))
        .min()
}

fn strip_homura_message_envelope(content: &str) -> String {
    let trimmed = content.trim();
    let Some(message_pos) = trimmed.find("<|message|>") else {
        return content.to_string();
    };
    if !homura_envelope_prefix_is_valid(&trimmed[..message_pos]) {
        return content.to_string();
    }
    let mut body = &trimmed[message_pos + "<|message|>".len()..];
    if let Some(end) = find_homura_terminal(body) {
        body = &body[..end];
    }
    body.trim().to_string()
}

enum HomuraStreamMode {
    Prefix,
    Body,
    Passthrough,
    Done,
}

struct HomuraStreamCleaner {
    mode: HomuraStreamMode,
    buf: String,
}

impl HomuraStreamCleaner {
    fn new() -> Self {
        Self {
            mode: HomuraStreamMode::Prefix,
            buf: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> String {
        if matches!(self.mode, HomuraStreamMode::Done) {
            return String::new();
        }
        self.buf.push_str(text);
        match self.mode {
            HomuraStreamMode::Prefix => self.drain_prefix(),
            HomuraStreamMode::Body => self.drain_body(false),
            HomuraStreamMode::Passthrough => std::mem::take(&mut self.buf),
            HomuraStreamMode::Done => String::new(),
        }
    }

    fn finish(&mut self) -> String {
        match self.mode {
            HomuraStreamMode::Done => String::new(),
            HomuraStreamMode::Prefix => {
                let out = self.drain_prefix();
                if out.is_empty() {
                    std::mem::take(&mut self.buf)
                } else {
                    out + &self.drain_body(true)
                }
            }
            HomuraStreamMode::Body => self.drain_body(true),
            HomuraStreamMode::Passthrough => std::mem::take(&mut self.buf),
        }
    }

    fn drain_prefix(&mut self) -> String {
        let trim_bytes = self.buf.len() - self.buf.trim_start().len();
        let visible = &self.buf[trim_bytes..];
        if let Some(message_pos) = visible.find("<|message|>") {
            if homura_envelope_prefix_is_valid(&visible[..message_pos]) {
                let body_start = trim_bytes + message_pos + "<|message|>".len();
                self.buf = self.buf[body_start..].trim_start().to_string();
                self.mode = HomuraStreamMode::Body;
                return self.drain_body(false);
            }
        }
        if homura_prefix_might_be_envelope(visible) && visible.len() <= 512 {
            return String::new();
        }
        self.mode = HomuraStreamMode::Passthrough;
        std::mem::take(&mut self.buf)
    }

    fn drain_body(&mut self, flush_tail: bool) -> String {
        if let Some(end) = find_homura_terminal(&self.buf) {
            let out = self.buf[..end].trim_end().to_string();
            self.buf.clear();
            self.mode = HomuraStreamMode::Done;
            return out;
        }
        if flush_tail {
            return drain_body_flush_tail(&mut self.buf);
        }
        drain_all_but_marker_tail(&mut self.buf)
    }
}

fn homura_terminal_prefix_tail_len(buf: &str) -> usize {
    ["<|eot|>", "<|end|>"]
        .iter()
        .flat_map(|marker| (1..marker.len()).map(move |n| &marker[..n]))
        .filter(|prefix| buf.ends_with(*prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

fn drain_body_flush_tail(buf: &mut String) -> String {
    let hold = homura_terminal_prefix_tail_len(buf);
    let split_at = buf.len().saturating_sub(hold);
    let out = buf[..split_at].to_string();
    buf.clear();
    out
}

fn drain_all_but_marker_tail(buf: &mut String) -> String {
    let hold = homura_terminal_prefix_tail_len(buf);
    if hold == buf.len() {
        return String::new();
    }
    let split_at = buf.len() - hold;
    let tail = buf[split_at..].to_string();
    let out = buf[..split_at].to_string();
    *buf = tail;
    out
}

/// Outcome of a canary probe. Tri-state on purpose: "inconclusive" (busy/hung/transient) must be
/// distinguished from "recovered", because only a proven `NonEmpty` may clear quarantine — an
/// inconclusive probe leaves a parked engine parked (it never proves recovery), and never restarts.
enum CanaryOutcome {
    /// Engine answered fast with EMPTY content — the zombie signature. Restart-eligible.
    Empty,
    /// Engine answered with real output — proven healthy. Clears quarantine.
    NonEmpty,
    /// Timeout or transport error — busy/hung/dead regime owned by the wedge + /health watchdogs.
    /// A canary queued behind full `--parallel` slots lands here, so it can NOT be read as "dead"
    /// (else an attacker could fill slots with long jobs + a few empties to force a restart) NOR as
    /// "recovered" (else a still-dead engine could un-park on a single slow probe). Do nothing.
    Inconclusive,
}

/// Canary probe: ask our OWN engine one fixed question with a short cap. A user cannot forge the
/// engine's own answer, which is what makes this the ground truth that gates restart/un-park.
async fn run_canary(engine: &InferenceEngine) -> CanaryOutcome {
    let messages = serde_json::json!([
        { "role": "user", "content": "Reply with the single word: ok" }
    ]);
    let probe = engine.chat_completion(messages, 0.0, 8, None, None);
    match tokio::time::timeout(std::time::Duration::from_secs(20), probe).await {
        Ok(Ok(r)) => {
            if r.content.trim().is_empty() {
                CanaryOutcome::Empty
            } else {
                CanaryOutcome::NonEmpty
            }
        }
        Ok(Err(e)) => {
            tracing::warn!("empty-completion canary errored — inconclusive (health/wedge watchdogs own this): {e}");
            CanaryOutcome::Inconclusive
        }
        Err(_) => {
            tracing::warn!("empty-completion canary timed out — inconclusive (engine busy/hung; wedge/health watchdogs own this)");
            CanaryOutcome::Inconclusive
        }
    }
}

use crate::config::{self, NodeConfig};
use crate::crypto::NodeKeypair;
use crate::inference::{ChatMessage, InferenceEngine, InferenceEngineConfig};
use crate::orchestrator::{OrchestratorClient, PendingJob};
use crate::tee;

pub struct ResourceConfig {
    pub threads: u32,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub max_jobs: u32,
    pub batch_size: u32,
    pub heartbeat_interval: u64,
    pub resource_percent: u8,
    /// Confidential token streaming. ALWAYS enabled: the node always advertises
    /// the `streaming` capability and serves stream jobs. Whether a given request
    /// streams is the CALLER's choice (`stream: true`), not the operator's — a
    /// provider can't silently disable it. The legacy `--enable-streaming` flag is
    /// now a deprecated no-op kept only so old service definitions still parse.
    pub streaming_enabled: bool,
}

impl ResourceConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn from_args(
        resource_percent: u8,
        threads: Option<u32>,
        gpu_layers: Option<u32>,
        context_size: u32,
        max_jobs: u32,
        batch_size: u32,
        heartbeat_interval: u64,
        _streaming_enabled: bool, // deprecated no-op: streaming is always on (see below)
    ) -> Self {
        let total_cpus = std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(4);

        let computed_threads =
            ((total_cpus as f64 * resource_percent as f64 / 100.0).ceil() as u32).max(1);
        let computed_gpu_layers = if resource_percent >= 50 {
            99
        } else {
            (99.0 * resource_percent as f64 / 100.0).round() as u32
        };
        // Windows default: AUTO (omit -ngl, llama.cpp fits offload itself) instead of
        // pinning 99 layers — consumer GPU drivers there are too variable to force max
        // offload (a pinned -ngl 99 fed the first tester's Vulkan crash loop). Explicit
        // --gpu-layers still wins on every platform.
        let default_gpu_layers = if cfg!(windows) {
            crate::inference::GPU_LAYERS_AUTO
        } else {
            computed_gpu_layers
        };

        Self {
            threads: threads.unwrap_or(computed_threads),
            gpu_layers: gpu_layers.unwrap_or(default_gpu_layers),
            context_size,
            max_jobs,
            batch_size,
            heartbeat_interval,
            resource_percent,
            // Always on — providers always stream; the caller opts in per request.
            streaming_enabled: true,
        }
    }

    pub fn load_factor(&self) -> f64 {
        1.0 - (self.resource_percent as f64 / 100.0)
    }
}

/// Bounded de-dup set of job ids the node has already handled. A job can arrive
/// via WS push AND a REST heartbeat poll during transitions — this ensures each
/// runs exactly once. Capped so it can't grow without bound.
struct SeenJobs {
    set: HashSet<String>,
    order: VecDeque<String>,
    cap: usize,
}

impl SeenJobs {
    fn new() -> Self {
        Self {
            set: HashSet::new(),
            order: VecDeque::new(),
            cap: 1024,
        }
    }

    /// Returns true if the id is new (caller should handle it); false if a duplicate.
    fn check_and_insert(&mut self, id: &str) -> bool {
        if self.set.contains(id) {
            return false;
        }
        if self.order.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        self.set.insert(id.to_string());
        self.order.push_back(id.to_string());
        true
    }
}

/// De-dup + capacity check, then spawn job processing. Shared by the REST poll
/// loop and the WS push callback so both transports are equivalent and safe.
#[allow(clippy::too_many_arguments)]
fn maybe_spawn_job(
    job: PendingJob,
    client: Arc<OrchestratorClient>,
    engine: Option<Arc<InferenceEngine>>,
    node_secret: [u8; 32],
    streaming_enabled: bool,
    active_jobs: Arc<AtomicU32>,
    inflight: Arc<Mutex<HashSet<String>>>,
    seen: Arc<Mutex<SeenJobs>>,
    max_jobs: u32,
    last_activity: Arc<AtomicU64>,
    completions: Arc<AtomicU64>,
    empty_health: Arc<EmptyHealth>,
) {
    // Empty-suspect quarantine: while the heartbeat loop is canary-probing (or a
    // canary-confirmed-dead engine is parked), refuse new work so no request hits a suspect
    // engine during the probe/restart window. Not marked seen → the orchestrator re-routes it.
    // This closes the gap where the prior heartbeat's advertisement is still live at the
    // orchestrator during the ≤20s canary (de-advertising alone only takes effect next heartbeat).
    if empty_health.is_quarantined() {
        tracing::debug!("node quarantined (empty-suspect) — deferring job {}", job.id);
        return;
    }
    // Atomically reserve a slot (capacity check + increment in one CAS) so
    // concurrent WS-push + REST-poll arrivals can't exceed max_jobs.
    if active_jobs
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
            if c < max_jobs {
                Some(c + 1)
            } else {
                None
            }
        })
        .is_err()
    {
        // At capacity. Not marked seen, so the next REST poll retries it.
        tracing::warn!("At max concurrent jobs ({max_jobs}), deferring job {}", job.id);
        return;
    }
    // De-dup; roll back the reserved slot if this id was already handled.
    {
        let mut s = seen.lock().unwrap();
        if !s.check_and_insert(&job.id) {
            active_jobs.fetch_sub(1, Ordering::Relaxed);
            tracing::debug!("Duplicate job {} ignored", job.id);
            return;
        }
    }
    // Record the REAL in-flight job id so the heartbeat can report exactly what we're
    // running (#119). The orchestrator uses this to clear ghost slots — dispatched jobs
    // we are NOT processing — without ever killing a job we ARE processing.
    inflight.lock().unwrap().insert(job.id.clone());
    // Stamp activity on accept AND completion — the watchdog treats "busy but no
    // activity for too long" as a wedged engine (jobs hang while /health still passes).
    last_activity.store(now_ms(), Ordering::Relaxed);
    tracing::info!("Accepted job {} (type: {})", job.id, job.job_type);
    let job_id = job.id.clone();
    tokio::spawn(async move {
        process_job(&client, &engine, &job, &node_secret, streaming_enabled, &empty_health).await;
        // Release the slot ONLY if this job is still tracked. The watchdog may have already
        // abandoned it (removed it from inflight + released its slot) when the engine wedged.
        // Gating decrement + completion on inflight membership makes slot-release idempotent,
        // preventing an active_jobs underflow (u32 wrap) and a false progress signal.
        let still_tracked = inflight.lock().unwrap().remove(&job_id);
        if still_tracked {
            active_jobs.fetch_sub(1, Ordering::Relaxed);
            // Genuine progress (the engine responded) — resets the watchdog's abort backstop.
            completions.fetch_add(1, Ordering::Relaxed);
        }
        last_activity.store(now_ms(), Ordering::Relaxed);
    });
}

pub async fn init(
    config_dir: &Path,
    orchestrator_url: &str,
    wallet: &str,
    tee_type: &str,
    models: &[String],
) -> Result<(), String> {
    let cfg_path = config::config_path(config_dir);
    if cfg_path.exists() {
        return Err(format!(
            "Node already initialized. Config at: {}\nTo reinitialize, delete the config directory first.",
            cfg_path.display()
        ));
    }

    let caps = tee::detect();
    tee::print_capabilities(&caps);
    println!();

    tracing::info!("Generating ed25519 keypair...");
    let keypair = NodeKeypair::generate();
    let kp_path = config::keypair_path(config_dir);
    keypair.save(&kp_path)?;
    tracing::info!("Keypair saved to {}", kp_path.display());

    let public_key = keypair.public_key_bs58();
    tracing::info!("Public key: {public_key}");

    tracing::info!("Registering with orchestrator at {orchestrator_url}...");
    let client = OrchestratorClient::new(orchestrator_url, None);
    let registration = client
        .register(wallet, None, tee_type, models, &public_key, &caps)
        .await?;

    tracing::info!("Registered! Node ID: {}", registration.node_id);

    let node_config = NodeConfig {
        node_id: registration.node_id,
        auth_token: registration.auth_token,
        wallet_address: wallet.to_string(),
        tee_type: tee_type.to_string(),
        orchestrator_url: orchestrator_url.to_string(),
        keypair_path: kp_path.to_string_lossy().to_string(),
    };

    config::save_config(config_dir, &node_config)?;
    tracing::info!("Config saved to {}", cfg_path.display());
    tracing::info!("Node initialized. Run `sgl start` to begin processing jobs.");
    tracing::info!("Run `sgl attest` to verify identity before receiving jobs.");

    Ok(())
}

/// `sgl login` — browser device-authorization flow.
pub async fn login(
    config_dir: &Path,
    orchestrator_url: &str,
    tee_type: &str,
    models: &[String],
) -> Result<(), String> {
    let cfg_path = config::config_path(config_dir);
    if cfg_path.exists() {
        return Err(format!(
            "Node already initialized. Config at: {}",
            cfg_path.display()
        ));
    }

    let caps = tee::detect();
    tee::print_capabilities(&caps);
    println!();

    let keypair = NodeKeypair::generate();
    let kp_path = config::keypair_path(config_dir);
    keypair.save(&kp_path)?;
    let public_key = keypair.public_key_bs58();

    let client = OrchestratorClient::new(orchestrator_url, None);
    let session = client.device_start().await?;

    println!("\n  Open to link this node:\n      {}\n  Approve with your staked Solana wallet (code: {}).\n", session.verify_url, session.user_code);
    // Only hand a real web URL to the OS opener (the verify link is always https from
    // our orchestrator). This guards the platform openers against an unexpected scheme.
    if session.verify_url.starts_with("https://") || session.verify_url.starts_with("http://") {
        #[cfg(target_os = "windows")]
        {
            // `rundll32 url.dll,FileProtocolHandler <url>` hands the URL to the OS URL
            // handler (default browser) as a single CreateProcess argument — no `cmd`
            // shell parsing/injection, and unlike `explorer.exe <url>` it still works on
            // Windows 11 24H2+ (explorer there ignores URLs and opens a file window).
            let _ = std::process::Command::new("rundll32")
                .args(["url.dll,FileProtocolHandler", &session.verify_url])
                .spawn();
        }
        #[cfg(not(target_os = "windows"))]
        {
            let opener = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            let _ = std::process::Command::new(opener)
                .arg(&session.verify_url)
                .spawn();
        }
    }

    let interval = session.interval.max(2);
    let max_polls = if session.expires_in > 0 {
        (session.expires_in / interval) + 2
    } else {
        200
    };
    tracing::info!("Waiting for approval in the browser...");

    let mut reg_code: Option<String> = None;
    let mut wallet: Option<String> = None;
    for _ in 0..max_polls {
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        match client.device_poll(&session.device_code).await {
            Ok(p) if p.status == "approved" => {
                reg_code = p.registration_code;
                wallet = p.wallet_address;
                break;
            }
            Ok(p) if p.status == "expired" => {
                return Err("Login session expired. Run `sgl login` again.".to_string())
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("poll error (retrying): {e}"),
        }
    }

    let reg_code = reg_code.ok_or("Login timed out waiting for approval.")?;
    let wallet = wallet.unwrap_or_default();
    tracing::info!("Approved by wallet {wallet}. Registering node...");

    let registration = client
        .register(
            &wallet,
            Some(&reg_code),
            tee_type,
            models,
            &public_key,
            &caps,
        )
        .await?;

    let node_config = NodeConfig {
        node_id: registration.node_id,
        auth_token: registration.auth_token,
        wallet_address: wallet,
        tee_type: tee_type.to_string(),
        orchestrator_url: orchestrator_url.to_string(),
        keypair_path: kp_path.to_string_lossy().to_string(),
    };
    config::save_config(config_dir, &node_config)?;
    tracing::info!(
        "Linked! Node ID: {}. Run `sgl attest`, then `sgl start --model-path <model.gguf>`.",
        node_config.node_id
    );
    Ok(())
}

/// Headless login for one-click machine deploys (cloud-init): registers with a
/// single-use provision code issued by the deploy pipeline and bound to the buyer's
/// wallet, instead of the interactive browser device flow. Identical trust model to
/// `login` — the keypair is generated ON this machine (private key never leaves it)
/// and the orchestrator validates the code + wallet + stake server-side before
/// returning the auth token. Only a short-lived, one-time code transits cloud-init.
pub async fn login_headless(
    config_dir: &Path,
    orchestrator_url: &str,
    tee_type: &str,
    models: &[String],
    code: &str,
    wallet: &str,
) -> Result<(), String> {
    let cfg_path = config::config_path(config_dir);
    if cfg_path.exists() {
        return Err(format!(
            "Node already initialized. Config at: {}",
            cfg_path.display()
        ));
    }
    if code.trim().is_empty() || wallet.trim().is_empty() {
        return Err("Provision code and wallet must be non-empty.".to_string());
    }

    let caps = tee::detect();
    tee::print_capabilities(&caps);

    let keypair = NodeKeypair::generate();
    let kp_path = config::keypair_path(config_dir);
    keypair.save(&kp_path)?;
    let public_key = keypair.public_key_bs58();

    let client = OrchestratorClient::new(orchestrator_url, None);
    tracing::info!("Registering node headlessly for wallet {wallet}...");
    let registration = client
        .register(wallet, Some(code), tee_type, models, &public_key, &caps)
        .await?;

    let node_config = NodeConfig {
        node_id: registration.node_id,
        auth_token: registration.auth_token,
        wallet_address: wallet.to_string(),
        tee_type: tee_type.to_string(),
        orchestrator_url: orchestrator_url.to_string(),
        keypair_path: kp_path.to_string_lossy().to_string(),
    };
    config::save_config(config_dir, &node_config)?;
    tracing::info!(
        "Linked! Node ID: {}. Run `sgl start --model-path <model.gguf>`.",
        node_config.node_id
    );
    Ok(())
}

/// Continuous-batching concurrency: how many requests this node serves in parallel.
/// llama-server runs N slots (`--parallel N --cont-batching`) over ONE loaded model — the
/// weights load once, so the only per-slot RAM cost is the KV cache. We keep each slot's
/// usable context at the operator's configured `context_size` by setting the server's total
/// `-c` to `slots * context_size` (llama-server divides total context across slots), so
/// concurrency trades RAM, not per-request context.
///
/// RAM-aware + conservative so a small operator Mac never OOMs:
///   fixed    = model weights (≈ GGUF file size) + headroom for OS/runtime/app
///   per-slot = KV cache at `context_size` (estimated from model size, rounded up)
///   slots    = clamp(free_for_kv / per_slot_kv, 1, MAX_SLOTS)
/// Boxes under MIN_RAM stay at 1. An explicit `--max-jobs N>0` caps (never raises) the auto
/// value; the baked default of 0 means "auto". This keeps `--max-jobs 1` literal.
fn compute_parallel_slots(
    memory_gb: f64,
    model_path: &Path,
    context_size: u32,
    requested_max_jobs: u32,
) -> u32 {
    // Ceiling on auto-sized slots. Raised 2 → 4 (2026-07-03): the per-slot KV estimate below
    // is deliberately HIGH and the RAM budget still gates the actual count per machine+model
    // (a 14B on 16GB still gets 1; a 2-3B on 24GB can use the headroom). Verified locally at
    // 3 concurrent sequences on a 16GB M4 (in-process, gemma-2-2b) with correct outputs and
    // deterministic token counts. Apple Silicon is unified memory, so model weights + KV
    // share ONE pool — we never assume 100% of RAM is ours.
    const MAX_SLOTS: u32 = 4;
    const OVERHEAD_GB: f64 = 4.0; // OS + llama runtime + node app + mmap slack
    const MIN_RAM_FOR_BATCHING_GB: f64 = 16.0;
    const USABLE_FRACTION: f64 = 0.85; // headroom for the rest of the machine

    let model_gb = std::fs::metadata(model_path)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(6.0); // unknown size → assume large, stay safe

    if memory_gb < MIN_RAM_FOR_BATCHING_GB {
        return 1; // too small to risk concurrency
    }

    // KV per slot at the configured context, estimated HIGH so we under-provision slots
    // rather than risk OOM (the estimate ignores exact dtype/arch, so we round up).
    let per_slot_kv_gb = (model_gb * 0.20).max(0.4) * (context_size as f64 / 8192.0).max(0.25);
    let usable = memory_gb * USABLE_FRACTION;
    let free_for_kv = (usable - model_gb - OVERHEAD_GB).max(0.0);
    let auto = ((free_for_kv / per_slot_kv_gb).floor() as u32).clamp(1, MAX_SLOTS);

    if requested_max_jobs > 0 {
        auto.min(requested_max_jobs) // explicit override caps, never raises past RAM-safe
    } else {
        auto
    }
}

/// How long a slot may be "busy with zero progress" before the watchdog treats the engine
/// as wedged. MUST exceed the orchestrator's model-aware stuck-job SLA (≈360s for 7–12B,
/// 900s for 13–34B, 1800s for 65B+) so a legitimately slow large-model inference is never
/// killed early. Tiered by GGUF size (a node serves one model) + margin over the SLA.
/// In-process concurrency (advertised capacity). DYNAMIC by default: the same machine- and
/// model-aware budget the server engine uses (total RAM, model file size, per-request
/// context) decides how many sequences batch concurrently — a small model on a big box gets
/// more slots, a big model on a small box gets 1. `SGL_INPROCESS_SLOTS=<n>` is a manual
/// override for operators/benchmarks, still capped by the RAM budget so a custom value can
/// never exceed what memory can safely hold.
fn inprocess_slots(
    memory_gb: f64,
    model_path: &Path,
    context_size: u32,
    requested_max_jobs: u32,
) -> u32 {
    let ram_cap =
        compute_parallel_slots(memory_gb, model_path, context_size, requested_max_jobs).max(1);
    match std::env::var("SGL_INPROCESS_SLOTS").ok().and_then(|s| s.trim().parse::<u32>().ok()) {
        Some(n) if n >= 1 => n.min(ram_cap), // custom, RAM-capped
        _ => ram_cap,                        // dynamic default (machine + model aware)
    }
}

fn wedge_timeout_ms(model_path: &Path) -> u64 {
    let gb = std::fs::metadata(model_path)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(6.0);
    if gb < 11.0 {
        600_000 // ~7–12B: SLA 360s + margin
    } else if gb < 40.0 {
        1_200_000 // ~13–34B: SLA 900s + margin
    } else {
        2_100_000 // 65B+: SLA 1800s + margin
    }
}

pub async fn start(
    config_dir: &Path,
    orchestrator_url: &str,
    model_path: Option<&str>,
    model_name: Option<&str>,
    mmproj_path: Option<&str>,
    image_max_tokens: Option<u32>,
    inference_port: u16,
    rc: &ResourceConfig,
) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let keypair = NodeKeypair::load(&config::keypair_path(config_dir))?;

    let client = Arc::new(OrchestratorClient::new(
        orchestrator_url,
        Some(cfg.auth_token.clone()),
    ));

    let total_cpus = std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4);

    tracing::info!(
        "Starting node {} (wallet: {})",
        cfg.node_id,
        cfg.wallet_address
    );
    tracing::info!("Public key: {}", keypair.public_key_bs58());
    tracing::info!("Resource config:");
    tracing::info!("  Preset:       {}%", rc.resource_percent);
    tracing::info!("  Threads:      {}/{}", rc.threads, total_cpus);
    tracing::info!("  GPU layers:   {}", rc.gpu_layers);
    tracing::info!("  Context:      {} tokens", rc.context_size);
    tracing::info!("  Batch size:   {}", rc.batch_size);
    tracing::info!(
        "  Max jobs:     {}",
        if rc.max_jobs == 0 { "auto".to_string() } else { rc.max_jobs.to_string() }
    );
    tracing::info!("  Streaming:    {}", if rc.streaming_enabled { "enabled" } else { "disabled" });

    let mut engine: Option<Arc<InferenceEngine>> = None;
    let mut models: Vec<String> = vec![];
    // Continuous-batching concurrency: how many requests this node serves at once.
    // Defaults to 1 (no model, or in-process engine); the server engine computes a
    // RAM-aware value below. Used for the LOCAL capacity gate AND advertised to the
    // orchestrator so it dispatches up to this many concurrent jobs.
    let mut effective_slots: u32 = 1;
    // #159 watchdog timeout, model-aware (set from the GGUF size once we know the model).
    let mut wedge_ms: u64 = 600_000;

    if let Some(path) = model_path {
        let name = model_name.map(|s| s.to_string()).unwrap_or_else(|| {
            Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        });

        // Crash-loop engine auto-swap: if the OS service keeps restarting a young-dying
        // node on the GPU (Vulkan) engine, swap to the CPU build before creating the
        // engine. No-op on in-process/mac installs (no engine.variant marker).
        crate::setup::crashloop_autoswap().await;

        // Engine selection: SGL_ENGINE=server|inprocess. DEFAULT = in-process on builds
        // that ship it (macOS: the model runs INSIDE this attested process — no separate
        // llama-server child that can die while the wrapper heartbeats "healthy", killing
        // the zombie-node class by design). SGL_ENGINE=server remains the escape hatch.
        // Builds without the feature (Linux, for now) default to the server engine.
        let engine_explicit = std::env::var("SGL_ENGINE")
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        let engine_mode = match std::env::var("SGL_ENGINE").ok().as_deref() {
            Some(s) if !s.is_empty() => crate::inference::EngineMode::parse(s)?,
            #[cfg(feature = "inprocess")]
            _ => crate::inference::EngineMode::InProcess,
            #[cfg(not(feature = "inprocess"))]
            _ => crate::inference::EngineMode::Server,
        };
        // Vision (multimodal) requires the SERVER engine — the in-process engine can't load an
        // mmproj. Force it whenever an mmproj is provided (even on the macOS/in-process default or
        // an explicit SGL_ENGINE=inprocess) so a vision node actually serves images instead of
        // silently degrading to text-only and omitting its `vision` capability.
        let engine_mode = match engine_mode {
            crate::inference::EngineMode::InProcess if mmproj_path.is_some() => {
                tracing::warn!(
                    "--mmproj-path set: using the server engine (in-process can't serve vision)"
                );
                crate::inference::EngineMode::Server
            }
            m => m,
        };

        let model_pb = PathBuf::from(path);
        // Continuous batching applies to BOTH engines, sized by the same machine+model-aware
        // budget (RAM, model size, per-request context) so a node advertises the same honest
        // capacity whichever engine it runs. In-process additionally honors the
        // SGL_INPROCESS_SLOTS override (custom, still RAM-capped) — see inprocess_slots().
        effective_slots = match engine_mode {
            crate::inference::EngineMode::Server => {
                compute_parallel_slots(tee::detect().memory_gb, &model_pb, rc.context_size, rc.max_jobs)
            }
            crate::inference::EngineMode::InProcess => inprocess_slots(
                tee::detect().memory_gb,
                &model_pb,
                rc.context_size,
                rc.max_jobs,
            ),
        };
        wedge_ms = wedge_timeout_ms(&model_pb);
        // llama-server divides its total `-c` across slots, so pass slots × context_size
        // to keep each slot at the operator's configured per-request context.
        let total_ctx = effective_slots
            .saturating_mul(rc.context_size)
            .max(rc.context_size);

        tracing::info!("Loading model: {name} from {path}");
        tracing::info!(
            "  Parallel slots: {effective_slots} (continuous batching) — total ctx {total_ctx} ({} per slot)",
            rc.context_size
        );
        let eng_config = InferenceEngineConfig {
            model_path: model_pb,
            model_name: name.clone(),
            port: inference_port,
            threads: rc.threads,
            gpu_layers: rc.gpu_layers,
            context_size: total_ctx,
            batch_size: rc.batch_size,
            parallel_slots: effective_slots,
            mmproj_path: mmproj_path.map(PathBuf::from),
            image_max_tokens,
        };
        tracing::info!("Inference engine mode: {engine_mode:?}");
        // New-arch fallback: the embedded llama.cpp (llama-cpp-2 crate) trails the
        // llama-server pin, so a model whose arch landed upstream recently (e.g.
        // muse-glimmer needs >= b10353) loads fine under the server engine but fails
        // in-process. When the DEFAULTED in-process engine can't create, retry once
        // as a server child instead of dying. Deliberately narrow: an EXPLICIT
        // SGL_ENGINE=inprocess still fails loudly (operator asked for that engine),
        // and embedding models are excluded — create() routes them to the dedicated
        // embed engine before engine_mode applies, so their failures are unrelated.
        let eng = match InferenceEngine::create(eng_config.clone(), engine_mode).await {
            Ok(e) => e,
            Err(err)
                if engine_mode == crate::inference::EngineMode::InProcess
                    && !engine_explicit
                    && !crate::embed_catalog::is_embedding_model(&eng_config.model_name) =>
            {
                tracing::warn!(
                    "In-process engine failed to load {name} ({err}) — retrying with the \
                     server engine (llama-server carries newer model archs)"
                );
                // macOS self-provisioning: the server retry needs a llama-server whose
                // llama.cpp is at least as new as our pin — a stale Homebrew copy would
                // fail on the same new arch that broke in-process. Install the managed,
                // hash-verified pinned build (one-time ~11 MB) so app-only operators
                // never touch Homebrew. Best-effort: if the download fails we still
                // retry with whatever find_llama_server() can locate.
                #[cfg(target_os = "macos")]
                if !crate::setup::managed_server_healthy() {
                    if let Err(e) = crate::setup::run(false).await {
                        tracing::warn!("llama.cpp self-provision failed ({e}) — trying existing installs");
                    }
                }
                InferenceEngine::create(eng_config, crate::inference::EngineMode::Server).await?
            }
            Err(err) => return Err(err),
        };
        models.push(name);
        engine = Some(Arc::new(eng));
        tracing::info!("Inference engine ready ({engine_mode:?})");
    } else {
        tracing::warn!("No model specified — node will register but cannot process inference jobs");
        tracing::warn!("Use --model-path <path.gguf> --model-name <name> to enable inference");
    }

    tracing::info!("Heartbeat interval: {}s", rc.heartbeat_interval);

    // Node's X25519 encryption key (derived from its ed25519 seed). Published on
    // every REST heartbeat so the orchestrator can seal prompts to it (E2E).
    let node_secret = keypair.signing_key.to_bytes();
    let node_enc_pubkey =
        crate::encryption::EncryptionKeypair::from_ed25519_seed(&node_secret).public_key_bs58();
    tracing::info!("X25519 encryption key: {node_enc_pubkey}");

    // #94: sign the keybind blob once (it's static per node — node_id + ed25519 +
    // x25519 + key_version are all fixed) and publish it on every heartbeat. Clients
    // verify this against the node's on-chain identity before sealing, so a malicious
    // orchestrator can't substitute its own key. Backward-compatible: if signing
    // can't happen (e.g. node_id isn't a UUID) we just publish the key unsigned.
    const KEY_VERSION: u32 = 1;
    let node_enc_pubkey_bytes =
        crate::encryption::EncryptionKeypair::from_ed25519_seed(&node_secret).public_key_bytes();
    let keybind_sig =
        crate::crypto::sign_keybind_v1(&node_secret, &cfg.node_id, &node_enc_pubkey_bytes, KEY_VERSION);
    let key_version_opt = keybind_sig.as_ref().map(|_| KEY_VERSION);
    match &keybind_sig {
        Some(_) => tracing::info!("Signed key identity published (keybind v1, key_version={KEY_VERSION})"),
        None => tracing::warn!("Could not sign keybind (node_id not a UUID?) — publishing key unsigned"),
    }

    // sha256 of the running binary, computed ONCE (reading the whole exe every
    // heartbeat would be wasteful — it can't change under a live process). Sent on
    // every heartbeat so the orchestrator tracks the build that's actually serving
    // and re-gates it against the allowlist (the attest-time hash goes stale the
    // moment `sgl update` swaps the binary). Empty string → send None.
    let binary_hash = {
        let h = crate::tee::detect_binary_hash();
        if h.is_empty() { None } else { Some(h) }
    };

    let active_jobs = Arc::new(AtomicU32::new(0));
    // Real in-flight job ids (#119) — reported in each heartbeat so the orchestrator can
    // clear ghost slots safely. Distinct from `active_jobs` (the capacity CAS counter).
    let inflight: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let seen_jobs = Arc::new(Mutex::new(SeenJobs::new()));
    // #159 watchdog input: unix-ms of the last job accept/completion. The heartbeat loop
    // treats "busy (active_jobs>0) but no activity for WEDGE_MS" as a wedged engine —
    // catches the case where llama-server answers /health but inference hangs forever.
    let last_activity = Arc::new(AtomicU64::new(now_ms()));
    // Monotonic count of genuinely-completed jobs — the watchdog resets its abort
    // backstop only when this advances (real progress, not a forced slot-reset).
    let completions = Arc::new(AtomicU64::new(0));
    // Empty-completion self-heal state (shared across both dispatch paths + the heartbeat loop).
    // Catches the "answers /health but returns 0 tokens" zombie the other watchdogs miss.
    let empty_health = Arc::new(EmptyHealth::load(config_dir));

    // ── WebSocket push-dispatch (additive fast-path) ──────────────────
    // Connects to the orchestrator and processes jobs the instant they're pushed,
    // removing the heartbeat pickup delay. If the socket is down the REST loop
    // below keeps serving (fallback). Jobs are de-duplicated by id across both.
    let ws_state = Arc::new(crate::ws::WsState::new());
    {
        let base = orchestrator_url.to_string();
        let node_id = cfg.node_id.clone();
        let client_ws = Arc::clone(&client);
        let client_job = Arc::clone(&client);
        let client_tok = Arc::clone(&client);
        let engine_ws = engine.clone();
        let secret = node_secret;
        let se = rc.streaming_enabled;
        let aj = Arc::clone(&active_jobs);
        let inf = Arc::clone(&inflight);
        let sj = Arc::clone(&seen_jobs);
        let mj = effective_slots;
        let la = Arc::clone(&last_activity);
        let co = Arc::clone(&completions);
        let eh = Arc::clone(&empty_health);
        let st = Arc::clone(&ws_state);
        let cfg_tok = cfg.clone();
        let config_dir_buf = config_dir.to_path_buf();
        tokio::spawn(async move {
            crate::ws::run(
                base,
                node_id,
                client_ws,
                st,
                move |job| {
                    maybe_spawn_job(
                        job,
                        Arc::clone(&client_job),
                        engine_ws.clone(),
                        secret,
                        se,
                        Arc::clone(&aj),
                        Arc::clone(&inf),
                        Arc::clone(&sj),
                        mj,
                        Arc::clone(&la),
                        Arc::clone(&co),
                        Arc::clone(&eh),
                    );
                },
                move |new_tok, _exp| {
                    client_tok.update_auth_token(new_tok.clone());
                    let mut updated = cfg_tok.clone();
                    updated.auth_token = new_tok;
                    if let Err(e) = config::save_config(&config_dir_buf, &updated) {
                        tracing::error!("Failed to save WS-rotated token: {e}");
                    } else {
                        tracing::info!("Auth token rotated over WS");
                    }
                },
            )
            .await;
        });
    }

    // Liveness-gated advertising: only advertise the model while llama-server actually
    // answers /health. If the engine has crashed/OOM'd mid-run (e.g. a 14B model on a
    // too-small box) we stop advertising after 2 consecutive failed probes, so the
    // orchestrator routes elsewhere instead of dispatching jobs this node would ghost.
    // A single transient blip is tolerated, and advertising resumes automatically the
    // moment /health is OK again (self-healing on operator restart or engine recovery).
    // Auto-restart supervision: when the engine has been unhealthy for 2+ checks we
    // KILL + RELAUNCH llama-server (instead of waiting for the operator). Bounded by a
    // restart cap so a model that can't run on this box (e.g. a 14B OOM-loop) doesn't
    // thrash forever — after the cap we stay unadvertised until a manual fix/recovery.
    // The counter resets the moment the engine is healthy again, so transient crashes
    // get unlimited future restarts. This is what lets a dead node self-heal.
    const MAX_ENGINE_RESTARTS: u32 = 5;
    let mut unhealthy_streak: u32 = 0;
    let mut restart_attempts: u32 = 0;
    // #159 inference-progress watchdog. `wedge_ms` (set above per model size) is always
    // LONGER than the orchestrator's stuck-job SLA for this model, so a legitimately slow
    // inference is never mistaken for a wedge — only a truly hung slot trips it.
    const MAX_WEDGE_RESTARTS: u32 = 3; // restart up to this many times, THEN abort the process
    let mut wedge_restarts: u32 = 0;
    let mut last_completions: u64 = 0;

    // Empty-completion self-heal (the fast-empty zombie: /health OK but 0-token replies). Unlike
    // /health + wedge, this trip has its OWN budget + cooldown because those reset on /health OK
    // (a zombie passes /health), so reusing them would leave the empty-loop effectively uncapped.
    // Default ON; operators can disable with SGL_EMPTY_RESTART=0. Canary-confirmed before any
    // restart, so it can't be weaponized by crafted client requests.
    const EMPTY_TRIP_THRESHOLD: u32 = 3; // consecutive empty completions before we investigate
    const MAX_EMPTY_RESTARTS: u32 = 5; // cap on empty-triggered restarts within the window below
    const EMPTY_RESTART_MIN_INTERVAL_MS: u64 = 600_000; // >=10 min between empty restarts
    const EMPTY_RESTART_BUDGET_WINDOW_MS: u64 = 3_600_000; // restarts older than 1h don't count
    // Max time to stay parked on ONLY-inconclusive probes before releasing to real traffic. Bounds
    // the alive-but-canary-slow case (which /health can't restart) so the node can't go dark forever.
    const MAX_QUARANTINE_MS: u64 = 300_000; // 5 min
    let empty_restart_enabled = std::env::var("SGL_EMPTY_RESTART")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true);

    // #231: advertise the node's modality once (static per engine). An embedding node reports
    // kind="embedding" + its native dim so the orchestrator routes embeddings vs chat correctly
    // and can surface embeddings separately in /v1/models. Chat nodes send neither (byte-identical).
    let embed_dim: Option<u32> = engine.as_ref().and_then(|e| e.embedding_dim());
    let node_kind: Option<&str> = if embed_dim.is_some() { Some("embedding") } else { None };
    // Vision (multimodal): advertise `vision: true` only when actually serving an mmproj model
    // (chat/embedding nodes omit it) so the orchestrator routes image requests only here.
    let node_vision: Option<bool> =
        engine.as_ref().and_then(|e| if e.is_vision() { Some(true) } else { None });

    loop {
        // ── #159 inference-progress watchdog ──────────────────────────────
        // The /health restart below only fires when llama-server stops answering /health.
        // But a slot can WEDGE while /health still passes — the job hangs, no tokens, no
        // completion (seen live with qwen-7b). Detect it independently: if we are busy
        // (active_jobs>0) yet nothing has started or finished for WEDGE_MS, the engine is
        // stuck → kill+relaunch llama-server to free the slot and abandon the hung job (the
        // orchestrator's stuck-reaper terminalizes it; it never completed so it is never
        // billed). Backstop: if the wedge survives MAX_WEDGE_RESTARTS with no real
        // completion in between, abort the process for a clean OS relaunch (anti-zombie).
        {
            let done = completions.load(Ordering::Relaxed);
            if done > last_completions {
                last_completions = done;
                wedge_restarts = 0; // real progress happened → reset the backstop
            }
            if let Some(ref eng) = engine {
                let busy = active_jobs.load(Ordering::Relaxed) > 0;
                let idle_ms = now_ms().saturating_sub(last_activity.load(Ordering::Relaxed));
                if busy && idle_ms > wedge_ms {
                    wedge_restarts += 1;
                    tracing::error!(
                        "inference WEDGED: busy {}s with no progress — restarting engine (wedge {wedge_restarts}/{MAX_WEDGE_RESTARTS})",
                        idle_ms / 1000
                    );
                    if wedge_restarts > MAX_WEDGE_RESTARTS {
                        tracing::error!("engine still wedged after {MAX_WEDGE_RESTARTS} restarts — aborting for a clean OS relaunch");
                        std::process::abort();
                    }
                    let _ = eng.restart().await;
                    // Abandon the hung job(s): drop them from inflight and release EXACTLY that
                    // many slots (saturating). Their tasks are still awaiting the now-killed
                    // llama-server; when they error out they'll find their id already gone and
                    // skip their own decrement (see maybe_spawn_job) — so no double-release and
                    // no underflow. The orchestrator's stuck-reaper terminalizes the jobs; they
                    // never completed, so they're never billed.
                    let abandoned = {
                        let mut g = inflight.lock().unwrap();
                        let n = g.len() as u32;
                        g.clear();
                        n
                    };
                    if abandoned > 0 {
                        let _ = active_jobs.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
                            Some(c.saturating_sub(abandoned))
                        });
                    }
                    last_activity.store(now_ms(), Ordering::Relaxed);
                }
            }
        }

        // ── empty-completion self-heal (canary-confirmed, sticky quarantine) ──
        // The wedge watchdog above only catches busy+no-progress; the /health path only catches a
        // dead server. A live server that returns FAST empty replies passes both, keeps advertising,
        // and serves empties — the observed zombie. On EMPTY_TRIP_THRESHOLD consecutive empties we
        // QUARANTINE (de-advertise + refuse new jobs) and CANARY-CONFIRM with our own probe; only a
        // confirmed FAST-EMPTY canary restarts the engine (cause-scoped: no Vulkan→CPU swap). This
        // turns an attacker/user-controllable signal into node-controlled ground truth. Quarantine
        // is STICKY: a canary-confirmed-dead engine stays parked (never resumes serving empties)
        // until a canary produces real output — even after the restart budget is exhausted. Budget +
        // cooldown are dedicated (the crash/wedge budgets reset on /health OK → no cap here) and
        // PERSISTED (an in-process abort() would otherwise reset them and abort-loop forever).
        if empty_restart_enabled {
            if let Some(ref eng) = engine {
                // Enter when a fresh streak trips OR we're already quarantined (sticky re-check).
                let tripped = empty_health.take_trip(EMPTY_TRIP_THRESHOLD);
                if tripped || empty_health.is_quarantined() {
                    let now = now_ms();
                    // Park immediately: de-advertise AND make maybe_spawn_job refuse work while we
                    // probe (≤20s), so no request hits a suspect engine during the window. Stamps
                    // the park-start time on first entry (for the max-age escape below).
                    empty_health.enter_quarantine(now);
                    match run_canary(eng).await {
                        CanaryOutcome::Empty => {
                            let prior = empty_health.effective_restarts(now, EMPTY_RESTART_BUDGET_WINDOW_MS);
                            let since = now.saturating_sub(empty_health.last_restart_ms());
                            if prior >= MAX_EMPTY_RESTARTS {
                                // Budget exhausted for this window: the engine is confirmed dead but
                                // restarting isn't fixing it. STAY PARKED (quarantined) — do NOT resume
                                // serving empties — until it recovers on its own or an operator acts.
                                tracing::error!(
                                    "empty-completion engine still dead after {MAX_EMPTY_RESTARTS} restarts this window — staying PARKED (de-advertised); needs operator attention"
                                );
                            } else if prior > 0 && since < EMPTY_RESTART_MIN_INTERVAL_MS {
                                // Confirmed dead but within cooldown: stay parked, don't thrash-restart.
                                tracing::warn!(
                                    "empty-completion engine still dead but within {}s cooldown — staying parked this cycle",
                                    EMPTY_RESTART_MIN_INTERVAL_MS / 1000
                                );
                            } else {
                                tracing::error!(
                                    "canary confirms engine produces nothing — restarting (empty-cause, no variant swap)"
                                );
                                // Persist the restart BEFORE restarting: an in-process restart is abort(),
                                // which never returns, so the budget must already be on disk.
                                empty_health.note_restart(now);
                                let _ = eng.restart_empty().await;
                                // Reuse the wedge path's cleanup EXACTLY: abandon inflight jobs and
                                // release exactly that many slots (saturating). Their tasks error out
                                // against the killed server and skip their own decrement (id already
                                // gone), so no double-release/underflow; the orchestrator reaps them
                                // (never completed → never billed). Stay quarantined — next cycle's
                                // canary confirms recovery before we re-advertise.
                                let abandoned = {
                                    let mut g = inflight.lock().unwrap();
                                    let n = g.len() as u32;
                                    g.clear();
                                    n
                                };
                                if abandoned > 0 {
                                    let _ = active_jobs.fetch_update(
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                        |c| Some(c.saturating_sub(abandoned)),
                                    );
                                }
                                last_activity.store(now_ms(), Ordering::Relaxed);
                            }
                        }
                        CanaryOutcome::NonEmpty => {
                            // Proven healthy → the empties were request-shaped (or transient), not a
                            // fault. Clear quarantine and resume advertising.
                            tracing::info!(
                                "empty-completion canary produced output — engine healthy; clearing quarantine"
                            );
                            empty_health.clear_quarantine();
                        }
                        CanaryOutcome::Inconclusive => {
                            // Busy/hung/transient — neither proven dead nor proven recovered. Do NOT
                            // restart. Normally stay quarantined and re-probe next cycle (parking
                            // sheds new load, so the next canary usually gets through), and a truly
                            // DEAD engine is meanwhile caught by the /health auto-restart which now
                            // always runs. But if we've been parked on ONLY inconclusive probes past
                            // MAX_QUARANTINE_MS (an alive-but-canary-slow engine /health won't
                            // restart), release so real traffic re-tests it instead of going dark.
                            let parked_ms = now.saturating_sub(empty_health.quarantined_since());
                            if parked_ms > MAX_QUARANTINE_MS {
                                tracing::warn!(
                                    "empty-completion quarantine exceeded {}s on inconclusive probes — releasing to let real traffic re-test",
                                    MAX_QUARANTINE_MS / 1000
                                );
                                empty_health.clear_quarantine();
                            } else {
                                tracing::warn!(
                                    "empty-completion canary inconclusive — staying quarantined, re-probing next cycle"
                                );
                            }
                        }
                    }
                }
            }
        }
        let empty_quarantined = empty_health.is_quarantined();

        // Snapshot the jobs we're actually running right now (#119) so the orchestrator can
        // clear any ghost slots. Always sent (even empty) so an idle node frees its slots.
        let active_job_ids: Vec<String> = inflight.lock().unwrap().iter().cloned().collect();

        // /health supervision ALWAYS runs — even while empty-quarantined — so a quarantined engine
        // that then dies outright is still caught by the /health auto-restart (the empty-canary
        // returns Inconclusive on a dead engine and won't restart it, and the wedge watchdog can't
        // fire because quarantine refuses new jobs → never busy). Quarantine only SUPPRESSES the
        // advertisement afterward; it must not disable the crash self-heal.
        let health_advertised: Vec<String> = if let Some(ref eng) = engine {
            if eng.is_healthy().await {
                if unhealthy_streak > 0 {
                    tracing::info!("llama-server healthy again — resuming model advertisement");
                }
                unhealthy_streak = 0;
                restart_attempts = 0; // healthy → future crashes get a fresh restart budget
                models.clone()
            } else {
                unhealthy_streak += 1;
                if unhealthy_streak >= 2 {
                    // Engine is down: stop advertising (grid routes elsewhere) AND try to
                    // self-heal by relaunching llama-server, up to the cap.
                    if restart_attempts < MAX_ENGINE_RESTARTS {
                        restart_attempts += 1;
                        tracing::warn!(
                            "llama-server not responding ({unhealthy_streak} checks) — auto-restarting engine (attempt {restart_attempts}/{MAX_ENGINE_RESTARTS})"
                        );
                        match eng.restart().await {
                            Ok(()) => tracing::info!("engine restarted; re-verifying health next cycle"),
                            Err(e) => tracing::error!("engine restart failed: {e}"),
                        }
                    } else {
                        tracing::error!(
                            "llama-server still down after {MAX_ENGINE_RESTARTS} restarts — staying unadvertised (likely the model can't run on this box; pick a smaller model)"
                        );
                    }
                    Vec::new()
                } else {
                    models.clone() // tolerate one transient blip before pulling the model
                }
            }
        } else {
            models.clone()
        };
        // Empty-suspect (canary running, just restarted, or parked): advertise nothing so the
        // orchestrator routes elsewhere. A zombie passes /health, so this is independent of it —
        // but the health supervision above still ran, so a dead quarantined engine self-heals.
        let advertised: Vec<String> = if empty_quarantined {
            Vec::new()
        } else {
            health_advertised
        };

        match client
            .heartbeat(
                &cfg.node_id,
                &advertised,
                rc.load_factor(),
                Some(&node_enc_pubkey),
                keybind_sig.as_deref(),
                key_version_opt,
                rc.streaming_enabled,
                rc.context_size,
                active_job_ids,
                binary_hash.clone(),
                effective_slots,
                node_kind,
                embed_dim,
                node_vision,
            )
            .await
        {
            Ok(resp) => {
                tracing::debug!("Heartbeat OK — status: {}", resp.status);

                // Handle token rotation
                if let Some(new_token) = &resp.new_auth_token {
                    tracing::info!("Auth token rotated by orchestrator, saving new token...");
                    let mut updated_cfg = cfg.clone();
                    updated_cfg.auth_token = new_token.clone();
                    if let Err(e) = config::save_config(config_dir, &updated_cfg) {
                        tracing::error!("Failed to save rotated token: {e}");
                    } else {
                        client.update_auth_token(new_token.clone());
                        tracing::info!(
                            "New token saved (expires: {})",
                            resp.token_expires_at.as_deref().unwrap_or("unknown")
                        );
                    }
                }

                // Process jobs concurrently (REST fallback path; de-duped against
                // anything already picked up via WS push).
                for job in resp.pending_jobs {
                    maybe_spawn_job(
                        job,
                        Arc::clone(&client),
                        engine.clone(),
                        node_secret,
                        rc.streaming_enabled,
                        Arc::clone(&active_jobs),
                        Arc::clone(&inflight),
                        Arc::clone(&seen_jobs),
                        effective_slots,
                        Arc::clone(&last_activity),
                        Arc::clone(&completions),
                        Arc::clone(&empty_health),
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Heartbeat failed: {e}");
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(rc.heartbeat_interval)).await;
    }
}

pub async fn status(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let keypair = NodeKeypair::load(&config::keypair_path(config_dir))?;

    println!("=== SGL Node Status ===");
    println!("Node ID:    {}", cfg.node_id);
    println!("Wallet:     {}", cfg.wallet_address);
    println!("TEE type:   {}", cfg.tee_type);
    println!("Public key: {}", keypair.public_key_bs58());
    println!("Config:     {}", config::config_path(config_dir).display());
    println!();

    let caps = tee::detect();
    tee::print_capabilities(&caps);
    println!();

    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    match client.get_node_status(&cfg.node_id).await {
        Ok(info) => {
            println!("--- Orchestrator ---");
            println!("Status:       {}", info.status);
            println!("Attested:     {}", info.attestation_verified);
            if let Some(score) = info.reputation_score {
                println!("Reputation:   {:.1}", score);
            }
            if let Some(completed) = info.jobs_completed {
                println!("Jobs done:    {completed}");
            }
            if let Some(failed) = info.jobs_failed {
                println!("Jobs failed:  {failed}");
            }
        }
        Err(e) => {
            println!("Could not reach orchestrator: {e}");
        }
    }

    Ok(())
}

/// Toggle off-grid (maintenance) mode. Off-grid removes the node from job
/// dispatch for planned downtime — no jobs are routed to it and it isn't
/// penalized for being offline. Tamper slashing is unaffected.
pub async fn set_off_grid(config_dir: &Path, orchestrator_url: &str, off_grid: bool) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.set_off_grid(&cfg.node_id, off_grid).await?;
    if off_grid {
        println!("🔌 Node is now OFF-GRID (maintenance).");
        println!("   It won't receive new jobs and won't be penalized for being offline.");
        println!("   Run `sgl on-grid` when you're ready to serve again.");
    } else {
        println!("✅ Node is back ON-GRID — eligible to receive jobs again.");
    }
    Ok(())
}

/// Show this node's per-model pricing (custom vs platform suggested, + the band).
pub async fn show_prices(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    let data = client.get_prices(&cfg.node_id).await?;
    let prices = data.get("prices").and_then(|p| p.as_array()).cloned().unwrap_or_default();
    if prices.is_empty() {
        println!("This node isn't serving any models yet.");
        return Ok(());
    }
    println!("Per-model pricing (USD per 1M tokens):\n");
    for p in prices {
        let model = p.get("model").and_then(|m| m.as_str()).unwrap_or("?");
        let eff = p.get("effective");
        let custom = p.get("custom").map(|c| !c.is_null()).unwrap_or(false);
        let r = p.get("reference");
        let g = |v: Option<&serde_json::Value>, k: &str| v.and_then(|o| o.get(k)).and_then(|n| n.as_f64()).unwrap_or(0.0);
        println!(
            "  {model:<20} in ${:.6} / out ${:.6}  [{}]   (suggested in ${:.6} / out ${:.6})",
            g(eff, "inputPerM"), g(eff, "outputPerM"),
            if custom { "custom" } else { "suggested" },
            g(r, "inputPerM"), g(r, "outputPerM"),
        );
    }
    Ok(())
}

/// Set a custom per-token price for a model (USD per 1M tokens). Band-enforced server-side.
pub async fn set_price(config_dir: &Path, orchestrator_url: &str, model: &str, input_per_m: f64, output_per_m: f64) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.set_price(&cfg.node_id, model, input_per_m, output_per_m).await?;
    println!("✅ Price set for {model}: in ${input_per_m}/1M · out ${output_per_m}/1M. You earn 80% of what you charge.");
    Ok(())
}

/// Reset a model's price back to the platform suggested rate.
pub async fn reset_price(config_dir: &Path, orchestrator_url: &str, model: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.reset_price(&cfg.node_id, model).await?;
    println!("✅ {model} reset to the platform suggested price.");
    Ok(())
}

pub async fn attest(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let keypair = NodeKeypair::load(&config::keypair_path(config_dir))?;

    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));

    tracing::info!("Requesting attestation challenge...");
    let challenge = client.request_challenge(&cfg.node_id).await?;
    let expiry_owned;
    let expiry = match challenge.expires_at.as_deref() {
        Some(at) => at,
        None => {
            expiry_owned = challenge
                .expires_in_seconds
                .map(|s| format!("{s}s"))
                .unwrap_or_else(|| "unknown".to_string());
            &expiry_owned
        }
    };
    tracing::info!("Challenge received (expires: {expiry})");

    // Build the hardware report (TEE type, SIP status, binary self-hash). The
    // orchestrator gates on SIP + binary-hash allowlist before activating.
    let report = crate::tee::generate_attestation_report();
    let report_hash = report.report_hash.clone();
    tracing::info!(
        "Hardware report: sip_enabled={}, binary_hash={}…",
        report.sip_enabled,
        &report.binary_hash[..report.binary_hash.len().min(12)]
    );

    // Sign the plain challenge (proves key ownership). The hardware report is
    // delivered over the authenticated node session and gated server-side.
    let _ = report_hash;
    let signature = keypair.sign_message(challenge.challenge.as_bytes());
    tracing::info!("Challenge signed, submitting with hardware report...");

    // Derive the node's X25519 encryption key (for E2E-encrypted prompts) from
    // the same ed25519 seed and publish it during attestation.
    let enc_keypair =
        crate::encryption::EncryptionKeypair::from_ed25519_seed(&keypair.signing_key.to_bytes());
    let encryption_public_key = enc_keypair.public_key_bs58();
    tracing::info!("Publishing X25519 encryption key: {encryption_public_key}");

    let report_json = serde_json::to_value(&report).ok();
    let result = client
        .verify_attestation(
            &cfg.node_id,
            &signature,
            Some(encryption_public_key),
            report_json,
        )
        .await?;

    if result.verified {
        println!("✅ Attestation verified — node status: {}", result.status);
    } else {
        return Err("Attestation verification failed".to_string());
    }

    Ok(())
}

async fn process_job(
    client: &OrchestratorClient,
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    node_secret: &[u8; 32],
    streaming_enabled: bool,
    empty_health: &EmptyHealth,
) {
    tracing::info!("Processing job {} (type: {})", job.id, job.job_type);

    // If the prompt is sealed (E2E), decrypt it with the node's X25519 key and
    // remember the caller's response key so we can seal the reply back.
    let mut response_pubkey: Option<[u8; 32]> = None;
    let mut enc_version = crate::encryption::EncVersion::V1;
    let mut effective_job = job.clone();
    if let Some(payload) = &job.input_payload {
        match crate::encryption::unseal_input(payload, node_secret) {
            Ok((inner, resp, version)) => {
                response_pubkey = resp;
                enc_version = version;
                if resp.is_some() {
                    effective_job.input_payload = Some(inner);
                }
            }
            Err(e) => {
                tracing::error!("Failed to unseal job {}: {e}", job.id);
                let _ = client
                    .fail_job(&job.id, &format!("decrypt failed: {e}"))
                    .await;
                return;
            }
        }
    }

    // Streaming path requires three independent agreements, so a single party
    // can't force it:
    //   1. this node has streaming enabled locally (`streaming_enabled`)
    //   2. the orchestrator set the cleartext dispatch marker (it set up the SSE
    //      relay) — read from the ORIGINAL job payload, alongside `enc`
    //   3. the client asked for it in the AUTHENTICATED, sealed payload — read from
    //      the DECRYPTED inner payload (a relay can't flip this)
    let dispatch_stream = job
        .input_payload
        .as_ref()
        .and_then(|p| p.get("stream"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let sealed_stream = effective_job
        .input_payload
        .as_ref()
        .and_then(|p| p.get("stream"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if streaming_enabled
        && dispatch_stream
        && sealed_stream
        && enc_version == crate::encryption::EncVersion::V2
        && effective_job.job_type == "inference"
    {
        if let Some(resp_pub) = response_pubkey {
            process_inference_stream(client, engine, &effective_job, node_secret, &resp_pub, empty_health).await;
            return;
        }
    }

    let result = match effective_job.job_type.as_str() {
        "inference" => execute_inference(engine, &effective_job, empty_health).await,
        "embedding" => execute_embedding(engine, &effective_job).await,
        _ => {
            tracing::warn!("Unsupported job type: {}", effective_job.job_type);
            Err(format!("Unsupported job type: {}", effective_job.job_type))
        }
    };

    match result {
        Ok(output) => {
            if let Some(resp_pub) = response_pubkey {
                // ── E2E: seal the result to the caller's response key ──
                let result_bytes = output.to_string();
                let usage = output.get("usage").cloned();
                // Reply in the SAME version the caller used (v2 = HKDF + AAD).
                let sealed = if enc_version == crate::encryption::EncVersion::V2 {
                    crate::encryption::encrypt_for_recipient_v2(&resp_pub, result_bytes.as_bytes())
                } else {
                    crate::encryption::encrypt_for_recipient(&resp_pub, result_bytes.as_bytes())
                };
                let algo = if enc_version == crate::encryption::EncVersion::V2 {
                    crate::encryption::ALGO_V2
                } else {
                    "x25519-xchacha20poly1305"
                };
                match sealed {
                    Ok((sealed, ephemeral_pub)) => {
                        // base64 (linear-time), NOT base58 (O(n^2) big-int math): an embedding
                        // batch result is ~100KB of ciphertext, which base58 took ~13s to encode
                        // and the orchestrator another ~13s to decode — dwarfing the ~100ms
                        // compute and 503-ing large batches. `encoding` tells decoders which
                        // alphabet this envelope uses; old envelopes (no field) remain base58.
                        // The envelope signature covers the encoded string verbatim, so the
                        // orchestrator's verify (hash of the received string) is unchanged.
                        use base64::Engine as _;
                        let ciphertext_b64 =
                            base64::engine::general_purpose::STANDARD.encode(&sealed);
                        // Sign an envelope over the *public* ciphertext + job id so the
                        // orchestrator can prove which node produced this result for this
                        // job (anti-replay) without ever seeing the plaintext.
                        let env_sig = crate::crypto::sign_result_envelope(
                            node_secret,
                            &job.id,
                            "sealed",
                            ciphertext_b64.as_bytes(),
                        );
                        let sealed_result = serde_json::json!({
                            "ciphertext": ciphertext_b64,
                            "encoding": "base64",
                            "ephemeral_public_key": bs58::encode(ephemeral_pub).into_string(),
                            "algorithm": algo,
                        });
                        if let Err(e) = client
                            .complete_job_sealed(&job.id, sealed_result, usage, Some(env_sig))
                            .await
                        {
                            tracing::error!("Failed to report sealed completion: {e}");
                        } else {
                            tracing::info!("Job {} completed (E2E sealed, REST)", job.id);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to seal result for job {}: {e}", job.id);
                        let _ = client.fail_job(&job.id, &format!("seal failed: {e}")).await;
                    }
                }
            } else {
                let result_str = output.to_string();
                let env_sig = crate::crypto::sign_result_envelope(
                    node_secret,
                    &job.id,
                    "plain",
                    result_str.as_bytes(),
                );
                if let Err(e) = client.complete_job(&job.id, &output, Some(env_sig)).await {
                    tracing::error!("Failed to report job completion: {e}");
                } else {
                    tracing::info!("Job {} completed", job.id);
                }
            }
        }
        Err(reason) => {
            if let Err(e) = client.fail_job(&job.id, &reason).await {
                tracing::error!("Failed to report job failure: {e}");
            } else {
                tracing::warn!("Job {} failed: {reason}", job.id);
            }
        }
    }
}

/// Parse + bound the inference parameters from a (decrypted) job payload. Shared
/// by the non-streaming and streaming paths so both apply identical validation.
/// Validated inference request pulled off a job payload. `messages` is kept as opaque JSON so
/// tool-calling round-trips survive verbatim (assistant `tool_calls`, `tool`-role messages,
/// null content); `tools`/`tool_choice` are forwarded to llama-server (which needs `--jinja`).
struct InferenceParams {
    messages: serde_json::Value,
    temperature: f64,
    max_tokens: i32,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
}

fn parse_inference_params(
    payload: Option<&serde_json::Value>,
) -> Result<InferenceParams, String> {
    let payload = payload.ok_or("Job has no input payload")?;

    // Forward messages opaquely (don't destructure to {role,content} — that would drop tool
    // fields on agent-loop turns). Accept a `prompt` string as a convenience shorthand.
    let messages: serde_json::Value = if let Some(msgs) = payload.get("messages") {
        if !msgs.is_array() {
            return Err("'messages' must be an array".to_string());
        }
        msgs.clone()
    } else if let Some(prompt) = payload.get("prompt").and_then(|p| p.as_str()) {
        serde_json::json!([{ "role": "user", "content": prompt }])
    } else {
        return Err("Payload must contain 'messages' array or 'prompt' string".to_string());
    };

    // Bound untrusted input before handing it to the inference server.
    const MAX_MESSAGES: usize = 256;
    const MAX_TOTAL_PROMPT_BYTES: usize = 4 * 1024 * 1024; // 4 MiB
    let msg_count = messages.as_array().map(|a| a.len()).unwrap_or(0);
    if msg_count == 0 {
        return Err("'messages' must not be empty".to_string());
    }
    if msg_count > MAX_MESSAGES {
        return Err(format!("too many messages ({msg_count} > {MAX_MESSAGES})"));
    }
    // Serialized size is a safe upper bound on the prompt bytes (incl. tool_calls / tool results).
    let total_bytes = messages.to_string().len();
    if total_bytes > MAX_TOTAL_PROMPT_BYTES {
        return Err(format!(
            "prompt too large ({total_bytes} bytes > {MAX_TOTAL_PROMPT_BYTES})"
        ));
    }

    let temperature = payload
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.7)
        .clamp(0.0, 2.0);

    let max_tokens = payload
        .get("max_tokens")
        .and_then(|t| t.as_i64())
        .unwrap_or(2048)
        .clamp(1, 8192) as i32;

    // Tool-calling passthrough: forward `tools` only when it's a non-empty array (ignore junk so
    // we never confuse the chat template), and `tool_choice` only alongside it.
    let tools = payload
        .get("tools")
        .filter(|t| t.as_array().map(|a| !a.is_empty()).unwrap_or(false))
        .cloned();
    if let Some(t) = &tools {
        // Bound the tool schema independently of the prompt — an unbounded `tools` array would
        // otherwise sail past the message-size guard above and OOM the server. Tool defs are
        // small in practice; 512 KiB / 128 tools is generous.
        const MAX_TOOLS: usize = 128;
        const MAX_TOOLS_BYTES: usize = 512 * 1024;
        let n = t.as_array().map(|a| a.len()).unwrap_or(0);
        if n > MAX_TOOLS {
            return Err(format!("too many tools ({n} > {MAX_TOOLS})"));
        }
        let bytes = t.to_string().len();
        if bytes > MAX_TOOLS_BYTES {
            return Err(format!("tools schema too large ({bytes} bytes > {MAX_TOOLS_BYTES})"));
        }
    }
    let tool_choice = if tools.is_some() {
        payload.get("tool_choice").cloned()
    } else {
        None
    };

    Ok(InferenceParams {
        messages,
        temperature,
        max_tokens,
        tools,
        tool_choice,
    })
}

/// Fallback tool-call extraction. Some models (verified live: Qwen2.5-Coder) wrap tool calls in
/// nonstandard tags — `<function-call>`, `<function_call>`, `<xml>`, ```json fences — that
/// llama.cpp's parser doesn't recognise (it extracts only `<tool_call>`), so the call arrives as
/// TEXT in `content` and agentic clients (OpenCode/Cline) see "no tool call" and stall
/// (upstream: ggml-org/llama.cpp#12279). When tools were requested and no structured tool_calls
/// came back, scan the text for `{"name": ..., "arguments": ...}` objects whose name matches a
/// REQUESTED tool (never invent calls) and synthesize the OpenAI tool_calls array.
fn extract_text_tool_calls(
    content: &str,
    allowed: &[String],
    job_id: &str,
) -> Option<serde_json::Value> {
    if allowed.is_empty() || content.is_empty() {
        return None;
    }
    let id8 = job_id.get(..8).unwrap_or(job_id);
    let mut calls: Vec<serde_json::Value> = Vec::new();
    let mut from = 0usize;
    // Walk every '{' and try to parse a JSON value there; serde stops at the value's end, so
    // surrounding tags/fences/prose don't matter. '{' is ASCII → indices stay on char bounds.
    while let Some(rel) = content[from..].find('{') {
        let start = from + rel;
        let mut iter =
            serde_json::Deserializer::from_str(&content[start..]).into_iter::<serde_json::Value>();
        match iter.next() {
            Some(Ok(v)) => {
                let consumed = iter.byte_offset().max(1);
                let name = v.get("name").and_then(|n| n.as_str());
                let args = v.get("arguments").or_else(|| v.get("parameters"));
                if let (Some(name), Some(args)) = (name, args) {
                    if allowed.iter().any(|a| a == name) {
                        // OpenAI shape: arguments is a STRING of JSON.
                        let args_str = match args.as_str() {
                            Some(s) => s.to_string(),
                            None => args.to_string(),
                        };
                        calls.push(serde_json::json!({
                            "id": format!("call_{id8}_{}", calls.len()),
                            "type": "function",
                            "function": { "name": name, "arguments": args_str },
                        }));
                    }
                }
                from = start + consumed;
            }
            _ => {
                from = start + 1;
            }
        }
    }
    // Fallback B — XML-attribute style: `<function name="X" arguments='{...}' />`.
    // Under tool_choice:auto (no forced grammar), some Qwen-Coder builds emit the call as this
    // tag instead of a JSON `{"name":..,"arguments":..}` block, so Fallback A above misses it
    // (the only `{...}` present is the arguments value, which has no `name` key). This is what
    // OpenCode hit. Only runs when the JSON walk found nothing, and still gated to allowed names.
    if calls.is_empty() {
        let mut fpos = 0usize;
        while let Some(rel) = content[fpos..].find("<function") {
            let tstart = fpos + rel;
            let tend = content[tstart..]
                .find('>')
                .map(|e| tstart + e + 1)
                .unwrap_or(content.len());
            let tag = &content[tstart..tend];
            if let (Some(name), Some(args)) = (
                xml_attr(tag, "name"),
                xml_attr(tag, "arguments").or_else(|| xml_attr(tag, "parameters")),
            ) {
                if allowed.iter().any(|a| a == name) {
                    // arguments is a JSON string; keep verbatim if it parses, else JSON-encode it.
                    let args_str = if serde_json::from_str::<serde_json::Value>(args).is_ok() {
                        args.to_string()
                    } else {
                        serde_json::Value::String(args.to_string()).to_string()
                    };
                    calls.push(serde_json::json!({
                        "id": format!("call_{id8}_{}", calls.len()),
                        "type": "function",
                        "function": { "name": name, "arguments": args_str },
                    }));
                }
            }
            fpos = tend;
        }
    }
    if calls.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(calls))
    }
}

/// Extract an XML attribute value (`attr="..."` or `attr='...'`) from a tag slice. Returns the
/// raw inner string. Tolerant of surrounding whitespace around `=`. ASCII quotes only.
fn xml_attr<'a>(tag: &'a str, attr: &str) -> Option<&'a str> {
    let mut search = 0usize;
    while let Some(rel) = tag[search..].find(attr) {
        let i = search + rel;
        let after = tag[i + attr.len()..].trim_start();
        if let Some(rest) = after.strip_prefix('=') {
            let rest = rest.trim_start();
            let q = rest.chars().next();
            if q == Some('"') || q == Some('\'') {
                let quote = q.unwrap();
                if let Some(end) = rest[1..].find(quote) {
                    return Some(&rest[1..1 + end]);
                }
            }
        }
        search = i + attr.len();
    }
    None
}

async fn execute_inference(
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    empty_health: &EmptyHealth,
) -> Result<serde_json::Value, String> {
    let engine = engine
        .as_ref()
        .ok_or("No inference engine configured — start with --model-path")?;

    let p = parse_inference_params(job.input_payload.as_ref())?;
    // Whether THIS request should have produced output — captured before `p.messages` is moved
    // into the engine call, for the empty-completion health signal below.
    let expects_output = request_expects_output(&p.messages, p.max_tokens);
    // Names of the requested tools — kept for the fallback extractor (only synthesizes calls to
    // tools the CLIENT asked for; a hallucinated name never becomes a tool_call).
    let tool_names: Vec<String> = p
        .tools
        .as_ref()
        .and_then(|t| t.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default();
    let tools_requested = p.tools.is_some();

    let result = engine
        .chat_completion(p.messages, p.temperature, p.max_tokens, p.tools, p.tool_choice)
        .await?;

    let homura_model = is_homura_model(job);
    let mut content = result.content;
    let mut tool_calls = result.tool_calls;
    let mut finish_reason = result.finish_reason;
    if homura_model {
        content = strip_homura_message_envelope(&content);
    }
    // Empty-completion health signal (non-streaming). Content-emptiness is the signal — NOT
    // completion_tokens, which some server builds omit (0 tokens with real content) and which a
    // 1-EOS-token zombie would sneak past. Exempt only a genuine tool-call turn (its content is
    // "" by design — inference.rs normalises it) and, via `expects_output`, empty-prompt /
    // max_tokens<=1. A request that merely *offers* tools but got neither content nor a tool_call
    // IS a zombie (agent clients always send tools — they must not be blanket-exempt). For HOMURA,
    // check AFTER protocol cleanup so a reply containing only `to=user<|message|><|eot|>` cannot
    // become a billed blank success.
    // An EXEMPT request (empty prompt / max_tokens<=1) is a no-op: it proves nothing about engine
    // health, so it must neither trip nor RESET the streak (else a client could indefinitely defer
    // detection by interleaving max_tokens:1 health-check probes with real traffic).
    if !expects_output {
        // exempt — leave the streak untouched
    } else if tool_calls.is_none() && content.trim().is_empty() {
        empty_health.record_empty();
        if homura_model {
            return Err("Inference produced no user-visible output after HOMURA protocol cleanup".to_string());
        }
    } else {
        empty_health.record_ok();
    }
    // Tolerant fallback: model emitted the tool call as text → synthesize structured tool_calls.
    if tools_requested && tool_calls.is_none() {
        if let Some(tc) = extract_text_tool_calls(&content, &tool_names, &job.id) {
            tracing::info!("job {}: extracted tool_calls from text reply (nonstandard format)", job.id);
            tool_calls = Some(tc);
            finish_reason = Some("tool_calls".to_string());
            // OpenAI semantics: a tool-call turn carries no user-facing content.
            content = String::new();
        }
    }

    let mut out = serde_json::json!({
        "content": content,
        "model": result.model,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
    });
    // Surface tool-calling fields only when present (keeps plain-chat replies byte-identical).
    if let Some(tc) = tool_calls {
        out["tool_calls"] = tc;
    }
    if let Some(fr) = finish_reason {
        out["finish_reason"] = serde_json::Value::String(fr);
    }
    Ok(out)
}

/// Execute an embedding job (#231). Parses the (decrypted) payload's `input` (string | string[]),
/// optional `input_type` (query/document) + `dimensions` (Matryoshka), runs the dedicated
/// embedding engine, and returns an OpenAI-compatible embeddings response. The orchestrator
/// structurally validates every vector (dim/finite/non-zero) before it bills, so a wrong shape
/// here fails the job (input-only billing → nothing charged), never a silently-wrong embedding.
async fn execute_embedding(
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
) -> Result<serde_json::Value, String> {
    {
        let engine = engine
            .as_ref()
            .ok_or("No embedding engine configured — start with an embedding --model-path")?;
        if !engine.is_embedding() {
            return Err("this node is not an embedding node".to_string());
        }
        let payload = job.input_payload.as_ref().ok_or("Job has no input payload")?;

        // `input`: string | string[] (OpenAI shape). Reject anything else.
        let inputs: Vec<String> = match payload.get("input") {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => {
                let mut v = Vec::with_capacity(a.len());
                for item in a {
                    let s = item
                        .as_str()
                        .ok_or("'input' array must contain only strings")?;
                    v.push(s.to_string());
                }
                v
            }
            _ => return Err("'input' must be a string or an array of strings".to_string()),
        };
        if inputs.is_empty() {
            return Err("'input' must not be empty".to_string());
        }

        let input_type = match payload.get("input_type").and_then(|v| v.as_str()) {
            Some("query") => crate::embed_catalog::InputType::Query,
            Some("document") => crate::embed_catalog::InputType::Document,
            _ => crate::embed_catalog::InputType::Unspecified,
        };
        // `dimensions` (Matryoshka): reject a present-but-invalid value rather than truncating a
        // huge JSON number via `as u32` (which would wrap, Codex #2). Absent = native dim.
        let dimensions = match payload.get("dimensions") {
            None => None,
            Some(v) => match v.as_u64().and_then(|d| u32::try_from(d).ok()) {
                Some(d) => Some(d),
                None => return Err("'dimensions' must be a positive 32-bit integer".to_string()),
            },
        };

        let out = engine.embed(inputs, input_type, dimensions).await?;

        let data: Vec<serde_json::Value> = out
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| serde_json::json!({ "object": "embedding", "index": i, "embedding": v }))
            .collect();

        Ok(serde_json::json!({
            "object": "list",
            "data": data,
            "model": job.model.clone().unwrap_or_default(),
            // Input-only billing: total_tokens == prompt_tokens (no output). The orchestrator
            // caps the billed count at its pre-quoted upper bound regardless.
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "total_tokens": out.prompt_tokens,
            }
        }))
    }
}

#[cfg(test)]
mod tool_extract_tests {
    use super::extract_text_tool_calls;

    fn allowed() -> Vec<String> {
        vec!["get_weather".to_string()]
    }

    #[test]
    fn extracts_from_function_call_tags() {
        let c = "<function-call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</function-call>";
        let tc = extract_text_tool_calls(c, &allowed(), "jobid123").unwrap();
        assert_eq!(tc[0]["function"]["name"], "get_weather");
        assert!(tc[0]["function"]["arguments"].as_str().unwrap().contains("Tokyo"));
    }

    #[test]
    fn extracts_from_xml_and_fences() {
        for c in [
            "<xml>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</xml>",
            "```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```",
            "{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Paris\\\"}\"}",
        ] {
            let tc = extract_text_tool_calls(c, &allowed(), "jobid123").unwrap();
            assert_eq!(tc.as_array().unwrap().len(), 1, "case: {c}");
        }
    }

    #[test]
    fn ignores_unrequested_tool_names_and_plain_text() {
        assert!(extract_text_tool_calls(
            "{\"name\": \"rm_rf\", \"arguments\": {}}",
            &allowed(),
            "j"
        )
        .is_none());
        assert!(extract_text_tool_calls("Here is code: `{}` and {\"a\":1}", &allowed(), "j").is_none());
        assert!(extract_text_tool_calls("no braces at all", &allowed(), "j").is_none());
    }

    #[test]
    fn extracts_xml_attribute_function_tags() {
        // The exact shape Qwen-Coder emitted under tool_choice:auto (OpenCode's default).
        // Real emitted shape: double-quoted name, single-quoted JSON arguments (JSON needs "").
        for c in [
            "<function name=\"get_weather\" arguments='{\"city\": \"Tokyo\"}' />",
            "<function name='get_weather' arguments='{\"city\": \"Tokyo\"}'>",
            "Sure, I'll check.\n<function name=\"get_weather\" arguments='{\"city\":\"Tokyo\"}'/>",
        ] {
            let tc = extract_text_tool_calls(c, &allowed(), "jobid123").unwrap();
            assert_eq!(tc.as_array().unwrap().len(), 1, "case: {c}");
            assert_eq!(tc[0]["function"]["name"], "get_weather", "case: {c}");
            assert!(
                tc[0]["function"]["arguments"].as_str().unwrap().contains("Tokyo"),
                "case: {c}"
            );
        }
    }

    #[test]
    fn xml_tags_still_gate_on_allowed_names() {
        // An unrequested tool in XML form is never synthesized.
        assert!(extract_text_tool_calls(
            "<function name=\"rm_rf\" arguments='{}' />",
            &allowed(),
            "j"
        )
        .is_none());
    }
}

#[cfg(test)]
mod homura_normalizer_tests {
    use super::{strip_homura_message_envelope, HomuraStreamCleaner};

    #[test]
    fn strips_plain_homura_message_envelope() {
        assert_eq!(
            strip_homura_message_envelope(
                "to=user<|message|>Hey fam! What's spinning in your head lately?<|eot|>",
            ),
            "Hey fam! What's spinning in your head lately?"
        );
    }

    #[test]
    fn strips_start_and_channel_envelope() {
        assert_eq!(
            strip_homura_message_envelope(
                "<|start|>assistant to=user <|channel|>final <|message|>Hi there<|end|>",
            ),
            "Hi there"
        );
    }

    #[test]
    fn leaves_normal_text_untouched() {
        let text = "Here is a literal token later: to=user<|message|>, not an envelope.";
        assert_eq!(strip_homura_message_envelope(text), text);
    }

    #[test]
    fn stream_cleaner_strips_split_prefix_and_terminal() {
        let mut c = HomuraStreamCleaner::new();
        assert_eq!(c.push("to=us"), "");
        assert_eq!(c.push("er<|mess"), "");
        assert_eq!(c.push("age|>Hey fam"), "Hey fam");
        assert_eq!(c.push("! What's up?<|e"), "! What's up?");
        assert_eq!(c.push("ot|> ignored"), "");
        assert_eq!(c.finish(), "");
    }

    #[test]
    fn stream_cleaner_passes_non_enveloped_text() {
        let mut c = HomuraStreamCleaner::new();
        assert_eq!(c.push("Hello there"), "Hello there");
        assert_eq!(c.finish(), "");
    }

    #[test]
    fn stream_cleaner_drops_partial_terminal_on_finish() {
        let mut c = HomuraStreamCleaner::new();
        assert_eq!(c.push("to=user<|message|>Hello<|e"), "Hello");
        assert_eq!(c.finish(), "");
    }
}

#[cfg(test)]
mod empty_health_tests {
    use super::{request_expects_output, EmptyHealth};
    use serde_json::json;

    // ── request_expects_output: the false-positive gate ──────────────────

    #[test]
    fn normal_prompt_expects_output() {
        let msgs = json!([{ "role": "user", "content": "Write a haiku about the sea." }]);
        assert!(request_expects_output(&msgs, 512));
    }

    #[test]
    fn max_tokens_one_is_exempt() {
        // A 1-token cap can legitimately produce an empty/EOS reply — never a zombie signal.
        let msgs = json!([{ "role": "user", "content": "hello" }]);
        assert!(!request_expects_output(&msgs, 1));
    }

    #[test]
    fn whitespace_or_empty_prompt_is_exempt() {
        assert!(!request_expects_output(&json!([{ "role": "user", "content": "   " }]), 512));
        assert!(!request_expects_output(&json!([{ "role": "user", "content": "" }]), 512));
        // No string content at all (e.g. only a tool/assistant scaffold) → exempt.
        assert!(!request_expects_output(&json!([{ "role": "assistant", "content": null }]), 512));
    }

    #[test]
    fn any_message_with_text_counts() {
        // A system+user pair where the user carries the real ask.
        let msgs = json!([
            { "role": "system", "content": "" },
            { "role": "user", "content": "Explain TCP." }
        ]);
        assert!(request_expects_output(&msgs, 512));
    }

    #[test]
    fn array_of_parts_content_is_recognized() {
        // OpenAI multimodal shape: content is an array of typed parts. A text part with real text
        // must count as expecting output (llama-server serves these; they must not be blanket-exempt).
        let msgs = json!([{ "role": "user", "content": [{ "type": "text", "text": "Describe this." }] }]);
        assert!(request_expects_output(&msgs, 512));
        // An array with only empty text is exempt.
        let empty = json!([{ "role": "user", "content": [{ "type": "text", "text": "  " }] }]);
        assert!(!request_expects_output(&empty, 512));
    }

    // ── EmptyHealth state machine ────────────────────────────────────────

    #[test]
    fn streak_advances_and_resets() {
        let h = EmptyHealth::new_for_test();
        assert_eq!(h.consecutive(), 0);
        h.record_empty();
        h.record_empty();
        assert_eq!(h.consecutive(), 2);
        // A single good (or exempt-legit-empty) reply clears the streak — no false trip on
        // alternating empty/non-empty traffic.
        h.record_ok();
        assert_eq!(h.consecutive(), 0);
        h.record_empty();
        assert_eq!(h.consecutive(), 1);
    }

    #[test]
    fn take_trip_consumes_only_at_threshold() {
        let h = EmptyHealth::new_for_test();
        h.record_empty();
        h.record_empty();
        // Below threshold: no trip, streak preserved.
        assert!(!h.take_trip(3));
        assert_eq!(h.consecutive(), 2);
        h.record_empty();
        assert_eq!(h.consecutive(), 3);
        // At threshold: trips once and clears the streak.
        assert!(h.take_trip(3));
        assert_eq!(h.consecutive(), 0);
        // Restart budget is independent and starts empty.
        assert_eq!(h.restarts(), 0);
        assert_eq!(h.last_restart_ms(), 0);
    }

    #[test]
    fn quarantine_flag_toggles_and_stamps_entry_once() {
        let h = EmptyHealth::new_for_test();
        assert!(!h.is_quarantined());
        assert_eq!(h.quarantined_since(), 0);
        h.enter_quarantine(5_000);
        assert!(h.is_quarantined());
        assert_eq!(h.quarantined_since(), 5_000);
        // Re-entering while already quarantined must NOT re-stamp — the max-age is measured from
        // the first entry, not each re-probe cycle.
        h.enter_quarantine(9_000);
        assert_eq!(h.quarantined_since(), 5_000);
        h.clear_quarantine();
        assert!(!h.is_quarantined());
        assert_eq!(h.quarantined_since(), 0);
        // A fresh entry after clearing stamps the new time.
        h.enter_quarantine(12_000);
        assert_eq!(h.quarantined_since(), 12_000);
    }

    #[test]
    fn note_restart_tracks_budget_and_cooldown_stamp() {
        let h = EmptyHealth::new_for_test();
        h.note_restart(1000);
        assert_eq!(h.restarts(), 1);
        assert_eq!(h.last_restart_ms(), 1000);
        h.note_restart(700_000);
        assert_eq!(h.restarts(), 2);
        assert_eq!(h.last_restart_ms(), 700_000);
    }

    #[test]
    fn effective_restarts_decays_after_window() {
        let h = EmptyHealth::new_for_test();
        h.note_restart(1_000_000);
        assert_eq!(h.restarts(), 1);
        // Within window: still counts.
        assert_eq!(h.effective_restarts(1_500_000, 3_600_000), 1);
        // Past the window: decays to 0 (and clears the persisted count).
        assert_eq!(h.effective_restarts(1_000_000 + 3_600_001, 3_600_000), 0);
        assert_eq!(h.restarts(), 0);
    }
}

/// Streaming inference: run llama-server with streaming, seal each token batch as
/// an ordered chunk, and POST it to the orchestrator (which relays it to the
/// client over SSE). Each chunk's AAD binds its seq + final flag; the node also
/// signs an envelope per chunk so the orchestrator can attribute it. Billing
/// happens on the final chunk's usage; if the client aborts, chunk POSTs start
/// failing and we stop early (no final → no charge).
/// Seal one stream chunk, sign its envelope, and POST it. Returns `Ok(true)` if
/// the orchestrator reports the client is gone (stop early), `Ok(false)` to keep
/// going, `Err` on a hard failure.
#[allow(clippy::too_many_arguments)]
async fn seal_post_chunk(
    client: &OrchestratorClient,
    sealer: &crate::encryption::StreamSealer,
    node_secret: &[u8; 32],
    job_id: &str,
    eph_b58: &str,
    seq: u64,
    is_final: bool,
    plaintext: &[u8],
    usage: Option<serde_json::Value>,
) -> Result<bool, String> {
    let ct = sealer.seal_chunk(plaintext, seq, is_final)?;
    let kind = format!("stream:{seq}:{}", if is_final { 1 } else { 0 });
    let sig = crate::crypto::sign_result_envelope(node_secret, job_id, &kind, ct.as_bytes());
    let eph = if seq == 0 { Some(eph_b58) } else { None };
    client
        .post_chunk(job_id, seq, is_final, eph, &ct, usage, Some(sig))
        .await
}

async fn process_inference_stream(
    client: &OrchestratorClient,
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    node_secret: &[u8; 32],
    resp_pub: &[u8; 32],
    empty_health: &EmptyHealth,
) {
    let engine = match engine {
        Some(e) => e.clone(),
        None => {
            let _ = client
                .fail_job(&job.id, "No inference engine configured")
                .await;
            return;
        }
    };

    let p = match parse_inference_params(job.input_payload.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            let _ = client.fail_job(&job.id, &e).await;
            return;
        }
    };
    let (temperature, max_tokens) = (p.temperature, p.max_tokens);
    // Whether THIS request should have produced output — captured before `p.messages` is moved,
    // for the empty-completion health signal recorded at the terminal Done chunk below.
    let expects_output = request_expects_output(&p.messages, p.max_tokens);
    // Streaming is chat-only in v1 (tool-calling uses the non-streaming path), so render the
    // opaque messages into the {role,content} shape the stream engine consumes.
    let messages: Vec<ChatMessage> = match serde_json::from_value(p.messages) {
        Ok(m) => m,
        Err(e) => {
            let _ = client
                .fail_job(&job.id, &format!("Invalid messages format: {e}"))
                .await;
            return;
        }
    };

    // Per-request nonce chosen by the client (inside the sealed prompt) — bound
    // into every chunk's AAD so a stream can't be spliced into another request.
    let req_nonce = job
        .input_payload
        .as_ref()
        .and_then(|p| p.get("nonce"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let sealer = match crate::encryption::StreamSealer::new(resp_pub, req_nonce) {
        Ok(s) => s,
        Err(e) => {
            let _ = client
                .fail_job(&job.id, &format!("stream seal init failed: {e}"))
                .await;
            return;
        }
    };
    let eph_b58 = sealer.ephemeral_public_b58().to_string();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::inference::StreamEvent>(64);
    let engine2 = engine.clone();
    let inf = tokio::spawn(async move {
        engine2
            .chat_completion_stream(messages, temperature, max_tokens, tx)
            .await
    });

    let mut seq: u64 = 0;
    let mut emitted_tokens: u32 = 0;
    // Did the stream emit any NON-whitespace content? The Server engine counts a whitespace-only
    // delta (e.g. "\n") toward emitted_tokens, so emitted_tokens>0 alone wouldn't catch a
    // whitespace-only zombie; track real content separately for the empty-completion signal.
    let mut emitted_nonws = false;
    let mut final_sent = false;
    let mut client_gone = false;
    let mut homura_cleaner = is_homura_model(job).then(HomuraStreamCleaner::new);
    let mut stream_failure_reason: Option<String> = None;

    while let Some(ev) = rx.recv().await {
        match ev {
            crate::inference::StreamEvent::Delta { text, tokens } => {
                emitted_tokens = emitted_tokens.saturating_add(tokens);
                let out_text = if let Some(cleaner) = homura_cleaner.as_mut() {
                    cleaner.push(&text)
                } else {
                    text
                };
                if out_text.is_empty() {
                    continue;
                }
                if !out_text.trim().is_empty() {
                    emitted_nonws = true;
                }
                match seal_post_chunk(
                    client, &sealer, node_secret, &job.id, &eph_b58, seq, false,
                    out_text.as_bytes(), None,
                )
                .await
                {
                    Ok(false) => seq += 1,
                    Ok(true) => {
                        client_gone = true;
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("Job {} chunk {seq} post failed: {e}", job.id);
                        client_gone = true;
                        break;
                    }
                }
            }
            crate::inference::StreamEvent::Done {
                prompt_tokens,
                completion_tokens,
            } => {
                if let Some(cleaner) = homura_cleaner.as_mut() {
                    let tail = cleaner.finish();
                    if !tail.is_empty() {
                        if !tail.trim().is_empty() {
                            emitted_nonws = true;
                        }
                        match seal_post_chunk(
                            client, &sealer, node_secret, &job.id, &eph_b58, seq, false,
                            tail.as_bytes(), None,
                        )
                        .await
                        {
                            Ok(false) => seq += 1,
                            Ok(true) => {
                                client_gone = true;
                                break;
                            }
                            Err(e) => {
                                tracing::warn!("Job {} cleanup tail chunk post failed: {e}", job.id);
                                client_gone = true;
                                break;
                            }
                        }
                    }
                }
                if homura_cleaner.is_some() && expects_output && !emitted_nonws {
                    empty_health.record_empty();
                    stream_failure_reason = Some(
                        "Inference produced no user-visible output after HOMURA protocol cleanup"
                            .to_string(),
                    );
                    break;
                }
                let usage = serde_json::json!({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                });
                // Only treat as success if the final chunk was actually accepted;
                // otherwise fall through to the failure path below.
                match seal_post_chunk(
                    client, &sealer, node_secret, &job.id, &eph_b58, seq, true, b"", Some(usage),
                )
                .await
                {
                    Ok(_) => {
                        final_sent = true;
                        // Empty-completion health signal (streaming). `emitted_nonws` is true iff any
                        // non-whitespace content was streamed — the streaming twin of content
                        // emptiness, independent of whether the build reported usage. Only counted
                        // on an accepted final (this arm); the client-gone/partial and stream-failure
                        // paths below are NOT counted (already correct-by-construction). An EXEMPT
                        // request is a no-op — neither trips nor resets (see the non-stream site).
                        if !expects_output {
                            // exempt — leave the streak untouched
                        } else if !emitted_nonws {
                            empty_health.record_empty();
                        } else {
                            empty_health.record_ok();
                        }
                    }
                    Err(e) => tracing::warn!("Job {} final chunk post failed: {e}", job.id),
                }
                break;
            }
        }
    }

    if final_sent {
        inf.abort(); // generation already finished; ensure the task is reaped
        tracing::info!("Job {} completed (E2E stream, {} chunk(s))", job.id, seq);
        return;
    }

    if client_gone {
        // Client disconnected mid-stream. Stop the generator FIRST (drop the
        // receiver so llama-server reads stop unblocking the task, then abort it),
        // then settle the partial so the generated tokens aren't free. Prompt
        // tokens are unknown without [DONE]; bill completion tokens only
        // (conservative, favors the user).
        drop(rx);
        inf.abort();
        let usage = serde_json::json!({
            "prompt_tokens": 0,
            "completion_tokens": emitted_tokens,
            "total_tokens": emitted_tokens,
        });
        let _ = seal_post_chunk(
            client, &sealer, node_secret, &job.id, &eph_b58, seq, true, b"", Some(usage),
        )
        .await;
        tracing::warn!(
            "Job {} client aborted after {} chunk(s); settled partial ({} tokens)",
            job.id,
            seq,
            emitted_tokens
        );
        return;
    }

    // Generation failure (upstream EOF without [DONE], or the final post failed) —
    // abort the client stream and fail the job. NO billing.
    let reason = match stream_failure_reason {
        Some(reason) => {
            let _ = inf.await;
            reason
        }
        None => {
            let inf_res = inf
                .await
                .unwrap_or_else(|_| Err("inference task panicked".to_string()));
            inf_res
                .err()
                .unwrap_or_else(|| "stream ended without completion".to_string())
        }
    };
    if let Err(e) = client.report_stream_error(&job.id).await {
        tracing::warn!("stream error report failed for {}: {e}", job.id);
        let _ = client.fail_job(&job.id, &format!("stream failed: {reason}")).await;
    }
    tracing::warn!("Job {} stream failed: {reason}", job.id);
}
