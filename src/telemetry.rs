//! Routing telemetry reported on every REST heartbeat as ONE additive `telemetry` object.
//!
//! Compatibility contract (both directions):
//! - Old orchestrators parse the heartbeat as loose JSON and never read `telemetry`, so it is
//!   ignored: no DB write, no effect on heartbeat coalescing. Nothing that already exists in the
//!   heartbeat (`current_load`, `capabilities`, `max_concurrent_jobs`, `available_models`) changes
//!   meaning or value because of this module.
//! - New orchestrators must treat the whole object as OPTIONAL (old nodes omit it) and every
//!   field inside it as optional and untrusted. It is self-reported by the node: a node can lie
//!   about being fast to attract traffic, so it may inform load-avoidance but must never be the
//!   only input to a preference ranking or to eligibility. Eligibility stays `available_models`
//!   plus the orchestrator's own capacity gate.
//!
//! Cost: counters are relaxed atomics; timing samples take one uncontended mutex per completed
//! job. Nothing here runs per token, and the snapshot is built once per heartbeat. Operators can
//! stop sending it with `SGL_HEARTBEAT_TELEMETRY=0`.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Bump when a field changes meaning. Adding a field does not need a bump.
pub const SCHEMA_VERSION: u32 = 1;

/// Weight of the newest sample in each moving average. 0.2 ≈ the last ~10 jobs dominate.
const EWMA_ALPHA: f64 = 0.2;

/// A job shorter than this cannot give a meaningful tokens/second figure, so we skip it.
const MIN_TPS_ELAPSED: Duration = Duration::from_millis(20);

/// Kill switch. Default ON: the object is inert until an orchestrator reads it.
fn enabled() -> bool {
    std::env::var("SGL_HEARTBEAT_TELEMETRY")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

#[derive(Default)]
struct Rolling {
    job_ms: Option<f64>,
    output_tps: Option<f64>,
    stream_output_tps: Option<f64>,
    last_job: Option<Instant>,
}

fn ewma(prev: Option<f64>, sample: f64) -> f64 {
    match prev {
        Some(p) => p + EWMA_ALPHA * (sample - p),
        None => sample,
    }
}

/// How a job ended, for the counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JobEnd {
    Ok,
    Failed,
    /// The caller went away mid-stream. Neither a success nor an engine fault.
    ClientGone,
}

pub struct Telemetry {
    enabled: bool,
    started: Instant,
    deferred_at_capacity: AtomicU64,
    jobs_ok: AtomicU64,
    jobs_failed: AtomicU64,
    rolling: Mutex<Rolling>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    pub fn new() -> Self {
        let enabled = enabled();
        if !enabled {
            tracing::info!("heartbeat telemetry disabled (SGL_HEARTBEAT_TELEMETRY=0)");
        }
        Self {
            enabled,
            started: Instant::now(),
            deferred_at_capacity: AtomicU64::new(0),
            jobs_ok: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            rolling: Mutex::new(Rolling::default()),
        }
    }

    /// A dispatch delivery was refused because every local slot was busy. The REST poll
    /// re-delivers a refused job, so one job can be counted more than once.
    pub fn note_deferred_at_capacity(&self) {
        self.deferred_at_capacity.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a buffered (non-stream) job. `elapsed` must cover engine work only (not the
    /// result upload), so `job_ms_ewma` means engine latency. `output_tokens` is None when the
    /// job type has no output tokens (embeddings) or the engine did not report usage.
    pub fn record_buffered(&self, end: JobEnd, elapsed: Duration, output_tokens: Option<u64>) {
        self.count(end);
        let mut r = self.rolling.lock().unwrap_or_else(|p| p.into_inner());
        r.last_job = Some(Instant::now());
        if end != JobEnd::Ok {
            return;
        }
        r.job_ms = Some(ewma(r.job_ms, elapsed.as_secs_f64() * 1000.0));
        if let Some(tps) = tokens_per_second(output_tokens, elapsed) {
            r.output_tps = Some(ewma(r.output_tps, tps));
        }
    }

    /// Record a sealed stream job. `elapsed` includes the per-chunk relay POSTs to the
    /// orchestrator, so this rate is what the caller received, not raw engine speed. That is
    /// why it has its own field and does not mix into `output_tps_ewma`.
    pub fn record_stream(&self, end: JobEnd, elapsed: Duration, output_tokens: Option<u64>) {
        self.count(end);
        let mut r = self.rolling.lock().unwrap_or_else(|p| p.into_inner());
        r.last_job = Some(Instant::now());
        if end != JobEnd::Ok {
            return;
        }
        if let Some(tps) = tokens_per_second(output_tokens, elapsed) {
            r.stream_output_tps = Some(ewma(r.stream_output_tps, tps));
        }
    }

    fn count(&self, end: JobEnd) {
        match end {
            JobEnd::Ok => {
                self.jobs_ok.fetch_add(1, Ordering::Relaxed);
            }
            JobEnd::Failed => {
                self.jobs_failed.fetch_add(1, Ordering::Relaxed);
            }
            JobEnd::ClientGone => {}
        }
    }

    /// Buffered-job convenience: engine time plus the job's result. Embedding usage carries no
    /// output tokens, so embeddings only feed the latency average.
    pub fn record_buffered_result(
        &self,
        job_type: &str,
        elapsed: Duration,
        result: &Result<serde_json::Value, String>,
    ) {
        match result {
            Ok(out) => {
                let tokens = (job_type != "embedding")
                    .then(|| output_tokens_from_result(out))
                    .flatten();
                self.record_buffered(JobEnd::Ok, elapsed, tokens)
            }
            Err(_) => self.record_buffered(JobEnd::Failed, elapsed, None),
        }
    }

    /// Build this heartbeat's object, or None when the operator turned it off.
    pub fn snapshot(&self, s: SnapshotInputs<'_>) -> Option<HeartbeatTelemetry> {
        self.enabled.then(|| self.build(s))
    }

    fn build(&self, s: SnapshotInputs<'_>) -> HeartbeatTelemetry {
        let r = self.rolling.lock().unwrap_or_else(|p| p.into_inner());
        HeartbeatTelemetry {
            schema: SCHEMA_VERSION,
            runtime: runtime_label(s.has_engine, s.has_sidecar),
            accelerator: s.accelerator,
            state: EngineState::derive(
                s.has_engine,
                s.has_sidecar,
                s.unhealthy_streak,
                s.quarantined,
            ),
            models: s.models.to_vec(),
            slots_busy: s.slots_busy.min(s.slots_total),
            slots_total: s.slots_total,
            deferred_at_capacity: self.deferred_at_capacity.load(Ordering::Relaxed),
            jobs_ok: self.jobs_ok.load(Ordering::Relaxed),
            jobs_failed: self.jobs_failed.load(Ordering::Relaxed),
            job_ms_ewma: r
                .job_ms
                .map(|v| v.round().clamp(0.0, u32::MAX as f64) as u32),
            output_tps_ewma: r.output_tps.map(round_tps),
            stream_output_tps_ewma: r.stream_output_tps.map(round_tps),
            last_job_age_s: r.last_job.map(|t| t.elapsed().as_secs()),
            uptime_s: self.started.elapsed().as_secs(),
        }
    }
}

fn tokens_per_second(output_tokens: Option<u64>, elapsed: Duration) -> Option<f64> {
    let n = output_tokens.filter(|n| *n > 0)?;
    if elapsed < MIN_TPS_ELAPSED {
        return None;
    }
    Some(n as f64 / elapsed.as_secs_f64())
}

fn round_tps(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// Output tokens from a buffered job's result: `usage.completion_tokens` (chat) or
/// `usage.output_tokens` (System One).
pub fn output_tokens_from_result(output: &serde_json::Value) -> Option<u64> {
    let usage = output.get("usage")?;
    usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(|v| v.as_u64())
}

/// Engine readiness as the heartbeat loop sees it this cycle. Model ids in `available_models`
/// are only routable when this is `Ready` (or `Degraded`, one failed probe tolerated) — the
/// other states already empty `available_models`; this just says WHY.
#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum EngineState {
    /// Engine answered its health probe this cycle.
    Ready,
    /// One failed probe; still advertised (a single blip is tolerated).
    Degraded,
    /// Two or more failed probes; de-advertised and self-restarting.
    Down,
    /// Empty-completion quarantine; de-advertised while a canary decides.
    Quarantined,
    /// Serving through an external sidecar the node does not health-probe (System One).
    Unprobed,
    /// No model configured.
    NoModel,
}

/// "llama.cpp" (both engines are llama.cpp), "systemone-sidecar", or "none".
pub fn runtime_label(has_engine: bool, has_sidecar: bool) -> &'static str {
    if has_engine {
        "llama.cpp"
    } else if has_sidecar {
        "systemone-sidecar"
    } else {
        "none"
    }
}

impl EngineState {
    /// Derived from the heartbeat loop's own supervision state AFTER its health block runs:
    /// `unhealthy_streak` is 0 after a healthy probe, 1 after a single tolerated blip, and >= 2
    /// once the engine is de-advertised. A down engine outranks quarantine.
    pub fn derive(
        has_engine: bool,
        has_sidecar: bool,
        unhealthy_streak: u32,
        quarantined: bool,
    ) -> Self {
        if !has_engine {
            return if has_sidecar {
                EngineState::Unprobed
            } else {
                EngineState::NoModel
            };
        }
        match unhealthy_streak {
            0 if quarantined => EngineState::Quarantined,
            0 => EngineState::Ready,
            1 if quarantined => EngineState::Quarantined,
            1 => EngineState::Degraded,
            _ => EngineState::Down,
        }
    }
}

/// Best-effort backend of a llama-server binary, computed once per engine (re)start. Never
/// used for trust or security. `variant` is the `sgl setup` marker, which only describes the
/// MANAGED binary, so it is ignored for any other path.
pub fn server_accelerator(binary: &str, gpu_layers: u32, variant: Option<&str>) -> &'static str {
    if gpu_layers == 0 {
        return "cpu";
    }
    if cfg!(target_os = "macos") {
        // Every llama.cpp build we find on macOS (managed, Homebrew, PATH) enables Metal.
        return "metal";
    }
    let managed = std::path::Path::new(binary)
        .parent()
        .is_some_and(|dir| dir.ends_with(std::path::Path::new("sgl-node").join("bin")));
    if managed {
        return match variant {
            Some("vulkan") => "vulkan",
            Some("cpu") => "cpu",
            _ => "unknown",
        };
    }
    // One-click Linux GPU machines: cloud-init installs a pinned CUDA build here.
    if binary == "/opt/sgl/bin/llama-server" {
        return "cuda";
    }
    "unknown"
}

/// Backend the in-process llama.cpp was BUILT with. A Vulkan build runs on CPU when the host
/// has no Vulkan device, so "vulkan" means GPU-capable, not GPU-proven.
#[cfg(feature = "inprocess")]
pub fn inprocess_accelerator(gpu_layers: u32) -> &'static str {
    if gpu_layers == 0 {
        "cpu"
    } else if cfg!(all(target_os = "macos", feature = "metal")) {
        "metal"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else {
        "cpu"
    }
}

pub struct SnapshotInputs<'a> {
    pub has_engine: bool,
    pub has_sidecar: bool,
    /// The heartbeat loop's consecutive failed-/health count (0 = healthy this cycle).
    pub unhealthy_streak: u32,
    /// Empty-completion quarantine is active this cycle.
    pub quarantined: bool,
    pub accelerator: Option<&'static str>,
    pub models: &'a [String],
    pub slots_busy: u32,
    pub slots_total: u32,
}

/// Wire shape of the heartbeat `telemetry` object. See the module docs for the contract.
#[derive(Serialize, Debug)]
pub struct HeartbeatTelemetry {
    pub schema: u32,
    /// Serving runtime: "llama.cpp" (either engine), "systemone-sidecar", or "none".
    pub runtime: &'static str,
    /// Best-effort compute backend: "metal" | "cuda" | "vulkan" | "cpu" | "unknown".
    /// "vulkan" means a Vulkan-capable build: ggml falls back to CPU when there is no device.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<&'static str>,
    pub state: EngineState,
    /// Models this node is CONFIGURED to serve, whether or not they are advertised right now.
    /// Diagnostic only; routing eligibility is `available_models`.
    pub models: Vec<String>,
    /// Local slots in use right now (the node's own capacity counter).
    pub slots_busy: u32,
    /// Local slot limit; equals the top-level `max_concurrent_jobs`.
    pub slots_total: u32,
    /// Monotonic since process start: dispatch deliveries refused because every slot was busy
    /// (a re-delivered job counts again). The node does not queue; the orchestrator keeps it.
    pub deferred_at_capacity: u64,
    /// Monotonic since process start.
    pub jobs_ok: u64,
    /// Monotonic since process start.
    pub jobs_failed: u64,
    /// Moving average of engine time for successful buffered jobs, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_ms_ewma: Option<u32>,
    /// Moving average of output tokens/second for successful buffered jobs (engine speed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tps_ewma: Option<f64>,
    /// Moving average of output tokens/second for completed sealed streams (delivered speed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_output_tps_ewma: Option<f64>,
    /// Seconds since the last job finished.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_job_age_s: Option<u64>,
    pub uptime_s: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(models: &[String]) -> SnapshotInputs<'_> {
        SnapshotInputs {
            has_engine: true,
            has_sidecar: false,
            unhealthy_streak: 0,
            quarantined: false,
            accelerator: Some("metal"),
            models,
            slots_busy: 1,
            slots_total: 2,
        }
    }

    #[test]
    fn fresh_node_omits_every_timing_field() {
        let t = Telemetry::new();
        let models = vec!["qwen".to_string()];
        let v = serde_json::to_value(t.build(inputs(&models))).unwrap();
        assert_eq!(v["schema"], 1);
        assert_eq!(v["runtime"], "llama.cpp");
        assert_eq!(v["accelerator"], "metal");
        assert_eq!(v["state"], "ready");
        assert_eq!(v["models"], serde_json::json!(["qwen"]));
        assert_eq!(v["slots_busy"], 1);
        assert_eq!(v["slots_total"], 2);
        assert_eq!(v["deferred_at_capacity"], 0);
        for k in [
            "job_ms_ewma",
            "output_tps_ewma",
            "stream_output_tps_ewma",
            "last_job_age_s",
        ] {
            assert!(v.get(k).is_none(), "{k} must be omitted before any job");
        }
    }

    #[test]
    fn buffered_success_feeds_latency_and_tps() {
        let t = Telemetry::new();
        t.record_buffered(JobEnd::Ok, Duration::from_millis(1000), Some(50));
        t.record_buffered(JobEnd::Ok, Duration::from_millis(2000), Some(50));
        let s = t.build(inputs(&[]));
        // 1000 then 2000 with alpha 0.2 → 1200.
        assert_eq!(s.job_ms_ewma, Some(1200));
        // 50 tok/s then 25 tok/s → 45.
        assert_eq!(s.output_tps_ewma, Some(45.0));
        assert_eq!(s.stream_output_tps_ewma, None);
        assert_eq!(s.jobs_ok, 2);
        assert_eq!(s.jobs_failed, 0);
        assert_eq!(s.last_job_age_s, Some(0));
    }

    #[test]
    fn failures_count_but_never_move_the_averages() {
        let t = Telemetry::new();
        t.record_buffered(JobEnd::Ok, Duration::from_millis(1000), Some(10));
        t.record_buffered(JobEnd::Failed, Duration::from_millis(1), None);
        t.record_stream(JobEnd::Failed, Duration::from_millis(1), None);
        t.record_stream(JobEnd::ClientGone, Duration::from_millis(1), Some(3));
        let s = t.build(inputs(&[]));
        assert_eq!(s.job_ms_ewma, Some(1000));
        assert_eq!(s.output_tps_ewma, Some(10.0));
        assert_eq!(s.stream_output_tps_ewma, None);
        assert_eq!(s.jobs_ok, 1);
        assert_eq!(s.jobs_failed, 2, "client-gone is not an engine failure");
    }

    #[test]
    fn stream_rate_stays_separate_from_engine_rate() {
        let t = Telemetry::new();
        t.record_stream(JobEnd::Ok, Duration::from_secs(2), Some(40));
        let s = t.build(inputs(&[]));
        assert_eq!(s.stream_output_tps_ewma, Some(20.0));
        assert_eq!(s.output_tps_ewma, None);
        assert_eq!(
            s.job_ms_ewma, None,
            "stream time includes relay; not engine latency"
        );
    }

    #[test]
    fn tiny_or_tokenless_jobs_do_not_produce_a_rate() {
        let t = Telemetry::new();
        t.record_buffered(JobEnd::Ok, Duration::from_millis(5), Some(100));
        t.record_buffered(JobEnd::Ok, Duration::from_millis(500), Some(0));
        t.record_buffered(JobEnd::Ok, Duration::from_millis(500), None);
        assert_eq!(t.build(inputs(&[])).output_tps_ewma, None);
    }

    #[test]
    fn busy_is_clamped_to_total_and_deferrals_count() {
        let t = Telemetry::new();
        t.note_deferred_at_capacity();
        t.note_deferred_at_capacity();
        let mut i = inputs(&[]);
        i.slots_busy = 9;
        let s = t.build(i);
        assert_eq!(s.slots_busy, 2);
        assert_eq!(s.deferred_at_capacity, 2);
    }

    #[test]
    fn output_tokens_read_from_either_usage_shape() {
        let chat = serde_json::json!({"usage": {"prompt_tokens": 3, "completion_tokens": 7}});
        let s1 = serde_json::json!({"usage": {"input_tokens": 0, "output_tokens": 4}});
        let emb = serde_json::json!({"usage": {"prompt_tokens": 12}});
        assert_eq!(output_tokens_from_result(&chat), Some(7));
        assert_eq!(output_tokens_from_result(&s1), Some(4));
        assert_eq!(output_tokens_from_result(&emb), None);
        assert_eq!(output_tokens_from_result(&serde_json::json!({})), None);
    }

    #[test]
    fn states_serialize_snake_case() {
        let v = serde_json::to_value([
            EngineState::Ready,
            EngineState::Degraded,
            EngineState::Down,
            EngineState::Quarantined,
            EngineState::Unprobed,
            EngineState::NoModel,
        ])
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!([
                "ready",
                "degraded",
                "down",
                "quarantined",
                "unprobed",
                "no_model"
            ])
        );
    }

    #[test]
    fn state_follows_the_loop_supervision_counters() {
        use EngineState::*;
        assert_eq!(EngineState::derive(true, false, 0, false), Ready);
        assert_eq!(EngineState::derive(true, false, 1, false), Degraded);
        assert_eq!(EngineState::derive(true, false, 2, false), Down);
        assert_eq!(EngineState::derive(true, false, 0, true), Quarantined);
        assert_eq!(EngineState::derive(true, false, 1, true), Quarantined);
        assert_eq!(
            EngineState::derive(true, false, 5, true),
            Down,
            "down outranks quarantine"
        );
        assert_eq!(EngineState::derive(false, true, 0, false), Unprobed);
        assert_eq!(EngineState::derive(false, false, 0, false), NoModel);
    }

    #[test]
    fn runtime_labels() {
        assert_eq!(runtime_label(true, false), "llama.cpp");
        assert_eq!(runtime_label(true, true), "llama.cpp");
        assert_eq!(runtime_label(false, true), "systemone-sidecar");
        assert_eq!(runtime_label(false, false), "none");
    }

    #[test]
    fn buffered_result_helper_skips_embedding_tokens() {
        let t = Telemetry::new();
        let emb = Ok(serde_json::json!({"usage": {"completion_tokens": 99}}));
        t.record_buffered_result("embedding", Duration::from_secs(1), &emb);
        assert_eq!(t.build(inputs(&[])).output_tps_ewma, None);
        let chat = Ok(serde_json::json!({"usage": {"completion_tokens": 30}}));
        t.record_buffered_result("inference", Duration::from_secs(1), &chat);
        assert_eq!(t.build(inputs(&[])).output_tps_ewma, Some(30.0));
        t.record_buffered_result("inference", Duration::from_secs(1), &Err("x".into()));
        assert_eq!(t.build(inputs(&[])).jobs_failed, 1);
    }

    #[test]
    fn zero_gpu_layers_is_always_cpu() {
        assert_eq!(
            server_accelerator("/opt/sgl/bin/llama-server", 0, None),
            "cpu"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_llama_server_is_metal() {
        assert_eq!(server_accelerator("llama-server", 99, None), "metal");
        assert_eq!(
            server_accelerator("/opt/homebrew/bin/llama-server", 99, None),
            "metal"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn marker_only_describes_the_managed_binary() {
        let managed = std::path::Path::new("/home/op/.local/share")
            .join("sgl-node")
            .join("bin")
            .join("llama-server");
        let managed = managed.to_str().unwrap();
        assert_eq!(server_accelerator(managed, 99, Some("vulkan")), "vulkan");
        assert_eq!(server_accelerator(managed, 99, Some("cpu")), "cpu");
        assert_eq!(server_accelerator(managed, 99, None), "unknown");
        // A PATH binary is not what the marker describes.
        assert_eq!(
            server_accelerator("llama-server", 99, Some("vulkan")),
            "unknown"
        );
        assert_eq!(
            server_accelerator("/opt/sgl/bin/llama-server", 99, None),
            "cuda"
        );
    }
}
