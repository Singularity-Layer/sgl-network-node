//! In-process inference engine (v1.5.0). Embeds llama.cpp via `llama-cpp-2` — no child
//! process, no localhost HTTP, no IPC. A single dedicated OS worker thread owns the
//! `!Send` model + context and serves one request at a time; async callers submit a job
//! over a channel and await the result on a oneshot.
//!
//! Why a worker thread (not spawn_blocking): `LlamaContext` is `!Send`, so the context
//! must live on one thread for its whole life. Concurrency is 1 by design on a single-GPU
//! Mac (advertise capacity = worker count). A fatal inference fault should take the whole
//! process down so launchd/systemd relaunches it clean — the anti-zombie property.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use crate::inference::{ChatMessage, StreamEvent};

/// Flush a streamed batch every N decoded tokens (mirrors the server path's cadence).
const FLUSH_EVERY: u32 = 6;

/// If a job is in flight but the worker hasn't made token progress in this long, treat
/// it as WEDGED (deadlocked / hung native call) → is_healthy() goes false so the node
/// stops advertising. Generous so a legitimately slow prefill/token never trips it.
const WEDGE_MS: u64 = 120_000;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Result of one non-streaming completion, with billing-critical token counts.
pub struct GenOut {
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct InProcessConfig {
    pub model_path: PathBuf,
    pub model_name: String,
    pub n_ctx: u32,
    pub n_gpu_layers: u32,
}

enum JobKind {
    /// Collect the whole completion and return it.
    NonStream(tokio::sync::oneshot::Sender<Result<GenOut, String>>),
    /// Stream `StreamEvent`s over `tokens`; signal terminal success/failure on `done`.
    Stream {
        tokens: tokio::sync::mpsc::Sender<StreamEvent>,
        done: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

struct Job {
    messages: Vec<ChatMessage>,
    max_tokens: i32,
    temperature: f32,
    kind: JobKind,
}

pub struct InProcessEngine {
    job_tx: Sender<Job>,
    healthy: Arc<AtomicBool>,
    /// True while a job is being generated. Idle (false) is always healthy.
    processing: Arc<AtomicBool>,
    /// Unix-ms of the last token (or prefill) progress; used to detect a wedged worker.
    last_progress_ms: Arc<AtomicU64>,
    model_name: String,
    _worker: JoinHandle<()>,
}

impl InProcessEngine {
    /// Spawn the worker, load the model, and block until ready (or fail).
    pub async fn start(cfg: InProcessConfig) -> Result<Self, String> {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let healthy = Arc::new(AtomicBool::new(false));
        let processing = Arc::new(AtomicBool::new(false));
        let last_progress_ms = Arc::new(AtomicU64::new(0));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let model_name = cfg.model_name.clone();

        let w_healthy = Arc::clone(&healthy);
        let w_processing = Arc::clone(&processing);
        let w_progress = Arc::clone(&last_progress_ms);
        let worker = std::thread::Builder::new()
            .name("sgl-inference".into())
            .spawn(move || worker_main(cfg, job_rx, w_healthy, w_processing, w_progress, ready_tx))
            .map_err(|e| format!("failed to spawn inference worker: {e}"))?;

        match ready_rx.await {
            Ok(Ok(())) => Ok(Self { job_tx, healthy, processing, last_progress_ms, model_name, _worker: worker }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("inference worker died during startup".to_string()),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Non-streaming completion.
    pub async fn chat_completion(
        &self,
        messages: &[ChatMessage],
        max_tokens: i32,
        temperature: f32,
    ) -> Result<GenOut, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.job_tx
            .send(Job {
                messages: messages.to_vec(),
                max_tokens,
                temperature,
                kind: JobKind::NonStream(reply),
            })
            .map_err(|_| "inference worker is gone".to_string())?;
        rx.await
            .map_err(|_| "inference worker dropped the request".to_string())?
    }

    /// Streaming completion. Forwards `StreamEvent`s over `tokens` (the caller seals +
    /// relays each), ending with a `Done` carrying token counts. Returns when generation
    /// finishes (Ok) or fails (Err); a dropped receiver stops generation early.
    pub async fn chat_completion_stream(
        &self,
        messages: &[ChatMessage],
        max_tokens: i32,
        temperature: f32,
        tokens: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<(), String> {
        let (done, rx) = tokio::sync::oneshot::channel();
        self.job_tx
            .send(Job {
                messages: messages.to_vec(),
                max_tokens,
                temperature,
                kind: JobKind::Stream { tokens, done },
            })
            .map_err(|_| "inference worker is gone".to_string())?;
        rx.await
            .map_err(|_| "inference worker dropped the request".to_string())?
    }

    /// Healthy = model loaded AND (idle OR the in-flight job is still making token
    /// progress). A wedged worker (in flight but no progress for WEDGE_MS) reads as
    /// UNHEALTHY so the heartbeat loop de-advertises it — closing the in-process
    /// equivalent of the zombie.
    pub fn is_healthy(&self) -> bool {
        if !self.healthy.load(Ordering::Relaxed) {
            return false;
        }
        if !self.processing.load(Ordering::Relaxed) {
            return true; // idle, model loaded
        }
        now_ms().saturating_sub(self.last_progress_ms.load(Ordering::Relaxed)) < WEDGE_MS
    }
}

/// Worker thread: owns the backend + model for its whole life, serves jobs serially.
fn worker_main(
    cfg: InProcessConfig,
    job_rx: Receiver<Job>,
    healthy: Arc<AtomicBool>,
    processing: Arc<AtomicBool>,
    progress: Arc<AtomicU64>,
    ready_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
) {
    let backend = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("llama backend init failed: {e}")));
            return;
        }
    };
    let model_params = LlamaModelParams::default().with_n_gpu_layers(cfg.n_gpu_layers);
    let model = match LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("model load failed: {e}")));
            return;
        }
    };

    healthy.store(true, Ordering::Relaxed);
    let _ = ready_tx.send(Ok(()));

    while let Ok(job) = job_rx.recv() {
        let Job { messages, max_tokens, temperature, kind } = job;
        let stream_sink = match &kind {
            JobKind::Stream { tokens, .. } => Some(tokens.clone()),
            JobKind::NonStream(_) => None,
        };
        // Mark in-flight + seed progress so the watchdog (is_healthy) distinguishes a
        // slow-but-alive generation from a wedged one.
        progress.store(now_ms(), Ordering::Relaxed);
        processing.store(true, Ordering::Relaxed);
        // A Rust panic mid-generation means a broken invariant — make it FATAL so the OS
        // service relaunches the node clean (the anti-zombie property). Native llama.cpp
        // aborts/segfaults already kill the process; this covers the Rust-panic case
        // without forcing global panic=abort on the rest of the node.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generate(&backend, &model, cfg.n_ctx, &messages, max_tokens, temperature, stream_sink.as_ref(), &progress)
        }));
        processing.store(false, Ordering::Relaxed);
        let result = match outcome {
            Ok(r) => r,
            Err(_) => {
                tracing::error!("inference worker panicked mid-generation — aborting for a clean OS restart");
                std::process::abort();
            }
        };
        match kind {
            JobKind::NonStream(reply) => {
                let _ = reply.send(result);
            }
            JobKind::Stream { done, .. } => {
                let _ = done.send(result.map(|_| ()));
            }
        }
    }
    healthy.store(false, Ordering::Relaxed);
}

/// Core generation. Renders the chat template, decodes the prompt, samples until EOG or
/// the token cap. When `stream` is Some, forwards `StreamEvent::Delta` batches and a final
/// `Done`; otherwise accumulates the full text. Returns exact token counts either way.
fn generate(
    backend: &LlamaBackend,
    model: &LlamaModel,
    n_ctx: u32,
    messages: &[ChatMessage],
    max_tokens: i32,
    temperature: f32,
    stream: Option<&tokio::sync::mpsc::Sender<StreamEvent>>,
    progress: &AtomicU64,
) -> Result<GenOut, String> {
    // Honor the configured context window (matches `llama-server -c <n_ctx>`); without
    // this the context would silently use the model/crate default.
    let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(n_ctx));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("context create failed: {e}"))?;

    let prompt = render_chat_prompt(model, messages)?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let prompt_tokens = tokens.len() as u32;
    if tokens.is_empty() {
        return Err("empty prompt after templating".to_string());
    }
    if prompt_tokens >= n_ctx {
        return Err("prompt exceeds context window".to_string());
    }

    let mut batch = LlamaBatch::new(n_ctx.max(512) as usize, 1);
    let last = tokens.len() - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch
            .add(*tok, i as i32, &[0], i == last)
            .map_err(|e| format!("batch add failed: {e}"))?;
    }
    ctx.decode(&mut batch).map_err(|e| format!("prompt decode failed: {e}"))?;
    progress.store(now_ms(), Ordering::Relaxed); // prefill done

    let mut sampler = if temperature > 0.0 {
        LlamaSampler::chain_simple([LlamaSampler::temp(temperature), LlamaSampler::dist(0)])
    } else {
        LlamaSampler::greedy()
    };

    let max_new = max_tokens.max(1);
    let mut n_cur = batch.n_tokens();
    let mut completion_tokens: u32 = 0;
    let mut out = String::new();
    let mut pending = String::new();
    let mut batched: u32 = 0;

    for _ in 0..max_new {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            completion_tokens += 1; // llama-server counts the terminal stop token; match it for billing parity
            break;
        }
        progress.store(now_ms(), Ordering::Relaxed); // token progress (watchdog)
        let piece = model.token_to_str(token, Special::Plaintext).unwrap_or_default();
        completion_tokens += 1;

        if let Some(tx) = stream {
            pending.push_str(&piece);
            batched += 1;
            if batched >= FLUSH_EVERY {
                if tx
                    .blocking_send(StreamEvent::Delta { text: std::mem::take(&mut pending), tokens: batched })
                    .is_err()
                {
                    return Ok(GenOut { content: String::new(), prompt_tokens, completion_tokens }); // receiver gone
                }
                batched = 0;
            }
        } else {
            out.push_str(&piece);
        }

        if n_cur as u32 >= n_ctx {
            break;
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("gen batch add failed: {e}"))?;
        n_cur += 1;
        ctx.decode(&mut batch).map_err(|e| format!("gen decode failed: {e}"))?;
    }

    if let Some(tx) = stream {
        if !pending.is_empty() {
            let _ = tx.blocking_send(StreamEvent::Delta { text: std::mem::take(&mut pending), tokens: batched });
        }
        let _ = tx.blocking_send(StreamEvent::Done { prompt_tokens, completion_tokens });
    }

    Ok(GenOut { content: out, prompt_tokens, completion_tokens })
}

/// Render the chat prompt EXACTLY like llama-server: parse the GGUF's jinja chat template
/// (tokenizer.chat_template) with minijinja and feed it the same context — `messages`,
/// `add_generation_prompt=true`, the model's `bos_token`, plus `strftime_now` so Llama-3's
/// "Today Date" preamble matches. This closes the legacy-vs-jinja parity gap (prompt token
/// counts + output) so billing is identical across the server and in-process engines.
fn render_chat_prompt(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String, String> {
    let tmpl_str = model
        .meta_val_str("tokenizer.chat_template")
        .map_err(|e| format!("model has no chat_template metadata: {e}"))?;
    // BOS as text (e.g. "<|begin_of_text|>"); the template emits it via {{ bos_token }} and
    // str_to_token(parse_special) maps it back to the BOS id, so we tokenize with AddBos::Never.
    let bos_token = model
        .token_to_str(model.token_bos(), Special::Tokenize)
        .unwrap_or_default();

    let mut env = minijinja::Environment::new();
    env.add_function("strftime_now", |fmt: String| {
        chrono::Local::now().format(&fmt).to_string()
    });
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        },
    );
    env.add_template("chat", &tmpl_str)
        .map_err(|e| format!("chat template parse failed: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("chat template load failed: {e}"))?;

    tmpl.render(minijinja::context! {
        messages => messages,
        add_generation_prompt => true,
        bos_token => bos_token,
    })
    .map_err(|e| format!("chat template render failed: {e}"))
}
