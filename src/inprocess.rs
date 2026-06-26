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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use crate::inference::{ChatMessage, StreamEvent};

/// Flush a streamed batch every N decoded tokens (mirrors the server path's cadence).
const FLUSH_EVERY: u32 = 6;

/// Result of one non-streaming completion, with billing-critical token counts.
pub struct GenOut {
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct InProcessConfig {
    pub model_path: PathBuf,
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
    _worker: JoinHandle<()>,
}

impl InProcessEngine {
    /// Spawn the worker, load the model, and block until ready (or fail).
    pub async fn start(cfg: InProcessConfig) -> Result<Self, String> {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let healthy = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        let worker_healthy = Arc::clone(&healthy);
        let worker = std::thread::Builder::new()
            .name("sgl-inference".into())
            .spawn(move || worker_main(cfg, job_rx, worker_healthy, ready_tx))
            .map_err(|e| format!("failed to spawn inference worker: {e}"))?;

        match ready_rx.await {
            Ok(Ok(())) => Ok(Self { job_tx, healthy, _worker: worker }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("inference worker died during startup".to_string()),
        }
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

    /// Healthy = model loaded and the worker thread alive.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

/// Worker thread: owns the backend + model for its whole life, serves jobs serially.
fn worker_main(
    cfg: InProcessConfig,
    job_rx: Receiver<Job>,
    healthy: Arc<AtomicBool>,
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
        match job.kind {
            JobKind::NonStream(reply) => {
                let r = generate(&backend, &model, cfg.n_ctx, &job.messages, job.max_tokens, job.temperature, None);
                let _ = reply.send(r);
            }
            JobKind::Stream { tokens, done } => {
                let r = generate(&backend, &model, cfg.n_ctx, &job.messages, job.max_tokens, job.temperature, Some(&tokens));
                let _ = done.send(r.map(|_| ()));
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
) -> Result<GenOut, String> {
    let mut ctx = model
        .new_context(backend, LlamaContextParams::default())
        .map_err(|e| format!("context create failed: {e}"))?;

    let template = model
        .chat_template(None)
        .map_err(|e| format!("chat template unavailable: {e}"))?;
    let chat: Vec<LlamaChatMessage> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("bad chat message: {e}"))?;
    let prompt = model
        .apply_chat_template(&template, &chat, true)
        .map_err(|e| format!("apply chat template failed: {e}"))?;

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
            break;
        }
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
