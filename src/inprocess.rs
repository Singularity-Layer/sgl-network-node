//! In-process inference engine (v1.5.0). Embeds llama.cpp via `llama-cpp-2` — no child
//! process, no localhost HTTP, no IPC. A single dedicated OS worker thread owns the
//! `!Send` model + context and serves one request at a time; async callers submit a job
//! over a channel and await the result on a oneshot.
//!
//! Why a worker thread (not spawn_blocking): `LlamaContext` is `!Send`, so the context
//! must live on one thread for its whole life. Concurrency is 1 by design on a single-GPU
//! Mac (advertise capacity = worker count). A fatal inference fault should take the whole
//! process down so launchd/systemd relaunches it clean — that is the anti-zombie property.
//!
//! Non-streaming first (per the Codex design). Streaming is added next.

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

use crate::inference::ChatMessage;

/// Result of one completion, with the billing-critical token counts.
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

/// One unit of work handed to the worker thread.
struct Job {
    messages: Vec<ChatMessage>,
    max_tokens: i32,
    temperature: f32,
    reply: tokio::sync::oneshot::Sender<Result<GenOut, String>>,
}

pub struct InProcessEngine {
    job_tx: Sender<Job>,
    healthy: Arc<AtomicBool>,
    _worker: JoinHandle<()>,
}

impl InProcessEngine {
    /// Spawn the worker, load the model, and block until it's ready (or fail).
    pub async fn start(cfg: InProcessConfig) -> Result<Self, String> {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let healthy = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        let worker_healthy = Arc::clone(&healthy);
        let worker = std::thread::Builder::new()
            .name("sgl-inference".into())
            .spawn(move || worker_main(cfg, job_rx, worker_healthy, ready_tx))
            .map_err(|e| format!("failed to spawn inference worker: {e}"))?;

        // Wait for the worker to finish loading the model.
        match ready_rx.await {
            Ok(Ok(())) => Ok(Self { job_tx, healthy, _worker: worker }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("inference worker died during startup".to_string()),
        }
    }

    /// Non-streaming completion. Submits the job and awaits the worker's result.
    pub async fn chat_completion(
        &self,
        messages: &[ChatMessage],
        max_tokens: i32,
        temperature: f32,
    ) -> Result<GenOut, String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.job_tx
            .send(Job { messages: messages.to_vec(), max_tokens, temperature, reply })
            .map_err(|_| "inference worker is gone".to_string())?;
        rx.await
            .map_err(|_| "inference worker dropped the request".to_string())?
    }

    /// Healthy = model loaded and the worker thread is alive.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

/// Worker thread entry: owns the backend + model for its whole life, serves jobs serially.
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

    // Serve jobs until all senders drop (engine stopped).
    while let Ok(job) = job_rx.recv() {
        let result = run_generation(&backend, &model, cfg.n_ctx, &job);
        let _ = job.reply.send(result);
    }
    healthy.store(false, Ordering::Relaxed);
}

/// Render the chat template, decode the prompt, and greedily/temperature-sample until EOG
/// or the token cap. Returns exact prompt/completion token counts for billing parity.
fn run_generation(
    backend: &LlamaBackend,
    model: &LlamaModel,
    n_ctx: u32,
    job: &Job,
) -> Result<GenOut, String> {
    // Fresh context per request (isolated KV cache). Concurrency is 1, so this is safe.
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("context create failed: {e}"))?;

    // Render the model's own chat template so behavior matches llama-server.
    let template = model
        .chat_template(None)
        .map_err(|e| format!("chat template unavailable: {e}"))?;
    let chat: Vec<LlamaChatMessage> = job
        .messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("bad chat message: {e}"))?;
    let prompt = model
        .apply_chat_template(&template, &chat, true)
        .map_err(|e| format!("apply chat template failed: {e}"))?;

    // Template already carries BOS + special tokens (str_to_token parses special).
    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let prompt_tokens = tokens.len() as u32;
    if tokens.is_empty() {
        return Err("empty prompt after templating".to_string());
    }
    if tokens.len() as u32 >= n_ctx {
        return Err("prompt exceeds context window".to_string());
    }

    // Decode the prompt; only the last token needs logits.
    let mut batch = LlamaBatch::new(n_ctx.max(512) as usize, 1);
    let last = tokens.len() - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch
            .add(*tok, i as i32, &[0], i == last)
            .map_err(|e| format!("batch add failed: {e}"))?;
    }
    ctx.decode(&mut batch).map_err(|e| format!("prompt decode failed: {e}"))?;

    let mut sampler = if job.temperature > 0.0 {
        LlamaSampler::chain_simple([LlamaSampler::temp(job.temperature), LlamaSampler::dist(0)])
    } else {
        LlamaSampler::greedy()
    };

    let max_new = job.max_tokens.max(1);
    let mut n_cur = batch.n_tokens();
    let mut completion_tokens: u32 = 0;
    let mut out = String::new();

    for _ in 0..max_new {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        out.push_str(&model.token_to_str(token, Special::Plaintext).unwrap_or_default());
        completion_tokens += 1;

        if n_cur as u32 >= n_ctx {
            break; // hit the context window
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| format!("gen batch add failed: {e}"))?;
        n_cur += 1;
        ctx.decode(&mut batch).map_err(|e| format!("gen decode failed: {e}"))?;
    }

    Ok(GenOut { content: out, prompt_tokens, completion_tokens })
}
