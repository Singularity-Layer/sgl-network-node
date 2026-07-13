//! In-process embedding engine (#231). Loads an embedding GGUF via `llama-cpp-2` with a
//! POOLING-enabled context and serves OpenAI-compatible `/v1/embeddings`. Separate from the
//! chat `InProcessEngine`: a node is embeddings-OR-chat (one loaded llama.cpp context), so a
//! plain CPU box can dedicate itself to embeddings and earn without a GPU.
//!
//! Like the chat engine, a single dedicated OS worker thread owns the `!Send` model + context
//! for its whole life; a fatal native fault takes the process down so the OS service relaunches
//! it clean (the anti-zombie property). Embeddings are stateless one-shots (no streaming, no KV
//! reuse across requests), so the worker is simpler than the chat scheduler: it pulls a job (a
//! batch of input strings), INPUT-BATCHES them (packs many sequences into one encode pass, capped
//! to the model's context so KV memory is unchanged), and replies with one vector per input.
//!
//! The node-side model manifest (pooling / normalize / dim / prefix) MUST match the orchestrator
//! catalog (`SGLNetwork_Orchestrator/src/lib/embeddings.ts`) — the orchestrator structurally
//! validates every vector (dim, finite, non-zero) and REJECTS mismatches, so a wrong pooling or
//! dim here surfaces as a failed job, never a silently-wrong (billed) embedding.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

// ─── Node-side embedding model manifest (mirrors the orchestrator catalog) ────────────────────

/// Pooling strategy — how the per-token hidden states become one sentence vector. MUST match the
/// orchestrator catalog per model or the produced vector won't match what clients expect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // `Last` is a valid llama.cpp pooling type kept for future last-token models
pub enum Pooling {
    Cls,
    Mean,
    Last,
}

impl Pooling {
    fn to_llama(self) -> LlamaPoolingType {
        match self {
            Pooling::Cls => LlamaPoolingType::Cls,
            Pooling::Mean => LlamaPoolingType::Mean,
            Pooling::Last => LlamaPoolingType::Last,
        }
    }
}

/// One embedding model the node can serve. `id` is the grid model id (matches the orchestrator
/// catalog). `prefix_query`/`prefix_document` are the per-family retrieval prefixes applied
/// node-side by `input_type` (e5 needs `query:`/`passage:`; bge uses an instruction; nomic uses
/// `search_query:`/`search_document:`; some models want none).
#[derive(Clone, Copy, Debug)]
pub struct EmbedModelSpec {
    pub id: &'static str,
    pub pooling: Pooling,
    /// L2-normalize the output vector (so a valid embedding is never all-zero and cosine == dot).
    pub normalize: bool,
    /// Native output dimension (== `model.n_embd()`; used to size + validate).
    pub dim: u32,
    /// Extra Matryoshka dims this model can be truncated to (empty = fixed dim).
    pub allowed_dimensions: &'static [u32],
    /// Max input tokens per item (also sizes the context window).
    pub max_input_tokens: u32,
    pub prefix_query: &'static str,
    pub prefix_document: &'static str,
    /// Extra ids that resolve to this spec (aliases the orchestrator may route by).
    pub aliases: &'static [&'static str],
}

/// Authoritative node-side catalog. Keep in lock-step with `src/lib/embeddings.ts` in the
/// orchestrator. Adding a model here is what lets an operator serve it.
pub const EMBED_CATALOG: &[EmbedModelSpec] = &[
    EmbedModelSpec {
        id: "nomic-embed-text-v1.5",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 768,
        allowed_dimensions: &[768, 512, 256, 128, 64],
        max_input_tokens: 8192,
        prefix_query: "search_query: ",
        prefix_document: "search_document: ",
        aliases: &["nomic-embed-text", "text-embedding-3-small"],
    },
    EmbedModelSpec {
        id: "bge-small-en-v1.5",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 384,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "bge-base-en-v1.5",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 768,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "bge-large-en-v1.5",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "all-minilm-l6-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 384,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "",
        prefix_document: "",
        aliases: &["all-minilm", "minilm-l6"],
    },
    EmbedModelSpec {
        id: "e5-small-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 384,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "e5-base-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 768,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "e5-large-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "mxbai-embed-large-v1",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[1024, 512, 256],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "multilingual-e5-large",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &["e5-multilingual"],
    },
    EmbedModelSpec {
        id: "bge-m3",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 8192,
        prefix_query: "",
        prefix_document: "",
        aliases: &["bge-m3-multilingual"],
    },
];

/// Normalize a model id the way the orchestrator does (strip quant/`.gguf` suffixes), lowercased.
fn normalize_id(id: &str) -> String {
    let mut s = id.to_ascii_lowercase();
    if let Some(pos) = s.find("-q") {
        // Strip a trailing quant tag like "-q4_k_m" only if it looks like one (digit after -q).
        if s[pos + 2..].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            s.truncate(pos);
        }
    }
    if let Some(stripped) = s.strip_suffix(".gguf") {
        s = stripped.to_string();
    }
    s.trim().to_string()
}

/// Strict lookup — `None` for anything not in the catalog (NO default fallback, mirrors the
/// orchestrator: an unknown embedding model is rejected, never served with guessed params).
pub fn embed_model_spec(id: &str) -> Option<&'static EmbedModelSpec> {
    let want = id.to_ascii_lowercase();
    let want_norm = normalize_id(id);
    EMBED_CATALOG.iter().find(|m| {
        m.id.eq_ignore_ascii_case(&want)
            || m.id.eq_ignore_ascii_case(&want_norm)
            || m.aliases.iter().any(|a| a.eq_ignore_ascii_case(&want) || a.eq_ignore_ascii_case(&want_norm))
    })
}

/// True iff the given model id is a known embedding model — the node uses this to pick the
/// embedding engine over the chat engine at startup.
pub fn is_embedding_model(id: &str) -> bool {
    embed_model_spec(id).is_some()
}

// ─── Engine ───────────────────────────────────────────────────────────────────────────────────

/// If the worker is inside a native call for this long without progress, it's WEDGED → unhealthy
/// so the heartbeat de-advertises the node (mirrors the chat engine's watchdog). Embedding
/// decodes are fast, so this is generous.
const WEDGE_MS: u64 = 60_000;

/// Hard cap on inputs per job (defense-in-depth; the orchestrator already caps the batch).
const MAX_INPUTS_PER_JOB: usize = 4096;

/// Max sequences packed into a single `encode()` pass (input batching). The real limiter is the
/// per-group token budget (== `n_ctx`), so this is just an upper bound on the pooled-output buffer
/// (n_seq_max × n_embd floats — a few MB at most). It never raises KV-cache memory, so batching
/// here is safe on tight 16GB Macs — a group is still capped to the model's existing context.
const MAX_SEQS_PER_BATCH: usize = 256;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Which retrieval prefix to apply to an input (OpenAI extension; catalog encodes the strings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputType {
    Query,
    Document,
    /// No explicit type given — use the document prefix (the common indexing default).
    Unspecified,
}

pub struct EmbedConfig {
    pub model_path: PathBuf,
    /// Grid model id (must be in EMBED_CATALOG).
    pub model_name: String,
    pub n_gpu_layers: u32,
}

/// Result of embedding one batch. `vectors` is one row per input, in input order.
pub struct EmbedOut {
    pub vectors: Vec<Vec<f32>>,
    /// Total input tokens across the whole batch (the billable count — input only).
    pub prompt_tokens: u32,
}

struct EmbedJob {
    inputs: Vec<String>,
    input_type: InputType,
    /// Requested output dimension (Matryoshka truncation). `None` = native dim.
    dimensions: Option<u32>,
    reply: tokio::sync::oneshot::Sender<Result<EmbedOut, String>>,
}

pub struct EmbedEngine {
    /// `Option` so `Drop` can close the channel (worker exits) before joining.
    job_tx: Option<Sender<EmbedJob>>,
    healthy: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
    last_progress_ms: Arc<AtomicU64>,
    spec: &'static EmbedModelSpec,
    worker: Option<JoinHandle<()>>,
}

impl EmbedEngine {
    /// Spawn the worker, load the model + embedding context, block until ready (or fail).
    pub async fn start(cfg: EmbedConfig) -> Result<Self, String> {
        let spec = embed_model_spec(&cfg.model_name)
            .ok_or_else(|| format!("'{}' is not a known embedding model", cfg.model_name))?;

        let (job_tx, job_rx) = std::sync::mpsc::channel::<EmbedJob>();
        let healthy = Arc::new(AtomicBool::new(false));
        let busy = Arc::new(AtomicBool::new(false));
        let last_progress_ms = Arc::new(AtomicU64::new(0));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        let w_healthy = Arc::clone(&healthy);
        let w_busy = Arc::clone(&busy);
        let w_progress = Arc::clone(&last_progress_ms);
        let worker = std::thread::Builder::new()
            .name("sgl-embed".into())
            .spawn(move || worker_main(cfg, spec, job_rx, w_healthy, w_busy, w_progress, ready_tx))
            .map_err(|e| format!("failed to spawn embedding worker: {e}"))?;

        // Bound the startup wait — model load + context create are native calls that can wedge
        // (a clean timeout turns a silent infinite hang into a returned Err so the caller can exit
        // and the OS service relaunch). Generous for a slow first-run Metal compile.
        const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
        match tokio::time::timeout(STARTUP_TIMEOUT, ready_rx).await {
            Ok(Ok(Ok(()))) => Ok(Self {
                job_tx: Some(job_tx),
                healthy,
                busy,
                last_progress_ms,
                spec,
                worker: Some(worker),
            }),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Err(_)) => Err("embedding worker died during startup".to_string()),
            Err(_) => Err(format!(
                "embedding worker startup timed out after {STARTUP_TIMEOUT:?} (model/context load wedged)"
            )),
        }
    }

    /// Native output dimension (advertised to the orchestrator as `dim`).
    pub fn dim(&self) -> u32 {
        self.spec.dim
    }

    /// Embed a batch of inputs. Returns one vector per input (input order) + total input tokens.
    pub async fn embed(
        &self,
        inputs: Vec<String>,
        input_type: InputType,
        dimensions: Option<u32>,
    ) -> Result<EmbedOut, String> {
        if inputs.is_empty() {
            return Err("no inputs to embed".to_string());
        }
        if inputs.len() > MAX_INPUTS_PER_JOB {
            return Err("too many inputs in one embedding job".to_string());
        }
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.job_tx
            .as_ref()
            .ok_or_else(|| "embedding worker is gone".to_string())?
            .send(EmbedJob { inputs, input_type, dimensions, reply })
            .map_err(|_| "embedding worker is gone".to_string())?;
        rx.await
            .map_err(|_| "embedding worker dropped the request".to_string())?
    }

    /// Healthy = model loaded AND (idle OR still making progress inside the current job). A wedged
    /// worker (busy but no progress for WEDGE_MS) reads UNHEALTHY so the heartbeat de-advertises it.
    pub fn is_healthy(&self) -> bool {
        if !self.healthy.load(Ordering::Relaxed) {
            return false;
        }
        if !self.busy.load(Ordering::Relaxed) {
            return true;
        }
        now_ms().saturating_sub(self.last_progress_ms.load(Ordering::Relaxed)) < WEDGE_MS
    }
}

impl Drop for EmbedEngine {
    fn drop(&mut self) {
        drop(self.job_tx.take());
        let Some(handle) = self.worker.take() else { return };
        // Bounded join: a healthy worker exits fast; a worker wedged in a native call can't be
        // unblocked from Rust, so abort rather than hang the process on exit (OS relaunches clean).
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let joiner = std::thread::Builder::new()
            .name("sgl-embed-join".into())
            .spawn(move || {
                let _ = handle.join();
                let _ = tx.send(());
            });
        if joiner.is_err() {
            return;
        }
        if rx.recv_timeout(Duration::from_secs(20)).is_err() {
            tracing::error!("embedding worker did not shut down in 20s (wedged native call) — aborting");
            std::process::abort();
        }
    }
}

/// Worker thread: owns the backend + model + one pooling-enabled context for its whole life.
fn worker_main(
    cfg: EmbedConfig,
    spec: &'static EmbedModelSpec,
    job_rx: Receiver<EmbedJob>,
    healthy: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
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

    // Verify the loaded model's real dim matches the catalog — a mismatch means the operator
    // pointed us at the wrong GGUF; fail LOUD at startup rather than emit wrong-dim vectors the
    // orchestrator would reject on every job.
    let real_dim = model.n_embd();
    if real_dim as i64 != spec.dim as i64 {
        let _ = ready_tx.send(Err(format!(
            "model '{}' reports dim {} but catalog expects {} — wrong GGUF for this model id",
            spec.id, real_dim, spec.dim
        )));
        return;
    }

    // Non-causal embedding models (BERT-style: bge/e5/minilm) require the WHOLE sequence in one
    // ubatch, so size n_batch == n_ubatch == n_ctx to the model's max input. Pooling produces one
    // vector per sequence, read via embeddings_seq_ith. `n_seq_max` lets us pack MULTIPLE input
    // sequences into ONE encode() pass (input batching) — the group is still capped to n_ctx tokens
    // so KV-cache memory is unchanged; only the tiny pooled-output buffer scales with n_seq_max.
    let n_ctx = spec.max_input_tokens.max(512);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_n_seq_max(MAX_SEQS_PER_BATCH as u32)
        .with_embeddings(true)
        .with_pooling_type(spec.pooling.to_llama());
    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("context create failed: {e}")));
            return;
        }
    };

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);

    healthy.store(true, Ordering::Relaxed);
    let _ = ready_tx.send(Ok(()));

    // Serve jobs until every engine handle is dropped.
    loop {
        let job = match job_rx.recv() {
            Ok(j) => j,
            Err(_) => break, // all senders dropped → shut down
        };

        busy.store(true, Ordering::Relaxed);
        progress.store(now_ms(), Ordering::Relaxed);

        // A Rust panic mid-embed means a broken invariant on the shared context — make it FATAL so
        // the OS service relaunches the node clean (anti-zombie), matching the chat engine.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_embed_job(&model, &mut ctx, &mut batch, spec, n_ctx, &progress, job)
        }));
        if outcome.is_err() {
            tracing::error!("embedding worker panicked mid-job — aborting for a clean OS restart");
            std::process::abort();
        }

        busy.store(false, Ordering::Relaxed);
    }

    healthy.store(false, Ordering::Relaxed);
    busy.store(false, Ordering::Relaxed);
}

/// Embed one job (a batch of inputs), replying on its oneshot. Inputs are INPUT-BATCHED: multiple
/// sequences are packed into one `encode()` pass (distinct seq_id per input), greedily grouped so
/// each group's total tokens stay within `n_ctx`. This turns an N-input request from N model passes
/// into ~N/(seqs-per-group) passes — the batching win — at no extra KV memory (a group never
/// exceeds the model's existing context). Pooled vectors are read per-seq via embeddings_seq_ith
/// and returned in the original input order.
fn run_embed_job(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    batch: &mut LlamaBatch,
    spec: &EmbedModelSpec,
    n_ctx: u32,
    progress: &AtomicU64,
    job: EmbedJob,
) {
    let EmbedJob { inputs, input_type, dimensions, reply } = job;

    // Resolve the effective output dim (Matryoshka truncation). Reject an unsupported request
    // BEFORE embedding anything (mirrors the orchestrator, which also validates, but fail-fast here).
    let eff_dim = match dimensions {
        None => spec.dim,
        Some(d) if d == spec.dim => spec.dim,
        Some(d) if spec.allowed_dimensions.contains(&d) => d,
        Some(d) => {
            let _ = reply.send(Err(format!(
                "model '{}' cannot produce {}-dim vectors",
                spec.id, d
            )));
            return;
        }
    };

    let prefix = match input_type {
        InputType::Query => spec.prefix_query,
        InputType::Document | InputType::Unspecified => spec.prefix_document,
    };

    // Tokenize + validate EVERY input up front (with the family prefix + special tokens). Doing this
    // before any compute lets us reject a bad/oversized input without half-embedding the batch, and
    // gives us the per-input token counts needed to pack groups.
    let mut tokenized: Vec<Vec<llama_cpp_2::token::LlamaToken>> = Vec::with_capacity(inputs.len());
    let mut prompt_tokens: u32 = 0;
    for input in &inputs {
        let text = if prefix.is_empty() {
            input.clone()
        } else {
            format!("{prefix}{input}")
        };
        let tokens = match model.str_to_token(&text, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                let _ = reply.send(Err(format!("tokenize failed: {e}")));
                return;
            }
        };
        if tokens.is_empty() {
            let _ = reply.send(Err("empty input after tokenization".to_string()));
            return;
        }
        if tokens.len() as u32 > n_ctx {
            let _ = reply.send(Err(format!(
                "input exceeds {}-token context for {}",
                n_ctx, spec.id
            )));
            return;
        }
        prompt_tokens = prompt_tokens.saturating_add(tokens.len() as u32);
        tokenized.push(tokens);
    }

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(tokenized.len());

    // Walk the inputs, packing each greedy group up to (n_ctx tokens, MAX_SEQS_PER_BATCH seqs), then
    // encode the whole group in one pass. A single input longer than the remaining budget always
    // starts its own group (the `end > start` guard), so a big input degrades to one-per-pass rather
    // than being wrongly rejected.
    let mut start = 0usize;
    while start < tokenized.len() {
        let mut end = start;
        let mut group_tokens = 0usize;
        while end < tokenized.len() {
            let t = tokenized[end].len();
            if end - start >= MAX_SEQS_PER_BATCH {
                break;
            }
            if group_tokens + t > n_ctx as usize && end > start {
                break;
            }
            group_tokens += t;
            end += 1;
        }

        // Fresh group: clear ALL sequence KV, then add each input as its own seq_id (0..group_len).
        // EVERY token is marked as an output so pooling covers the full sequence (llama.cpp otherwise
        // warns "some input tokens were not marked as outputs -> overriding").
        let _ = ctx.clear_kv_cache_seq(None, None, None);
        batch.clear();
        for (local, seq) in tokenized[start..end].iter().enumerate() {
            for (pos, tok) in seq.iter().enumerate() {
                if let Err(e) = batch.add(*tok, pos as i32, &[local as i32], true) {
                    let _ = reply.send(Err(format!("batch add failed: {e}")));
                    return;
                }
            }
        }
        // Encoder (non-causal, BERT-style) embedding models are run with encode(); decode() would
        // auto-redirect but logs a warning. Fall back to decode() for any (causal) embedding model
        // where encode() isn't the right call, so the engine works for both families.
        if let Err(enc_err) = ctx.encode(&mut *batch) {
            if let Err(dec_err) = ctx.decode(&mut *batch) {
                let _ = reply.send(Err(format!("embed compute failed (encode: {enc_err}; decode: {dec_err})")));
                return;
            }
        }
        progress.store(now_ms(), Ordering::Relaxed);

        // Read each sequence's pooled embedding (seq_id == local index within the group), in order.
        for local in 0..(end - start) {
            let raw = match ctx.embeddings_seq_ith(local as i32) {
                Ok(v) => v,
                Err(e) => {
                    let _ = reply.send(Err(format!("read embedding failed: {e:?}")));
                    return;
                }
            };
            let mut vec: Vec<f32> = raw.to_vec();

            // Matryoshka: truncate to the requested dim, then renormalize the shortened vector.
            if (eff_dim as usize) < vec.len() {
                vec.truncate(eff_dim as usize);
            }
            if spec.normalize {
                l2_normalize(&mut vec);
            }
            vectors.push(vec);
        }

        start = end;
    }

    let _ = reply.send(Ok(EmbedOut { vectors, prompt_tokens }));
}

/// L2-normalize in place (no-op for a zero vector). Guarantees a unit vector so cosine == dot and
/// the orchestrator's all-zero-vector rejection never trips on a legitimate embedding.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lookup_by_id_alias_and_normalized() {
        assert_eq!(embed_model_spec("bge-base-en-v1.5").unwrap().dim, 768);
        assert_eq!(embed_model_spec("BGE-BASE-EN-V1.5").unwrap().dim, 768);
        assert_eq!(embed_model_spec("text-embedding-3-small").unwrap().id, "nomic-embed-text-v1.5");
        assert_eq!(embed_model_spec("bge-base-en-v1.5-q4_k_m").unwrap().dim, 768);
        assert_eq!(embed_model_spec("bge-base-en-v1.5.gguf").unwrap().dim, 768);
        assert!(embed_model_spec("gpt-4").is_none());
        assert!(embed_model_spec("qwen-2.5-7b").is_none());
    }

    #[test]
    fn embedding_model_detection() {
        assert!(is_embedding_model("e5-large-v2"));
        assert!(!is_embedding_model("llama-3.2-3b"));
    }

    #[test]
    fn l2_normalize_makes_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_noop() {
        let mut v = vec![0.0f32, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
