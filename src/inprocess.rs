//! In-process inference engine (v1.6.0). Embeds llama.cpp via `llama-cpp-2` — no child
//! process, no localhost HTTP, no IPC. A single dedicated OS worker thread owns the
//! `!Send` model + context for its whole life and serves up to `max_slots` requests
//! CONCURRENTLY via continuous batching: every decode step advances one token for each
//! in-flight sequence in a single `ctx.decode`, exactly like `llama-server --parallel N
//! --cont-batching`. Async callers submit a job over a channel and await on a oneshot.
//!
//! Why a worker thread (not spawn_blocking): `LlamaContext` is `!Send`, so the context
//! must live on one thread for its whole life. Sequences share ONE context (one KV cache
//! partitioned by seq_id) — this is the in-process analogue of llama-server's slots. A
//! fatal inference fault takes the whole process down so launchd/systemd relaunches it
//! clean — the anti-zombie property. Concurrency `max_slots` is advertised as capacity so
//! the orchestrator dispatches at most that many jobs at once.

use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use tokio::sync::mpsc::error::TrySendError;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::inference::{ChatMessage, StreamEvent};

/// Flush a streamed batch every N decoded tokens (mirrors the server path's cadence).
const FLUSH_EVERY: u32 = 6;

/// If a stream consumer falls this far behind (its channel stays Full while we buffer this
/// many bytes of undelivered text), treat it as gone and drop the stream. Prevents one slow
/// or stalled consumer from pinning memory — and, because sends are non-blocking, it can
/// never stall the OTHER slots sharing the single worker thread.
const MAX_STREAM_PENDING_BYTES: usize = 1024 * 1024;

/// Terminal (tail Delta + Done) best-effort delivery budget: try for up to this many short
/// steps before giving up. Bounds the worst-case worker stall at a slot's completion when the
/// consumer's channel is momentarily Full, without ever blocking indefinitely.
const TERMINAL_SEND_TRIES: u32 = 20;
const TERMINAL_SEND_STEP: Duration = Duration::from_millis(10);

/// Prefill the prompt in chunks of this many tokens per `decode` so a long prompt never
/// exceeds the context's logical batch (`n_batch`). Steady-state decode adds at most
/// `max_slots` tokens per step, which is far smaller, so one buffer of this size serves both.
const PREFILL_CHUNK: usize = 512;

/// If a job is in flight but the worker hasn't made token progress in this long, treat it
/// as WEDGED (deadlocked / hung native call) → is_healthy() goes false so the node stops
/// advertising. Generous so a legitimately slow prefill/token never trips it.
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
    /// TOTAL context across all slots (== `max_slots * per_slot_ctx`). This is the KV
    /// cache the context allocates; each sequence uses up to `per_slot_ctx` of it.
    pub n_ctx: u32,
    pub n_gpu_layers: u32,
    /// Continuous-batching concurrency: how many requests decode at once (>= 1).
    pub max_slots: u32,
    /// Per-request context window (the operator's configured context_size).
    pub per_slot_ctx: u32,
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

/// One in-flight sequence occupying a batching slot. Owns its own sampler + KV positions;
/// finished slots are freed (KV cells removed) so a new job can reuse the seq_id.
struct Slot {
    seq_id: i32,
    sampler: LlamaSampler,
    /// Next KV position to write for this sequence (also == tokens decoded so far).
    n_past: i32,
    /// The token to decode on the next step (the previously sampled, already-emitted token).
    cur_token: LlamaToken,
    max_new: i32,
    prompt_tokens: u32,
    completion_tokens: u32,
    kind: JobKind,
    /// PERSISTENT UTF-8 decoder for this request. token_to_str would create a fresh decoder
    /// per token, silently dropping multi-byte characters split across token boundaries
    /// (e.g. emoji); a per-slot decoder carries the partial bytes to the next token.
    decoder: encoding_rs::Decoder,
    // Streaming batching state (per slot).
    pending: String,
    batched: u32,
    /// True once the stream consumer dropped — finish the slot and settle the partial.
    stream_dropped: bool,
    // Non-streaming accumulation.
    out: String,
}

pub struct InProcessEngine {
    /// `Option` so `Drop` can close the channel (last Sender gone → worker loop exits) BEFORE
    /// joining the worker. Always `Some` for a live engine.
    job_tx: Option<Sender<Job>>,
    healthy: Arc<AtomicBool>,
    /// True while the worker is inside a scheduler iteration doing native work (admit/prefill
    /// OR decode). Set BEFORE the native calls so a wedge during a slot's prefill — when no
    /// slot is "active" yet — is still covered by the watchdog. Idle (false) is always healthy.
    busy: Arc<AtomicBool>,
    /// Unix-ms of the last token (or prefill) progress; used to detect a wedged worker.
    last_progress_ms: Arc<AtomicU64>,
    model_name: String,
    max_slots: u32,
    /// `Option` so `Drop` can `take()` + join it. Joining lets the worker fully release its
    /// llama.cpp context/model/backend BEFORE the process runs C++ static destructors at
    /// exit — otherwise ggml-metal's global device teardown asserts (rsets not empty) and
    /// aborts a clean shutdown. Harmless to a long-lived node, but we keep exit clean.
    worker: Option<JoinHandle<()>>,
}

impl InProcessEngine {
    /// Spawn the worker, load the model, and block until ready (or fail).
    pub async fn start(cfg: InProcessConfig) -> Result<Self, String> {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let healthy = Arc::new(AtomicBool::new(false));
        let busy = Arc::new(AtomicBool::new(false));
        let last_progress_ms = Arc::new(AtomicU64::new(0));
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let model_name = cfg.model_name.clone();
        let max_slots = cfg.max_slots.max(1);

        let w_healthy = Arc::clone(&healthy);
        let w_busy = Arc::clone(&busy);
        let w_progress = Arc::clone(&last_progress_ms);
        let worker = std::thread::Builder::new()
            .name("sgl-inference".into())
            .spawn(move || worker_main(cfg, job_rx, w_healthy, w_busy, w_progress, ready_tx))
            .map_err(|e| format!("failed to spawn inference worker: {e}"))?;

        // Bound the startup wait. Model load + context creation are native llama.cpp/Metal
        // calls that CAN wedge (e.g. a cold Metal shader compile, or a KV-cache alloc that
        // never returns). Without a timeout a wedge hangs the WHOLE node forever — it never
        // registers or heartbeats and looks dead with no error in the log. A clean timeout
        // turns that silent infinite hang into a returned Err, so the caller can exit (and
        // the OS service relaunch) instead of stranding the node. Generous (10 min) so a
        // legitimately slow first-run Metal compile + big-model load never trips it.
        const STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);
        match tokio::time::timeout(STARTUP_TIMEOUT, ready_rx).await {
            Ok(Ok(Ok(()))) => Ok(Self {
                job_tx: Some(job_tx),
                healthy,
                busy,
                last_progress_ms,
                model_name,
                max_slots,
                worker: Some(worker),
            }),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Err(_)) => Err("inference worker died during startup".to_string()),
            Err(_) => Err(format!(
                "inference worker startup timed out after {STARTUP_TIMEOUT:?} (model/context load wedged)"
            )),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Continuous-batching concurrency (advertised capacity).
    pub fn max_slots(&self) -> u32 {
        self.max_slots
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
            .as_ref()
            .ok_or_else(|| "inference worker is gone".to_string())?
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
            .as_ref()
            .ok_or_else(|| "inference worker is gone".to_string())?
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

    /// Healthy = model loaded AND (idle OR the worker is still making progress inside its
    /// current scheduler iteration). A wedged worker (busy but no progress for WEDGE_MS —
    /// including a wedge during a job's PREFILL, before any slot is "active") reads as
    /// UNHEALTHY so the heartbeat loop de-advertises it — closing the in-process zombie.
    pub fn is_healthy(&self) -> bool {
        if !self.healthy.load(Ordering::Relaxed) {
            return false;
        }
        if !self.busy.load(Ordering::Relaxed) {
            return true; // idle, model loaded
        }
        now_ms().saturating_sub(self.last_progress_ms.load(Ordering::Relaxed)) < WEDGE_MS
    }
}

impl Drop for InProcessEngine {
    fn drop(&mut self) {
        // Close the job channel first so the worker sees the disconnect, tears down its
        // in-flight + queued jobs (failing their awaits, never billing a partial), releases
        // its llama.cpp context/model/backend, and exits — all within roughly one decode step.
        // Joining before we return lets that teardown finish before the process runs C++ static
        // destructors at exit; skipping it lets ggml-metal's global device teardown race the
        // still-live context and abort a clean shutdown.
        drop(self.job_tx.take());
        let Some(handle) = self.worker.take() else { return };

        // Bounded join: a healthy worker exits fast, but a worker WEDGED inside a native
        // llama.cpp call can never be unblocked from Rust. Rather than hang the process on
        // exit forever, join in a helper and, if it doesn't finish, abort — the OS service
        // relaunches the node clean (the same anti-zombie outcome the watchdog aims for).
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let joiner = std::thread::Builder::new()
            .name("sgl-inference-join".into())
            .spawn(move || {
                let _ = handle.join();
                let _ = tx.send(());
            });
        if joiner.is_err() {
            return; // couldn't spawn the joiner (shutting down); best-effort, don't hang
        }
        if rx.recv_timeout(Duration::from_secs(20)).is_err() {
            tracing::error!("in-process inference worker did not shut down in 20s (wedged native call) — aborting");
            std::process::abort();
        }
    }
}

/// Worker thread: owns the backend + model + one multi-sequence context for its whole life,
/// serving up to `max_slots` jobs concurrently via continuous batching.
fn worker_main(
    cfg: InProcessConfig,
    job_rx: Receiver<Job>,
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

    let max_slots = cfg.max_slots.max(1);

    // Create the inference context ONCE, here at startup — NOT lazily per request. This is
    // the billing-/liveness-critical step: it allocates the KV cache (n_ctx tokens across
    // `max_slots` sequences) on the GPU, which on Metal can be the slowest/heaviest part of
    // bring-up. Doing it before we signal `ready` means "ready" guarantees the model AND a
    // working context, so a context wedge surfaces at startup (behind start()'s timeout)
    // instead of on the first paying request. The context is REUSED for every job; each
    // sequence's KV cells are removed when its slot finishes.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(cfg.n_ctx))
        .with_n_seq_max(max_slots);
    let mut ctx = match model.new_context(&backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("context create failed: {e}")));
            return;
        }
    };

    // One reusable batch buffer. Sized to hold either a prefill chunk (PREFILL_CHUNK) or a
    // full decode step (one token per slot) — whichever is larger. `n_seq_max` here is the
    // max seq_ids PER TOKEN (always 1: a token belongs to exactly one sequence).
    let batch_cap = PREFILL_CHUNK.max(max_slots as usize);
    let mut batch = LlamaBatch::new(batch_cap, 1);

    let mut slots: Vec<Option<Slot>> = (0..max_slots).map(|_| None).collect();
    let mut waiting: VecDeque<Job> = VecDeque::new();

    healthy.store(true, Ordering::Relaxed);
    let _ = ready_tx.send(Ok(()));

    let mut active_count = 0usize;
    loop {
        // When fully idle, block for at least one job so we never busy-spin. A recv() error
        // means every engine handle (Sender) was dropped → begin a bounded, clean shutdown.
        if active_count == 0 && waiting.is_empty() {
            match job_rx.recv() {
                Ok(j) => waiting.push_back(j),
                Err(_) => break, // idle + disconnected → nothing to drain, exit
            }
        }
        // Drain anything else already queued (non-blocking). If the channel is disconnected
        // AND we still have in-flight/queued work, tear down promptly: fail every queued and
        // in-flight job (so callers' awaits resolve to Err instead of hanging), free the KV,
        // and exit. Abandoning in-flight generations on shutdown is the money-SAFE choice —
        // a dropped reply bills nothing, versus risking a partial/forged completion.
        let mut disconnected = false;
        loop {
            match job_rx.try_recv() {
                Ok(j) => waiting.push_back(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            for slot in slots.iter_mut() {
                if let Some(s) = slot.take() {
                    finish_err(&mut ctx, s, "node shutting down");
                }
            }
            while let Some(job) = waiting.pop_front() {
                let _ = finish_admit_err(job.kind, "node shutting down".to_string());
            }
            break;
        }

        // Mark busy + seed the progress timestamp BEFORE the native work so a wedge anywhere
        // in this iteration (INCLUDING a prefill decode, when no slot is "active" yet) trips
        // the is_healthy() watchdog after WEDGE_MS instead of reading as idle-healthy forever.
        busy.store(true, Ordering::Relaxed);
        progress.store(now_ms(), Ordering::Relaxed);

        // Run one scheduler iteration (admit into free slots + one unified decode step).
        // A Rust panic mid-iteration means a broken invariant on the SHARED context — make
        // it FATAL so the OS service relaunches the node clean (the anti-zombie property).
        // Native llama.cpp aborts/segfaults already kill the process; this covers the
        // Rust-panic case without forcing global panic=abort on the rest of the node.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_iteration(&mut ctx, &mut batch, &mut slots, &mut waiting, &cfg, &progress);
        }));
        if outcome.is_err() {
            tracing::error!("inference worker panicked mid-iteration — aborting for a clean OS restart");
            std::process::abort();
        }

        busy.store(false, Ordering::Relaxed);
        active_count = slots.iter().filter(|s| s.is_some()).count();
    }
    healthy.store(false, Ordering::Relaxed);
    busy.store(false, Ordering::Relaxed);
}

/// One scheduler iteration: admit waiting jobs into free slots (prefill + first token), then
/// advance every active slot by exactly one token in a single batched `decode`.
fn run_iteration(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    slots: &mut [Option<Slot>],
    waiting: &mut VecDeque<Job>,
    cfg: &InProcessConfig,
    progress: &AtomicU64,
) {
    // 1. Admit: fill free slots from the waiting queue.
    for idx in 0..slots.len() {
        if slots[idx].is_some() {
            continue;
        }
        let Some(job) = waiting.pop_front() else { break };
        let seq_id = idx as i32;
        match admit(ctx, batch, seq_id, cfg, progress, job) {
            AdmitOutcome::Active(slot) => slots[idx] = Some(slot),
            AdmitOutcome::Finished => { /* replied already; slot stays free */ }
        }
    }

    // 2. Unified decode step: one token per active slot in a single decode call.
    batch.clear();
    let mut order: Vec<usize> = Vec::with_capacity(slots.len());
    for (idx, slot) in slots.iter().enumerate() {
        if let Some(slot) = slot {
            if batch
                .add(slot.cur_token, slot.n_past, &[slot.seq_id], true)
                .is_err()
            {
                // Should never happen (batch sized for all slots); skip this step defensively.
                continue;
            }
            order.push(idx);
        }
    }
    if order.is_empty() {
        return;
    }

    if let Err(e) = ctx.decode(batch) {
        // A decode failure poisons the whole in-flight batch (shared context) — fail every
        // sequence in this step with the error and free their slots. The node stays up.
        let msg = format!("decode failed: {e}");
        for &idx in &order {
            if let Some(slot) = slots[idx].take() {
                finish_err(ctx, slot, &msg);
            }
        }
        return;
    }

    let now = now_ms();
    for (i, &idx) in order.iter().enumerate() {
        // Sample the NEXT token for this sequence from its logits row (batch index i).
        let (token, is_eog) = {
            let slot = slots[idx].as_mut().expect("active slot present");
            let tok = slot.sampler.sample(&*ctx, i as i32);
            slot.sampler.accept(tok);
            slot.n_past += 1; // cur_token now occupies its position; advance the write head
            (tok, ctx.model.is_eog_token(tok))
        };
        progress.store(now, Ordering::Relaxed);

        // End-of-generation: llama-server counts the terminal stop token for billing parity.
        if is_eog {
            let mut slot = slots[idx].take().expect("active slot present");
            slot.completion_tokens += 1;
            finish_ok(ctx, slot);
            continue;
        }

        // Emit the sampled token, then decide whether to stop (length / context cap).
        let mut done = false;
        {
            let slot = slots[idx].as_mut().expect("active slot present");
            slot.completion_tokens += 1;
            // Persistent per-slot decoder: multi-byte chars split across tokens survive.
            let piece = ctx
                .model
                .token_to_piece(token, &mut slot.decoder, false, None)
                .unwrap_or_default();
            emit(slot, piece);
            slot.cur_token = token;

            if slot.stream_dropped
                || slot.completion_tokens as i32 >= slot.max_new
                || slot.n_past as u32 >= cfg.per_slot_ctx
            {
                done = true;
            }
        }
        if done {
            let slot = slots[idx].take().expect("active slot present");
            finish_ok(ctx, slot);
        }
    }
}

enum AdmitOutcome {
    Active(Slot),
    Finished,
}

/// Render + tokenize + prefill a new job into `seq_id`, then sample its first token. Returns
/// an active Slot ready for the decode loop, or Finished if it completed immediately (empty
/// prompt / error / instant EOG / max_tokens==1) — in which case the reply is already sent.
fn admit(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    seq_id: i32,
    cfg: &InProcessConfig,
    progress: &AtomicU64,
    job: Job,
) -> AdmitOutcome {
    let Job { messages, max_tokens, temperature, kind } = job;
    let model = ctx.model;

    // Defensive: clear any stale KV cells for this seq_id before reuse (positions restart
    // at 0 for every new job, so leftover state would corrupt positions + billing).
    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);

    let prompt = match render_chat_prompt(model, &messages) {
        Ok(p) => p,
        Err(e) => return finish_admit_err(kind, e),
    };
    let tokens = match model.str_to_token(&prompt, AddBos::Never) {
        Ok(t) => t,
        Err(e) => return finish_admit_err(kind, format!("tokenize failed: {e}")),
    };
    let prompt_tokens = tokens.len() as u32;
    if tokens.is_empty() {
        return finish_admit_err(kind, "empty prompt after templating".to_string());
    }
    if prompt_tokens >= cfg.per_slot_ctx {
        return finish_admit_err(kind, "prompt exceeds context window".to_string());
    }

    // Chunked prefill: decode the prompt in PREFILL_CHUNK-sized pieces so a long prompt
    // never exceeds the context's logical batch. Only the final token needs logits.
    // EVERY error return after the first chunk decoded must clear this seq's KV — leaving
    // partial prefill cells behind would poison the next job that reuses the seq_id.
    let last = tokens.len() - 1;
    let mut pos = 0usize;
    while pos < tokens.len() {
        let end = (pos + PREFILL_CHUNK).min(tokens.len());
        batch.clear();
        for i in pos..end {
            if let Err(e) = batch.add(tokens[i], i as i32, &[seq_id], i == last) {
                let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                return finish_admit_err(kind, format!("prefill batch add failed: {e}"));
            }
        }
        if let Err(e) = ctx.decode(batch) {
            let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
            return finish_admit_err(kind, format!("prompt decode failed: {e}"));
        }
        pos = end;
    }
    progress.store(now_ms(), Ordering::Relaxed); // prefill done

    let mut sampler = if temperature > 0.0 {
        LlamaSampler::chain_simple([LlamaSampler::temp(temperature), LlamaSampler::dist(0)])
    } else {
        LlamaSampler::greedy()
    };

    let max_new = max_tokens.max(1);
    // Sample the first generated token from the last prompt token's logits.
    let first = sampler.sample(&*ctx, batch.n_tokens() - 1);
    sampler.accept(first);
    let n_past = tokens.len() as i32; // positions 0..last consumed; next write head = len

    let mut slot = Slot {
        seq_id,
        sampler,
        n_past,
        cur_token: first,
        max_new,
        prompt_tokens,
        completion_tokens: 0,
        kind,
        decoder: encoding_rs::UTF_8.new_decoder(),
        pending: String::new(),
        batched: 0,
        stream_dropped: false,
        out: String::new(),
    };

    // First token is EOG → empty completion, count the stop token (server parity).
    if model.is_eog_token(first) {
        slot.completion_tokens = 1;
        finish_ok(ctx, slot);
        return AdmitOutcome::Finished;
    }

    // Emit the first token (persistent per-slot decoder — see Slot::decoder).
    slot.completion_tokens = 1;
    let piece = model
        .token_to_piece(first, &mut slot.decoder, false, None)
        .unwrap_or_default();
    emit(&mut slot, piece);
    progress.store(now_ms(), Ordering::Relaxed);

    // max_tokens == 1 (or the stream consumer already vanished) → finish now.
    if slot.stream_dropped || slot.completion_tokens as i32 >= slot.max_new {
        finish_ok(ctx, slot);
        return AdmitOutcome::Finished;
    }

    AdmitOutcome::Active(slot)
}

/// Append a decoded piece to a slot: stream it (flushing every FLUSH_EVERY tokens) or
/// accumulate it for a non-streaming reply. Streaming uses a NON-BLOCKING `try_send` so a
/// slow/stalled consumer can never block the shared worker thread (which serves every other
/// slot). If the consumer's channel stays Full past MAX_STREAM_PENDING_BYTES, or is closed,
/// the stream is marked dropped and the slot settles the partial.
fn emit(slot: &mut Slot, piece: String) {
    // Clone the Sender out so we can mutate `slot` (pending/batched/stream_dropped) without
    // holding an immutable borrow of `slot.kind`.
    let tokens = match &slot.kind {
        JobKind::Stream { tokens, .. } => tokens.clone(),
        JobKind::NonStream(_) => {
            slot.out.push_str(&piece);
            return;
        }
    };
    slot.pending.push_str(&piece);
    slot.batched += 1;
    if slot.batched >= FLUSH_EVERY {
        flush_stream_nonblocking(slot, &tokens);
    }
}

/// Try (non-blocking) to hand the buffered deltas to the consumer. On Full, keep the buffer
/// intact and retry on the next flush; only give up (drop) once the backlog exceeds the cap.
fn flush_stream_nonblocking(slot: &mut Slot, tokens: &tokio::sync::mpsc::Sender<StreamEvent>) {
    if slot.pending.is_empty() {
        return;
    }
    let ev = StreamEvent::Delta { text: slot.pending.clone(), tokens: slot.batched };
    match tokens.try_send(ev) {
        Ok(()) => {
            slot.pending.clear();
            slot.batched = 0;
        }
        Err(TrySendError::Full(_)) => {
            if slot.pending.len() > MAX_STREAM_PENDING_BYTES {
                slot.stream_dropped = true;
            }
        }
        Err(TrySendError::Closed(_)) => {
            slot.stream_dropped = true;
        }
    }
}

/// Best-effort terminal send (tail Delta / Done). Bounded retry so a momentarily-Full channel
/// still gets the final counts, but the worker never blocks indefinitely. Returns false if the
/// consumer is gone or stayed Full past the budget.
fn send_terminal(tokens: &tokio::sync::mpsc::Sender<StreamEvent>, ev: StreamEvent) -> bool {
    let mut ev = ev;
    for _ in 0..TERMINAL_SEND_TRIES {
        match tokens.try_send(ev) {
            Ok(()) => return true,
            Err(TrySendError::Full(returned)) => {
                ev = returned;
                std::thread::sleep(TERMINAL_SEND_STEP);
            }
            Err(TrySendError::Closed(_)) => return false,
        }
    }
    false
}

/// Successful terminal: flush any pending stream tail + send the final counts, or return the
/// accumulated non-stream text. Frees the sequence's KV cells for reuse.
fn finish_ok(ctx: &mut LlamaContext, slot: Slot) {
    let Slot {
        seq_id,
        prompt_tokens,
        completion_tokens,
        kind,
        mut pending,
        batched,
        stream_dropped,
        out,
        ..
    } = slot;
    let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
    match kind {
        JobKind::NonStream(reply) => {
            let _ = reply.send(Ok(GenOut { content: out, prompt_tokens, completion_tokens }));
        }
        JobKind::Stream { tokens, done } => {
            if !stream_dropped {
                // Done is gated on the tail Delta actually delivering: sending Done after a
                // failed tail would bill the FULL count for visibly truncated text. If the
                // tail (or Done itself) can't be delivered, report a DELIVERY FAILURE on the
                // done channel — the node's defined stream-failure path: no forged terminal,
                // nothing billed (fail-safe for the buyer).
                let tail_delivered = if pending.is_empty() {
                    true
                } else {
                    send_terminal(
                        &tokens,
                        StreamEvent::Delta { text: std::mem::take(&mut pending), tokens: batched },
                    )
                };
                let finished = tail_delivered
                    && send_terminal(&tokens, StreamEvent::Done { prompt_tokens, completion_tokens });
                let _ = done.send(if finished {
                    Ok(())
                } else {
                    Err("stream delivery failed: consumer stalled at completion".to_string())
                });
            } else {
                // Consumer vanished mid-generation (client abort): the caller's client_gone
                // path settles the partial from the deltas it already relayed.
                let _ = done.send(Ok(()));
            }
        }
    }
}

/// Failed terminal: report the error to the caller (NO forged `Done` on the stream path, so
/// a client never sees a truncated generation as success). Frees the sequence's KV cells.
fn finish_err(ctx: &mut LlamaContext, slot: Slot, msg: &str) {
    let _ = ctx.clear_kv_cache_seq(Some(slot.seq_id as u32), None, None);
    match slot.kind {
        JobKind::NonStream(reply) => {
            let _ = reply.send(Err(msg.to_string()));
        }
        JobKind::Stream { done, .. } => {
            let _ = done.send(Err(msg.to_string()));
        }
    }
}

/// Error before a slot was ever placed (during admit, no KV to clear beyond the caller's).
fn finish_admit_err(kind: JobKind, msg: String) -> AdmitOutcome {
    match kind {
        JobKind::NonStream(reply) => {
            let _ = reply.send(Err(msg));
        }
        JobKind::Stream { done, .. } => {
            let _ = done.send(Err(msg));
        }
    }
    AdmitOutcome::Finished
}

/// Render the chat prompt EXACTLY like llama-server: parse the GGUF's jinja chat template
/// (tokenizer.chat_template) with minijinja and feed it the same context — `messages`,
/// `add_generation_prompt=true`, the model's `bos_token`, plus `strftime_now` so Llama-3's
/// "Today Date" preamble matches. This closes the legacy-vs-jinja parity gap (prompt token
/// counts + output) so billing is identical across the server and in-process engines.
///
/// SYSTEM-ROLE FALLBACK: some templates (e.g. Gemma) `raise_exception` on a `system`
/// message. llama.cpp's legacy path handles those models by MERGING the system prompt
/// into the first user message — so if the faithful render fails and the conversation
/// has system messages, we fold them into the first user turn and render once more.
/// Matches llama-server behavior instead of failing every chat that sets a system prompt.
fn render_chat_prompt(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String, String> {
    match try_render_chat_prompt(model, messages) {
        Ok(p) => Ok(p),
        Err(first_err) => {
            if !messages.iter().any(|m| m.role == "system") {
                return Err(first_err);
            }
            let folded = fold_system_into_first_user(messages);
            try_render_chat_prompt(model, &folded).map_err(|_| first_err)
        }
    }
}

/// Merge all `system` messages into the first non-system message as a prefixed preamble
/// (mirrors llama.cpp's legacy handling for templates without a system role). If the
/// conversation is ONLY system messages, they become a single user message.
fn fold_system_into_first_user(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let system_text = messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut injected = false;
    for m in messages {
        if m.role == "system" {
            continue;
        }
        if !injected {
            out.push(ChatMessage {
                role: m.role.clone(),
                content: format!("{system_text}\n\n{}", m.content),
            });
            injected = true;
        } else {
            out.push(m.clone());
        }
    }
    if !injected {
        out.push(ChatMessage { role: "user".to_string(), content: system_text });
    }
    out
}

fn try_render_chat_prompt(model: &LlamaModel, messages: &[ChatMessage]) -> Result<String, String> {
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
