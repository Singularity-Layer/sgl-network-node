use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command};
use tokio::time::{sleep, Duration};

/// Sentinel for `gpu_layers` meaning "auto-fit": the server engine omits `-ngl` so llama.cpp
/// sizes the GPU offload to available VRAM (spilling the rest to system RAM — it can't OOM),
/// and the in-process engine offloads all layers (Mac unified memory). Set by `sgl start
/// --gpu-layers-auto`. This is the universal replacement for hardcoded per-model VRAM caps.
pub const GPU_LAYERS_AUTO: u32 = u32::MAX;

fn inference_transport_error(err: reqwest::Error) -> String {
    tracing::warn!("local inference backend request failed: {err}");

    if err.is_connect() || err.is_timeout() {
        "Inference backend unavailable; retry shortly".to_string()
    } else {
        "Inference backend request failed".to_string()
    }
}

fn retryable_inference_transport_error(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}

pub struct InferenceEngineConfig {
    pub model_path: PathBuf,
    pub model_name: String,
    pub port: u16,
    pub threads: u32,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub batch_size: u32,
    pub parallel_slots: u32,
}

/// The original child-process engine (spawns `llama-server`, talks HTTP). Kept as one
/// arm of the `InferenceEngine` enum for A/B during the in-process rollout.
pub struct ServerEngine {
    // Behind a Mutex so the engine can be shared across the heartbeat loop + job handlers
    // AND still be (re)started/stopped via &self. The guard is never held across an await
    // (we lock only to swap the Child handle).
    server_process: std::sync::Mutex<Option<Child>>,
    client: Client,
    base_url: String,
    config: InferenceEngineConfig,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    // Opaque JSON array, forwarded verbatim to llama-server. Passing it through (rather than
    // re-serializing a strict {role,content}) preserves tool-calling round-trips: assistant
    // messages carrying `tool_calls`, `tool`-role messages with `tool_call_id`/`name`, and
    // null `content` all survive — which is what an agent loop (OpenCode/Cline) needs.
    messages: serde_json::Value,
    temperature: f64,
    max_tokens: i32,
    stream: bool,
    // OpenAI tool-calling passthrough (needs llama-server `--jinja`). Omitted when absent so
    // non-tool requests are byte-identical to before.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatCompletionStreamRequest {
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: i32,
    stream: bool,
    stream_options: StreamOptions,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
}

#[derive(Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
}

/// One event from a streaming completion: a batch of decoded text (with the
/// number of tokens it carries, for partial billing), or the terminal marker
/// carrying final token counts. A generation failure is signalled by the
/// function returning `Err` WITHOUT a `Done` — never a forged terminal.
pub enum StreamEvent {
    Delta { text: String, tokens: u32 },
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: RespMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Assistant reply from llama-server. `content` is Option because a tool-call-only reply sets
/// it to null; `tool_calls` carries the OpenAI function-call array verbatim when the model
/// invokes a tool (populated only when the server ran with `--jinja`).
#[derive(Deserialize)]
struct RespMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[allow(dead_code)] // total_tokens is part of the llama.cpp response shape
struct UsageInfo {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct HealthResponse {
    status: Option<String>,
}

impl ServerEngine {
    pub fn new(mut config: InferenceEngineConfig) -> Self {
        let port = find_available_port(config.port);
        config.port = port;
        let base_url = format!("http://127.0.0.1:{}", port);
        Self {
            server_process: std::sync::Mutex::new(None),
            client: Client::new(),
            base_url,
            config,
        }
    }

    pub async fn start(&self) -> Result<(), String> {
        // Non-reentrant: kill any existing child first so a second start() (or a manual
        // call) can never orphan a running llama-server. No-op on the initial start.
        self.stop();
        if !self.config.model_path.exists() {
            return Err(format!(
                "Model file not found: {}",
                self.config.model_path.display()
            ));
        }

        // Qwen-family GGUFs (verified live on qwen-coder) ship chat templates WITHOUT a tools
        // section, so llama-server can't grammar-constrain or parse tool calls — the model
        // freestyles `<function-call>`/`<xml>` text (upstream ggml-org/llama.cpp#12279). Override
        // with the official Qwen2.5 tool-enabled template (teaches `<tool_call>`, which the
        // server parses natively into structured tool_calls). Excludes R1/QwQ distills — those
        // need their own reasoning templates. Written to a temp file each start.
        let template_file: Option<String> = if qwen_template_override(&self.config.model_path) {
            match write_qwen_tools_template() {
                Ok(p) => {
                    tracing::info!("using Qwen2.5 tool-enabled chat template override");
                    Some(p)
                }
                Err(e) => {
                    tracing::warn!("failed to write Qwen template override (continuing without): {e}");
                    None
                }
            }
        } else {
            None
        };

        let llama_server = find_llama_server()?;
        tracing::info!(
            "Starting llama-server with model: {}",
            self.config.model_path.display()
        );

        let port_str = self.config.port.to_string();
        let threads_str = self.config.threads.to_string();
        let gpu_layers_str = self.config.gpu_layers.to_string();
        let ctx_str = self.config.context_size.to_string();
        let batch_str = self.config.batch_size.to_string();
        let parallel_str = self.config.parallel_slots.to_string();

        let model_str = self.config.model_path.to_string_lossy();
        let mut args: Vec<&str> = vec![
            "-m", &model_str,
            "--host", "127.0.0.1",
            "--port", &port_str,
            "-c", &ctx_str,
            "-t", &threads_str,
            "-b", &batch_str,
            "--parallel", &parallel_str,
            // Continuous batching: interleave multiple in-flight requests in one
            // decode loop so the node serves `--parallel` jobs concurrently instead
            // of one-at-a-time. This is what stops "busy after a single request".
            "--cont-batching",
            // Keep reasoning-model thought tags INLINE in `content` (<think>...</think>)
            // instead of the default, which extracts them into a separate `reasoning_content`
            // field that we don't forward — so the browser never saw the reasoning. With
            // `none`, DeepSeek-R1 / QwQ etc. stream their thinking inline and the playground's
            // reasoning parser renders the "Thinking" box. No-op for non-reasoning models.
            "--reasoning-format", "none",
            // Use the model's embedded chat template (minijinja) so OpenAI `tools`/`tool_choice`
            // activate the tool grammar and the reply carries structured `tool_calls` — the
            // unlock for agentic coding clients (OpenCode/Cline/Roo). Without --jinja, tool
            // definitions are ignored and the model answers as plain text. NOTE: this switches
            // EVERY served model to its GGUF-embedded template, so each model must be verified
            // on a real build before release (chat parity + reasoning-inline + tool_calls).
            "--jinja",
        ];
        // Auto-fit: OMIT -ngl so llama.cpp sizes the GPU offload to available VRAM (can't OOM —
        // spills the rest to system RAM). Otherwise pin the requested layer count.
        if self.config.gpu_layers != GPU_LAYERS_AUTO {
            args.push("-ngl");
            args.push(&gpu_layers_str);
        }
        if let Some(p) = &template_file {
            args.push("--chat-template-file");
            args.push(p);
        }
        let mut child = Command::new(&llama_server)
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start llama-server: {e}"))?;

        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(line) if !line.trim().is_empty() => {
                            tracing::warn!(target: "llama_server", "{line}");
                        }
                        Ok(_) => {}
                        Err(err) => {
                            tracing::warn!(target: "llama_server", "stderr read failed: {err}");
                            break;
                        }
                    }
                }
            });
        }

        // Store the new child (replacing any prior handle). Lock is dropped before the
        // await loop below — never hold a std Mutex guard across .await.
        { *self.server_process.lock().unwrap() = Some(child); }

        for i in 0..30 {
            sleep(Duration::from_secs(1)).await;
            if self.health_check().await {
                tracing::info!("llama-server ready after {i}s");
                return Ok(());
            }
        }

        self.stop();
        Err("llama-server failed to start within 30s".to_string())
    }

    pub fn stop(&self) {
        // take() the child out under the lock, then kill+wait (all sync; no await).
        let child = self.server_process.lock().unwrap().take();
        if let Some(mut child) = child {
            let _ = child.kill();
            let _ = child.wait();
            tracing::info!("llama-server stopped");
        }
    }

    /// Supervised restart: kill the (crashed/hung) llama-server and start a fresh one.
    /// Called by the heartbeat loop when /health has failed repeatedly, so a node whose
    /// engine OOM'd or crashed mid-run self-heals WITHOUT the operator restarting it.
    /// Returns Ok once the new server answers /health (start() waits up to 30s).
    pub async fn restart(&self) -> Result<(), String> {
        tracing::warn!("llama-server unhealthy — restarting the inference engine");
        self.stop();
        // Brief pause so the OS reclaims the port + GPU/unified memory before relaunch.
        sleep(Duration::from_secs(2)).await;
        self.start().await
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        // Bounded so start()'s 30s readiness wait can't hang on a /health that accepts
        // the connection but never responds.
        match self.client.get(&url).timeout(Duration::from_secs(3)).send().await {
            Ok(resp) => {
                if let Ok(h) = resp.json::<HealthResponse>().await {
                    h.status.as_deref() == Some("ok")
                } else {
                    false
                }
            }
            Err(_) => false,
        }
    }

    /// Liveness probe for the heartbeat loop, with a SHORT timeout so a hung or dead
    /// llama-server is detected fast (instead of blocking). The node uses this to stop
    /// advertising its model when the engine isn't actually serving — which is what
    /// prevents "ghost" jobs (a node that heartbeats fine but whose llama-server has
    /// crashed/OOM'd mid-run, e.g. a 14B model on a too-small box). Self-healing: once
    /// the engine answers /health again, the node re-advertises automatically.
    pub async fn is_healthy(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(3))
            .send()
            .await
        {
            Ok(resp) => resp
                .json::<HealthResponse>()
                .await
                .map(|h| h.status.as_deref() == Some("ok"))
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    pub async fn chat_completion(
        &self,
        messages: serde_json::Value,
        temperature: f64,
        max_tokens: i32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
    ) -> Result<InferenceResult, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let body = ChatCompletionRequest {
            messages,
            temperature,
            max_tokens,
            stream: false,
            tools,
            tool_choice,
        };

        let mut resp = None;
        for attempt in 0..3 {
            match self.client.post(&url).json(&body).send().await {
                Ok(r) => {
                    resp = Some(r);
                    break;
                }
                Err(e) if retryable_inference_transport_error(&e) && attempt < 2 => {
                    tracing::warn!(
                        "local inference backend transient request failure (attempt {}/3): {e}",
                        attempt + 1
                    );
                    sleep(Duration::from_millis(250 * (attempt + 1) as u64)).await;
                }
                Err(e) => return Err(inference_transport_error(e)),
            }
        }
        let resp = resp.ok_or_else(|| "Inference backend unavailable; retry shortly".to_string())?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Inference failed: {text}"));
        }

        let result: ChatCompletionResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse inference response: {e}"))?;

        let choice = result.choices.first().ok_or("No completion returned")?;

        Ok(InferenceResult {
            // A tool-call-only reply has null content — normalise to "" so downstream stays simple.
            content: choice.message.content.clone().unwrap_or_default(),
            model: self.config.model_name.clone(),
            prompt_tokens: result
                .usage
                .as_ref()
                .and_then(|u| u.prompt_tokens)
                .unwrap_or(0),
            completion_tokens: result
                .usage
                .as_ref()
                .and_then(|u| u.completion_tokens)
                .unwrap_or(0),
            tool_calls: choice.message.tool_calls.clone(),
            finish_reason: choice.finish_reason.clone(),
        })
    }

    /// Streaming chat completion. Calls llama-server with `stream:true`, reads the
    /// SSE response incrementally (via `resp.chunk()` — no extra deps), batches
    /// decoded tokens, and forwards them over `tx` as `StreamEvent`s. The caller
    /// seals each batch and relays it. Ends with a `Done` carrying token counts.
    ///
    /// Returns early (Ok) if the receiver is dropped — that means the consumer
    /// (and ultimately the browser) went away, so we stop generating.
    pub async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f64,
        max_tokens: i32,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<(), String> {
        // Flush a batch every N decoded deltas. Keeps the POST count bounded
        // (~tokens/N requests) while still feeling live. Time-based flushing is a
        // later refinement; count-based is enough for current model speeds.
        const FLUSH_EVERY: u32 = 6;
        const MAX_SSE_BUF: usize = 2 * 1024 * 1024;

        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatCompletionStreamRequest {
            messages,
            temperature,
            max_tokens,
            stream: true,
            stream_options: StreamOptions {
                include_usage: true,
            },
        };

        let mut resp = None;
        for attempt in 0..3 {
            match self.client.post(&url).json(&body).send().await {
                Ok(r) => {
                    resp = Some(r);
                    break;
                }
                Err(e) if retryable_inference_transport_error(&e) && attempt < 2 => {
                    tracing::warn!(
                        "local inference backend transient stream request failure (attempt {}/3): {e}",
                        attempt + 1
                    );
                    sleep(Duration::from_millis(250 * (attempt + 1) as u64)).await;
                }
                Err(e) => return Err(inference_transport_error(e)),
            }
        }
        let mut resp =
            resp.ok_or_else(|| "Inference backend unavailable; retry shortly".to_string())?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Inference failed: {text}"));
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut pending = String::new();
        let mut batched: u32 = 0; // tokens accumulated in `pending`
        let mut emitted: u32 = 0; // total content deltas seen (fallback count)
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;

        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| format!("stream read failed: {e}"))?
        {
            if buf.len() + chunk.len() > MAX_SSE_BUF {
                return Err("SSE buffer exceeded maximum size".to_string());
            }
            buf.extend_from_slice(&chunk);

            // Drain complete newline-delimited lines from the buffer.
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let data = match line.strip_prefix("data:") {
                    Some(d) => d.trim(),
                    None => continue,
                };
                if data == "[DONE]" {
                    // Clean end of stream → flush the tail, then the terminal Done.
                    if !pending.is_empty() {
                        let _ = tx
                            .send(StreamEvent::Delta {
                                text: std::mem::take(&mut pending),
                                tokens: batched,
                            })
                            .await;
                    }
                    let c = if completion_tokens > 0 { completion_tokens } else { emitted };
                    let _ = tx
                        .send(StreamEvent::Done {
                            prompt_tokens,
                            completion_tokens: c,
                        })
                        .await;
                    return Ok(());
                }
                let parsed: StreamChunk = match serde_json::from_str(data) {
                    Ok(p) => p,
                    Err(_) => continue, // ignore keep-alives / non-JSON lines
                };
                if let Some(u) = parsed.usage {
                    prompt_tokens = u.prompt_tokens.unwrap_or(prompt_tokens);
                    completion_tokens = u.completion_tokens.unwrap_or(completion_tokens);
                }
                for ch in parsed.choices {
                    if let Some(c) = ch.delta.content {
                        if !c.is_empty() {
                            pending.push_str(&c);
                            batched += 1;
                            emitted += 1;
                            if batched >= FLUSH_EVERY {
                                if tx
                                    .send(StreamEvent::Delta {
                                        text: std::mem::take(&mut pending),
                                        tokens: batched,
                                    })
                                    .await
                                    .is_err()
                                {
                                    // Consumer dropped (client aborted). Stop
                                    // generating; the node settles the partial.
                                    return Ok(());
                                }
                                batched = 0;
                            }
                        }
                    }
                }
            }
        }

        // Upstream ended WITHOUT a `[DONE]` sentinel → treat as a generation failure
        // (truncation), NOT a successful completion. The caller fails the job and
        // aborts the client stream; the client never sees a forged `final`.
        Err("inference stream ended without [DONE] (truncated)".to_string())
    }
}

impl Drop for ServerEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct InferenceResult {
    pub content: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// OpenAI `tool_calls` array when the model invoked a tool (else None). Forwarded verbatim
    /// to the orchestrator → client so agentic clients can drive the tool loop.
    pub tool_calls: Option<serde_json::Value>,
    /// llama-server's `finish_reason` ("stop" | "tool_calls" | "length" | ...). None on the
    /// in-process engine (which doesn't surface it).
    pub finish_reason: Option<String>,
}

/// Which inference backend the node runs (`--engine`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineMode {
    /// Spawn `llama-server` as a child process and talk HTTP (legacy).
    Server,
    /// Embed llama.cpp in-process (no child, no localhost, no IPC). Requires the
    /// `inprocess` build feature; the node is one process so a crash relaunches clean.
    InProcess,
}

impl EngineMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "server" => Ok(EngineMode::Server),
            "inprocess" | "in-process" => Ok(EngineMode::InProcess),
            other => Err(format!("unknown --engine '{other}' (use 'server' or 'inprocess')")),
        }
    }
}

/// The node's inference engine. Dispatches to the legacy child-process `ServerEngine` or
/// the in-process `llama-cpp-2` engine. Public API matches what node.rs / ws.rs already
/// call, so the rest of the node is engine-agnostic.
pub enum InferenceEngine {
    Server(ServerEngine),
    #[cfg(feature = "inprocess")]
    InProcess(crate::inprocess::InProcessEngine),
}

impl InferenceEngine {
    /// Build + start the chosen engine.
    pub async fn create(config: InferenceEngineConfig, mode: EngineMode) -> Result<Self, String> {
        match mode {
            EngineMode::Server => {
                let e = ServerEngine::new(config);
                e.start().await?;
                Ok(InferenceEngine::Server(e))
            }
            EngineMode::InProcess => {
                #[cfg(feature = "inprocess")]
                {
                    // `config.context_size` is the TOTAL context (slots × per-slot). Recover
                    // the per-slot window so the engine enforces per-request limits and does
                    // continuous batching across `parallel_slots` sequences.
                    let max_slots = config.parallel_slots.max(1);
                    let per_slot_ctx = (config.context_size / max_slots).max(1);
                    let e = crate::inprocess::InProcessEngine::start(crate::inprocess::InProcessConfig {
                        model_path: config.model_path,
                        model_name: config.model_name,
                        n_ctx: config.context_size,
                        // Auto-fit on the in-process engine (Mac unified memory): offload all layers.
                        n_gpu_layers: if config.gpu_layers == GPU_LAYERS_AUTO { 999 } else { config.gpu_layers },
                        max_slots,
                        per_slot_ctx,
                    })
                    .await?;
                    Ok(InferenceEngine::InProcess(e))
                }
                #[cfg(not(feature = "inprocess"))]
                {
                    let _ = config;
                    Err("--engine=inprocess requires a build with the `inprocess` feature".to_string())
                }
            }
        }
    }

    pub async fn is_healthy(&self) -> bool {
        match self {
            InferenceEngine::Server(e) => e.is_healthy().await,
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(e) => e.is_healthy(),
        }
    }

    pub fn stop(&self) {
        match self {
            InferenceEngine::Server(e) => e.stop(),
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => { /* worker stops when the engine is dropped */ }
        }
    }

    pub async fn restart(&self) -> Result<(), String> {
        match self {
            InferenceEngine::Server(e) => e.restart().await,
            // In-process: there is no child process to restart — the engine IS this process,
            // and a wedged native llama.cpp call can never be recovered from Rust. The node
            // only calls restart() after a sustained unhealthy streak, so the correct (and
            // only) recovery is a clean process relaunch by the OS service / desktop app —
            // the anti-zombie property. Returning Ok(()) here would make the health loop
            // believe a "restart" happened and let a wedged process linger unadvertised.
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => {
                tracing::error!(
                    "in-process engine unhealthy — aborting for a clean OS relaunch (in-place restart is not possible)"
                );
                std::process::abort();
            }
        }
    }

    pub async fn chat_completion(
        &self,
        messages: serde_json::Value,
        temperature: f64,
        max_tokens: i32,
        tools: Option<serde_json::Value>,
        tool_choice: Option<serde_json::Value>,
    ) -> Result<InferenceResult, String> {
        match self {
            InferenceEngine::Server(e) => {
                e.chat_completion(messages, temperature, max_tokens, tools, tool_choice).await
            }
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(e) => {
                // In-process (macOS unified-memory) engine is chat-only for v1 — tool-calling is
                // a server-engine (Linux GPU) feature. Deserialize the opaque messages into the
                // strict {role,content} shape it renders; tools are ignored here.
                let _ = (&tools, &tool_choice);
                let msgs: Vec<ChatMessage> = serde_json::from_value(messages)
                    .map_err(|e| format!("Invalid messages format: {e}"))?;
                let out = e.chat_completion(&msgs, max_tokens, temperature as f32).await?;
                Ok(InferenceResult {
                    content: out.content,
                    model: e.model_name().to_string(),
                    prompt_tokens: out.prompt_tokens,
                    completion_tokens: out.completion_tokens,
                    tool_calls: None,
                    finish_reason: None,
                })
            }
        }
    }

    pub async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f64,
        max_tokens: i32,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<(), String> {
        match self {
            InferenceEngine::Server(e) => {
                e.chat_completion_stream(messages, temperature, max_tokens, tx).await
            }
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(e) => {
                e.chat_completion_stream(&messages, max_tokens, temperature as f32, tx).await
            }
        }
    }
}

/// Official Qwen2.5 tool-enabled chat template (from llama.cpp `models/templates/`, Apache-2.0).
/// Embedded so the override needs no network or extra release asset.
const QWEN25_TOOLS_TEMPLATE: &str = include_str!("templates/qwen2.5-tools.jinja");

/// Should this model get the Qwen tool-template override? Qwen chat/coder models: yes (their
/// GGUF templates lack a tools branch). R1/QwQ distills: NO — they carry reasoning templates
/// that the override would clobber (breaking `<think>` output).
fn qwen_template_override(model_path: &std::path::Path) -> bool {
    let name = model_path
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    name.contains("qwen") && !name.contains("r1") && !name.contains("qwq")
}

/// Write the embedded template to a temp file for `--chat-template-file`.
fn write_qwen_tools_template() -> Result<String, String> {
    let p = std::env::temp_dir().join("sgl-qwen25-tools.jinja");
    std::fs::write(&p, QWEN25_TOOLS_TEMPLATE).map_err(|e| e.to_string())?;
    Ok(p.to_string_lossy().into_owned())
}

#[cfg(test)]
mod template_override_tests {
    use super::qwen_template_override;
    use std::path::Path;

    #[test]
    fn qwen_models_get_override_but_reasoning_distills_do_not() {
        assert!(qwen_template_override(Path::new("/opt/sgl/models/qwen-coder-7b.gguf")));
        assert!(qwen_template_override(Path::new("/opt/sgl/models/qwen-2.5-14b.gguf")));
        assert!(!qwen_template_override(Path::new("/opt/sgl/models/r1-qwen-14b.gguf")));
        assert!(!qwen_template_override(Path::new("/opt/sgl/models/r1-llama-8b.gguf")));
        assert!(!qwen_template_override(Path::new("/opt/sgl/models/qwq-32b.gguf")));
        assert!(!qwen_template_override(Path::new("/opt/sgl/models/gemma-2-2b.gguf")));
    }
}

pub fn find_available_port(preferred: u16) -> u16 {
    if preferred != 0 {
        if let Ok(listener) = std::net::TcpListener::bind(("127.0.0.1", preferred)) {
            drop(listener);
            return preferred;
        }
    }
    // Bind to port 0 to let the OS assign a random available port
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("Failed to bind to any port");
    listener.local_addr().unwrap().port()
}

fn find_llama_server() -> Result<String, String> {
    // Windows: llama.cpp ships `llama-server.exe`. Look on PATH and in the usual
    // per-user install locations (%LOCALAPPDATA%\sgl-node\bin is where the desktop
    // shell / one-click installer drops the bundled build).
    #[cfg(windows)]
    {
        let mut candidates: Vec<String> = vec![
            "llama-server.exe".to_string(),
            "llama-cli.exe".to_string(),
        ];
        if let Some(local) = dirs::data_local_dir() {
            candidates.push(
                local
                    .join("sgl-node")
                    .join("bin")
                    .join("llama-server.exe")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        if let Ok(pf) = std::env::var("ProgramFiles") {
            candidates.push(format!(r"{pf}\llama.cpp\llama-server.exe"));
        }
        for cmd in &candidates {
            if Command::new(cmd).arg("--help").output().is_ok() {
                return Ok(cmd.clone());
            }
        }
        return Err(
            "llama-server.exe not found. Install llama.cpp for Windows (https://github.com/ggerganov/llama.cpp/releases) \
             and put llama-server.exe on PATH or in %LOCALAPPDATA%\\sgl-node\\bin. \
             Or run with --engine=inprocess on a build compiled with the `inprocess` feature.".to_string()
        );
    }

    #[cfg(not(windows))]
    {
        let candidates = [
            "llama-server",
            "llama-cli",
            "/usr/local/bin/llama-server",
            #[cfg(target_os = "macos")]
            "/opt/homebrew/bin/llama-server",
            // Headless Linux GPU machines (one-click deploys): cloud-init installs a
            // pinned CUDA llama-server here.
            #[cfg(target_os = "linux")]
            "/opt/sgl/bin/llama-server",
        ];

        for cmd in &candidates {
            if Command::new(cmd).arg("--help").output().is_ok() {
                return Ok(cmd.to_string());
            }
        }

        Err(
            "llama-server not found. Install llama.cpp:\n  brew install llama.cpp\n  # or build from source: https://github.com/ggerganov/llama.cpp".to_string()
        )
    }
}
