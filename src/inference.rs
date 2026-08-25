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

#[derive(Clone)]
pub struct InferenceEngineConfig {
    pub model_path: PathBuf,
    pub model_name: String,
    pub port: u16,
    pub threads: u32,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub batch_size: u32,
    pub parallel_slots: u32,
    /// Vision (multimodal): path to the model's mmproj (vision projector) GGUF. When set,
    /// llama-server is launched with `--mmproj` so it accepts OpenAI image_url content. The
    /// app/provisioner downloads + hash-verifies it (like the main GGUF) and passes the path.
    /// Only meaningful for the server engine — the in-process engine can't do mmproj.
    pub mmproj_path: Option<PathBuf>,
    /// Cap on tokens a single image may cost (llama-server `--image-max-tokens`). None = default.
    pub image_max_tokens: Option<u32>,
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
    /// Set when the configured model is a known embedding model: llama-server is then
    /// launched in `--embedding` mode with the catalog's pooling, and embedding jobs are
    /// forwarded to its native /v1/embeddings endpoint. This is what lets NON-`inprocess`
    /// builds (Windows) serve embedding models — the in-process engine handles them on
    /// macOS/Linux builds and takes priority in InferenceEngine::create.
    embed_spec: Option<&'static crate::embed_catalog::EmbedModelSpec>,
    /// Supervised-restart counter for the engine-variant auto-swap (see restart()).
    restart_count: std::sync::atomic::AtomicU32,
}

/// Which llama.cpp variant `sgl setup` installed ("vulkan" | "cpu"), from the marker it
/// writes beside llama-server. Unknown/legacy installs return None.
fn installed_engine_variant() -> Option<String> {
    let base = dirs::data_local_dir()?;
    std::fs::read_to_string(base.join("sgl-node").join("bin").join("engine.variant"))
        .ok()
        .map(|s| s.trim().to_string())
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
        let embed_spec = crate::embed_catalog::embed_model_spec(&config.model_name);
        Self {
            server_process: std::sync::Mutex::new(None),
            client: Client::new(),
            base_url,
            config,
            embed_spec,
            restart_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// True iff this server engine runs llama-server in embedding mode.
    pub fn is_embedding(&self) -> bool {
        self.embed_spec.is_some()
    }

    /// Native embedding dimension (catalog value) when in embedding mode.
    pub fn embedding_dim(&self) -> Option<u32> {
        self.embed_spec.map(|s| s.dim)
    }

    /// Forward an embedding batch to llama-server's native /v1/embeddings (embedding mode
    /// only). Mirrors the in-process engine's contract: catalog prefixes by input_type,
    /// Matryoshka truncate + renormalize, L2-normalize per the spec, input-only usage.
    /// The orchestrator structurally validates every vector (dim/finite/non-zero) before
    /// billing, so any mismatch here fails the job — never a silently-wrong embedding.
    pub async fn embed(
        &self,
        inputs: Vec<String>,
        input_type: crate::embed_catalog::InputType,
        dimensions: Option<u32>,
    ) -> Result<crate::embed_catalog::EmbedOut, String> {
        use crate::embed_catalog::{l2_normalize, InputType};
        let spec = self.embed_spec.ok_or("this node is not an embedding node")?;
        // Resolve the effective output dim (Matryoshka truncation). Reject an unsupported
        // request rather than guessing (mirrors the in-process engine + orchestrator).
        let eff_dim = match dimensions {
            None => spec.dim,
            Some(d) if d == spec.dim || spec.allowed_dimensions.contains(&d) => d,
            Some(d) => {
                return Err(format!(
                    "unsupported dimensions {d} for {} (native {}, allowed {:?})",
                    spec.id, spec.dim, spec.allowed_dimensions
                ))
            }
        };
        let prefix = match input_type {
            InputType::Query => spec.prefix_query,
            InputType::Document | InputType::Unspecified => spec.prefix_document,
        };
        let n = inputs.len();
        let texts: Vec<String> = inputs
            .into_iter()
            .map(|i| if prefix.is_empty() { i } else { format!("{prefix}{i}") })
            .collect();

        let url = format!("{}/v1/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(120))
            .json(&serde_json::json!({ "input": texts, "model": spec.id }))
            .send()
            .await
            .map_err(|e| format!("embedding request to llama-server failed: {e}"))?;
        let status = resp.status();
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("invalid embedding response from llama-server: {e}"))?;
        if !status.is_success() {
            let msg = body
                .pointer("/error/message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(format!("llama-server embedding error ({status}): {msg}"));
        }

        let data = body
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or("llama-server embedding response missing 'data'")?;
        if data.len() != n {
            return Err(format!(
                "llama-server returned {} embeddings for {n} inputs",
                data.len()
            ));
        }
        // Re-order by `index` (OpenAI contract) so vectors line up with inputs.
        let mut rows: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n);
        for item in data {
            let idx = item
                .get("index")
                .and_then(|i| i.as_u64())
                .unwrap_or(rows.len() as u64) as usize;
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or("llama-server embedding item missing 'embedding'")?;
            let mut vec: Vec<f32> = Vec::with_capacity(emb.len());
            for v in emb {
                vec.push(v.as_f64().ok_or("non-numeric embedding value")? as f32);
            }
            if vec.len() != spec.dim as usize {
                return Err(format!(
                    "llama-server produced dim {} but {} expects {}",
                    vec.len(),
                    spec.id,
                    spec.dim
                ));
            }
            rows.push((idx, vec));
        }
        // `index` must be an exact permutation of 0..n — a duplicate/out-of-range index
        // would silently misalign vectors with inputs (Codex MED).
        let mut seen = vec![false; n];
        for (idx, _) in &rows {
            if *idx >= n || seen[*idx] {
                return Err(
                    "llama-server embedding response has duplicate or out-of-range indexes"
                        .to_string(),
                );
            }
            seen[*idx] = true;
        }
        rows.sort_by_key(|(i, _)| *i);
        let mut vectors = Vec::with_capacity(n);
        for (_, mut vec) in rows {
            if eff_dim < spec.dim {
                // Matryoshka: truncate to the requested dim, then renormalize the shortened vector.
                vec.truncate(eff_dim as usize);
                l2_normalize(&mut vec);
            } else if spec.normalize {
                // llama-server normalizes by default; keep the guarantee explicit (idempotent).
                l2_normalize(&mut vec);
            }
            vectors.push(vec);
        }
        let prompt_tokens = body
            .pointer("/usage/prompt_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;
        Ok(crate::embed_catalog::EmbedOut { vectors, prompt_tokens })
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
        let template_file: Option<String> = if self.embed_spec.is_none()
            && qwen_template_override(&self.config.model_path)
        {
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
        // Vision projector (mmproj) + optional per-image token cap. Owned here so the &str
        // borrows live for the whole args vec. Only used on the chat path (below the embed
        // early-return), so an embedding model never gets --mmproj.
        let mmproj_str = self
            .config
            .mmproj_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());
        let image_max_tokens_str = self.config.image_max_tokens.map(|n| n.to_string());
        // Embedding mode (non-`inprocess` builds, e.g. Windows): llama-server pools the
        // hidden states itself. c/b/ub are all sized to the model's max input — pooled
        // (non-causal) models require the WHOLE sequence to fit in one ubatch, and BERT
        // context is small anyway. No chat flags (--jinja/--parallel/templates).
        let embed_ctx_str = self
            .embed_spec
            .map(|s| s.max_input_tokens.to_string())
            .unwrap_or_default();
        if let Some(spec) = self.embed_spec {
            let args: Vec<&str> = vec![
                "-m", &model_str,
                "--host", "127.0.0.1",
                "--port", &port_str,
                "-t", &threads_str,
                "-c", &embed_ctx_str,
                "-b", &embed_ctx_str,
                "-ub", &embed_ctx_str,
                "--embedding",
                "--pooling", spec.pooling.as_server_flag(),
            ];
            return self.spawn_server(&llama_server, &args).await;
        }
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
        // KV-cache quantization (opt-in, default OFF). `--cache-type-k q8_0` roughly halves
        // K-cache memory for long contexts on RAM-tight boxes, reducing spill/pressure. V-cache
        // quant additionally needs flash-attention (Metal head-size dependent), so we ship K-only
        // until per-model validation and gate it behind SGL_KV_QUANT=1. No-op when unset.
        if std::env::var("SGL_KV_QUANT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            args.push("--cache-type-k");
            args.push("q8_0");
        }
        if let Some(p) = &template_file {
            args.push("--chat-template-file");
            args.push(p);
        }
        // Vision: load the mmproj so llama-server accepts image content on /v1/chat/completions.
        // Guarded by mmproj_path being set (a vision model + a verified projector on disk).
        if let Some(mm) = &mmproj_str {
            args.push("--mmproj");
            args.push(mm);
            if let Some(n) = &image_max_tokens_str {
                args.push("--image-max-tokens");
                args.push(n);
            }
        }
        self.spawn_server(&llama_server, &args).await
    }

    /// True when this engine is serving a vision (multimodal) model — an mmproj is loaded
    /// and it is a chat (not embedding) model. Drives the node's `vision` heartbeat capability
    /// so the orchestrator only routes image requests here.
    pub fn is_vision(&self) -> bool {
        self.embed_spec.is_none() && self.config.mmproj_path.is_some()
    }

    /// Spawn llama-server with the given args, stream its stderr into tracing, store the
    /// child handle, and wait (up to 30s) for /health. Shared by the chat and embedding
    /// launch paths.
    async fn spawn_server(&self, llama_server: &str, args: &[&str]) -> Result<(), String> {
        let mut cmd = Command::new(llama_server);
        cmd.args(args);
        // Windows: the node may itself run windowless (app-spawned / Scheduled Task);
        // without this flag each llama-server child pops a console window at the operator.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = cmd
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
        self.restart_inner(true).await
    }

    /// Restart WITHOUT counting toward the crash restart budget / Vulkan→CPU auto-swap.
    /// Used by the empty-completion self-heal: an engine that stays up and answers /health
    /// but PRODUCES EMPTY output (canary-confirmed) is a different failure from a crash, and
    /// must never trip the GPU→CPU downgrade — otherwise a user could weaponize crafted empty
    /// replies to permanently degrade a healthy GPU node. Same kill+relaunch, no swap counter.
    pub async fn restart_no_swap(&self) -> Result<(), String> {
        self.restart_inner(false).await
    }

    async fn restart_inner(&self, allow_variant_swap: bool) -> Result<(), String> {
        tracing::warn!("restarting the inference engine (variant_swap={allow_variant_swap})");
        self.stop();
        // Engine-variant auto-swap: a GPU (Vulkan) build that passes --version/health but
        // CRASHES under real inference is a driver problem the operator can't fix — seen
        // live on the first external Windows tester. After the 2nd supervised restart,
        // swap to the CPU build via the same verified `sgl setup` pipeline and keep
        // serving. One swap direction only (vulkan → cpu), once per install: if CPU also
        // fails, the normal restart/watchdog path continues and surfaces the error.
        // ONLY crash/wedge restarts feed this counter — empty-completion restarts pass
        // allow_variant_swap=false so a produce-nothing engine can't force the downgrade.
        if allow_variant_swap {
            let n = self
                .restart_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            if n == 2 && installed_engine_variant().as_deref() == Some("vulkan") {
                tracing::warn!(
                    "engine crashed twice — auto-swapping the GPU (Vulkan) build for the CPU build"
                );
                match crate::setup::run(true).await {
                    Ok(_) => tracing::info!("CPU engine installed — relaunching"),
                    Err(e) => tracing::warn!("engine auto-swap failed (continuing with current build): {e}"),
                }
            }
        }
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
    /// Dedicated embedding engine (#231). A node is embeddings-OR-chat; when the configured model
    /// is a known embedding model this variant is created instead of a chat engine. Requires the
    /// `inprocess` feature (llama-cpp-2 with pooling).
    #[cfg(feature = "inprocess")]
    Embed(crate::embed::EmbedEngine),
}

impl InferenceEngine {
    /// Build + start the chosen engine.
    pub async fn create(config: InferenceEngineConfig, mode: EngineMode) -> Result<Self, String> {
        // Embedding models get the dedicated pooling engine regardless of `--engine` (they can't
        // run through the chat path). Only available on an `inprocess` build.
        #[cfg(feature = "inprocess")]
        if crate::embed::is_embedding_model(&config.model_name) {
            let e = crate::embed::EmbedEngine::start(crate::embed::EmbedConfig {
                model_path: config.model_path,
                model_name: config.model_name,
                n_gpu_layers: if config.gpu_layers == GPU_LAYERS_AUTO { 999 } else { config.gpu_layers },
            })
            .await?;
            return Ok(InferenceEngine::Embed(e));
        }
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
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(e) => e.is_healthy(),
        }
    }

    /// True iff this engine serves embeddings (routes `/v1/embeddings` jobs) rather than chat.
    pub fn is_embedding(&self) -> bool {
        match self {
            // Server engine in --embedding mode (the non-`inprocess`/Windows path).
            InferenceEngine::Server(e) => e.is_embedding(),
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => true,
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => false,
        }
    }

    /// True iff this engine serves a vision (multimodal) model. Only the server engine can —
    /// the in-process (Mac) engine has no mmproj support, and embedding engines never do.
    pub fn is_vision(&self) -> bool {
        match self {
            InferenceEngine::Server(e) => e.is_vision(),
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => false,
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => false,
        }
    }

    /// Native embedding dimension when this is an embedding engine (else None) — advertised to the
    /// orchestrator so it can meter + surface `dim` per model.
    pub fn embedding_dim(&self) -> Option<u32> {
        match self {
            InferenceEngine::Server(e) => e.embedding_dim(),
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(e) => Some(e.dim()),
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => None,
        }
    }

    pub fn stop(&self) {
        match self {
            InferenceEngine::Server(e) => e.stop(),
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => { /* worker stops when the engine is dropped */ }
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => { /* worker stops when the engine is dropped */ }
        }
    }

    /// Embed a batch of inputs (only valid on an embedding engine). Ungated: the server
    /// engine forwards to llama-server's /v1/embeddings on builds without `inprocess`.
    pub async fn embed(
        &self,
        inputs: Vec<String>,
        input_type: crate::embed_catalog::InputType,
        dimensions: Option<u32>,
    ) -> Result<crate::embed_catalog::EmbedOut, String> {
        match self {
            InferenceEngine::Server(e) => e.embed(inputs, input_type, dimensions).await,
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(e) => e.embed(inputs, input_type, dimensions).await,
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => Err("this node is not an embedding node".to_string()),
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
            // Same anti-zombie recovery as the in-process chat engine: a wedged native call can't
            // be recovered from Rust, so abort for a clean OS relaunch.
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => {
                tracing::error!(
                    "embedding engine unhealthy — aborting for a clean OS relaunch (in-place restart is not possible)"
                );
                std::process::abort();
            }
        }
    }

    /// Restart driven by the empty-completion self-heal (canary-confirmed produce-nothing).
    /// Server engine: kill+relaunch WITHOUT feeding the Vulkan→CPU auto-swap counter (a
    /// produce-nothing engine is not a crash, and empties are attacker-inducible). In-process /
    /// embedding engines have no child to bounce, so recovery is the same clean OS relaunch
    /// (abort) as `restart()` — the canary + cooldown in the heartbeat loop gate it.
    pub async fn restart_empty(&self) -> Result<(), String> {
        match self {
            InferenceEngine::Server(e) => e.restart_no_swap().await,
            #[cfg(feature = "inprocess")]
            InferenceEngine::InProcess(_) => self.restart().await,
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => self.restart().await,
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
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => {
                Err("this node serves embeddings, not chat completions".to_string())
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
            #[cfg(feature = "inprocess")]
            InferenceEngine::Embed(_) => {
                let _ = tx;
                Err("this node serves embeddings, not chat completions".to_string())
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
    // Qwen3's GGUF template already ships a tools section AND thinking controls
    // (enable_thinking / <think>), so it must keep its EMBEDDED --jinja template —
    // forcing the Qwen2.5 tool template onto it would clobber the thinking behavior.
    // Only the older Qwen2.5-family GGUFs (which lack the tools section) need the override.
    name.contains("qwen")
        && !name.contains("qwen3")
        && !name.contains("r1")
        && !name.contains("qwq")
}

/// Write the embedded template to a temp file for `--chat-template-file`.
fn write_qwen_tools_template() -> Result<String, String> {
    let p = std::env::temp_dir().join("sgl-qwen25-tools.jinja");
    std::fs::write(&p, QWEN25_TOOLS_TEMPLATE).map_err(|e| e.to_string())?;
    Ok(p.to_string_lossy().into_owned())
}

// Helper fns intentionally live below this module; reordering a 1.2k-line file to
// satisfy an organizational lint would churn history for no behavioural gain.
// Allowed narrowly so `clippy -D warnings` can gate on lints that actually matter.
#[allow(clippy::items_after_test_module)]
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
    // Windows: llama.cpp ships `llama-server.exe`. Prefer the `sgl setup`-installed, hash-VERIFIED
    // build (%LOCALAPPDATA%\sgl-node\bin) FIRST, so a stale/incompatible llama-server.exe on PATH
    // can't shadow it (Codex MED). Fall back to PATH / ProgramFiles only if the verified one is
    // absent.
    #[cfg(windows)]
    {
        let mut candidates: Vec<String> = Vec::new();
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
        candidates.push("llama-server.exe".to_string());
        candidates.push("llama-cli.exe".to_string());
        if let Ok(pf) = std::env::var("ProgramFiles") {
            candidates.push(format!(r"{pf}\llama.cpp\llama-server.exe"));
        }
        for cmd in &candidates {
            // CREATE_NO_WINDOW: each probe is a console app — without the flag every probe
            // flashes a console window at the operator (seen live on the Windows beta).
            use std::os::windows::process::CommandExt;
            let probe = Command::new(cmd)
                .arg("--help")
                .creation_flags(0x0800_0000)
                .output();
            if probe.is_ok() {
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
        let mut candidates: Vec<String> = Vec::new();
        // The `sgl setup`-installed, hash-VERIFIED pinned build FIRST — same rule as
        // Windows above. On macOS this is what the engine fallback self-provisions;
        // without this ordering a stale Homebrew llama-server (predating a model's
        // arch) would shadow the managed copy and the fallback would fail anyway.
        if let Some(local) = dirs::data_local_dir() {
            candidates.push(local.join("sgl-node").join("bin").join("llama-server").to_string_lossy().into_owned());
        }
        candidates.push("llama-server".to_string());
        candidates.push("llama-cli".to_string());
        candidates.push("/usr/local/bin/llama-server".to_string());
        #[cfg(target_os = "macos")]
        candidates.push("/opt/homebrew/bin/llama-server".to_string());
        // Headless Linux GPU machines (one-click deploys): cloud-init installs a
        // pinned CUDA llama-server here.
        #[cfg(target_os = "linux")]
        candidates.push("/opt/sgl/bin/llama-server".to_string());

        for cmd in &candidates {
            if Command::new(cmd).arg("--help").output().is_ok() {
                return Ok(cmd.clone());
            }
        }

        Err(
            "llama-server not found. Install llama.cpp:\n  brew install llama.cpp\n  # or build from source: https://github.com/ggerganov/llama.cpp".to_string()
        )
    }
}
