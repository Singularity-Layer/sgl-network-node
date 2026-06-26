use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command};
use tokio::time::{sleep, Duration};

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

pub struct InferenceEngine {
    // Behind a Mutex so the engine can be shared as Arc<InferenceEngine> across the
    // heartbeat loop + job handlers AND still be (re)started/stopped via &self. The
    // guard is never held across an await (we lock only to swap the Child handle).
    server_process: std::sync::Mutex<Option<Child>>,
    client: Client,
    base_url: String,
    config: InferenceEngineConfig,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: i32,
    stream: bool,
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
    message: ChatMessage,
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

impl InferenceEngine {
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

        let child = Command::new(&llama_server)
            .args([
                "-m",
                &self.config.model_path.to_string_lossy(),
                "--host",
                "127.0.0.1",
                "--port",
                &port_str,
                "-ngl",
                &gpu_layers_str,
                "-c",
                &ctx_str,
                "-t",
                &threads_str,
                "-b",
                &batch_str,
                "--parallel",
                &parallel_str,
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start llama-server: {e}"))?;

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
        messages: Vec<ChatMessage>,
        temperature: f64,
        max_tokens: i32,
    ) -> Result<InferenceResult, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let body = ChatCompletionRequest {
            messages,
            temperature,
            max_tokens,
            stream: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Inference request failed: {e}"))?;

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
            content: choice.message.content.clone(),
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

        let mut resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Inference request failed: {e}"))?;

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

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct InferenceResult {
    pub content: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
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
    let candidates = [
        "llama-server",
        "llama-cli",
        "/usr/local/bin/llama-server",
        "/opt/homebrew/bin/llama-server",
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
