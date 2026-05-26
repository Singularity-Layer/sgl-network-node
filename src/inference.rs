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
    server_process: Option<Child>,
    client: Client,
    base_url: String,
    config: InferenceEngineConfig,
}

#[derive(Serialize)]
struct CompletionRequest {
    prompt: String,
    n_predict: i32,
    temperature: f64,
    stop: Vec<String>,
    stream: bool,
}

#[derive(Deserialize)]
struct CompletionResponse {
    content: String,
    tokens_predicted: Option<u32>,
    tokens_evaluated: Option<u32>,
    generation_settings: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: i32,
    stream: bool,
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
    pub fn new(config: InferenceEngineConfig) -> Self {
        let base_url = format!("http://127.0.0.1:{}", config.port);
        Self {
            server_process: None,
            client: Client::new(),
            base_url,
            config,
        }
    }

    pub async fn start(&mut self) -> Result<(), String> {
        if !self.config.model_path.exists() {
            return Err(format!("Model file not found: {}", self.config.model_path.display()));
        }

        let llama_server = find_llama_server()?;
        tracing::info!("Starting llama-server with model: {}", self.config.model_path.display());

        let port_str = self.config.port.to_string();
        let threads_str = self.config.threads.to_string();
        let gpu_layers_str = self.config.gpu_layers.to_string();
        let ctx_str = self.config.context_size.to_string();
        let batch_str = self.config.batch_size.to_string();
        let parallel_str = self.config.parallel_slots.to_string();

        let child = Command::new(&llama_server)
            .args([
                "-m", &self.config.model_path.to_string_lossy(),
                "--host", "127.0.0.1",
                "--port", &port_str,
                "-ngl", &gpu_layers_str,
                "-c", &ctx_str,
                "-t", &threads_str,
                "-b", &batch_str,
                "--parallel", &parallel_str,
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start llama-server: {e}"))?;

        self.server_process = Some(child);

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

    pub fn stop(&mut self) {
        if let Some(ref mut child) = self.server_process {
            let _ = child.kill();
            let _ = child.wait();
            tracing::info!("llama-server stopped");
        }
        self.server_process = None;
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        match self.client.get(&url).send().await {
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

        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Inference request failed: {e}"))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Inference failed: {text}"));
        }

        let result: ChatCompletionResponse = resp.json().await
            .map_err(|e| format!("Failed to parse inference response: {e}"))?;

        let choice = result.choices.first()
            .ok_or("No completion returned")?;

        Ok(InferenceResult {
            content: choice.message.content.clone(),
            model: self.config.model_name.clone(),
            prompt_tokens: result.usage.as_ref().and_then(|u| u.prompt_tokens).unwrap_or(0),
            completion_tokens: result.usage.as_ref().and_then(|u| u.completion_tokens).unwrap_or(0),
        })
    }

    pub async fn completion(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f64,
    ) -> Result<InferenceResult, String> {
        let url = format!("{}/completion", self.base_url);

        let body = CompletionRequest {
            prompt: prompt.to_string(),
            n_predict: max_tokens,
            temperature,
            stop: vec!["</s>".to_string(), "<|eot_id|>".to_string()],
            stream: false,
        };

        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Inference request failed: {e}"))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Inference failed: {text}"));
        }

        let result: CompletionResponse = resp.json().await
            .map_err(|e| format!("Failed to parse response: {e}"))?;

        Ok(InferenceResult {
            content: result.content,
            model: self.config.model_name.clone(),
            prompt_tokens: result.tokens_evaluated.unwrap_or(0),
            completion_tokens: result.tokens_predicted.unwrap_or(0),
        })
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

