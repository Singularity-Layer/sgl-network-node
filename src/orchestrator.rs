use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;
use std::time::Duration;

use crate::tee::TeeCapabilities;

/// Percent-encode a value used as a single URL path segment. node_id/job_id come
/// from the orchestrator, but we never want a stray '/', '?', or '#' to silently
/// reshape the request path.
/// True only if the URL's HOST (not a substring) is loopback. Parses the authority
/// so `http://localhost.evil.test` or `http://x/?u=localhost` are NOT treated as local.
fn is_loopback_url(url: &str) -> bool {
    let after = match url.split_once("://") {
        Some((_, rest)) => rest,
        None => return false,
    };
    // authority = up to the first '/', '?', or '#'
    let authority = after.split(['/', '?', '#']).next().unwrap_or("");
    // strip any userinfo ("user:pass@")
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    // strip port, handling bracketed IPv6 ([::1]:port)
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };
    host == "localhost" || host == "127.0.0.1" || host == "::1"
}

fn enc_seg(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub struct OrchestratorClient {
    client: Client,
    base_url: String,
    auth_token: RwLock<Option<String>>,
}

#[derive(Serialize)]
struct RegisterRequest {
    wallet_address: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    registration_code: Option<String>,
    tee_type: String,
    available_models: Vec<String>,
    public_key: String,
    cpu_cores: u32,
    ram_gb: f64,
    gpu_model: String,
    supported_runtimes: Vec<String>,
}

#[derive(Serialize)]
struct DeviceStartRequest {}

#[derive(Deserialize)]
pub struct DeviceStartResponse {
    pub device_code: String,
    pub user_code: String,
    pub verify_url: String,
    #[serde(default = "default_interval")]
    pub interval: u64,
    #[serde(default)]
    pub expires_in: u64,
}

fn default_interval() -> u64 {
    3
}

#[derive(Serialize)]
struct DevicePollRequest {
    device_code: String,
}

#[derive(Deserialize)]
pub struct DevicePollResponse {
    pub status: String,
    #[serde(default)]
    pub registration_code: Option<String>,
    #[serde(default)]
    pub wallet_address: Option<String>,
}

#[derive(Deserialize)]
pub struct RegisterResponse {
    pub node_id: String,
    pub auth_token: String,
}

#[derive(Serialize)]
struct HeartbeatRequest {
    current_load: f64,
    available_models: Vec<String>,
    // Node's X25519 key for E2E-encrypted prompts (sent every heartbeat so the
    // orchestrator always has the current key without a separate attest step).
    #[serde(skip_serializing_if = "Option::is_none")]
    encryption_public_key: Option<String>,
    // #94: ed25519 signature binding the X25519 key above to this node's identity
    // (node_id + ed25519 + x25519 + key_version). The orchestrator stores + relays it
    // so clients can verify they're sealing to the genuine node (anti-MITM).
    #[serde(skip_serializing_if = "Option::is_none")]
    encryption_public_key_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    key_version: Option<u32>,
    // Feature capabilities the orchestrator uses for routing (e.g. only route
    // streaming requests to nodes that advertise `streaming: true`).
    capabilities: NodeCapabilities,
}

#[derive(Serialize)]
struct NodeCapabilities {
    streaming: bool,
    // Context window this node serves with (llama-server -c). The grid uses it to
    // size the pre-dispatch prompt check per node instead of assuming a default.
    context_size: u32,
}

#[derive(Deserialize)]
pub struct HeartbeatResponse {
    pub status: String,
    #[serde(default)]
    pub pending_jobs: Vec<PendingJob>,
    pub new_auth_token: Option<String>,
    pub token_expires_at: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct PendingJob {
    pub id: String,
    pub job_type: String,
    #[allow(dead_code)] // part of the dispatch payload; model selection is server-side
    pub model: Option<String>,
    pub input_payload: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct ChallengeResponse {
    pub challenge: String,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub expires_in_seconds: Option<u64>,
}

#[derive(Serialize)]
struct VerifyAttestationRequest {
    signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encryption_public_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hardware_report: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct AttestationResponse {
    pub verified: bool,
    pub status: String,
}

#[derive(Deserialize)]
#[serde(default)]
pub struct NodeStatusResponse {
    pub status: String,
    pub attestation_verified: bool,
    pub reputation_score: Option<f64>,
    #[serde(alias = "total_jobs_completed")]
    pub jobs_completed: Option<u64>,
    #[serde(alias = "total_jobs_failed")]
    pub jobs_failed: Option<u64>,
}

impl Default for NodeStatusResponse {
    fn default() -> Self {
        Self {
            status: "unknown".to_string(),
            attestation_verified: false,
            reputation_score: None,
            jobs_completed: None,
            jobs_failed: None,
        }
    }
}

#[derive(Deserialize)]
struct ErrorBody {
    error: String,
}

impl OrchestratorClient {
    pub fn new(base_url: &str, auth_token: Option<String>) -> Self {
        // Refuse to send the node auth token over plaintext: require https except
        // for an explicit loopback HOST (local dev). Fail fast on a misconfigured URL.
        if !base_url.starts_with("https://") && !is_loopback_url(base_url) {
            panic!(
                "Refusing insecure orchestrator URL '{base_url}': use https:// \
                 (only http://localhost is allowed for local dev)."
            );
        }
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token: RwLock::new(auth_token),
        }
    }

    /// Read a response body with a hard byte cap, streaming chunk-by-chunk so an
    /// unbounded/chunked (no Content-Length) hostile response can't exhaust memory.
    async fn read_body_capped(mut resp: reqwest::Response) -> Result<Vec<u8>, String> {
        const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024; // 8 MiB
                                                           // Reject early if the advertised length is already too large.
        if let Some(len) = resp.content_length() {
            if len as usize > MAX_RESPONSE_BYTES {
                return Err(format!("orchestrator response too large ({len} bytes)"));
            }
        }
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| format!("response read failed: {e}"))?
        {
            if buf.len() + chunk.len() > MAX_RESPONSE_BYTES {
                return Err("orchestrator response exceeded maximum size".to_string());
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf)
    }

    pub fn update_auth_token(&self, token: String) {
        let mut guard = self.auth_token.write().unwrap();
        *guard = Some(token);
    }

    fn get_token(&self) -> Result<String, String> {
        let guard = self.auth_token.read().unwrap();
        guard
            .clone()
            .ok_or_else(|| "No auth token configured".to_string())
    }

    /// Current auth token (for the WS client to read fresh on each reconnect).
    pub fn current_token(&self) -> Option<String> {
        self.auth_token.read().unwrap().clone()
    }

    pub async fn register(
        &self,
        wallet: &str,
        code: Option<&str>,
        tee_type: &str,
        models: &[String],
        public_key: &str,
        caps: &TeeCapabilities,
    ) -> Result<RegisterResponse, String> {
        let url = format!("{}/grid/nodes/register", self.base_url);

        let body = RegisterRequest {
            wallet_address: wallet.to_string(),
            registration_code: code.map(|s| s.to_string()),
            tee_type: tee_type.to_string(),
            available_models: models.to_vec(),
            public_key: public_key.to_string(),
            cpu_cores: caps.cpu_cores,
            ram_gb: caps.memory_gb,
            gpu_model: caps.gpu.clone(),
            supported_runtimes: vec!["llama.cpp".to_string()],
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Registration request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if let Ok(err) = serde_json::from_str::<ErrorBody>(&text) {
                return Err(format!("Registration failed ({status}): {}", err.error));
            }
            return Err(format!("Registration failed ({status}): {text}"));
        }

        resp.json::<RegisterResponse>()
            .await
            .map_err(|e| format!("Failed to parse registration response: {e}"))
    }

    /// Start a device-authorization session for `sgl login`.
    pub async fn device_start(&self) -> Result<DeviceStartResponse, String> {
        let url = format!("{}/grid/auth/device/start", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&DeviceStartRequest {})
            .send()
            .await
            .map_err(|e| format!("device/start failed: {e}"))?;
        if !resp.status().is_success() {
            let s = resp.status();
            return Err(format!(
                "device/start failed ({s}): {}",
                resp.text().await.unwrap_or_default()
            ));
        }
        resp.json::<DeviceStartResponse>()
            .await
            .map_err(|e| format!("Failed to parse device/start: {e}"))
    }

    /// Poll a device-authorization session for approval.
    pub async fn device_poll(&self, device_code: &str) -> Result<DevicePollResponse, String> {
        let url = format!("{}/grid/auth/device/poll", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&DevicePollRequest {
                device_code: device_code.to_string(),
            })
            .send()
            .await
            .map_err(|e| format!("device/poll failed: {e}"))?;
        if !resp.status().is_success() {
            let s = resp.status();
            return Err(format!(
                "device/poll failed ({s}): {}",
                resp.text().await.unwrap_or_default()
            ));
        }
        resp.json::<DevicePollResponse>()
            .await
            .map_err(|e| format!("Failed to parse device/poll: {e}"))
    }

    pub async fn heartbeat(
        &self,
        node_id: &str,
        models: &[String],
        current_load: f64,
        encryption_public_key: Option<&str>,
        encryption_public_key_signature: Option<&str>,
        key_version: Option<u32>,
        streaming: bool,
        context_size: u32,
    ) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/grid/nodes/heartbeat", self.base_url);
        let token = self.get_token()?;

        let body = HeartbeatRequest {
            current_load,
            available_models: models.to_vec(),
            encryption_public_key: encryption_public_key.map(|s| s.to_string()),
            encryption_public_key_signature: encryption_public_key_signature.map(|s| s.to_string()),
            key_version,
            capabilities: NodeCapabilities { streaming, context_size },
        };

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .header("X-Node-Id", node_id)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Heartbeat failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Heartbeat failed ({status}): {text}"));
        }

        let body = Self::read_body_capped(resp).await?;
        serde_json::from_slice::<HeartbeatResponse>(&body)
            .map_err(|e| format!("Failed to parse heartbeat response: {e}"))
    }

    pub async fn request_challenge(&self, node_id: &str) -> Result<ChallengeResponse, String> {
        let url = format!(
            "{}/grid/nodes/{}/challenge",
            self.base_url,
            enc_seg(node_id)
        );
        let token = self.get_token()?;

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .send()
            .await
            .map_err(|e| format!("Challenge request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Challenge request failed ({status}): {text}"));
        }

        resp.json::<ChallengeResponse>()
            .await
            .map_err(|e| format!("Failed to parse challenge response: {e}"))
    }

    /// Toggle off-grid (maintenance) mode for this node. When off-grid the node
    /// is excluded from job dispatch (planned downtime) and is not penalized for
    /// being offline. Tamper slashing is unaffected.
    pub async fn set_off_grid(&self, node_id: &str, off_grid: bool) -> Result<(), String> {
        let url = format!("{}/grid/nodes/{}/status", self.base_url, enc_seg(node_id));
        let token = self.get_token()?;

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&serde_json::json!({ "off_grid": off_grid }))
            .send()
            .await
            .map_err(|e| format!("Status update failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Status update failed ({status}): {text}"));
        }
        Ok(())
    }

    /// Fetch this node's per-model prices + the allowed band (public endpoint).
    pub async fn get_prices(&self, node_id: &str) -> Result<serde_json::Value, String> {
        let url = format!("{}/grid/nodes/{}/prices", self.base_url, enc_seg(node_id));
        let resp = self.client.get(&url).send().await.map_err(|e| format!("Price fetch failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Price fetch failed ({status}): {text}"));
        }
        resp.json().await.map_err(|e| format!("Bad price response: {e}"))
    }

    /// Set a custom per-token price for a model this node serves (X-Node-Auth; the
    /// orchestrator enforces the allowed band). Prices are USD per 1M tokens.
    pub async fn set_price(&self, node_id: &str, model: &str, input_per_m: f64, output_per_m: f64) -> Result<(), String> {
        let url = format!("{}/grid/nodes/{}/prices", self.base_url, enc_seg(node_id));
        let token = self.get_token()?;
        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&serde_json::json!({ "model": model, "input_per_m": input_per_m, "output_per_m": output_per_m }))
            .send()
            .await
            .map_err(|e| format!("Set price failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Set price failed ({status}): {text}"));
        }
        Ok(())
    }

    /// Reset a model's price back to the platform suggested rate.
    pub async fn reset_price(&self, node_id: &str, model: &str) -> Result<(), String> {
        let url = format!("{}/grid/nodes/{}/prices", self.base_url, enc_seg(node_id));
        let token = self.get_token()?;
        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&serde_json::json!({ "model": model, "reset": true }))
            .send()
            .await
            .map_err(|e| format!("Reset price failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Reset price failed ({status}): {text}"));
        }
        Ok(())
    }

    pub async fn verify_attestation(
        &self,
        node_id: &str,
        signature: &str,
        encryption_public_key: Option<String>,
        hardware_report: Option<serde_json::Value>,
    ) -> Result<AttestationResponse, String> {
        let url = format!(
            "{}/grid/nodes/{}/verify-attestation",
            self.base_url,
            enc_seg(node_id)
        );
        let token = self.get_token()?;

        let body = VerifyAttestationRequest {
            signature: signature.to_string(),
            encryption_public_key,
            hardware_report,
        };

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Attestation verification failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Attestation failed ({status}): {text}"));
        }

        resp.json::<AttestationResponse>()
            .await
            .map_err(|e| format!("Failed to parse attestation response: {e}"))
    }

    pub async fn get_node_status(&self, node_id: &str) -> Result<NodeStatusResponse, String> {
        let url = format!("{}/grid/nodes/{}", self.base_url, enc_seg(node_id));
        let token = self.get_token()?;

        let resp = self
            .client
            .get(&url)
            .header("X-Node-Auth", token)
            .send()
            .await
            .map_err(|e| format!("Status request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Status request failed ({status}): {text}"));
        }

        resp.json::<NodeStatusResponse>()
            .await
            .map_err(|e| format!("Failed to parse status response: {e}"))
    }

    pub async fn complete_job(
        &self,
        job_id: &str,
        result: &serde_json::Value,
        envelope_signature: Option<String>,
    ) -> Result<(), String> {
        let mut body = serde_json::json!({ "encrypted_result": result.to_string() });
        if let Some(sig) = envelope_signature {
            body["result_envelope_signature"] = serde_json::Value::String(sig);
            body["result_envelope_version"] = serde_json::Value::String("v1".to_string());
        }
        self.post_complete(job_id, body).await
    }

    /// Complete a sealed (E2E) job: result sealed to caller's key, usage cleartext.
    pub async fn complete_job_sealed(
        &self,
        job_id: &str,
        sealed_result: serde_json::Value,
        usage: Option<serde_json::Value>,
        envelope_signature: Option<String>,
    ) -> Result<(), String> {
        let mut body = serde_json::json!({ "sealed_result": sealed_result });
        if let Some(u) = usage {
            body["usage"] = u;
        }
        if let Some(sig) = envelope_signature {
            body["result_envelope_signature"] = serde_json::Value::String(sig);
            body["result_envelope_version"] = serde_json::Value::String("v1".to_string());
        }
        self.post_complete(job_id, body).await
    }

    /// Post one sealed stream chunk to the orchestrator, which relays it to the
    /// waiting client over SSE. `ephemeral_public_key` is sent only on seq 0 (the
    /// client derives the stream output key from it). On the final chunk, `usage`
    /// carries the token counts the orchestrator uses to bill.
    /// Returns `Ok(true)` when the orchestrator reports the client stream is closed
    /// (the browser went away) so the node can stop generating early; `Ok(false)`
    /// to keep going. `Err` on a real failure (e.g. signature rejected).
    #[allow(clippy::too_many_arguments)]
    pub async fn post_chunk(
        &self,
        job_id: &str,
        seq: u64,
        is_final: bool,
        ephemeral_public_key: Option<&str>,
        ciphertext_b58: &str,
        usage: Option<serde_json::Value>,
        envelope_signature: Option<String>,
    ) -> Result<bool, String> {
        let url = format!("{}/grid/jobs/{}/chunk", self.base_url, enc_seg(job_id));
        let token = self.get_token()?;

        let mut body = serde_json::json!({
            "seq": seq,
            "final": is_final,
            "ciphertext": ciphertext_b58,
            "algorithm": crate::encryption::ALGO_V2_STREAM,
        });
        if let Some(eph) = ephemeral_public_key {
            body["ephemeral_public_key"] = serde_json::Value::String(eph.to_string());
        }
        if let Some(u) = usage {
            body["usage"] = u;
        }
        if let Some(sig) = envelope_signature {
            body["result_envelope_signature"] = serde_json::Value::String(sig);
            body["result_envelope_version"] = serde_json::Value::String("v1".to_string());
        }

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("chunk post failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("chunk post failed ({status}): {text}"));
        }
        // 200 body: { ok, closed }. `closed` means the client is gone.
        let parsed: serde_json::Value = serde_json::from_slice(&Self::read_body_capped(resp).await?)
            .unwrap_or(serde_json::Value::Null);
        Ok(parsed.get("closed").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    /// Tell the orchestrator a streaming job failed to generate (e.g. the model
    /// truncated). It aborts the client's SSE stream and fails the job — no billing.
    pub async fn report_stream_error(&self, job_id: &str) -> Result<(), String> {
        let url = format!("{}/grid/jobs/{}/chunk", self.base_url, enc_seg(job_id));
        let token = self.get_token()?;
        let body = serde_json::json!({ "error": true });
        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("stream error report failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("stream error report failed ({})", resp.status()));
        }
        Ok(())
    }

    async fn post_complete(&self, job_id: &str, body: serde_json::Value) -> Result<(), String> {
        let url = format!("{}/grid/jobs/{}/complete", self.base_url, enc_seg(job_id));
        let token = self.get_token()?;

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Job completion failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Job completion failed ({status}): {text}"));
        }

        Ok(())
    }

    pub async fn fail_job(&self, job_id: &str, reason: &str) -> Result<(), String> {
        let url = format!("{}/grid/jobs/{}/fail", self.base_url, enc_seg(job_id));
        let token = self.get_token()?;

        let body = serde_json::json!({ "reason": reason });

        let resp = self
            .client
            .post(&url)
            .header("X-Node-Auth", token)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Job failure report failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Job failure report failed ({status}): {text}"));
        }

        Ok(())
    }
}
