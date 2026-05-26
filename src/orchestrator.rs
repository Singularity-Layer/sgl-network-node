use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;
use std::time::Duration;

use crate::tee::TeeCapabilities;

pub struct OrchestratorClient {
    client: Client,
    base_url: String,
    auth_token: RwLock<Option<String>>,
}

#[derive(Serialize)]
struct RegisterRequest {
    wallet_address: String,
    tee_type: String,
    available_models: Vec<String>,
    public_key: String,
    cpu_cores: u32,
    ram_gb: f64,
    gpu_model: String,
    supported_runtimes: Vec<String>,
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
}

#[derive(Deserialize)]
pub struct HeartbeatResponse {
    pub status: String,
    #[serde(default)]
    pub pending_jobs: Vec<PendingJob>,
    pub new_auth_token: Option<String>,
    pub token_expires_at: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct PendingJob {
    pub id: String,
    pub job_type: String,
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

    pub fn update_auth_token(&self, token: String) {
        let mut guard = self.auth_token.write().unwrap();
        *guard = Some(token);
    }

    fn get_token(&self) -> Result<String, String> {
        let guard = self.auth_token.read().unwrap();
        guard.clone().ok_or_else(|| "No auth token configured".to_string())
    }

    pub async fn register(
        &self,
        wallet: &str,
        tee_type: &str,
        models: &[String],
        public_key: &str,
        caps: &TeeCapabilities,
    ) -> Result<RegisterResponse, String> {
        let url = format!("{}/grid/nodes/register", self.base_url);

        let body = RegisterRequest {
            wallet_address: wallet.to_string(),
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

    pub async fn heartbeat(
        &self,
        node_id: &str,
        models: &[String],
        current_load: f64,
    ) -> Result<HeartbeatResponse, String> {
        let url = format!("{}/grid/nodes/heartbeat", self.base_url);
        let token = self.get_token()?;

        let body = HeartbeatRequest {
            current_load,
            available_models: models.to_vec(),
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

        resp.json::<HeartbeatResponse>()
            .await
            .map_err(|e| format!("Failed to parse heartbeat response: {e}"))
    }

    pub async fn request_challenge(&self, node_id: &str) -> Result<ChallengeResponse, String> {
        let url = format!("{}/grid/nodes/{}/challenge", self.base_url, node_id);
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

    pub async fn verify_attestation(
        &self,
        node_id: &str,
        signature: &str,
    ) -> Result<AttestationResponse, String> {
        let url = format!(
            "{}/grid/nodes/{}/verify-attestation",
            self.base_url, node_id
        );
        let token = self.get_token()?;

        let body = VerifyAttestationRequest {
            signature: signature.to_string(),
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

    pub async fn get_node_status(
        &self,
        node_id: &str,
    ) -> Result<NodeStatusResponse, String> {
        let url = format!("{}/grid/nodes/{}", self.base_url, node_id);
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
    ) -> Result<(), String> {
        let url = format!("{}/grid/jobs/{}/complete", self.base_url, job_id);
        let token = self.get_token()?;

        let body = serde_json::json!({ "encrypted_result": result.to_string() });

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
        let url = format!("{}/grid/jobs/{}/fail", self.base_url, job_id);
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
