use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use sha2::Digest;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use crate::config::NodeConfig;
use crate::crypto::NodeKeypair;
use crate::encryption;
use crate::inference::InferenceEngine;
use crate::orchestrator::PendingJob;

#[derive(serde::Deserialize, Debug)]
struct WSIncoming {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    job: Option<WSJob>,
}

#[derive(serde::Deserialize, Debug)]
struct WSJob {
    id: String,
    job_type: String,
    model: Option<String>,
    input_payload: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct WSHeartbeat {
    #[serde(rename = "type")]
    msg_type: &'static str,
    current_load: f64,
    available_models: Vec<String>,
    // Deterministic X25519 key clients/orchestrator encrypt prompts to. Published
    // on every heartbeat so it's always current without a separate attest step.
    encryption_public_key: String,
}

#[derive(serde::Serialize)]
struct WSJobComplete {
    #[serde(rename = "type")]
    msg_type: &'static str,
    job_id: String,
    encrypted_result: String,
    result_signature: String,
    attestation_proof: Option<serde_json::Value>,
    // E2E path: when the prompt was sealed, the result is sealed to the caller's
    // response key and the (non-sensitive) token usage is sent in cleartext so
    // the orchestrator can bill without reading content.
    #[serde(skip_serializing_if = "Option::is_none")]
    sealed_result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct WSJobFailed {
    #[serde(rename = "type")]
    msg_type: &'static str,
    job_id: String,
    reason: String,
}

pub async fn run_websocket(
    cfg: NodeConfig,
    orchestrator_url: &str,
    keypair: Arc<NodeKeypair>,
    engine: Option<Arc<InferenceEngine>>,
    models: Vec<String>,
    load_factor: f64,
    heartbeat_interval: u64,
) {
    let ws_url = build_ws_url(orchestrator_url, &cfg.node_id, &cfg.auth_token);
    let mut backoff = 1u64;
    let max_backoff = 60u64;

    loop {
        tracing::info!("Connecting WebSocket to orchestrator...");

        match tokio_tungstenite::connect_async(&ws_url).await {
            Ok((stream, _resp)) => {
                tracing::info!("WebSocket connected — real-time job dispatch active");
                backoff = 1;

                let (write, read) = stream.split();
                let (tx, rx) = mpsc::channel::<Message>(64);

                let write_handle = tokio::spawn(write_loop(write, rx));
                let read_handle = tokio::spawn(read_loop(
                    read,
                    tx.clone(),
                    keypair.clone(),
                    engine.clone(),
                ));

                let hb_enc_key = encryption::EncryptionKeypair::from_ed25519_seed(
                    &keypair.signing_key.to_bytes(),
                ).public_key_bs58();
                let heartbeat_handle = tokio::spawn(heartbeat_loop(
                    tx.clone(),
                    models.clone(),
                    load_factor,
                    heartbeat_interval,
                    hb_enc_key,
                ));

                tokio::select! {
                    r = read_handle => {
                        if let Err(e) = r {
                            tracing::warn!("WebSocket read task panicked: {e}");
                        }
                    }
                    r = write_handle => {
                        if let Err(e) = r {
                            tracing::warn!("WebSocket write task panicked: {e}");
                        }
                    }
                }

                heartbeat_handle.abort();
                tracing::warn!("WebSocket disconnected, reconnecting in {backoff}s...");
            }
            Err(e) => {
                tracing::warn!("WebSocket connection failed: {e}");
                tracing::info!("Retrying in {backoff}s...");
            }
        }

        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff * 2).min(max_backoff);
    }
}

fn build_ws_url(orchestrator_url: &str, node_id: &str, auth_token: &str) -> String {
    let base = orchestrator_url
        .replace("https://", "wss://")
        .replace("http://", "ws://");
    format!(
        "{}/grid/nodes/{}/ws?auth_token={}",
        base.trim_end_matches('/'),
        node_id,
        auth_token,
    )
}

async fn write_loop(
    mut write: futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        Message,
    >,
    mut rx: mpsc::Receiver<Message>,
) {
    while let Some(msg) = rx.recv().await {
        if let Err(e) = write.send(msg).await {
            tracing::error!("WebSocket send error: {e}");
            break;
        }
    }
}

async fn read_loop(
    mut read: futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
    tx: mpsc::Sender<Message>,
    keypair: Arc<NodeKeypair>,
    engine: Option<Arc<InferenceEngine>>,
) {
    while let Some(result) = read.next().await {
        match result {
            Ok(Message::Text(text)) => {
                let msg: WSIncoming = match serde_json::from_str(&text) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!("Invalid WebSocket message: {e}");
                        continue;
                    }
                };

                match msg.msg_type.as_str() {
                    "job" => {
                        if let Some(ws_job) = msg.job {
                            tracing::info!("Received job via WebSocket: {} ({})", ws_job.id, ws_job.job_type);
                            let tx_clone = tx.clone();
                            let kp_clone = keypair.clone();
                            let eng_clone = engine.clone();
                            tokio::spawn(async move {
                                handle_ws_job(ws_job, kp_clone, eng_clone, tx_clone).await;
                            });
                        }
                    }
                    "ping" => {
                        let pong = serde_json::json!({"type": "pong"});
                        let _ = tx.send(Message::Text(pong.to_string())).await;
                    }
                    "job_complete_ack" | "job_failed_ack" => {
                        tracing::debug!("Ack received: {text}");
                    }
                    "error" => {
                        tracing::warn!("Server error: {text}");
                    }
                    other => {
                        tracing::debug!("Unknown message type: {other}");
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let _ = tx.send(Message::Pong(data)).await;
            }
            Ok(Message::Close(_)) => {
                tracing::info!("Server sent close frame");
                break;
            }
            Err(e) => {
                tracing::warn!("WebSocket read error: {e}");
                break;
            }
            _ => {}
        }
    }
}

async fn heartbeat_loop(
    tx: mpsc::Sender<Message>,
    models: Vec<String>,
    load_factor: f64,
    interval_secs: u64,
    encryption_public_key: String,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
    loop {
        interval.tick().await;
        let hb = WSHeartbeat {
            msg_type: "heartbeat",
            current_load: load_factor,
            available_models: models.clone(),
            encryption_public_key: encryption_public_key.clone(),
        };
        let msg = serde_json::to_string(&hb).unwrap();
        if tx.send(Message::Text(msg)).await.is_err() {
            break;
        }
    }
}

async fn handle_ws_job(
    ws_job: WSJob,
    keypair: Arc<NodeKeypair>,
    engine: Option<Arc<InferenceEngine>>,
    tx: mpsc::Sender<Message>,
) {
    // Detect & decrypt a sealed (E2E) prompt; plaintext payloads pass through
    // unchanged so existing dispatch keeps working exactly as before.
    let node_secret = keypair.signing_key.to_bytes();
    let (effective_payload, response_pubkey) = match ws_job.input_payload.as_ref() {
        Some(p) => match encryption::unseal_input(p, &node_secret) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Failed to unseal job {}: {e}", ws_job.id);
                let failed = WSJobFailed { msg_type: "job_failed", job_id: ws_job.id.clone(), reason: format!("decrypt failed: {e}") };
                let _ = tx.send(Message::Text(serde_json::to_string(&failed).unwrap())).await;
                return;
            }
        },
        None => (serde_json::Value::Null, None),
    };

    let job = PendingJob {
        id: ws_job.id.clone(),
        job_type: ws_job.job_type.clone(),
        model: ws_job.model,
        input_payload: if effective_payload.is_null() { None } else { Some(effective_payload) },
    };

    let result = match job.job_type.as_str() {
        "inference" => execute_inference(&engine, &job).await,
        _ => Err(format!("Unsupported job type: {}", job.job_type)),
    };

    match result {
        Ok(output) => {
            let result_bytes = output.to_string();
            let signature = keypair.sign_message(result_bytes.as_bytes());
            // Token usage is non-sensitive and is always exposed in cleartext so
            // the orchestrator can bill without reading prompt/response content.
            let usage = output.get("usage").cloned();

            if let Some(resp_pub) = response_pubkey {
                // ── E2E path: seal the result to the caller's response key ──
                match encryption::encrypt_for_recipient(&resp_pub, result_bytes.as_bytes()) {
                    Ok((sealed, ephemeral_pub)) => {
                        let sealed_result = serde_json::json!({
                            "ciphertext": bs58::encode(&sealed).into_string(),
                            "ephemeral_public_key": bs58::encode(ephemeral_pub).into_string(),
                            "algorithm": "x25519-xchacha20poly1305",
                        });
                        let complete = WSJobComplete {
                            msg_type: "job_complete",
                            job_id: ws_job.id.clone(),
                            encrypted_result: String::new(),
                            result_signature: signature,
                            attestation_proof: None,
                            sealed_result: Some(sealed_result),
                            usage,
                        };
                        let msg = serde_json::to_string(&complete).unwrap();
                        if let Err(e) = tx.send(Message::Text(msg)).await {
                            tracing::error!("Failed to send sealed completion: {e}");
                        } else {
                            tracing::info!("Job {} completed (E2E sealed)", ws_job.id);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to seal result for job {}: {e}", ws_job.id);
                        let failed = WSJobFailed { msg_type: "job_failed", job_id: ws_job.id.clone(), reason: format!("seal failed: {e}") };
                        let _ = tx.send(Message::Text(serde_json::to_string(&failed).unwrap())).await;
                    }
                }
                return;
            }

            // ── Plaintext path (unchanged) ──
            let encrypted = encryption::encrypt_result(
                &keypair.signing_key.to_bytes(),
                result_bytes.as_bytes(),
            );

            let attestation_proof = match &encrypted {
                Ok(enc) => Some(serde_json::json!({
                    "result_signature": signature,
                    "encryption": enc,
                    "plaintext_hash": hex::encode(sha2::Sha256::digest(result_bytes.as_bytes())),
                })),
                Err(e) => {
                    tracing::warn!("Encryption failed: {e}");
                    None
                }
            };

            let complete = WSJobComplete {
                msg_type: "job_complete",
                job_id: ws_job.id.clone(),
                encrypted_result: result_bytes,
                result_signature: signature,
                attestation_proof,
                sealed_result: None,
                usage: None,
            };

            let msg = serde_json::to_string(&complete).unwrap();
            if let Err(e) = tx.send(Message::Text(msg)).await {
                tracing::error!("Failed to send job completion: {e}");
            } else {
                tracing::info!("Job {} completed via WebSocket (signed + encrypted)", ws_job.id);
            }
        }
        Err(reason) => {
            let failed = WSJobFailed {
                msg_type: "job_failed",
                job_id: ws_job.id.clone(),
                reason: reason.clone(),
            };
            let msg = serde_json::to_string(&failed).unwrap();
            if let Err(e) = tx.send(Message::Text(msg)).await {
                tracing::error!("Failed to send job failure: {e}");
            }
            tracing::warn!("Job {} failed: {reason}", ws_job.id);
        }
    }
}

use crate::inference::ChatMessage;

async fn execute_inference(
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
) -> Result<serde_json::Value, String> {
    let engine = engine.as_ref().ok_or("No inference engine configured")?;

    let payload = job.input_payload.as_ref().ok_or("Job has no input payload")?;

    let messages: Vec<ChatMessage> = if let Some(msgs) = payload.get("messages") {
        serde_json::from_value(msgs.clone())
            .map_err(|e| format!("Invalid messages format: {e}"))?
    } else if let Some(prompt) = payload.get("prompt").and_then(|p| p.as_str()) {
        vec![ChatMessage { role: "user".to_string(), content: prompt.to_string() }]
    } else {
        return Err("Payload must contain 'messages' array or 'prompt' string".to_string());
    };

    let temperature = payload.get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.7)
        .clamp(0.0, 2.0);

    let max_tokens = payload.get("max_tokens")
        .and_then(|t| t.as_i64())
        .unwrap_or(2048)
        .clamp(1, 8192) as i32;

    let result = engine.chat_completion(messages, temperature, max_tokens).await?;

    Ok(serde_json::json!({
        "content": result.content,
        "model": result.model,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
    }))
}
