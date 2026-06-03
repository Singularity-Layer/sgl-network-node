//! WebSocket transport to the orchestrator's NodeSession Durable Object.
//!
//! Primary use: **push job dispatch** — the orchestrator pushes a job the instant
//! it's created instead of waiting for the next REST heartbeat (which removes the
//! up-to-5s pickup delay). The node also answers pings and applies token rotations
//! delivered over the socket.
//!
//! This is an ADDITIVE fast-path. The REST heartbeat loop keeps running as the
//! control plane (DB liveness, token rotation, stake checks) and as the fallback
//! when the WS is down — so the node never goes dark if the socket drops. Jobs that
//! arrive via both transports are de-duplicated by id on the node side.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::orchestrator::{OrchestratorClient, PendingJob};

/// Shared connection state, readable by the rest of the node (e.g. to prefer WS).
pub struct WsState {
    pub connected: AtomicBool,
}

impl WsState {
    pub fn new() -> Self {
        Self {
            connected: AtomicBool::new(false),
        }
    }
}

/// Percent-encode a value for use in a URL query string (token may contain
/// non-unreserved characters).
fn query_encode(s: &str) -> String {
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

/// Run the WS client with automatic reconnect. `on_job` is invoked for each
/// dispatched job; `on_token` when the orchestrator rotates the auth token. The
/// auth token is read fresh from `token` on every (re)connect, so rotations applied
/// elsewhere (the REST heartbeat) take effect on the next reconnect too.
pub async fn run<FJob, FTok>(
    orchestrator_url: String,
    node_id: String,
    client: Arc<OrchestratorClient>,
    state: Arc<WsState>,
    on_job: FJob,
    on_token: FTok,
) where
    FJob: Fn(PendingJob) + Send + Sync + 'static,
    FTok: Fn(String, Option<String>) + Send + Sync + 'static,
{
    let ws_base = orchestrator_url
        .trim_end_matches('/')
        .replacen("https://", "wss://", 1)
        .replacen("http://", "ws://", 1);

    let mut backoff: u64 = 1;
    loop {
        let tok = client.current_token().unwrap_or_default();
        let url = format!(
            "{ws_base}/grid/nodes/{node_id}/ws?auth_token={}",
            query_encode(&tok)
        );

        match connect_async(&url).await {
            Ok((ws_stream, _resp)) => {
                backoff = 1;
                state.connected.store(true, Ordering::SeqCst);
                tracing::info!("WS connected to orchestrator (push dispatch active)");

                let (mut write, mut read) = ws_stream.split();

                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(txt)) => {
                            let v: serde_json::Value = match serde_json::from_str(&txt) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };
                            match v.get("type").and_then(|t| t.as_str()) {
                                Some("job") => {
                                    if let Some(job_val) = v.get("job") {
                                        match serde_json::from_value::<PendingJob>(job_val.clone()) {
                                            Ok(job) => on_job(job),
                                            Err(e) => {
                                                tracing::warn!("WS job parse failed: {e}")
                                            }
                                        }
                                    }
                                }
                                Some("ping") => {
                                    let _ = write
                                        .send(Message::Text("{\"type\":\"pong\"}".to_string()))
                                        .await;
                                }
                                Some("token_rotated") => {
                                    if let Some(nt) =
                                        v.get("new_auth_token").and_then(|t| t.as_str())
                                    {
                                        let exp = v
                                            .get("token_expires_at")
                                            .and_then(|t| t.as_str())
                                            .map(|s| s.to_string());
                                        on_token(nt.to_string(), exp);
                                    }
                                }
                                // job_complete_ack / job_failed_ack / stream_chunk_ack /
                                // error — informational; completion + streaming run over
                                // REST in this iteration.
                                _ => {}
                            }
                        }
                        Ok(Message::Ping(p)) => {
                            let _ = write.send(Message::Pong(p)).await;
                        }
                        Ok(Message::Close(_)) => break,
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("WS read error: {e}");
                            break;
                        }
                    }
                }

                state.connected.store(false, Ordering::SeqCst);
                tracing::warn!("WS disconnected; falling back to REST polling, will reconnect");
            }
            Err(e) => {
                tracing::warn!("WS connect failed ({e}); using REST polling");
            }
        }

        tokio::time::sleep(Duration::from_secs(backoff)).await;
        backoff = (backoff.saturating_mul(2)).min(30);
    }
}
