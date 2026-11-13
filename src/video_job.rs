//! Grid `video-v1` job glue: the seam between the node loop (`node.rs`) and the video engine
//! (`video.rs`). Kept out of node.rs on purpose (CLAUDE.md rule 6: the hot file must not grow).
//! Spec: x402Studio `GRID_VIDEO_V1.md` sections 5.2 to 5.8.
//!
//! Flow for one dispatched job:
//!   1. parse the cleartext binding (`input_payload.video_reservation`) and check it against
//!      the loaded model's contract (size, seconds, fps, frames, steps, cfg, pins, caps);
//!   2. require the sealed v2 envelope and unseal the canonical
//!      `{"protocol","request_id","model","prompt"}` plaintext; bind it to the job id, the
//!      model and the reservation (prompt byte length and sha256);
//!   3. apply the narrow node-side floor (sexual content involving minors only);
//!   4. generate and encode while reporting progress at most every 5 s; a `continue:false`
//!      answer kills the engine at once and the node stops without reporting anything;
//!   5. upload the MP4 (and the WebP thumbnail) streamed from disk, retrying 5xx and network
//!      errors with backoff, never a 4xx;
//!   6. sign over sha256(MP4) in the `video` result domain and complete over REST, retrying
//!      until the job deadline (the server completion is one idempotent transaction).
//!
//! The per-job folder (0700, under the node's config directory) is removed on every path.
//! Failure codes: `video_input_invalid` (client side), `video_content_refused` (the floor) and
//! `video_runtime_failed` (this node). WS `job_complete` is never used for video.
use crate::inference::InferenceEngine;
use crate::orchestrator::{CompletionAttemptError, OrchestratorClient, PendingJob};
use crate::video::{self, Progress, Stage, VideoError, VideoModel, VideoOutput, VideoRequest};
use serde::Deserialize;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const INPUT_INVALID: &str = "video_input_invalid";
pub const CONTENT_REFUSED: &str = "video_content_refused";
pub const RUNTIME_FAILED: &str = "video_runtime_failed";

/// Spec 5.3: at most one progress call per 5 s.
pub const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const UPLOAD_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(120);
const COMPLETE_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
/// No upload or completion attempt starts with less than this left before the deadline.
const MIN_ATTEMPT: Duration = Duration::from_secs(5);
/// The sealed request is tiny: canonical JSON plus at most 2000 prompt bytes (escaped).
const MAX_SEALED_INPUT: usize = 16 * 1024;

/// Cleartext reservation binding (spec 5.2). Every field is required; unknown fields are refused.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Reservation {
    pub protocol: String,
    pub request_id: String,
    pub model: String,
    pub model_sha256: String,
    pub width: u32,
    pub height: u32,
    pub seconds: u32,
    pub fps: u32,
    pub frames: u32,
    pub steps: u32,
    pub cfg_scale: f64,
    pub seed: u64,
    pub prompt_bytes: u64,
    pub prompt_sha256: String,
    pub deadline_seconds: u64,
    pub upload_max_bytes: u64,
    pub thumbnail_max_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dispatch {
    enc: serde_json::Value,
    video_reservation: serde_json::Value,
}

/// The sealed v2 envelope the node's unseal path expects (same as the image lane).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    algorithm: String,
    encoding: String,
    ciphertext: String,
    #[allow(dead_code)]
    client_ephemeral_pubkey: String,
    #[allow(dead_code)]
    client_response_pubkey: String,
}

/// Canonical sealed plaintext: `{"protocol","request_id","model","prompt"}`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedRequest {
    protocol: String,
    request_id: String,
    model: String,
    prompt: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

fn lower_hex64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Parse and check the cleartext binding against the loaded model before anything is
/// decrypted. Any mismatch is the client's (or the orchestrator's) problem: `video_input_invalid`.
pub fn parse_binding(job: &PendingJob, model: &VideoModel) -> Result<Reservation, &'static str> {
    // The dispatch row's model, when present, must agree; the reservation's model is always
    // checked below.
    if job.model.as_deref().is_some_and(|m| m != model.id) {
        return Err(INPUT_INVALID);
    }
    let payload = job.input_payload.as_ref().ok_or(INPUT_INVALID)?;
    let dispatch: Dispatch = serde_json::from_value(payload.clone()).map_err(|_| INPUT_INVALID)?;
    if !dispatch.enc.is_object() {
        return Err(INPUT_INVALID);
    }
    let r: Reservation =
        serde_json::from_value(dispatch.video_reservation).map_err(|_| INPUT_INVALID)?;
    let ok = r.protocol == video::PROTOCOL
        && crate::image::canonical_uuid_v4(&r.request_id)
        && r.request_id == job.id
        && r.model == model.id
        && r.model_sha256 == model.diffusion.sha256
        && model.size_allowed(r.width, r.height)
        && (video::MIN_SECONDS..=video::MAX_SECONDS).contains(&r.seconds)
        && r.fps == model.fps
        && r.frames == video::frames_for(r.seconds, model.fps)
        && r.steps == model.steps
        && r.cfg_scale == model.cfg_scale
        && r.seed <= u64::from(u32::MAX)
        && r.prompt_bytes >= 1
        && r.prompt_bytes <= video::MAX_PROMPT_BYTES as u64
        && lower_hex64(&r.prompt_sha256)
        && (video::MIN_DEADLINE_SECONDS..=video::MAX_DEADLINE_SECONDS)
            .contains(&r.deadline_seconds)
        && r.upload_max_bytes >= 1
        && r.upload_max_bytes <= video::MAX_VIDEO_BYTES
        && r.thumbnail_max_bytes <= video::MAX_THUMBNAIL_BYTES;
    if ok {
        Ok(r)
    } else {
        Err(INPUT_INVALID)
    }
}

/// Unseal the dispatch (v2 only, with a response key, strict envelope) and bind the canonical
/// plaintext to the job and the reservation. Returns the prompt.
pub fn unseal_prompt(
    job: &PendingJob,
    node_secret: &[u8; 32],
    binding: &Reservation,
) -> Result<String, &'static str> {
    let payload = job.input_payload.as_ref().ok_or(INPUT_INVALID)?;
    let envelope: Envelope = payload
        .get("enc")
        .cloned()
        .and_then(|enc| serde_json::from_value(enc).ok())
        .ok_or(INPUT_INVALID)?;
    if envelope.algorithm != crate::encryption::ALGO_V2
        || envelope.encoding != "base64"
        || envelope.ciphertext.len() > MAX_SEALED_INPUT.div_ceil(3) * 4 + 64
    {
        return Err(INPUT_INVALID);
    }
    let (inner, response, version) =
        crate::encryption::unseal_input(payload, node_secret).map_err(|_| INPUT_INVALID)?;
    if response.is_none() || version != crate::encryption::EncVersion::V2 {
        return Err(INPUT_INVALID);
    }
    let sealed: SealedRequest = serde_json::from_value(inner).map_err(|_| INPUT_INVALID)?;
    let prompt = sealed.prompt;
    let ok = sealed.protocol == video::PROTOCOL
        && sealed.request_id == job.id
        && sealed.request_id == binding.request_id
        && sealed.model == binding.model
        && prompt.len() as u64 == binding.prompt_bytes
        && sha256_hex(prompt.as_bytes()) == binding.prompt_sha256
        && !prompt.contains('\0')
        && !prompt.trim().is_empty();
    if ok {
        Ok(prompt)
    } else {
        Err(INPUT_INVALID)
    }
}

pub fn build_request(binding: &Reservation, prompt: String) -> VideoRequest {
    VideoRequest {
        job_id: binding.request_id.clone(),
        prompt,
        width: binding.width,
        height: binding.height,
        seconds: binding.seconds,
        frames: binding.frames,
        fps: binding.fps,
        steps: binding.steps,
        seed: binding.seed as u32,
        max_video_bytes: binding.upload_max_bytes,
        max_thumbnail_bytes: binding.thumbnail_max_bytes,
    }
}

/// The REST completion body (spec 5.5), signed over sha256 of the MP4 bytes in the `video`
/// result domain: `sgl-result-v1\n<job_id>\nvideo\n<sha256>`.
pub fn completion_body(
    node_secret: &[u8; 32],
    job_id: &str,
    binding: &Reservation,
    output: &VideoOutput,
    thumbnail_sha256: Option<&str>,
    gpu: &str,
) -> serde_json::Value {
    let signature =
        crate::crypto::sign_result_digest(node_secret, job_id, "video", &output.video.sha256);
    serde_json::json!({
        "video": {
            "sha256": output.video.sha256,
            "bytes": output.video.bytes,
            "content_type": video::CONTENT_TYPE_VIDEO,
            "width": binding.width,
            "height": binding.height,
            "seconds": binding.seconds,
            "frames": binding.frames,
            "fps": binding.fps,
            "thumbnail_sha256": thumbnail_sha256,
        },
        "usage": {
            "seconds": binding.seconds,
            "frames": binding.frames,
            "steps": binding.steps,
            "gen_ms": output.gen_ms,
            "encode_ms": output.encode_ms,
            "gpu": gpu,
        },
        "result_envelope_signature": signature,
        "result_envelope_version": "v1",
    })
}

/// Drives a future while posting progress at most once per interval. The first report goes out
/// at once (it moves the job to `running`). `continue:false` flips the cancel signal; the
/// driven future observes it and stops on its own (the engine kills its process group).
struct Reporter<'a> {
    client: &'a OrchestratorClient,
    job_id: &'a str,
    cancel: tokio::sync::watch::Sender<bool>,
    interval: Duration,
    next_at: Instant,
}

impl Reporter<'_> {
    fn stopped(&self) -> bool {
        *self.cancel.borrow()
    }

    async fn drive<F: std::future::Future>(
        &mut self,
        progress: &tokio::sync::watch::Receiver<Progress>,
        fut: F,
    ) -> F::Output {
        tokio::pin!(fut);
        loop {
            let wait = self.next_at.saturating_duration_since(Instant::now());
            let stopped = self.stopped();
            tokio::select! {
                out = &mut fut => return out,
                _ = tokio::time::sleep(wait), if !stopped => {
                    self.next_at = Instant::now() + self.interval;
                    let p = *progress.borrow();
                    let report = self.client.report_video_progress(
                        self.job_id, p.stage.as_str(), p.step, p.total);
                    tokio::pin!(report);
                    tokio::select! {
                        out = &mut fut => return out,
                        answer = &mut report => match answer {
                            Ok(false) => {
                                tracing::warn!("video job {} stopped by the orchestrator", self.job_id);
                                self.cancel.send_replace(true);
                            }
                            Ok(true) => {}
                            Err(e) => tracing::debug!("video progress for {}: {e}", self.job_id),
                        },
                    }
                }
            }
        }
    }
}

/// Wait `wait`, unless the cancel signal fires first. True when cancelled.
async fn sleep_or_cancel(wait: Duration, cancel: &mut tokio::sync::watch::Receiver<bool>) -> bool {
    if *cancel.borrow() {
        return true;
    }
    let sleep = tokio::time::sleep(wait);
    tokio::pin!(sleep);
    let signalled = tokio::select! {
        _ = &mut sleep => false,
        answer = cancel.wait_for(|stop| *stop) => answer.is_ok(),
    };
    if !signalled && !sleep.is_elapsed() {
        // The cancel sender is gone (nothing can cancel any more): just finish the wait.
        sleep.await;
    }
    signalled || *cancel.borrow()
}

/// Upload one artifact with exponential backoff (1, 2, 4, 8, 16 s) until the deadline. A 4xx
/// other than 408/429 ends it at once; so does a cancel.
async fn upload_with_retry(
    client: &OrchestratorClient,
    job_id: &str,
    variant: &str,
    content_type: &str,
    artifact: &video::Artifact,
    deadline: Instant,
    mut cancel: tokio::sync::watch::Receiver<bool>,
) -> Result<u32, String> {
    let mut attempt = 0u32;
    let mut last = String::from("no upload attempt fit before the deadline");
    loop {
        let wait = if attempt == 0 {
            Duration::ZERO
        } else {
            crate::image::retry_backoff(attempt)
        };
        let start = Instant::now() + wait;
        let remaining = deadline.saturating_duration_since(start);
        if remaining < MIN_ATTEMPT {
            return Err(last);
        }
        if sleep_or_cancel(wait, &mut cancel).await {
            return Err("cancelled".into());
        }
        attempt += 1;
        match client
            .upload_video_artifact(
                job_id,
                variant,
                &artifact.path,
                content_type,
                artifact.bytes,
                &artifact.sha256,
                remaining.min(UPLOAD_ATTEMPT_TIMEOUT),
            )
            .await
        {
            Ok(()) => return Ok(attempt),
            Err(CompletionAttemptError::Permanent(e)) => return Err(e),
            Err(CompletionAttemptError::Retryable(e)) => {
                tracing::warn!("video {variant} upload attempt {attempt} for {job_id} failed: {e}");
                last = e;
            }
        }
    }
}

/// REST completion with backoff until the deadline. A permanent refusal ends it; an abandoned
/// completion is never billed (the reaper fails the job and releases the hold).
async fn complete_with_retry(
    client: &OrchestratorClient,
    job_id: &str,
    body: &serde_json::Value,
    deadline: Instant,
) -> Result<u32, String> {
    let mut attempt = 0u32;
    let mut last = String::from("no completion attempt fit before the deadline");
    loop {
        let wait = if attempt == 0 {
            Duration::ZERO
        } else {
            crate::image::retry_backoff(attempt)
        };
        let remaining = deadline.saturating_duration_since(Instant::now() + wait);
        if remaining < MIN_ATTEMPT {
            return Err(last);
        }
        tokio::time::sleep(wait).await;
        attempt += 1;
        match client
            .complete_video_attempt(job_id, body, remaining.min(COMPLETE_ATTEMPT_TIMEOUT))
            .await
        {
            Ok(()) => return Ok(attempt),
            Err(CompletionAttemptError::Permanent(e)) => return Err(e),
            Err(CompletionAttemptError::Retryable(e)) => {
                tracing::warn!("video completion attempt {attempt} for {job_id} failed: {e}");
                last = e;
            }
        }
    }
}

/// How one job ended, node side.
#[derive(Debug, PartialEq)]
pub enum Outcome {
    /// Completed over REST after this many attempts.
    Completed(u32),
    /// Reported to `/fail` with this code.
    Failed(&'static str),
    /// The orchestrator said stop (`continue:false`): nothing more is reported.
    Stopped,
    /// Uploaded but the completion never landed before the deadline; the reaper owns it.
    Abandoned(String),
}

/// Time kept after generation for the uploads and the completion: a quarter of the deadline,
/// between 30 s and 3 min.
fn upload_reserve(deadline_seconds: u64) -> Duration {
    Duration::from_secs((deadline_seconds / 4).clamp(30, 180))
}

/// The whole job, without telemetry or the final `/fail` call (see `process`).
pub async fn run(
    client: &OrchestratorClient,
    engine: &video::VideoEngine,
    job: &PendingJob,
    node_secret: &[u8; 32],
    received: Instant,
    interval: Duration,
) -> Outcome {
    let binding = match parse_binding(job, engine.model()) {
        Ok(binding) => binding,
        Err(code) => return Outcome::Failed(code),
    };
    let prompt = match unseal_prompt(job, node_secret, &binding) {
        Ok(prompt) => prompt,
        Err(code) => return Outcome::Failed(code),
    };
    if video::asks_for_sexual_minor(&prompt) {
        return Outcome::Failed(CONTENT_REFUSED);
    }
    let deadline = received + Duration::from_secs(binding.deadline_seconds);
    let generate_deadline = deadline
        .checked_sub(upload_reserve(binding.deadline_seconds))
        .unwrap_or(deadline);
    let dir = match engine.job_dir(&job.id) {
        Ok(dir) => dir,
        Err(VideoError::InputInvalid(_)) => return Outcome::Failed(INPUT_INVALID),
        Err(_) => return Outcome::Failed(RUNTIME_FAILED),
    };
    let request = build_request(&binding, prompt);
    let (progress_tx, progress_rx) = tokio::sync::watch::channel(Progress {
        stage: Stage::Loading,
        step: 0,
        total: request.steps,
    });
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
    let mut reporter = Reporter {
        client,
        job_id: &job.id,
        cancel: cancel_tx,
        interval,
        next_at: Instant::now(),
    };
    let generated = reporter
        .drive(
            &progress_rx,
            engine.generate(
                &request,
                dir.path(),
                &progress_tx,
                cancel_rx.clone(),
                generate_deadline,
            ),
        )
        .await;
    let output = match generated {
        _ if reporter.stopped() => return Outcome::Stopped,
        Ok(output) => output,
        Err(VideoError::Cancelled) => return Outcome::Stopped,
        Err(VideoError::InputInvalid(_)) => return Outcome::Failed(INPUT_INVALID),
        Err(VideoError::RuntimeFailed(reason)) => {
            tracing::warn!("video job {} failed on this node: {reason}", job.id);
            return Outcome::Failed(RUNTIME_FAILED);
        }
    };
    tracing::info!(
        "video job {}: {} ms clip, {} bytes, generated in {} ms, encoded in {} ms",
        job.id,
        output.duration_ms,
        output.video.bytes,
        output.gen_ms,
        output.encode_ms
    );
    progress_tx.send_replace(Progress {
        stage: Stage::Uploading,
        step: 0,
        total: 1,
    });
    let uploaded = reporter
        .drive(
            &progress_rx,
            upload_with_retry(
                client,
                &job.id,
                "video",
                video::CONTENT_TYPE_VIDEO,
                &output.video,
                deadline,
                cancel_rx.clone(),
            ),
        )
        .await;
    if reporter.stopped() {
        return Outcome::Stopped;
    }
    if let Err(e) = uploaded {
        tracing::warn!("video upload for {} failed: {e}", job.id);
        return Outcome::Failed(RUNTIME_FAILED);
    }
    let mut thumbnail_sha256 = None;
    if let Some(thumb) = &output.thumbnail {
        let sent = reporter
            .drive(
                &progress_rx,
                upload_with_retry(
                    client,
                    &job.id,
                    "thumbnail",
                    video::CONTENT_TYPE_THUMBNAIL,
                    thumb,
                    deadline,
                    cancel_rx.clone(),
                ),
            )
            .await;
        match sent {
            Ok(_) => thumbnail_sha256 = Some(thumb.sha256.clone()),
            Err(e) => tracing::warn!("video thumbnail upload for {} skipped: {e}", job.id),
        }
    }
    if reporter.stopped() {
        return Outcome::Stopped;
    }
    let body = completion_body(
        node_secret,
        &job.id,
        &binding,
        &output,
        thumbnail_sha256.as_deref(),
        &engine.gpu().name,
    );
    // The job folder goes before the completion wait: nothing local is needed any more.
    drop(dir);
    match complete_with_retry(client, &job.id, &body, deadline).await {
        Ok(attempts) => Outcome::Completed(attempts),
        Err(e) => Outcome::Abandoned(e),
    }
}

/// The whole video job. Called from `process_job` before the generic unseal path.
pub async fn process(
    client: &OrchestratorClient,
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    node_secret: &[u8; 32],
    telemetry: &crate::telemetry::Telemetry,
) {
    let received = Instant::now();
    let outcome = match engine.as_ref().and_then(|e| e.video()) {
        Some(video) => run(client, video, job, node_secret, received, PROGRESS_INTERVAL).await,
        None => Outcome::Failed(RUNTIME_FAILED),
    };
    let summary = match &outcome {
        Outcome::Completed(attempts) => {
            Ok(serde_json::json!({ "video": true, "attempts": attempts }))
        }
        other => Err(format!("{other:?}")),
    };
    telemetry.record_buffered_result("video", received.elapsed(), &summary);
    match outcome {
        Outcome::Completed(attempts) => {
            tracing::info!("Job {} completed (video, {attempts} attempt(s))", job.id)
        }
        Outcome::Failed(code) => {
            if let Err(e) = client.fail_job(&job.id, code).await {
                tracing::error!("Failed to report video job failure: {e}");
            }
        }
        Outcome::Stopped => tracing::info!("Video job {} stopped on request", job.id),
        Outcome::Abandoned(e) => tracing::error!("Video completion for {} abandoned: {e}", job.id),
    }
}

// ── Node glue (keeps node.rs from growing) ───────────────────────────────────────────────────

/// Node `kind` for a dedicated media worker label (`None` for chat and embedding engines).
pub fn dedicated_kind(label: &str) -> Option<&'static str> {
    match label {
        "stt-worker" => Some("transcription"),
        "tts-worker" => Some("speech"),
        l if l == crate::image::ENGINE_LABEL => Some("image"),
        l if l == video::ENGINE_LABEL => Some("video"),
        _ => None,
    }
}

/// Dedicated workers have no transient-blip grace period in the health loop.
pub fn is_dedicated_worker(label: &str) -> bool {
    label == "embedding-worker" || dedicated_kind(label).is_some()
}

/// Heartbeat telemetry runtime label for a dedicated media worker.
pub fn worker_runtime(label: &str) -> Option<&'static str> {
    match dedicated_kind(label)? {
        "transcription" => Some(crate::stt::RUNTIME),
        "speech" => Some(crate::tts::RUNTIME),
        "image" => Some(crate::image::RUNTIME),
        _ => Some(video::RUNTIME),
    }
}

/// Busy watchdog threshold floor for media models (0 for anything else).
pub fn media_wedge_ms(model_name: &str) -> u64 {
    if crate::image::is_image_model(model_name) {
        crate::image::WEDGE_MS
    } else if video::is_video_model(model_name) {
        video::WEDGE_MS
    } else {
        0
    }
}

/// REST heartbeat parts for a video engine: (available_models, capabilities.video).
pub fn rest_snapshot(
    engine: Option<&Arc<InferenceEngine>>,
    models: &[String],
    active: &AtomicU32,
) -> Option<(Vec<String>, Option<video::Capabilities>)> {
    let snapshot = engine?.video()?.readiness_snapshot(models);
    let free = 1u32.saturating_sub(active.load(Ordering::Relaxed));
    Some((
        snapshot.available_models,
        snapshot.capabilities.map(|c| c.bind_heartbeat(free)),
    ))
}

/// WebSocket heartbeat closure for an image or video node (None for any other engine).
pub fn media_ws_heartbeat(
    engine: &Arc<InferenceEngine>,
    models: Vec<String>,
    encryption_public_key: String,
    signature: Option<String>,
    key_version: Option<u32>,
    current_load: f64,
    active: Arc<AtomicU32>,
) -> Option<Arc<dyn Fn() -> serde_json::Value + Send + Sync>> {
    if engine.mode_label() == crate::image::ENGINE_LABEL {
        return Some(crate::image_job::ws_heartbeat(
            engine.clone(),
            models,
            encryption_public_key,
            signature,
            key_version,
            current_load,
            active,
        ));
    }
    engine.video()?;
    let engine = engine.clone();
    Some(Arc::new(move || {
        let (available, cap) = rest_snapshot(Some(&engine), &models, &active).unwrap_or_default();
        serde_json::json!({"type":"heartbeat", "current_load":current_load,
            "available_models":available,
            "capabilities":crate::orchestrator::video_capability_wire(cap),
            "encryption_public_key":encryption_public_key,
            "encryption_public_key_signature":signature, "key_version":key_version,
            "max_concurrent_jobs":1})
    }))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const RID: &str = "6f1d2c3b-4a59-4e8f-9a7b-1c2d3e4f5a6b";
    const NODE_SEED: [u8; 32] = [0x55u8; 32];

    fn model() -> &'static VideoModel {
        video::model_by_id("wan2.1-t2v-1.3b").unwrap()
    }

    #[derive(Clone, Debug)]
    struct Seen {
        method: String,
        path: String,
        headers: HashMap<String, String>,
        body: Vec<u8>,
    }

    type Handler = Arc<dyn Fn(&Seen, usize) -> (u16, String) + Send + Sync>;

    /// A tiny HTTP/1.1 orchestrator stand-in. `handler(request, nth call to this path)`.
    async fn mock(handler: Handler) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let log = log.clone();
                let handler = handler.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 65536];
                    let header_end = loop {
                        if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break at + 4;
                        }
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&chunk[..n]),
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    let mut lines = head.split("\r\n");
                    let mut first = lines.next().unwrap_or("").split(' ');
                    let method = first.next().unwrap_or("").to_string();
                    let path = first.next().unwrap_or("").to_string();
                    let headers: HashMap<String, String> = lines
                        .filter_map(|l| l.split_once(':'))
                        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                        .collect();
                    let len: usize = headers
                        .get("content-length")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let mut body = buf[header_end..].to_vec();
                    while body.len() < len {
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => body.extend_from_slice(&chunk[..n]),
                        }
                    }
                    let request = Seen {
                        method,
                        path,
                        headers,
                        body,
                    };
                    let nth = {
                        let mut log = log.lock().unwrap();
                        let nth = log.iter().filter(|r| r.path == request.path).count();
                        log.push(request.clone());
                        nth
                    };
                    let (status, text) = handler(&request, nth);
                    let response = format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                        text.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        (format!("http://127.0.0.1:{}", address.port()), seen)
    }

    /// Answers every route like a healthy orchestrator, with optional overrides.
    fn orchestrator(
        progress_continue: bool,
        upload: impl Fn(usize) -> u16 + Send + Sync + 'static,
    ) -> Handler {
        Arc::new(move |r: &Seen, nth: usize| {
            if r.path.ends_with("/progress") {
                (200, format!(r#"{{"continue":{progress_continue}}}"#))
            } else if r.path.contains("/video?variant=") {
                (upload(nth), r#"{"ok":true}"#.into())
            } else {
                (200, r#"{"ok":true}"#.into())
            }
        })
    }

    fn reservation(prompt: &str) -> serde_json::Value {
        serde_json::json!({
            "protocol": "video-v1", "request_id": RID, "model": model().id,
            "model_sha256": model().diffusion.sha256, "width": 832, "height": 480,
            "seconds": 2, "fps": 16, "frames": 33, "steps": 20, "cfg_scale": 5.0,
            "seed": 42, "prompt_bytes": prompt.len(), "prompt_sha256": sha256_hex(prompt.as_bytes()),
            "deadline_seconds": 300, "upload_max_bytes": 62914560, "thumbnail_max_bytes": 524288,
        })
    }

    fn sealed(plaintext: &str) -> serde_json::Value {
        let node_pub =
            crate::encryption::EncryptionKeypair::from_ed25519_seed(&NODE_SEED).public_key_bs58();
        let response_pub =
            *x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from([0x66u8; 32]))
                .as_bytes();
        crate::encryption::test_client::seal_input_v2(
            &node_pub,
            [0x77u8; 32],
            &response_pub,
            plaintext.as_bytes(),
        )
    }

    fn canonical(prompt: &str) -> String {
        format!(
            r#"{{"protocol":"video-v1","request_id":"{RID}","model":"{}","prompt":{}}}"#,
            model().id,
            serde_json::to_string(prompt).unwrap()
        )
    }

    fn job(enc: serde_json::Value, reservation: serde_json::Value) -> PendingJob {
        PendingJob {
            id: RID.into(),
            job_type: "video".into(),
            model: Some(model().id.into()),
            input_payload: Some(serde_json::json!({"enc": enc, "video_reservation": reservation})),
        }
    }

    fn good_job(prompt: &str) -> PendingJob {
        job(sealed(&canonical(prompt)), reservation(prompt))
    }

    async fn engine(mode: video::FakeMode) -> (Arc<InferenceEngine>, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "sgl-video-job-test-{}-{:x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let rt = video::Runtime::fake(
            model(),
            root.join("work"),
            video::Fake {
                job: mode,
                smoke: video::FakeMode::Ok,
            },
        );
        let engine = video::VideoEngine::start(rt).await.unwrap();
        (Arc::new(InferenceEngine::Video(Box::new(engine))), root)
    }

    async fn run_job(
        engine: &Arc<InferenceEngine>,
        handler: Handler,
        job: &PendingJob,
    ) -> Vec<Seen> {
        let (url, seen) = mock(handler).await;
        let client = OrchestratorClient::new(&url, Some("node-token".into()));
        let telemetry = crate::telemetry::Telemetry::new();
        process_with(&client, engine, job, &telemetry, Instant::now()).await;
        let out = seen.lock().unwrap().clone();
        out
    }

    /// `process` with a fast progress interval and a chosen receipt time.
    async fn process_with(
        client: &OrchestratorClient,
        engine: &Arc<InferenceEngine>,
        job: &PendingJob,
        telemetry: &crate::telemetry::Telemetry,
        received: Instant,
    ) {
        let video = engine.video().unwrap();
        let outcome = run(
            client,
            video,
            job,
            &NODE_SEED,
            received,
            Duration::from_millis(50),
        )
        .await;
        let _ = telemetry;
        if let Outcome::Failed(code) = outcome {
            client.fail_job(&job.id, code).await.unwrap();
        }
    }

    fn fail_reason(seen: &[Seen]) -> Option<String> {
        seen.iter().find(|r| r.path.ends_with("/fail")).map(|r| {
            serde_json::from_slice::<serde_json::Value>(&r.body).unwrap()["reason"]
                .as_str()
                .unwrap()
                .to_string()
        })
    }

    fn job_dir_gone(root: &std::path::Path) -> bool {
        !root.join("work").join(RID).exists()
    }

    #[tokio::test]
    async fn happy_path_uploads_signs_and_completes() {
        let (engine, root) = engine(video::FakeMode::Ok).await;
        let prompt = "a lighthouse at dusk, \"cinematic\", waves -o /etc/passwd";
        let seen = run_job(&engine, orchestrator(true, |_| 200), &good_job(prompt)).await;
        assert_eq!(fail_reason(&seen), None);
        assert!(job_dir_gone(&root), "temp files removed after success");
        // First call is progress (moves the job to running), with X-Node-Auth.
        assert!(seen[0]
            .path
            .ends_with(&format!("/grid/jobs/{RID}/progress")));
        assert!(seen
            .iter()
            .all(|r| r.headers.get("x-node-auth").map(String::as_str) == Some("node-token")));
        let progress: serde_json::Value = serde_json::from_slice(&seen[0].body).unwrap();
        assert_eq!(progress["stage"], "loading");
        assert_eq!(progress["total"], 20);

        let put = seen
            .iter()
            .find(|r| r.path.ends_with("/video?variant=video"))
            .expect("video upload");
        assert_eq!(put.method, "PUT");
        assert_eq!(put.headers["content-type"], "video/mp4");
        assert_eq!(put.headers["content-length"], put.body.len().to_string());
        let sha = sha256_hex(&put.body);
        assert_eq!(put.headers["x-content-sha256"], sha);
        let info_path = root.join("check.mp4");
        std::fs::write(&info_path, &put.body).unwrap();
        let info = video::inspect_mp4(&info_path).unwrap();
        assert_eq!((info.width, info.height), (832, 480));
        assert!(info.comment.unwrap().contains(&format!("Job {RID}.")));
        let thumb = seen
            .iter()
            .find(|r| r.path.ends_with("/video?variant=thumbnail"))
            .expect("thumbnail");
        assert_eq!(thumb.headers["content-type"], "image/webp");
        assert!(video::is_webp(&thumb.body));

        let complete = seen
            .iter()
            .filter(|r| r.path.ends_with("/complete"))
            .collect::<Vec<_>>();
        assert_eq!(complete.len(), 1, "one completion");
        let body: serde_json::Value = serde_json::from_slice(&complete[0].body).unwrap();
        assert_eq!(body["video"]["sha256"], sha);
        assert_eq!(body["video"]["bytes"], put.body.len());
        assert_eq!(body["video"]["content_type"], "video/mp4");
        assert_eq!(body["video"]["width"], 832);
        assert_eq!(body["video"]["height"], 480);
        assert_eq!(body["video"]["seconds"], 2);
        assert_eq!(body["video"]["frames"], 33);
        assert_eq!(body["video"]["fps"], 16);
        assert_eq!(body["video"]["thumbnail_sha256"], sha256_hex(&thumb.body));
        assert_eq!(body["usage"]["steps"], 20);
        assert_eq!(body["usage"]["gpu"], "NVIDIA GeForce RTX 4090");
        assert_eq!(body["result_envelope_version"], "v1");
        let mut keys: Vec<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "result_envelope_signature",
                "result_envelope_version",
                "usage",
                "video"
            ]
        );

        // The signature verifies exactly as the orchestrator checks it.
        use ed25519_dalek::{Signature, SigningKey, Verifier};
        let message = format!("sgl-result-v1\n{RID}\nvideo\n{sha}");
        let raw: [u8; 64] = bs58::decode(body["result_envelope_signature"].as_str().unwrap())
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap();
        let key = SigningKey::from_bytes(&NODE_SEED).verifying_key();
        assert!(key
            .verify(message.as_bytes(), &Signature::from_bytes(&raw))
            .is_ok());
        assert!(
            key.verify(
                message.replace("\nvideo\n", "\nimage\n").as_bytes(),
                &Signature::from_bytes(&raw)
            )
            .is_err(),
            "the signature is bound to the video domain"
        );
        assert_eq!(
            body["result_envelope_signature"],
            crate::crypto::sign_result_envelope(&NODE_SEED, RID, "video", &put.body)
        );
        assert_eq!(
            crate::node::sign_sealed_result_envelope(&NODE_SEED, "video", RID, &put.body),
            crate::crypto::sign_result_envelope(&NODE_SEED, RID, "video", &put.body)
        );
        // Order: progress, uploads, then complete last.
        let complete_at = seen
            .iter()
            .position(|r| r.path.ends_with("/complete"))
            .unwrap();
        let put_at = seen
            .iter()
            .position(|r| r.path.ends_with("variant=video"))
            .unwrap();
        assert!(put_at < complete_at);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn every_binding_mismatch_is_input_invalid() {
        let prompt = "a red fox";
        let m = model();
        let ok = good_job(prompt);
        let binding = parse_binding(&ok, m).unwrap();
        assert_eq!(unseal_prompt(&ok, &NODE_SEED, &binding).unwrap(), prompt);

        let mutations: Vec<(&str, serde_json::Value)> = vec![
            ("protocol", "video-v2".into()),
            ("request_id", "7f1d2c3b-4a59-4e8f-9a7b-1c2d3e4f5a6b".into()),
            ("request_id", RID.to_ascii_uppercase().into()),
            ("model", "wan2.2-ti2v-5b".into()),
            ("model_sha256", "0".repeat(64).into()),
            ("width", 1280.into()),
            ("height", 832.into()),
            ("seconds", 1.into()),
            ("seconds", 6.into()),
            ("fps", 24.into()),
            ("frames", 32.into()),
            ("steps", 4.into()),
            ("cfg_scale", 6.0.into()),
            ("seed", (u64::from(u32::MAX) + 1).into()),
            ("seed", serde_json::Value::Null),
            ("prompt_bytes", 0.into()),
            ("prompt_bytes", 2001.into()),
            ("prompt_sha256", "A".repeat(64).into()),
            ("deadline_seconds", 30.into()),
            ("deadline_seconds", 7200.into()),
            ("upload_max_bytes", (video::MAX_VIDEO_BYTES + 1).into()),
            ("upload_max_bytes", 0.into()),
            (
                "thumbnail_max_bytes",
                (video::MAX_THUMBNAIL_BYTES + 1).into(),
            ),
            ("unknown_field", 1.into()),
        ];
        for (key, value) in mutations {
            let mut r = reservation(prompt);
            r[key] = value.clone();
            let j = job(sealed(&canonical(prompt)), r);
            assert_eq!(
                parse_binding(&j, m).unwrap_err(),
                INPUT_INVALID,
                "{key}={value}"
            );
        }
        let mut missing = reservation(prompt);
        missing.as_object_mut().unwrap().remove("seed");
        assert!(parse_binding(&job(sealed(&canonical(prompt)), missing), m).is_err());
        let mut wrong_type = ok.clone();
        wrong_type.model = Some("flux2-klein-4b-q4".into());
        assert!(parse_binding(&wrong_type, m).is_err());
        let mut no_model = ok.clone();
        no_model.model = None;
        assert!(
            parse_binding(&no_model, m).is_ok(),
            "the reservation carries the model"
        );
        let mut wrong_id = ok.clone();
        wrong_id.id = "7f1d2c3b-4a59-4e8f-9a7b-1c2d3e4f5a6b".into();
        assert!(parse_binding(&wrong_id, m).is_err());
        let mut extra = ok.clone();
        extra.input_payload.as_mut().unwrap()["image_reservation"] = serde_json::json!({});
        assert!(parse_binding(&extra, m).is_err());

        // Sealed plaintext bindings: prompt hash and length, request id, model, protocol, keys.
        let bind = |plaintext: &str, r: serde_json::Value| {
            let j = job(sealed(plaintext), r);
            let b = parse_binding(&j, m)?;
            unseal_prompt(&j, &NODE_SEED, &b)
        };
        assert_eq!(
            bind(&canonical("a red fix"), reservation(prompt)).unwrap_err(),
            INPUT_INVALID
        );
        assert!(bind(&canonical("a red foxes"), reservation(prompt)).is_err());
        let other_id = canonical(prompt).replace(RID, "7f1d2c3b-4a59-4e8f-9a7b-1c2d3e4f5a6b");
        assert!(bind(&other_id, reservation(prompt)).is_err());
        let other_model = canonical(prompt).replace("wan2.1-t2v-1.3b", "wan2.2-ti2v-5b");
        assert!(bind(&other_model, reservation(prompt)).is_err());
        let other_protocol = canonical(prompt).replace("video-v1", "image-v1");
        assert!(bind(&other_protocol, reservation(prompt)).is_err());
        let extra_key = canonical(prompt).replace(r#""prompt":"#, r#""steps":4,"prompt":"#);
        assert!(bind(&extra_key, reservation(prompt)).is_err());
        assert!(bind(&canonical("a\0b"), reservation("a\0b")).is_err());
        assert!(bind(&canonical("   "), reservation("   ")).is_err());
        // Multi-byte prompts count UTF-8 bytes.
        assert_eq!(
            bind(&canonical("café"), reservation("café")).unwrap(),
            "café"
        );
        // Envelope: v1, extra keys, wrong node key, plaintext dispatch.
        let mut v1 = sealed(&canonical(prompt));
        v1["algorithm"] = "x25519-xchacha20poly1305-v1".into();
        assert!(bind_env(v1, prompt).is_err());
        let mut extra_env = sealed(&canonical(prompt));
        extra_env["embedding_protocol"] = "x".into();
        assert!(bind_env(extra_env, prompt).is_err());
        let j = good_job(prompt);
        let b = parse_binding(&j, m).unwrap();
        assert!(unseal_prompt(&j, &[0x56u8; 32], &b).is_err());
    }

    fn bind_env(enc: serde_json::Value, prompt: &str) -> Result<String, &'static str> {
        let j = job(enc, reservation(prompt));
        let b = parse_binding(&j, model())?;
        unseal_prompt(&j, &NODE_SEED, &b)
    }

    #[tokio::test]
    async fn binding_mismatch_and_floor_are_reported_with_their_codes() {
        let (engine, root) = engine(video::FakeMode::Ok).await;
        let mut bad = reservation("a red fox");
        bad["steps"] = 4.into();
        let seen = run_job(
            &engine,
            orchestrator(true, |_| 200),
            &job(sealed(&canonical("a red fox")), bad),
        )
        .await;
        assert_eq!(fail_reason(&seen).as_deref(), Some(INPUT_INVALID));
        assert!(!seen
            .iter()
            .any(|r| r.path.contains("/video?") || r.path.ends_with("/complete")));

        let lora = "a cat <lora:../../secret:1>";
        let seen = run_job(&engine, orchestrator(true, |_| 200), &good_job(lora)).await;
        assert_eq!(
            fail_reason(&seen).as_deref(),
            Some(INPUT_INVALID),
            "LoRA syntax refused"
        );

        let seen = run_job(
            &engine,
            orchestrator(true, |_| 200),
            &good_job("nsfw photo of a child"),
        )
        .await;
        assert_eq!(fail_reason(&seen).as_deref(), Some(CONTENT_REFUSED));
        assert!(
            !seen.iter().any(|r| r.path.ends_with("/progress")),
            "refused before any work"
        );
        assert!(engine.video().unwrap().is_healthy());
        assert!(job_dir_gone(&root));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn engine_crash_and_bad_output_fail_without_uploading() {
        for mode in [
            video::FakeMode::Crash,
            video::FakeMode::Lie,
            video::FakeMode::NoLabel,
            video::FakeMode::Invalid,
        ] {
            let (engine, root) = engine(mode).await;
            let seen = run_job(&engine, orchestrator(true, |_| 200), &good_job("a red fox")).await;
            assert_eq!(
                fail_reason(&seen).as_deref(),
                Some(RUNTIME_FAILED),
                "{mode:?}"
            );
            assert!(
                !seen.iter().any(|r| r.path.contains("/video?")),
                "{mode:?}: rejected locally before upload"
            );
            assert!(!seen.iter().any(|r| r.path.ends_with("/complete")));
            assert!(job_dir_gone(&root), "{mode:?}: temp files removed");
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[tokio::test]
    async fn hang_past_the_deadline_is_killed_and_failed() {
        let (engine, root) = engine(video::FakeMode::Hang).await;
        let (url, seen) = mock(orchestrator(true, |_| 200)).await;
        let client = OrchestratorClient::new(&url, Some("t".into()));
        // deadline 300 s, upload reserve 75 s: pretend the job arrived 223 s ago.
        let received = Instant::now() - Duration::from_secs(223);
        let started = Instant::now();
        let telemetry = crate::telemetry::Telemetry::new();
        process_with(
            &client,
            &engine,
            &good_job("a red fox"),
            &telemetry,
            received,
        )
        .await;
        assert!(started.elapsed() < Duration::from_secs(10));
        let seen = seen.lock().unwrap().clone();
        assert_eq!(fail_reason(&seen).as_deref(), Some(RUNTIME_FAILED));
        assert!(job_dir_gone(&root));
        assert!(!engine.video().unwrap().is_healthy());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn continue_false_kills_the_engine_and_reports_nothing() {
        let (engine, root) = engine(video::FakeMode::Hang).await;
        let handler: Handler = Arc::new(|r: &Seen, nth: usize| {
            if r.path.ends_with("/progress") {
                // First report: keep going; the next one says the job was cancelled.
                (200, format!(r#"{{"continue":{}}}"#, nth == 0))
            } else {
                (200, "{}".into())
            }
        });
        let started = Instant::now();
        let seen = run_job(&engine, handler, &good_job("a red fox")).await;
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "killed at once, not at the deadline"
        );
        assert_eq!(
            fail_reason(&seen),
            None,
            "a stopped job is not reported as failed"
        );
        assert!(!seen
            .iter()
            .any(|r| r.path.contains("/video?") || r.path.ends_with("/complete")));
        assert!(job_dir_gone(&root));
        assert!(
            engine.video().unwrap().is_healthy(),
            "a cancel is not a fault"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn progress_errors_keep_going_only_gone_statuses_stop() {
        let (engine, root) = engine(video::FakeMode::Ok).await;
        for status in [401u16, 403, 429, 500, 503] {
            let handler: Handler = Arc::new(move |r: &Seen, _| {
                if r.path.ends_with("/progress") {
                    (status, r#"{"error":"nope"}"#.into())
                } else {
                    (200, "{}".into())
                }
            });
            let seen = run_job(&engine, handler, &good_job("a red fox")).await;
            assert_eq!(fail_reason(&seen), None, "{status}");
            assert_eq!(
                seen.iter()
                    .filter(|r| r.path.ends_with("/complete"))
                    .count(),
                1,
                "{status}: the job keeps going and completes"
            );
        }
        let (hang, hang_root) = engine_hang().await;
        for status in [404u16, 409, 410] {
            let handler: Handler = Arc::new(move |r: &Seen, _| {
                if r.path.ends_with("/progress") {
                    (status, "{}".into())
                } else {
                    (200, "{}".into())
                }
            });
            let started = Instant::now();
            let seen = run_job(&hang, handler, &good_job("a red fox")).await;
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "{status}: stopped at once"
            );
            assert_eq!(fail_reason(&seen), None, "{status}");
            assert!(!seen
                .iter()
                .any(|r| r.path.contains("/video?") || r.path.ends_with("/complete")));
        }
        assert!(crate::orchestrator::progress_status_stops(410));
        assert!(!crate::orchestrator::progress_status_stops(401));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(hang_root);
    }

    async fn engine_hang() -> (Arc<InferenceEngine>, std::path::PathBuf) {
        engine(video::FakeMode::Hang).await
    }

    #[tokio::test]
    async fn upload_4xx_is_not_retried_and_5xx_is() {
        let (engine, root) = engine(video::FakeMode::Ok).await;
        let seen = run_job(&engine, orchestrator(true, |_| 400), &good_job("a red fox")).await;
        let puts = seen
            .iter()
            .filter(|r| r.path.ends_with("variant=video"))
            .count();
        assert_eq!(puts, 1, "a 4xx is final");
        assert_eq!(fail_reason(&seen).as_deref(), Some(RUNTIME_FAILED));
        assert!(!seen.iter().any(|r| r.path.ends_with("/complete")));

        let seen = run_job(
            &engine,
            orchestrator(true, |nth| if nth == 0 { 503 } else { 200 }),
            &good_job("a red fox"),
        )
        .await;
        let puts: Vec<&Seen> = seen
            .iter()
            .filter(|r| r.path.ends_with("variant=video"))
            .collect();
        assert_eq!(puts.len(), 2, "a 5xx is retried");
        assert_eq!(puts[0].body, puts[1].body, "the same bytes again");
        assert_eq!(fail_reason(&seen), None);
        assert_eq!(
            seen.iter()
                .filter(|r| r.path.ends_with("/complete"))
                .count(),
            1
        );
        assert!(job_dir_gone(&root));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn completion_5xx_is_retried_once_landed_it_stops() {
        let (engine, root) = engine(video::FakeMode::Ok).await;
        let handler: Handler = Arc::new(|r: &Seen, nth: usize| {
            if r.path.ends_with("/progress") {
                (200, r#"{"continue":true}"#.into())
            } else if r.path.ends_with("/complete") && nth == 0 {
                (502, "{}".into())
            } else {
                (200, "{}".into())
            }
        });
        let seen = run_job(&engine, handler, &good_job("a red fox")).await;
        let completes: Vec<&Seen> = seen
            .iter()
            .filter(|r| r.path.ends_with("/complete"))
            .collect();
        assert_eq!(completes.len(), 2);
        assert_eq!(
            completes[0].body, completes[1].body,
            "an identical, idempotent retry"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn no_video_engine_is_a_runtime_failure() {
        let (url, seen) = mock(orchestrator(true, |_| 200)).await;
        let client = OrchestratorClient::new(&url, Some("t".into()));
        let telemetry = crate::telemetry::Telemetry::new();
        process(
            &client,
            &None,
            &good_job("a red fox"),
            &NODE_SEED,
            &telemetry,
        )
        .await;
        assert_eq!(
            fail_reason(&seen.lock().unwrap()).as_deref(),
            Some(RUNTIME_FAILED)
        );
    }

    #[tokio::test]
    async fn heartbeat_parts_follow_engine_health_and_busy_state() {
        let (engine, root) = engine(video::FakeMode::Crash).await;
        let models = vec![model().id.to_string()];
        let active = AtomicU32::new(0);
        let (available, cap) = rest_snapshot(Some(&engine), &models, &active).unwrap();
        assert_eq!(available, models);
        assert_eq!(cap.as_ref().unwrap().free_slots, 1);
        active.store(1, Ordering::Relaxed);
        let (_, cap) = rest_snapshot(Some(&engine), &models, &active).unwrap();
        assert_eq!(cap.unwrap().free_slots, 0);
        let wire = crate::orchestrator::video_capability_wire(
            rest_snapshot(Some(&engine), &models, &active).unwrap().1,
        );
        assert_eq!(wire["kind"], "video");
        assert_eq!(wire["engine"], "video-worker");
        assert_eq!(wire["video"]["protocol"], "video-v1");
        assert_eq!(wire["video"]["gpu"]["vram_mb"], 24564);
        assert!(wire.get("image").is_none());
        let beat = media_ws_heartbeat(
            &engine,
            models.clone(),
            "pk".into(),
            Some("sig".into()),
            Some(1),
            0.5,
            Arc::new(AtomicU32::new(0)),
        )
        .unwrap()();
        assert_eq!(beat["capabilities"]["video"]["free_slots"], 1);
        assert_eq!(beat["available_models"], serde_json::json!(models));
        // A crashed job de-advertises at once: no capability, no model.
        let _ = run_job(&engine, orchestrator(true, |_| 200), &good_job("a red fox")).await;
        let (available, cap) = rest_snapshot(Some(&engine), &models, &AtomicU32::new(0)).unwrap();
        assert!(available.is_empty() && cap.is_none());
        let gone = crate::orchestrator::video_capability_wire(None);
        assert!(gone.get("video").is_none() && gone.get("kind").is_none());
        assert_eq!(dedicated_kind("video-worker"), Some("video"));
        assert_eq!(dedicated_kind(crate::image::ENGINE_LABEL), Some("image"));
        assert_eq!(dedicated_kind("inprocess"), None);
        assert!(is_dedicated_worker("embedding-worker") && !is_dedicated_worker("server"));
        assert_eq!(worker_runtime("video-worker"), Some("sdcpp"));
        assert_eq!(worker_runtime("stt-worker"), Some(crate::stt::RUNTIME));
        assert_eq!(worker_runtime("inprocess"), None);
        assert_eq!(media_wedge_ms("wan2.1-t2v-1.3b"), video::WEDGE_MS);
        assert_eq!(media_wedge_ms("llama-3.2-3b"), 0);
        let _ = std::fs::remove_dir_all(root);
    }
}
