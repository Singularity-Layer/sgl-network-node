//! Dedicated EmbeddingGemma 2 process supervisor. No llama.cpp or network fallback.
//! A single owned worker is serialized, framed, timed out, killed and reaped on failure.
use crate::embed_catalog::InputType;
use crate::embedding_input::{EmbeddingBatch, MAX_ENCODED_BODY};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

pub const MODEL_ID: &str = "embeddinggemma-2";
pub const OFFICIAL_MODEL: &str = "google/embeddinggemma-2";
pub const OFFICIAL_REVISION: &str = "914f7f89142e33e77833254d9c9b90c3cef7303b";
pub const MLX_MODEL: &str = "mlx-community/embeddinggemma-2-bf16";
pub const MLX_MODEL_REVISION: &str = "1a4ffddb7905d3f63486748deabe091a01fb6201";
pub const MLX_VLM_REVISION: &str = "30f177f03cbcb42bc2f65496458de79f51b80c28";
pub const MLX_VERSION: &str = "0.32.3";
pub const TRANSFORMERS_VERSION: &str = "5.19.0";
pub const SENTENCE_TRANSFORMERS_VERSION: &str = "6.1.0";
pub const PROTOCOL: &str = "embedding-multimodal-v1";
pub const DIMENSIONS: &[u32] = &[768, 512, 256, 128];
const MAX_OUTPUT_FRAME: u64 = 1024 * 1024;

#[derive(Clone)]
pub struct WorkerConfig {
    pub program: String,
    pub args: Vec<String>,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
}

#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub embedding_dim: u32,
    pub embedding_ready: bool,
    pub input_envelope_encodings: Vec<String>,
    pub embedding_protocol: String,
    pub embedding_modalities: Vec<String>,
    pub embedding_dimensions: Vec<u32>,
    pub embedding_runtime: String,
    pub processor_revision: String,
    pub media_transport: Vec<String>,
}

#[derive(Default, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Usage {
    pub text: u32,
    pub image: u32,
    pub audio: u32,
    pub video: u32,
}
impl Usage {
    pub fn total(&self) -> Result<u32, String> {
        [self.text, self.image, self.audio, self.video]
            .iter()
            .try_fold(0u32, |sum, n| {
                sum.checked_add(*n)
                    .ok_or_else(|| "invalid embedding usage".into())
            })
    }
}

pub struct Output {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Usage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    #[serde(rename = "type")]
    frame_type: String,
    protocol: String,
    runtime: String,
    processor_revision: String,
    model_revision: String,
    modalities: Vec<String>,
    dimensions: Vec<u32>,
    smoke_vectors: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    #[serde(rename = "type")]
    frame_type: String,
    request_id: u64,
    vectors: Vec<Vec<f32>>,
    usage: Usage,
}

struct Job {
    id: u64,
    batch: EmbeddingBatch,
    input_type: InputType,
    dimensions: u32,
    reply: tokio::sync::oneshot::Sender<Result<Output, String>>,
}

pub struct Eg2Engine {
    tx: Mutex<Option<mpsc::SyncSender<Job>>>,
    config: WorkerConfig,
    restart_lock: tokio::sync::Mutex<()>,
    restarts: AtomicU64,
    assets: Option<PrivateWorkerTemp>,
    healthy: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    next_id: AtomicU64,
    capabilities: Capabilities,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Eg2Engine {
    /// Start one explicitly configured worker; no capability exists until smoke validation.
    pub async fn start(config: WorkerConfig) -> Result<Self, String> {
        let saved_config = config.clone();
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let healthy = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let (h, stop) = (healthy.clone(), stopped.clone());
        let handle = std::thread::Builder::new()
            .name("eg2-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, h, stop))
            .map_err(|_| "could not start embedding supervisor")?;
        let capabilities = match ready_rx.await {
            Ok(Ok(ready)) => ready,
            _ => {
                stopped.store(true, Ordering::Release);
                let _ = handle.join();
                return Err("embedding worker startup failed".into());
            }
        };
        Ok(Self {
            tx: Mutex::new(Some(tx)),
            config: saved_config,
            restart_lock: tokio::sync::Mutex::new(()),
            restarts: AtomicU64::new(0),
            assets: None,
            healthy,
            stopped,
            next_id: AtomicU64::new(1),
            capabilities,
            worker: Mutex::new(Some(handle)),
        })
    }

    /// Explicit model selection is required. Materialize the exact M4 canary worker and
    /// fixture bytes in owned storage; model assets and the dedicated venv stay read-only.
    pub async fn production(
        model_path: &std::path::Path,
        python: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let python = python
            .ok_or("EmbeddingGemma 2 requires --embedding-python for its dedicated runtime")?;
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err("EmbeddingGemma 2 candidate runtime requires Apple Silicon macOS".into());
        }
        if !model_path.is_dir() || !python.is_file() {
            return Err("EmbeddingGemma 2 model snapshot or interpreter unavailable".into());
        }
        let assets = materialize_assets()?;
        let config = WorkerConfig {
            program: python
                .to_str()
                .ok_or("embedding interpreter path is not UTF-8")?
                .into(),
            args: vec![
                "-u".into(),
                assets
                    .0
                    .join("embeddinggemma_worker.py")
                    .to_str()
                    .ok_or("embedding runtime path is not UTF-8")?
                    .into(),
                "--model-path".into(),
                model_path
                    .to_str()
                    .ok_or("embedding model path is not UTF-8")?
                    .into(),
                "--smoke-dir".into(),
                assets
                    .0
                    .join("smoke")
                    .to_str()
                    .ok_or("embedding smoke path is not UTF-8")?
                    .into(),
            ],
            startup_timeout: Duration::from_secs(180),
            request_timeout: Duration::from_secs(120),
        };
        let mut engine = Self::start(config).await?;
        engine.assets = Some(assets);
        Ok(engine)
    }

    /// Lifetime budget avoids crash loops. A successful relaunch repeats all startup smokes.
    pub async fn restart(&self) -> Result<(), String> {
        let _guard = self.restart_lock.lock().await;
        if self.restarts.fetch_add(1, Ordering::AcqRel) >= 3 {
            return Err("embedding worker restart budget exhausted".into());
        }
        self.stop();
        self.stopped.store(false, Ordering::Release);
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (healthy, stopped) = (self.healthy.clone(), self.stopped.clone());
        let config = self.config.clone();
        let handle = std::thread::Builder::new()
            .name("eg2-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, healthy, stopped))
            .map_err(|_| "could not restart embedding supervisor")?;
        *self.worker.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
        match ready_rx.await {
            Ok(Ok(_)) if self.is_healthy() => {
                *self.tx.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
                Ok(())
            }
            _ => {
                self.stop();
                Err("embedding worker restart failed".into())
            }
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && !self.stopped.load(Ordering::Acquire)
    }
    pub fn capabilities(&self) -> Option<Capabilities> {
        if self.is_healthy() {
            Some(self.capabilities.clone())
        } else {
            None
        }
    }

    pub async fn embed(
        &self,
        batch: EmbeddingBatch,
        input_type: InputType,
        dimensions: Option<u32>,
    ) -> Result<Output, String> {
        if !self.is_healthy() {
            return Err("embedding worker unavailable".into());
        }
        let dim = dimensions.unwrap_or(768);
        if !DIMENSIONS.contains(&dim) {
            return Err("unsupported embedding dimensions".into());
        }
        // Public typed structures can be built directly: validate again at the process boundary.
        let input = serde_json::to_value(&batch.items).map_err(|_| "invalid embedding input")?;
        let batch = EmbeddingBatch::parse(&input, true)?;
        let (reply, receive) = tokio::sync::oneshot::channel();
        let job = Job {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            batch,
            input_type,
            dimensions: dim,
            reply,
        };
        self.tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .ok_or("embedding worker unavailable")?
            .try_send(job)
            .map_err(|_| "embedding worker busy or unavailable")?;
        receive.await.map_err(|_| "embedding worker unavailable")?
    }

    pub fn stop(&self) {
        self.healthy.store(false, Ordering::Release);
        self.stopped.store(true, Ordering::Release);
        self.tx.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(handle) = self.worker.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = handle.join();
        }
    }
}
impl Drop for Eg2Engine {
    fn drop(&mut self) {
        self.stop();
    }
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct PrivateWorkerTemp(std::path::PathBuf);
impl PrivateWorkerTemp {
    fn create() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "sgl-eg2-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).map_err(|_| "embedding temporary storage unavailable")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).is_err() {
                let _ = std::fs::remove_dir(&path);
                return Err("embedding temporary storage permissions failed".into());
            }
        }
        Ok(Self(path))
    }
}
impl Drop for PrivateWorkerTemp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn materialize_assets() -> Result<PrivateWorkerTemp, String> {
    let assets = PrivateWorkerTemp::create()?;
    let smoke = assets.0.join("smoke");
    std::fs::create_dir(&smoke).map_err(|_| "embedding smoke storage unavailable")?;
    for (name, bytes) in [
        (
            "embeddinggemma_worker.py",
            include_bytes!("../scripts/embeddinggemma_worker.py").as_slice(),
        ),
        (
            "embeddinggemma_model_files.json",
            include_bytes!("../scripts/embeddinggemma_model_files.json").as_slice(),
        ),
        (
            "smoke/image.png",
            include_bytes!("../assets/embeddinggemma2/smoke/image.png").as_slice(),
        ),
        (
            "smoke/audio.wav",
            include_bytes!("../assets/embeddinggemma2/smoke/audio.wav").as_slice(),
        ),
        (
            "smoke/video.mp4",
            include_bytes!("../assets/embeddinggemma2/smoke/video.mp4").as_slice(),
        ),
        (
            "smoke/durations.json",
            include_bytes!("../assets/embeddinggemma2/smoke/durations.json").as_slice(),
        ),
    ] {
        std::fs::write(assets.0.join(name), bytes)
            .map_err(|_| "embedding runtime materialization failed")?;
    }
    Ok(assets)
}

fn supervise(
    config: WorkerConfig,
    jobs: mpsc::Receiver<Job>,
    ready: tokio::sync::oneshot::Sender<Result<Capabilities, String>>,
    healthy: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
) {
    let temporary = match PrivateWorkerTemp::create() {
        Ok(tmp) => tmp,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let result = Command::new(&config.program)
        .args(&config.args)
        .env("TMPDIR", &temporary.0)
        .env("TEMP", &temporary.0)
        .env("TMP", &temporary.0)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match result {
        Ok(child) => OwnedChild(child),
        Err(_) => {
            let _ = ready.send(Err("embedding worker spawn failed".into()));
            return;
        }
    };
    let mut stdin = child.0.stdin.take().unwrap();
    let (writes_tx, writes_rx) = mpsc::sync_channel::<(Vec<u8>, mpsc::SyncSender<bool>)>(1);
    let writer = std::thread::spawn(move || {
        while let Ok((bytes, ack)) = writes_rx.recv() {
            let success = stdin.write_all(&bytes).and_then(|_| stdin.flush()).is_ok();
            let _ = ack.try_send(success);
            if !success {
                break;
            }
        }
    });
    let stdout = child.0.stdout.take().unwrap();
    let (frames_tx, frames_rx) = mpsc::sync_channel(2);
    let h = healthy.clone();
    let output_open = Arc::new(AtomicBool::new(true));
    let open = output_open.clone();
    let reader = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut frame = Vec::new();
            let result = reader
                .by_ref()
                .take(MAX_OUTPUT_FRAME + 1)
                .read_until(b'\n', &mut frame);
            if !matches!(result, Ok(n) if n > 0 && n as u64 <= MAX_OUTPUT_FRAME)
                || frame.last() != Some(&b'\n')
            {
                h.store(false, Ordering::Release);
                break;
            }
            let valid_frame = serde_json::from_slice::<serde_json::Value>(&frame)
                .ok()
                .and_then(|v| {
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| matches!(t, "ready" | "result"))
                })
                .unwrap_or(false);
            if !valid_frame {
                h.store(false, Ordering::Release);
                break;
            }
            if frames_tx.try_send(frame).is_err() {
                h.store(false, Ordering::Release);
                break;
            }
        }
        open.store(false, Ordering::Release);
    });
    let startup = receive_frame(&frames_rx, config.startup_timeout, &stopped, None)
        .and_then(|frame| validate_ready(&frame));
    match startup {
        Ok(cap)
            if output_open.load(Ordering::Acquire)
                && !stopped.load(Ordering::Acquire)
                && child.0.try_wait().ok().flatten().is_none() =>
        {
            healthy.store(true, Ordering::Release);
            if ready.send(Ok(cap)).is_err() {
                stopped.store(true, Ordering::Release);
            }
        }
        _ => {
            let _ = ready.send(Err("invalid embedding readiness".into()));
            stopped.store(true, Ordering::Release);
        }
    }
    while output_open.load(Ordering::Acquire)
        && healthy.load(Ordering::Acquire)
        && !stopped.load(Ordering::Acquire)
    {
        let job = match jobs.recv_timeout(Duration::from_millis(25)) {
            Ok(job) => job,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if frames_rx.try_recv().is_ok() || !matches!(child.0.try_wait(), Ok(None)) {
                    healthy.store(false, Ordering::Release);
                }
                continue;
            }
            Err(_) => break,
        };
        if job.reply.is_closed() {
            continue;
        }
        let response = run_job(
            &job,
            &writes_tx,
            &frames_rx,
            &stopped,
            config.request_timeout,
        );
        if response.is_err() {
            healthy.store(false, Ordering::Release);
        }
        let _ = job.reply.send(response);
    }
    healthy.store(false, Ordering::Release);
    // Close the pipe, kill and reap before joining its EOF reader. No orphan worker.
    drop(writes_tx);
    drop(child);
    let _ = reader.join();
    let _ = writer.join();
}

fn receive_frame(
    frames: &mpsc::Receiver<Vec<u8>>,
    timeout: Duration,
    stopped: &AtomicBool,
    reply: Option<&tokio::sync::oneshot::Sender<Result<Output, String>>>,
) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if stopped.load(Ordering::Acquire) || reply.is_some_and(|reply| reply.is_closed()) {
            return Err("embedding request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("embedding worker timed out".into());
        }
        match frames.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(frame) => return Ok(frame),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => return Err("embedding worker closed output".into()),
        }
    }
}

fn validate_ready(frame: &[u8]) -> Result<Capabilities, String> {
    let ready: Ready = serde_json::from_slice(frame).map_err(|_| "invalid embedding readiness")?;
    if ready.frame_type != "ready"
        || ready.protocol != PROTOCOL
        || ready.runtime != "mlx-vlm"
        || ready.processor_revision != MLX_VLM_REVISION
        || ready.model_revision != MLX_MODEL_REVISION
        || ready.modalities != ["text", "image", "audio", "video"]
        || ready.dimensions != DIMENSIONS
        || ready.smoke_vectors.len() != 5
    {
        return Err("embedding worker identity or smoke mismatch".into());
    }
    for mut vector in ready.smoke_vectors {
        normalize_native(&mut vector, 768)?;
    }
    Ok(Capabilities {
        embedding_dim: 768,
        embedding_ready: true,
        input_envelope_encodings: vec!["base64".into()],
        embedding_protocol: PROTOCOL.into(),
        embedding_modalities: ready.modalities,
        embedding_dimensions: ready.dimensions,
        embedding_runtime: ready.runtime,
        processor_revision: ready.processor_revision,
        media_transport: vec!["inline-base64".into()],
    })
}

fn run_job(
    job: &Job,
    writes: &mpsc::SyncSender<(Vec<u8>, mpsc::SyncSender<bool>)>,
    frames: &mpsc::Receiver<Vec<u8>>,
    stopped: &AtomicBool,
    timeout: Duration,
) -> Result<Output, String> {
    let input_type = match job.input_type {
        InputType::Query => "query",
        InputType::Document => "document",
        InputType::Unspecified => "unspecified",
    };
    let request = serde_json::json!({"type":"embed","request_id":job.id,"protocol":PROTOCOL,"input":job.batch.items,"input_type":input_type,"dimensions":768});
    let mut bytes = serde_json::to_vec(&request).map_err(|_| "invalid embedding request")?;
    if bytes.len() > MAX_ENCODED_BODY {
        return Err("embedding request exceeds frame limit".into());
    }
    bytes.push(b'\n');
    let deadline = Instant::now() + timeout;
    let (ack_tx, ack_rx) = mpsc::sync_channel(1);
    writes
        .try_send((bytes, ack_tx))
        .map_err(|_| "embedding worker input unavailable")?;
    loop {
        if stopped.load(Ordering::Acquire) || job.reply.is_closed() {
            return Err("embedding request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("embedding worker input timed out".into());
        }
        match ack_rx.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(true) => break,
            Ok(false) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("embedding worker input closed".into())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
        }
    }
    let frame = receive_frame(
        frames,
        deadline.saturating_duration_since(Instant::now()),
        stopped,
        Some(&job.reply),
    )?;
    let mut result: Response =
        serde_json::from_slice(&frame).map_err(|_| "invalid embedding worker output")?;
    if result.frame_type != "result"
        || result.request_id != job.id
        || result.vectors.len() != job.batch.items.len()
    {
        return Err("embedding worker response mismatch".into());
    }
    let total = result.usage.total()?;
    if total == 0 || total > 8192 * job.batch.items.len() as u32 {
        return Err("embedding worker usage exceeds context budget".into());
    }
    let modalities = job.batch.modalities();
    for (kind, tokens) in [
        ("text", result.usage.text),
        ("image", result.usage.image),
        ("audio", result.usage.audio),
        ("video", result.usage.video),
    ] {
        if kind != "text"
            && ((!modalities.contains(&kind) && tokens != 0)
                || (modalities.contains(&kind) && tokens == 0))
        {
            return Err("embedding worker usage modality mismatch".into());
        }
    }
    for vector in &mut result.vectors {
        normalize_native(vector, job.dimensions)?;
    }
    Ok(Output {
        vectors: result.vectors,
        usage: result.usage,
    })
}

fn normalize_native(vector: &mut Vec<f32>, dimensions: u32) -> Result<(), String> {
    if vector.len() != 768 || vector.iter().any(|v| !v.is_finite()) {
        return Err("invalid embedding vector".into());
    }
    let native_norm = vector.iter().map(|v| v * v).sum::<f32>();
    if !native_norm.is_finite() || (native_norm - 1.0).abs() > 0.001 {
        return Err("native embedding is not unit normalized".into());
    }
    vector.truncate(dimensions as usize);
    // Scale first to avoid overflow/underflow while accumulating and normalizing in FP32.
    let scale = vector.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    if scale == 0.0 {
        return Err("zero embedding vector".into());
    }
    for value in vector.iter_mut() {
        *value /= scale;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    for value in vector.iter_mut() {
        *value /= norm;
    }
    if vector.iter().any(|v| !v.is_finite())
        || (vector.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() > 0.001
    {
        return Err("embedding normalization failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_mrl_dimension_is_unit_and_bad_native_outputs_fail() {
        for dim in DIMENSIONS {
            let mut vector = vec![1.0 / (768.0f32).sqrt(); 768];
            normalize_native(&mut vector, *dim).unwrap();
            assert_eq!(vector.len(), *dim as usize);
            assert!((vector.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 0.001);
        }
        for mut vector in [
            vec![0.0; 768],
            vec![1.0; 768],
            vec![f32::NAN; 768],
            vec![1.0; 767],
        ] {
            assert!(normalize_native(&mut vector, 128).is_err());
        }
        let mut trailing = vec![0.0; 768];
        trailing[767] = 1.0;
        assert!(normalize_native(&mut trailing, 128).is_err());
    }
    #[test]
    fn usage_overflow_is_rejected() {
        assert!(Usage {
            text: u32::MAX,
            image: 1,
            audio: 0,
            video: 0
        }
        .total()
        .is_err());
    }
}
