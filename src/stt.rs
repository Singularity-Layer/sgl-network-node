//! Dedicated whisper.cpp transcription process supervisor. No llama.cpp or network fallback.
//! A single owned worker is serialized, framed, timed out, killed and reaped on failure.
//! Audio is uncompressed mono 16 kHz signed 16-bit PCM; sample count is the duration authority.
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

pub const MODEL_ID: &str = "whisper-1";
pub const MODEL_FILE: &str = "ggml-small.bin";
pub const MODEL_REPOSITORY: &str = "ggerganov/whisper.cpp";
pub const MODEL_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
pub const MODEL_SHA256: &str = "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b";
pub const MODEL_BYTES: u64 = 487_601_967;
pub const RUNTIME: &str = "whisper.cpp";
pub const RUNTIME_REVISION: &str = "d1be6fde11ac6e0407606b4e42fe72d34add8037";
pub const RUNTIME_TAG: &str = "v1.9.5";
/// Approved founder-M4 artifact built from RUNTIME_REVISION. Other platforms stay
/// gated until reproducible hashes and real runtime canaries are reviewed.
pub const MACOS_AARCH64_RUNTIME_SHA256: &str =
    "4ac1f78373fa19037ff036c66586db2b775e0b21785c320953a1960bcb425405";

pub const MACOS_AARCH64_RUNTIME_BYTES: u64 = 7_760_160;

fn approved_runtime_sha256() -> Option<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some(MACOS_AARCH64_RUNTIME_SHA256)
    } else {
        None
    }
}
pub const PROTOCOL: &str = "transcription-v1";
pub const SAMPLE_RATE: u32 = 16_000;
pub const CHANNELS: u16 = 1;
pub const BITS: u16 = 16;
pub const MAX_DURATION_SECONDS: u32 = 60;
pub const MAX_SAMPLES: u32 = MAX_DURATION_SECONDS * SAMPLE_RATE;
/// 60 s of PCM = 1,920,000 bytes; base64 inflates by 4/3 plus JSON framing overhead.
pub const MAX_INPUT_FRAME: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_FRAME: u64 = 1024 * 1024;
pub const MAX_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_SEGMENTS: usize = 256;
const MAX_PCM_BASE64_BYTES: usize = (MAX_SAMPLES as usize * 2).div_ceil(3) * 4;
const REPLAY_CACHE_SIZE: usize = 65_536;
const REPLAY_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);
const SUPPORTED_LANGUAGES: &[&str] = &[
    "en", "fr", "de", "es", "it", "pt", "nl", "hi", "ja", "ko", "zh", "ru", "ar",
];

#[derive(Clone)]
pub struct WorkerConfig {
    pub program: String,
    pub args: Vec<String>,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    pub expected_runtime_sha256: String,
    pub expected_runtime_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct KeyBinding {
    pub encryption_public_key: String,
    pub encryption_public_key_signature: String,
    pub key_version: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub transcription_ready: bool,
    pub transcription_protocol: String,
    pub transcription_models: Vec<String>,
    pub transcription_runtime: String,
    pub runtime_revision: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub model_repository: String,
    pub model_bytes: u64,
    pub runtime_binary_sha256: String,
    pub runtime_binary_bytes: u64,
    pub os: String,
    pub architecture: String,
    pub model_license: String,
    pub runtime_license: String,
    pub audio_format: String,
    pub max_duration_seconds: u32,
    pub languages: Vec<String>,
    pub input_envelope_encodings: Vec<String>,
    pub media_transport: Vec<String>,
    pub transcription_streaming: bool,
    pub free_slots: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_binding: Option<KeyBinding>,
}

impl Capabilities {
    pub fn bind_heartbeat(
        mut self,
        free_slots: u32,
        encryption_public_key: &str,
        encryption_public_key_signature: Option<&str>,
        key_version: Option<u32>,
    ) -> Self {
        self.free_slots = free_slots.min(1);
        self.key_binding =
            encryption_public_key_signature
                .zip(key_version)
                .map(|(signature, version)| KeyBinding {
                    encryption_public_key: encryption_public_key.to_string(),
                    encryption_public_key_signature: signature.to_string(),
                    key_version: version,
                });
        if self.key_binding.is_none() {
            self.transcription_ready = false;
            self.free_slots = 0;
        }
        self
    }
}

#[derive(Default, Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub duration_seconds: f64,
    pub segments: Vec<Segment>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    #[serde(rename = "type")]
    frame_type: String,
    protocol: String,
    runtime: String,
    runtime_revision: String,
    model_revision: String,
    model_sha256: String,
    model_bytes: u64,
    runtime_binary_sha256: String,
    runtime_binary_bytes: u64,
    model_id: String,
    audio_format: String,
    max_duration_seconds: u32,
    smoke_transcript: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    #[serde(rename = "type")]
    frame_type: String,
    request_id: u64,
    text: String,
    language: Option<String>,
    duration_seconds: f64,
    segments: Vec<Segment>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestError {
    #[serde(rename = "type")]
    frame_type: String,
    request_id: u64,
    code: String,
}

enum WorkerFailure {
    InvalidInput,
    Fatal(String),
}
impl From<String> for WorkerFailure {
    fn from(reason: String) -> Self {
        Self::Fatal(reason)
    }
}
impl From<&str> for WorkerFailure {
    fn from(reason: &str) -> Self {
        Self::Fatal(reason.into())
    }
}

pub struct ReadinessSnapshot {
    pub available_models: Vec<String>,
    pub capabilities: Option<Capabilities>,
}

/// Bounded PCM audio request body validated at the process boundary.
#[derive(Debug, Clone)]
pub struct AudioRequest {
    pub pcm: Vec<u8>,
    pub sample_count: u32,
    pub language: Option<String>,
    pub request_id: String,
    pub model: String,
    pub model_revision: String,
    pub model_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalAudio {
    format: String,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    sample_count: u64,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalRequest {
    protocol: String,
    request_id: String,
    model: String,
    model_revision: String,
    model_sha256: String,
    language: String,
    audio: CanonicalAudio,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReservationBinding {
    pub protocol: String,
    pub request_id: String,
    pub model: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub language: String,
    pub sample_count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TranscriptionDispatch {
    enc: serde_json::Value,
    transcription_reservation: ReservationBinding,
}

impl ReservationBinding {
    pub fn parse_dispatch(value: &serde_json::Value) -> Result<Self, String> {
        let dispatch: TranscriptionDispatch = serde_json::from_value(value.clone())
            .map_err(|_| "transcription dispatch shape invalid".to_string())?;
        if !dispatch.enc.is_object() {
            return Err("transcription encrypted envelope missing".into());
        }
        let binding = dispatch.transcription_reservation;
        if binding.protocol != PROTOCOL
            || !canonical_uuid(&binding.request_id)
            || binding.model != MODEL_ID
            || binding.model_revision != MODEL_REVISION
            || binding.model_sha256 != MODEL_SHA256
            || binding.sample_count == 0
            || binding.sample_count > MAX_SAMPLES as u64
            || !valid_language(&binding.language)
        {
            return Err("transcription reservation binding invalid".into());
        }
        Ok(binding)
    }

    pub fn matches(&self, audio: &AudioRequest) -> bool {
        self.protocol == PROTOCOL
            && self.request_id == audio.request_id
            && self.model == audio.model
            && self.model_revision == audio.model_revision
            && self.model_sha256 == audio.model_sha256
            && self.sample_count == audio.sample_count as u64
            && match (&audio.language, self.language.as_str()) {
                (None, "auto") => true,
                (Some(actual), expected) => actual == expected,
                _ => false,
            }
    }
}

impl AudioRequest {
    pub fn parse(value: &serde_json::Value) -> Result<Self, String> {
        let request: CanonicalRequest = serde_json::from_value(value.clone())
            .map_err(|_| "transcription request shape invalid".to_string())?;
        if request.protocol != PROTOCOL {
            return Err("transcription protocol mismatch".into());
        }
        if !canonical_uuid(&request.request_id) {
            return Err("transcription request id invalid".into());
        }
        if request.model != MODEL_ID
            || request.model_revision != MODEL_REVISION
            || request.model_sha256 != MODEL_SHA256
        {
            return Err("transcription model binding mismatch".into());
        }
        if request.audio.format != "pcm_s16le" {
            return Err("transcription audio format must be pcm_s16le".into());
        }
        if request.audio.sample_rate != SAMPLE_RATE {
            return Err("transcription audio must be 16 kHz".into());
        }
        if request.audio.channels != CHANNELS {
            return Err("transcription audio must be mono".into());
        }
        if request.audio.bits_per_sample != BITS {
            return Err("transcription audio must be signed 16-bit".into());
        }
        let sample_count = request.audio.sample_count;
        if sample_count == 0 || sample_count > MAX_SAMPLES as u64 {
            return Err("transcription audio duration out of bounds".into());
        }
        if request.audio.data.len() > MAX_PCM_BASE64_BYTES {
            return Err("transcription audio base64 exceeds bound".into());
        }
        use base64::Engine;
        let pcm = base64::engine::general_purpose::STANDARD
            .decode(request.audio.data)
            .map_err(|_| "transcription audio base64 invalid".to_string())?;
        if pcm.len() % 2 != 0 || pcm.len() as u64 != sample_count * 2 {
            return Err("transcription audio length contradicts sample count".into());
        }
        let language = match request.language.as_str() {
            "auto" => None,
            code if SUPPORTED_LANGUAGES.contains(&code) => Some(code.to_string()),
            _ => return Err("transcription language unsupported".into()),
        };
        Ok(Self {
            pcm,
            sample_count: sample_count as u32,
            language,
            request_id: request.request_id,
            model: request.model,
            model_revision: request.model_revision,
            model_sha256: request.model_sha256,
        })
    }
}

fn valid_language(value: &str) -> bool {
    value == "auto" || SUPPORTED_LANGUAGES.contains(&value)
}

fn canonical_uuid(value: &str) -> bool {
    if value.len() != 36 {
        return false;
    }
    value.bytes().enumerate().all(|(index, byte)| match index {
        8 | 13 | 18 | 23 => byte == b'-',
        _ => byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase(),
    })
}

struct Job {
    id: u64,
    audio: AudioRequest,
    reply: tokio::sync::oneshot::Sender<Result<TranscriptionOutput, String>>,
}

struct SeenRequestIds {
    set: std::collections::HashSet<String>,
    order: std::collections::VecDeque<(String, Instant)>,
}

impl SeenRequestIds {
    fn new() -> Self {
        Self {
            set: std::collections::HashSet::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn insert(&mut self, request_id: &str) -> bool {
        let now = Instant::now();
        while self
            .order
            .front()
            .is_some_and(|(_, inserted)| now.duration_since(*inserted) >= REPLAY_WINDOW)
        {
            if let Some((expired, _)) = self.order.pop_front() {
                self.set.remove(&expired);
            }
        }
        if self.set.contains(request_id) {
            return false;
        }
        if self.order.len() >= REPLAY_CACHE_SIZE {
            if let Some((expired, _)) = self.order.pop_front() {
                self.set.remove(&expired);
            }
        }
        self.set.insert(request_id.to_string());
        self.order.push_back((request_id.to_string(), now));
        true
    }
}

pub struct SttEngine {
    tx: Mutex<Option<mpsc::SyncSender<Job>>>,
    config: WorkerConfig,
    restart_lock: tokio::sync::Mutex<()>,
    restarts: AtomicU64,
    assets: Option<PrivateWorkerTemp>,
    healthy: Arc<AtomicBool>,
    stopped: Arc<AtomicBool>,
    next_id: AtomicU64,
    capabilities: Capabilities,
    seen_request_ids: Mutex<SeenRequestIds>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl SttEngine {
    /// Start one explicitly configured worker; no capability exists until smoke validation.
    pub async fn start(config: WorkerConfig) -> Result<Self, String> {
        let saved_config = config.clone();
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let healthy = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let (h, stop) = (healthy.clone(), stopped.clone());
        let handle = std::thread::Builder::new()
            .name("stt-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, h, stop))
            .map_err(|_| "could not start transcription supervisor")?;
        let capabilities = match ready_rx.await {
            Ok(Ok(ready)) => ready,
            _ => {
                stopped.store(true, Ordering::Release);
                let _ = handle.join();
                return Err("transcription worker startup failed".into());
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
            seen_request_ids: Mutex::new(SeenRequestIds::new()),
            worker: Mutex::new(Some(handle)),
        })
    }

    /// Explicit model and whisper.cpp binary are required. The worker script ships in the
    /// binary; the model file and runtime stay operator-owned and read-only.
    pub async fn production(
        model_path: &std::path::Path,
        whisper: &std::path::Path,
        expected_runtime_sha256: &str,
        smoke_dir: &std::path::Path,
        python: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        if !model_path.is_file() {
            return Err(
                "whisper model file unavailable; --model-path must point at the pinned ggml file"
                    .into(),
            );
        }
        if !whisper.is_file() {
            return Err("whisper.cpp runtime unavailable; --stt-whisper must point at the whisper-cli binary".into());
        }
        if !valid_sha256(expected_runtime_sha256) {
            return Err("--stt-whisper-sha256 must be exactly 64 lowercase hex characters".into());
        }
        let approved = approved_runtime_sha256()
            .ok_or("transcription runtime is not approved on this platform")?;
        if expected_runtime_sha256 != approved {
            return Err(format!(
                "whisper runtime pin is not in the approved {RUNTIME_TAG} platform manifest"
            ));
        }
        verify_file(model_path, MODEL_BYTES, MODEL_SHA256, MODEL_FILE)?;
        verify_file(
            whisper,
            MACOS_AARCH64_RUNTIME_BYTES,
            expected_runtime_sha256,
            "whisper.cpp static runtime",
        )?;
        let program = python
            .map(|p| {
                p.to_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| "transcription interpreter path is not UTF-8".to_string())
            })
            .transpose()?
            .unwrap_or_else(|| "python3".into());
        if !smoke_dir.join("audio.wav").is_file() || !smoke_dir.join("smoke.json").is_file() {
            return Err("approved transcription startup fixture unavailable".into());
        }
        let assets = materialize_assets()?;
        let config = WorkerConfig {
            program,
            args: vec![
                "-I".into(),
                "-u".into(),
                assets
                    .0
                    .join("whisper_worker.py")
                    .to_str()
                    .ok_or("transcription runtime path is not UTF-8")?
                    .into(),
                "--model-path".into(),
                model_path
                    .to_str()
                    .ok_or("whisper model path is not UTF-8")?
                    .into(),
                "--whisper-bin".into(),
                whisper
                    .to_str()
                    .ok_or("whisper binary path is not UTF-8")?
                    .into(),
                "--whisper-sha256".into(),
                expected_runtime_sha256.to_string(),
                "--smoke-dir".into(),
                smoke_dir
                    .to_str()
                    .ok_or("whisper smoke path is not UTF-8")?
                    .into(),
            ],
            startup_timeout: Duration::from_secs(180),
            request_timeout: Duration::from_secs(120),
            expected_runtime_sha256: expected_runtime_sha256.to_string(),
            expected_runtime_bytes: MACOS_AARCH64_RUNTIME_BYTES,
        };
        let mut engine = Self::start(config).await?;
        engine.assets = Some(assets);
        Ok(engine)
    }

    /// Lifetime budget avoids crash loops. A successful relaunch repeats all startup smokes.
    pub async fn restart(&self) -> Result<(), String> {
        let _guard = self.restart_lock.lock().await;
        if self.restarts.fetch_add(1, Ordering::AcqRel) >= 3 {
            return Err("transcription worker restart budget exhausted".into());
        }
        self.stop();
        self.stopped.store(false, Ordering::Release);
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (healthy, stopped) = (self.healthy.clone(), self.stopped.clone());
        let config = self.config.clone();
        let handle = std::thread::Builder::new()
            .name("stt-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, healthy, stopped))
            .map_err(|_| "could not restart transcription supervisor")?;
        *self.worker.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
        match ready_rx.await {
            Ok(Ok(_)) if self.is_healthy() => {
                *self.tx.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
                Ok(())
            }
            _ => {
                self.stop();
                Err("transcription worker restart failed".into())
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

    pub fn readiness_snapshot(&self, models: &[String]) -> ReadinessSnapshot {
        let capabilities = self.capabilities();
        let available_models = if capabilities.is_some() {
            models.to_vec()
        } else {
            vec![]
        };
        ReadinessSnapshot {
            available_models,
            capabilities,
        }
    }

    pub async fn transcribe(&self, audio: AudioRequest) -> Result<TranscriptionOutput, String> {
        if !self.is_healthy() {
            return Err("transcription_runtime_failed".into());
        }
        if audio.pcm.len() > MAX_SAMPLES as usize * 2 {
            return Err("transcription_input_invalid".into());
        }
        if !self
            .seen_request_ids
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(&audio.request_id)
        {
            return Err("transcription_input_invalid".into());
        }
        let (reply, receive) = tokio::sync::oneshot::channel();
        let job = Job {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            audio,
            reply,
        };
        self.tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .ok_or("transcription_runtime_failed")?
            .try_send(job)
            .map_err(|_| "transcription_runtime_failed")?;
        receive.await.map_err(|_| "transcription_runtime_failed")?
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

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn verify_file(
    path: &std::path::Path,
    expected_bytes: u64,
    expected_sha256: &str,
    label: &str,
) -> Result<(), String> {
    if std::fs::metadata(path).map(|m| m.len()).ok() != Some(expected_bytes) {
        return Err(format!("{label} size does not match its pin"));
    }
    let mut file = std::fs::File::open(path).map_err(|_| format!("{label} unavailable"))?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| format!("{label} could not be hashed"))?;
        if read == 0 {
            break;
        }
        use sha2::Digest;
        hasher.update(&buffer[..read]);
    }
    use sha2::Digest;
    let actual = hex::encode(hasher.finalize());
    if actual != expected_sha256 {
        return Err(format!("{label} sha256 does not match its pin"));
    }
    Ok(())
}
impl Drop for SttEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            // The Python supervisor is the process-group leader, and whisper-cli inherits
            // that group. Kill the complete tree before reaping the leader.
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        #[cfg(windows)]
        {
            // Windows has no stable std Job Object API. taskkill is part of the supported
            // Windows base image and /T terminates every descendant before /F returns.
            let _ = Command::new("taskkill")
                .args(["/PID", &self.0.id().to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct PrivateWorkerTemp(std::path::PathBuf);
impl PrivateWorkerTemp {
    fn create() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "sgl-stt-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).map_err(|_| "transcription temporary storage unavailable")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).is_err() {
                let _ = std::fs::remove_dir(&path);
                return Err("transcription temporary storage permissions failed".into());
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
    for (name, bytes) in [
        (
            "whisper_worker.py",
            include_bytes!("../scripts/whisper_worker.py").as_slice(),
        ),
        (
            "whisper_model_files.json",
            include_bytes!("../scripts/whisper_model_files.json").as_slice(),
        ),
    ] {
        std::fs::write(assets.0.join(name), bytes)
            .map_err(|_| "transcription runtime materialization failed")?;
    }
    Ok(assets)
}

fn blocked_worker_env(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("DYLD_")
        || name.starts_with("__XPC_DYLD_")
        || name.starts_with("LD_")
        || name.starts_with("PYTHON")
        || matches!(
            name.as_str(),
            "BASH_ENV" | "ENV" | "GGML_BACKEND_PATH" | "GGML_METAL_PATH_RESOURCES" | "GCONV_PATH"
        )
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
    let mut command = Command::new(&config.program);
    command
        .args(&config.args)
        .env("TMPDIR", &temporary.0)
        .env("TEMP", &temporary.0)
        .env("TMP", &temporary.0)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    // Remove loader and interpreter injection before Python itself starts.
    for (name, _) in std::env::vars_os() {
        if blocked_worker_env(&name.to_string_lossy()) {
            command.env_remove(name);
        }
    }
    let result = command.spawn();
    let mut child = match result {
        Ok(child) => OwnedChild(child),
        Err(_) => {
            let _ = ready.send(Err("transcription worker spawn failed".into()));
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
                        .map(|t| matches!(t, "ready" | "result" | "request_error"))
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
    let startup =
        receive_frame(&frames_rx, config.startup_timeout, &stopped, None).and_then(|frame| {
            validate_ready(
                &frame,
                &config.expected_runtime_sha256,
                config.expected_runtime_bytes,
            )
        });
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
            let _ = ready.send(Err("invalid transcription readiness".into()));
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
        let response = match response {
            Ok(output) => Ok(output),
            Err(WorkerFailure::InvalidInput) => Err("transcription_input_invalid".into()),
            Err(WorkerFailure::Fatal(_reason)) => {
                healthy.store(false, Ordering::Release);
                Err("transcription_runtime_failed".into())
            }
        };
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
    reply: Option<&tokio::sync::oneshot::Sender<Result<TranscriptionOutput, String>>>,
) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if stopped.load(Ordering::Acquire) || reply.is_some_and(|reply| reply.is_closed()) {
            return Err("transcription request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("transcription worker timed out".into());
        }
        match frames.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(frame) => return Ok(frame),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => return Err("transcription worker closed output".into()),
        }
    }
}

fn validate_ready(
    frame: &[u8],
    expected_runtime_sha256: &str,
    expected_runtime_bytes: u64,
) -> Result<Capabilities, String> {
    let ready: Ready =
        serde_json::from_slice(frame).map_err(|_| "invalid transcription readiness")?;
    if ready.frame_type != "ready"
        || ready.protocol != PROTOCOL
        || ready.runtime != RUNTIME
        || ready.runtime_revision != RUNTIME_REVISION
        || ready.model_revision != MODEL_REVISION
        || ready.model_sha256 != MODEL_SHA256
        || ready.model_bytes != MODEL_BYTES
        || ready.runtime_binary_sha256 != expected_runtime_sha256
        || ready.runtime_binary_bytes != expected_runtime_bytes
        || expected_runtime_bytes == 0
        || ready.model_id != MODEL_ID
        || ready.audio_format != "pcm_s16le_16k_mono"
        || ready.max_duration_seconds != MAX_DURATION_SECONDS
        || ready.smoke_transcript.len() < 20
    {
        return Err("transcription worker identity or smoke mismatch".into());
    }
    let lowered = ready.smoke_transcript.to_lowercase();
    for word in ["ready", "for", "service"] {
        if !lowered.contains(word) {
            return Err("transcription smoke transcript mismatch".into());
        }
    }
    Ok(Capabilities {
        transcription_ready: true,
        transcription_protocol: PROTOCOL.into(),
        transcription_models: vec![MODEL_ID.into()],
        transcription_runtime: RUNTIME.into(),
        runtime_revision: ready.runtime_revision,
        model_revision: ready.model_revision,
        model_sha256: ready.model_sha256,
        model_repository: MODEL_REPOSITORY.into(),
        model_bytes: ready.model_bytes,
        runtime_binary_sha256: ready.runtime_binary_sha256,
        runtime_binary_bytes: ready.runtime_binary_bytes,
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        model_license: "MIT".into(),
        runtime_license: "MIT".into(),
        audio_format: ready.audio_format,
        max_duration_seconds: ready.max_duration_seconds,
        languages: std::iter::once("auto".to_string())
            .chain(SUPPORTED_LANGUAGES.iter().map(|code| (*code).to_string()))
            .collect(),
        input_envelope_encodings: vec!["base64".into()],
        media_transport: vec!["inline-base64".into()],
        transcription_streaming: false,
        free_slots: 1,
        key_binding: None,
    })
}

fn run_job(
    job: &Job,
    writes: &mpsc::SyncSender<(Vec<u8>, mpsc::SyncSender<bool>)>,
    frames: &mpsc::Receiver<Vec<u8>>,
    stopped: &AtomicBool,
    timeout: Duration,
) -> Result<TranscriptionOutput, WorkerFailure> {
    use base64::Engine;
    let data = base64::engine::general_purpose::STANDARD.encode(&job.audio.pcm);
    let digest = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&job.audio.pcm);
        hex::encode(hasher.finalize())
    };
    let mut request = serde_json::json!({
        "type": "transcribe",
        "request_id": job.id,
        "protocol": PROTOCOL,
        "audio": {
            "encoding": "base64",
            "mime_type": "audio/pcm",
            "data": data,
            "sha256": digest,
            "sample_rate": SAMPLE_RATE,
            "sample_count": job.audio.sample_count,
            "channels": CHANNELS,
            "bits": BITS,
        },
    });
    if let Some(language) = &job.audio.language {
        request["audio"]["language"] = serde_json::json!(language);
    }
    let mut bytes = serde_json::to_vec(&request).map_err(|_| "invalid transcription request")?;
    if bytes.len() > MAX_INPUT_FRAME {
        return Err("transcription request exceeds frame limit".into());
    }
    bytes.push(b'\n');
    let deadline = Instant::now() + timeout;
    let (ack_tx, ack_rx) = mpsc::sync_channel(1);
    writes
        .try_send((bytes, ack_tx))
        .map_err(|_| "transcription worker input unavailable")?;
    loop {
        if stopped.load(Ordering::Acquire) || job.reply.is_closed() {
            return Err("transcription request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("transcription worker input timed out".into());
        }
        match ack_rx.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(true) => break,
            Ok(false) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("transcription worker input closed".into())
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
    let value: serde_json::Value =
        serde_json::from_slice(&frame).map_err(|_| "invalid transcription worker frame")?;
    if value.get("type").and_then(|v| v.as_str()) == Some("request_error") {
        let error: RequestError = serde_json::from_value(value)
            .map_err(|_| "invalid transcription request-error frame")?;
        if error.frame_type != "request_error"
            || error.request_id != job.id
            || error.code != "invalid_input"
        {
            return Err("transcription request-error mismatch".into());
        }
        return Err(WorkerFailure::InvalidInput);
    }
    let result: Response =
        serde_json::from_slice(&frame).map_err(|_| "invalid transcription worker output")?;
    if result.frame_type != "result" || result.request_id != job.id {
        return Err("transcription worker response mismatch".into());
    }
    validate_output(&result, job.audio.sample_count)?;
    Ok(TranscriptionOutput {
        text: result.text,
        language: result.language,
        duration_seconds: result.duration_seconds,
        segments: result.segments,
    })
}

fn validate_output(result: &Response, sample_count: u32) -> Result<(), String> {
    if result.text.len() > MAX_TEXT_BYTES
        || (result.text.trim().is_empty() && !result.segments.is_empty())
    {
        return Err("transcription text exceeds bound".into());
    }
    if result
        .text
        .chars()
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err("transcription text contains control characters".into());
    }
    let expected_duration = sample_count as f64 / SAMPLE_RATE as f64;
    if !(0.0..=(expected_duration + 1.0).min(MAX_DURATION_SECONDS as f64 + 1.0))
        .contains(&result.duration_seconds)
        || result.duration_seconds <= 0.0
    {
        return Err("transcription duration contradicts input samples".into());
    }
    if let Some(language) = &result.language {
        if !SUPPORTED_LANGUAGES.contains(&language.as_str()) {
            return Err("transcription language invalid".into());
        }
    }
    if result.segments.len() > MAX_SEGMENTS {
        return Err("transcription segment count exceeds bound".into());
    }
    let mut last_end = 0.0f64;
    let mut segment_text_bytes = 0usize;
    for segment in &result.segments {
        segment_text_bytes += segment.text.len();
        if segment_text_bytes > MAX_TEXT_BYTES
            || segment
                .text
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            return Err("transcription segment text invalid".into());
        }
        if !segment.start.is_finite()
            || !segment.end.is_finite()
            || segment.start < 0.0
            || segment.end <= segment.start
            || segment.end > expected_duration + 0.05
            || segment.start < last_end - 0.05
        {
            return Err("transcription segment timestamps not monotonic".into());
        }
        last_end = segment.end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_value(sample_count: u64, pcm: &[u8]) -> serde_json::Value {
        use base64::Engine;
        serde_json::json!({
            "protocol": PROTOCOL,
            "request_id": "00000000-0000-4000-8000-000000000001",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA256,
            "language": "auto",
            "audio": {
                "format": "pcm_s16le",
                "data": base64::engine::general_purpose::STANDARD.encode(pcm),
                "sample_rate": SAMPLE_RATE,
                "sample_count": sample_count,
                "channels": CHANNELS,
                "bits_per_sample": BITS,
            }
        })
    }

    #[test]
    fn audio_request_accepts_valid_pcm_and_rejects_drift() {
        let pcm = vec![0u8; 3200]; // 100 ms
        let value = audio_value(1600, &pcm);
        let parsed = AudioRequest::parse(&value).unwrap();
        assert_eq!(parsed.sample_count, 1600);
        assert!(parsed.language.is_none());

        assert!(AudioRequest::parse(&audio_value(1601, &pcm)).is_err());
        assert!(AudioRequest::parse(&audio_value(0, &pcm)).is_err());
        assert!(AudioRequest::parse(&audio_value(MAX_SAMPLES as u64 + 1, &pcm)).is_err());

        let mut wrong_rate = value.clone();
        wrong_rate["audio"]["sample_rate"] = serde_json::json!(44100);
        assert!(AudioRequest::parse(&wrong_rate).is_err());
        let mut wrong_mime = value.clone();
        wrong_mime["audio"]["format"] = serde_json::json!("audio/wav");
        assert!(AudioRequest::parse(&wrong_mime).is_err());
        let mut bad_sha = value.clone();
        bad_sha["audio"]["sha256"] = serde_json::json!("00".repeat(32));
        assert!(AudioRequest::parse(&bad_sha).is_err());
    }

    #[test]
    fn language_hint_must_be_two_lowercase_letters() {
        let pcm = vec![1u8; 3200];
        let mut value = audio_value(1600, &pcm);
        value["language"] = serde_json::json!("en");
        assert_eq!(
            AudioRequest::parse(&value).unwrap().language.as_deref(),
            Some("en")
        );
        value["language"] = serde_json::json!("EN");
        assert!(AudioRequest::parse(&value).is_err());
        value["language"] = serde_json::json!("eng");
        assert!(AudioRequest::parse(&value).is_err());
        value["language"] = serde_json::json!("");
        assert!(AudioRequest::parse(&value).is_err());
    }

    #[tokio::test]
    async fn operator_hash_cannot_extend_the_approved_runtime_manifest() {
        let path =
            std::env::temp_dir().join(format!("sgl-stt-pin-test-{:016x}", rand::random::<u64>()));
        std::fs::write(&path, b"unapproved artifact").unwrap();
        let result = SttEngine::production(
            &path,
            &path,
            &"11".repeat(32),
            std::env::temp_dir().as_path(),
            None,
        )
        .await;
        std::fs::remove_file(path).unwrap();
        assert!(result.is_err());
        let error = result.err().unwrap();
        assert!(error.contains("approved"), "{error}");
    }

    #[test]
    fn canonical_contract_and_reservation_fail_closed() {
        let value = audio_value(1600, &vec![0u8; 3200]);
        for (field, wrong) in [
            ("protocol", "transcription-v0"),
            ("request_id", "not-a-uuid"),
            ("model", "embeddinggemma-2"),
            ("model_revision", MODEL_SHA256),
            ("model_sha256", "wrong"),
            ("language", "xx"),
        ] {
            let mut changed = value.clone();
            changed[field] = serde_json::json!(wrong);
            assert!(AudioRequest::parse(&changed).is_err(), "{field}");
        }
        for field in ["url", "path", "sha256", "bits", "encoding"] {
            let mut changed = value.clone();
            changed["audio"][field] = serde_json::json!("unexpected");
            assert!(AudioRequest::parse(&changed).is_err(), "{field}");
        }
        let mut changed = value.clone();
        changed["extra"] = serde_json::json!(true);
        assert!(AudioRequest::parse(&changed).is_err());
        let binding = serde_json::json!({
            "protocol": PROTOCOL, "request_id": value["request_id"], "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "language": "auto", "sample_count": 1600,
        });
        let dispatch = serde_json::json!({"enc": {}, "transcription_reservation": binding});
        let parsed = ReservationBinding::parse_dispatch(&dispatch).unwrap();
        let audio = AudioRequest::parse(&value).unwrap();
        assert!(parsed.matches(&audio));
        let mut wrong = parsed.clone();
        wrong.sample_count += 1;
        assert!(!wrong.matches(&audio));
        let mut wrong = parsed.clone();
        wrong.request_id = "00000000-0000-4000-8000-000000000002".into();
        assert!(!wrong.matches(&audio));
        let mut wrong = parsed.clone();
        wrong.language = "en".into();
        assert!(!wrong.matches(&audio));
        let mut extra = dispatch.clone();
        extra["audio"] = value["audio"].clone();
        assert!(ReservationBinding::parse_dispatch(&extra).is_err());
        assert!(ReservationBinding::parse_dispatch(&value).is_err());
    }

    #[test]
    fn replay_cache_is_bounded_and_retains_recent_ids() {
        let mut seen = SeenRequestIds::new();
        for id in 0..REPLAY_CACHE_SIZE {
            assert!(seen.insert(&id.to_string()));
        }
        assert!(!seen.insert("0"));
        assert!(seen.insert("overflow"));
        assert_eq!(seen.set.len(), REPLAY_CACHE_SIZE);
        assert!(!seen.insert("overflow"));
        assert!(seen.insert("0"));
    }

    #[test]
    fn runtime_manifest_matches_the_compiled_static_pin() {
        let manifest: serde_json::Value =
            serde_json::from_slice(include_bytes!("../scripts/whisper_runtime_files.json"))
                .unwrap();
        assert_eq!(manifest["sha256"], MACOS_AARCH64_RUNTIME_SHA256);
        assert_eq!(manifest["bytes"], MACOS_AARCH64_RUNTIME_BYTES);
        assert_eq!(manifest["source_commit"], RUNTIME_REVISION);
        assert_eq!(manifest["cmake_definitions"]["BUILD_SHARED_LIBS"], "OFF");
        assert_eq!(manifest["cmake_definitions"]["GGML_BACKEND_DL"], "OFF");
        assert!(manifest["rpath"].as_array().unwrap().is_empty());
        for library in manifest["dynamic_dependencies"].as_array().unwrap() {
            let library = library.as_str().unwrap();
            assert!(library.starts_with("/System/Library/") || library.starts_with("/usr/lib/"));
        }
    }

    #[test]
    fn worker_env_blocks_native_and_python_loader_injection() {
        for name in [
            "DYLD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_FRAMEWORK_PATH",
            "__XPC_DYLD_LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "BASH_ENV",
            "GGML_BACKEND_PATH",
            "GGML_METAL_PATH_RESOURCES",
        ] {
            assert!(blocked_worker_env(name), "{name}");
        }
        assert!(!blocked_worker_env("PATH"));
        assert!(!blocked_worker_env("TMPDIR"));
    }

    #[test]
    fn replay_window_expires_without_losing_newer_ids() {
        let mut seen = SeenRequestIds::new();
        assert!(seen.insert("old"));
        seen.order.front_mut().unwrap().1 = Instant::now() - REPLAY_WINDOW;
        assert!(seen.insert("recent"));
        assert!(!seen.set.contains("old"));
        assert!(!seen.insert("recent"));
        assert!(seen.insert("old"));
    }

    #[test]
    fn output_validation_rejects_lies_and_enforces_monotonic_segments() {
        let mut response = Response {
            frame_type: "result".into(),
            request_id: 1,
            text: "hello world".into(),
            language: Some("en".into()),
            duration_seconds: 5.0,
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 2.0,
                    text: "hello".into(),
                },
                Segment {
                    start: 2.0,
                    end: 5.0,
                    text: "world".into(),
                },
            ],
        };
        assert!(validate_output(&response, 80_000).is_ok());

        response.duration_seconds = 999.0;
        assert!(validate_output(&response, 80_000).is_err());
        response.duration_seconds = 0.0;
        assert!(validate_output(&response, 80_000).is_err());
        response.duration_seconds = 5.0;

        response.segments[1].start = 1.0; // overlaps previous segment
        assert!(validate_output(&response, 80_000).is_err());
        response.segments[1].start = 2.0;

        response.language = Some("e".into());
        assert!(validate_output(&response, 80_000).is_err());
        response.language = Some("en".into());

        response.text = "x".repeat(MAX_TEXT_BYTES + 1);
        assert!(validate_output(&response, 80_000).is_err());
    }
}
