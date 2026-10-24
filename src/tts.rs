//! Dedicated mlx-audio text-to-speech worker supervisor for the Grid `speech-v1` lane
//! (registry: Kokoro 82M, Chatterbox Turbo; one model per node).
//! Mirrors `stt.rs`: one owned, serialized, framed worker that is timed out, killed and reaped
//! on failure. Text in (UTF-8, byte count bound in the reservation), m4a AAC audio out.
//! Spec: x402Studio documentation/product/sgl-network/GRID_TTS_SPEECH_V1.md and
//! GRID_TTS_MULTI_MODEL.md.
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

pub const MODEL_ID: &str = "kokoro-82m";
pub const MODEL_REPOSITORY: &str = "mlx-community/Kokoro-82M-bf16";
pub const MODEL_REVISION: &str = "a71e4d38b236d968966a2002c4c895dbd12b1c3c";
/// sha256 of the canonical file manifest (scripts/tts_model_files.json), not of one file.
pub const MODEL_SHA256: &str = "cccd8451bb6755bbc9e41fdb705391c0f15fc17b68f55b3b680d5f73e7041ddf";
pub const RUNTIME: &str = "mlx-audio";
pub const RUNTIME_REVISION: &str = "mlx-audio==0.6.1;mlx==0.32.4;misaki==0.9.4";
pub const PROTOCOL: &str = "speech-v1";
pub const OUTPUT_FORMAT: &str = "m4a_aac_24k_mono";
pub const OUTPUT_MIME: &str = "audio/mp4";
pub const SAMPLE_RATE: u32 = 24_000;
pub const MAX_INPUT_BYTES: usize = 1000;
pub const MAX_AUDIO_SECONDS: u32 = 90;
pub const MAX_AUDIO_BYTES: usize = 700_000;
pub const VOICES: &[&str] = &[
    "af_heart",
    "af_bella",
    "af_nicole",
    "bf_emma",
    "af_aoede",
    "af_kore",
    "af_sarah",
    "am_fenrir",
    "am_michael",
    "am_puck",
];
/// Sealed request: canonical JSON with base64 text; 1000 bytes of text is ~1.4 KB.
pub const MAX_SEALED_INPUT: usize = 16 * 1024;
const MAX_WORKER_INPUT_FRAME: usize = 8 * 1024;
/// 700,000 audio bytes as base64 (~933 KB) plus a small JSON frame.
const MAX_OUTPUT_FRAME: u64 = 1024 * 1024;
const REPLAY_CACHE_SIZE: usize = 65_536;
const REPLAY_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);
const MODEL_MANIFEST: &str = include_str!("../scripts/tts_model_files.json");
const CHATTERBOX_TURBO_MANIFEST: &str =
    include_str!("../scripts/tts_model_files_chatterbox_turbo.json");

/// One Grid speech model. Entry 0 is kokoro-82m and must stay exactly as above.
#[derive(Debug, PartialEq)]
pub struct SpeechModel {
    pub id: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    /// sha256 of the canonical file manifest.
    pub sha256: &'static str,
    pub license: &'static str,
    pub voices: &'static [&'static str],
    /// Speed is part of the binding; a fixed-speed model only accepts `"1.00"`.
    pub fixed_speed: bool,
    manifest: &'static str,
}

pub const MODELS: [SpeechModel; 2] = [
    SpeechModel {
        id: MODEL_ID,
        repository: MODEL_REPOSITORY,
        revision: MODEL_REVISION,
        sha256: MODEL_SHA256,
        license: "Apache-2.0",
        voices: VOICES,
        fixed_speed: false,
        manifest: MODEL_MANIFEST,
    },
    SpeechModel {
        id: "chatterbox-turbo",
        repository: "mlx-community/chatterbox-turbo-4bit",
        revision: "c63817725071d7b5269c7b558772d6e8cbf59cec",
        sha256: "196bd3c645a8b9969610d63d17663740ade3f5baabacf2e096623f53d34581fe",
        license: "MIT",
        voices: &["aria"],
        fixed_speed: true,
        manifest: CHATTERBOX_TURBO_MANIFEST,
    },
];

pub fn model_by_id(id: &str) -> Option<&'static SpeechModel> {
    MODELS.iter().find(|m| m.id == id)
}

/// Operator-facing lookup for `--model-name` (case-insensitive, like the rest of the CLI).
pub fn model_for_name(name: &str) -> Option<&'static SpeechModel> {
    MODELS.iter().find(|m| m.id.eq_ignore_ascii_case(name))
}

pub fn is_speech_model(name: &str) -> bool {
    model_for_name(name).is_some()
}

impl SpeechModel {
    fn pins_match(&self, model: &str, revision: &str, sha256: &str) -> bool {
        self.id == model && self.revision == revision && self.sha256 == sha256
    }
    fn speed_ok(&self, speed: &str) -> bool {
        valid_speed(speed) && (!self.fixed_speed || speed == "1.00")
    }
    fn voice_ok(&self, voice: &str) -> bool {
        self.voices.contains(&voice)
    }
}

fn platform_approved() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

#[derive(Clone)]
pub struct WorkerConfig {
    pub model: &'static SpeechModel,
    pub program: String,
    pub args: Vec<String>,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
}

#[derive(Clone, Debug, Serialize)]
pub struct KeyBinding {
    pub encryption_public_key: String,
    pub encryption_public_key_signature: String,
    pub key_version: u32,
}

/// Advertised under heartbeat `capabilities.speech`.
#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub kind: String,
    pub speech_ready: bool,
    pub speech_protocol: String,
    pub speech_models: Vec<String>,
    pub speech_runtime: String,
    pub runtime_revision: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub model_repository: String,
    pub os: String,
    pub architecture: String,
    pub model_license: String,
    pub voices: Vec<String>,
    pub max_input_bytes: usize,
    pub max_audio_seconds: u32,
    pub output_format: String,
    pub input_envelope_encodings: Vec<String>,
    pub media_transport: Vec<String>,
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
            self.speech_ready = false;
            self.free_slots = 0;
        }
        self
    }
}

pub struct ReadinessSnapshot {
    pub available_models: Vec<String>,
    pub capabilities: Option<Capabilities>,
}

/// Validated text request (from the decrypted canonical plaintext).
#[derive(Debug, Clone)]
pub struct SpeechRequest {
    pub text: Vec<u8>,
    pub request_id: String,
    pub model: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub voice: String,
    pub speed: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalInput {
    encoding: String,
    bytes: u64,
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
    voice: String,
    speed: String,
    input: CanonicalInput,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReservationBinding {
    pub protocol: String,
    pub request_id: String,
    pub model: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub voice: String,
    pub speed: String,
    pub input_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechDispatch {
    enc: serde_json::Value,
    speech_reservation: ReservationBinding,
}

/// `"0.50"`..=`"2.00"`, exactly one digit, a dot and two digits.
pub fn valid_speed(value: &str) -> bool {
    let b = value.as_bytes();
    if b.len() != 4 || b[1] != b'.' || ![b[0], b[2], b[3]].iter().all(u8::is_ascii_digit) {
        return false;
    }
    let hundredths = (b[0] - b'0') as u32 * 100 + (b[2] - b'0') as u32 * 10 + (b[3] - b'0') as u32;
    (50..=200).contains(&hundredths)
}

impl ReservationBinding {
    pub fn parse_dispatch(value: &serde_json::Value) -> Result<Self, String> {
        let dispatch: SpeechDispatch = serde_json::from_value(value.clone())
            .map_err(|_| "speech dispatch shape invalid".to_string())?;
        if !dispatch.enc.is_object() {
            return Err("speech encrypted envelope missing".into());
        }
        let binding = dispatch.speech_reservation;
        let model = model_by_id(&binding.model).ok_or("speech reservation binding invalid")?;
        if binding.protocol != PROTOCOL
            || !canonical_uuid(&binding.request_id)
            || !model.pins_match(
                &binding.model,
                &binding.model_revision,
                &binding.model_sha256,
            )
            || !model.voice_ok(&binding.voice)
            || !model.speed_ok(&binding.speed)
            || binding.input_bytes == 0
            || binding.input_bytes > MAX_INPUT_BYTES as u64
        {
            return Err("speech reservation binding invalid".into());
        }
        Ok(binding)
    }

    pub fn matches(&self, request: &SpeechRequest) -> bool {
        self.protocol == PROTOCOL
            && self.request_id == request.request_id
            && self.model == request.model
            && self.model_revision == request.model_revision
            && self.model_sha256 == request.model_sha256
            && self.voice == request.voice
            && self.speed == request.speed
            && self.input_bytes == request.text.len() as u64
    }
}

impl SpeechRequest {
    pub fn parse(value: &serde_json::Value) -> Result<Self, String> {
        let request: CanonicalRequest = serde_json::from_value(value.clone())
            .map_err(|_| "speech request shape invalid".to_string())?;
        if request.protocol != PROTOCOL {
            return Err("speech protocol mismatch".into());
        }
        if !canonical_uuid(&request.request_id) {
            return Err("speech request id invalid".into());
        }
        let model = model_by_id(&request.model)
            .filter(|m| {
                m.pins_match(
                    &request.model,
                    &request.model_revision,
                    &request.model_sha256,
                )
            })
            .ok_or("speech model binding mismatch")?;
        if !model.voice_ok(&request.voice) || !model.speed_ok(&request.speed) {
            return Err("speech voice or speed invalid".into());
        }
        if request.input.encoding != "base64"
            || request.input.bytes == 0
            || request.input.bytes > MAX_INPUT_BYTES as u64
            || request.input.data.len() > MAX_INPUT_BYTES.div_ceil(3) * 4
        {
            return Err("speech input out of bounds".into());
        }
        use base64::Engine;
        let text = base64::engine::general_purpose::STANDARD
            .decode(&request.input.data)
            .map_err(|_| "speech input base64 invalid".to_string())?;
        if text.len() as u64 != request.input.bytes {
            return Err("speech input length contradicts byte count".into());
        }
        let decoded = std::str::from_utf8(&text).map_err(|_| "speech input is not UTF-8")?;
        if decoded.trim().is_empty() || decoded.contains('\0') {
            return Err("speech input empty".into());
        }
        Ok(Self {
            text,
            request_id: request.request_id,
            model: request.model,
            model_revision: request.model_revision,
            model_sha256: request.model_sha256,
            voice: request.voice,
            speed: request.speed,
        })
    }
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

#[derive(Debug, Clone)]
pub struct SpeechOutput {
    pub audio: Vec<u8>,
    pub duration_seconds: f64,
    pub sha256: String,
}

#[derive(Serialize)]
struct ResultAudio<'a> {
    format: &'a str,
    mime_type: &'a str,
    sample_rate: u32,
    channels: u16,
    duration_seconds: f64,
    bytes: usize,
    sha256: &'a str,
}

/// Field order is the wire order (serde serializes struct fields in declaration order).
#[derive(Serialize)]
struct ResultHeader<'a> {
    object: &'a str,
    protocol: &'a str,
    request_id: &'a str,
    job_id: &'a str,
    model: &'a str,
    model_revision: &'a str,
    model_sha256: &'a str,
    voice: &'a str,
    speed: &'a str,
    input_bytes: usize,
    audio: ResultAudio<'a>,
}

/// `speech-result-v1` plaintext: one canonical JSON header line, `\n`, then the raw m4a bytes.
pub fn result_plaintext(job_id: &str, request: &SpeechRequest, output: &SpeechOutput) -> Vec<u8> {
    let header = ResultHeader {
        object: "speech",
        protocol: PROTOCOL,
        request_id: &request.request_id,
        job_id,
        // The request's own (registry-validated) model, never a hardcoded one: a chatterbox
        // result stamped with Kokoro's pins is rejected by every client.
        model: &request.model,
        model_revision: &request.model_revision,
        model_sha256: &request.model_sha256,
        voice: &request.voice,
        speed: &request.speed,
        input_bytes: request.text.len(),
        audio: ResultAudio {
            format: OUTPUT_FORMAT,
            mime_type: OUTPUT_MIME,
            sample_rate: SAMPLE_RATE,
            channels: 1,
            duration_seconds: output.duration_seconds,
            bytes: output.audio.len(),
            sha256: &output.sha256,
        },
    };
    let mut bytes = serde_json::to_vec(&header).expect("header serializes");
    bytes.push(b'\n');
    bytes.extend_from_slice(&output.audio);
    bytes
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    #[serde(rename = "type")]
    frame_type: String,
    protocol: String,
    runtime: String,
    runtime_revision: String,
    model_id: String,
    model_revision: String,
    model_sha256: String,
    voices: Vec<String>,
    output_format: String,
    max_input_bytes: usize,
    max_audio_seconds: u32,
    smoke_seconds: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    #[serde(rename = "type")]
    frame_type: String,
    request_id: u64,
    duration_seconds: f64,
    bytes: usize,
    sha256: String,
    audio_b64: String,
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

struct Job {
    id: u64,
    request: SpeechRequest,
    reply: tokio::sync::oneshot::Sender<Result<SpeechOutput, String>>,
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

pub struct TtsEngine {
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

impl TtsEngine {
    /// Start one explicitly configured worker; no capability exists until smoke validation.
    pub async fn start(config: WorkerConfig) -> Result<Self, String> {
        let saved_config = config.clone();
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let healthy = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let (h, stop) = (healthy.clone(), stopped.clone());
        let handle = std::thread::Builder::new()
            .name("tts-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, h, stop))
            .map_err(|_| "could not start speech supervisor")?;
        let capabilities = match ready_rx.await {
            Ok(Ok(ready)) => ready,
            _ => {
                stopped.store(true, Ordering::Release);
                let _ = handle.join();
                return Err("speech worker startup failed".into());
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

    /// Operator supplies the interpreter (a venv with the pinned mlx-audio stack; the desktop
    /// app passes the one it installs) and the local model directory. The node verifies every
    /// pinned model file before starting the worker, and the worker verifies them again.
    pub async fn production(
        model: &'static SpeechModel,
        model_dir: &std::path::Path,
        python: &std::path::Path,
    ) -> Result<Self, String> {
        if !platform_approved() {
            return Err(
                "speech runtime is not approved on this platform (Apple Silicon macOS only)".into(),
            );
        }
        if !model_dir.is_dir() {
            return Err(format!(
                "speech model directory unavailable; --tts-model-dir must point at the pinned {} folder",
                model.repository.rsplit('/').next().unwrap_or(model.repository)
            ));
        }
        if !python.is_file() {
            return Err("speech interpreter unavailable; --tts-python must point at the mlx-audio venv python".into());
        }
        verify_model_dir(model, model_dir)?;
        let assets = materialize_assets(model)?;
        let config = WorkerConfig {
            model,
            program: python
                .to_str()
                .ok_or("speech interpreter path is not UTF-8")?
                .into(),
            args: vec![
                "-I".into(),
                "-u".into(),
                assets
                    .0
                    .join("tts_worker.py")
                    .to_str()
                    .ok_or("speech runtime path is not UTF-8")?
                    .into(),
                "--model-dir".into(),
                model_dir
                    .to_str()
                    .ok_or("speech model path is not UTF-8")?
                    .into(),
                "--manifest".into(),
                assets
                    .0
                    .join("tts_model_files.json")
                    .to_str()
                    .ok_or("speech runtime path is not UTF-8")?
                    .into(),
                "--model-id".into(),
                model.id.into(),
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
            return Err("speech worker restart budget exhausted".into());
        }
        self.stop();
        self.stopped.store(false, Ordering::Release);
        let (tx, rx) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (healthy, stopped) = (self.healthy.clone(), self.stopped.clone());
        let config = self.config.clone();
        let handle = std::thread::Builder::new()
            .name("tts-supervisor".into())
            .spawn(move || supervise(config, rx, ready_tx, healthy, stopped))
            .map_err(|_| "could not restart speech supervisor")?;
        *self.worker.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
        match ready_rx.await {
            Ok(Ok(_)) if self.is_healthy() => {
                *self.tx.lock().unwrap_or_else(|p| p.into_inner()) = Some(tx);
                Ok(())
            }
            _ => {
                self.stop();
                Err("speech worker restart failed".into())
            }
        }
    }

    pub fn model(&self) -> &'static SpeechModel {
        self.config.model
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

    pub async fn speak(&self, request: SpeechRequest) -> Result<SpeechOutput, String> {
        if !self.is_healthy() {
            return Err("speech_runtime_failed".into());
        }
        if request.text.is_empty()
            || request.text.len() > MAX_INPUT_BYTES
            || request.model != self.model().id
        {
            return Err("speech_input_invalid".into());
        }
        if !self
            .seen_request_ids
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(&request.request_id)
        {
            return Err("speech_input_invalid".into());
        }
        let (reply, receive) = tokio::sync::oneshot::channel();
        let job = Job {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            request,
            reply,
        };
        self.tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .ok_or("speech_runtime_failed")?
            .try_send(job)
            .map_err(|_| "speech_runtime_failed")?;
        receive.await.map_err(|_| "speech_runtime_failed")?
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

impl Drop for TtsEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Deserialize)]
struct ManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    model_id: String,
    model_revision: String,
    model_sha256: String,
    files: Vec<ManifestFile>,
}

/// The canonical manifest digest: sorted `path size sha256` lines, LF endings.
fn manifest_digest(files: &[ManifestFile]) -> String {
    let mut lines: Vec<String> = files
        .iter()
        .map(|f| format!("{} {} {}", f.path, f.bytes, f.sha256))
        .collect();
    lines.sort();
    let text: String = lines.into_iter().map(|l| l + "\n").collect();
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(text.as_bytes()))
}

fn embedded_manifest(model: &SpeechModel) -> Result<Manifest, String> {
    let manifest: Manifest =
        serde_json::from_str(model.manifest).map_err(|_| "speech model manifest invalid")?;
    if manifest.model_id != model.id
        || manifest.model_revision != model.revision
        || manifest.model_sha256 != model.sha256
        || manifest_digest(&manifest.files) != model.sha256
    {
        return Err("speech model manifest does not match its pins".into());
    }
    Ok(manifest)
}

fn verify_model_dir(model: &SpeechModel, model_dir: &std::path::Path) -> Result<(), String> {
    let manifest = embedded_manifest(model)?;
    for file in &manifest.files {
        if file.path.starts_with('/') || file.path.split('/').any(|part| part == "..") {
            return Err("speech model manifest path invalid".into());
        }
        let path = model_dir.join(&file.path);
        if std::fs::symlink_metadata(&path)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(true)
        {
            return Err(format!("speech model file missing: {}", file.path));
        }
        verify_file(&path, file.bytes, &file.sha256, &file.path)?;
    }
    Ok(())
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
    use sha2::Digest;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| format!("{label} could not be hashed"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if hex::encode(hasher.finalize()) != expected_sha256 {
        return Err(format!("{label} sha256 does not match its pin"));
    }
    Ok(())
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            // The Python worker leads its own process group; afconvert children inherit it.
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct PrivateWorkerTemp(std::path::PathBuf);
impl PrivateWorkerTemp {
    fn create() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "sgl-tts-{}-{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).map_err(|_| "speech temporary storage unavailable")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).is_err() {
                let _ = std::fs::remove_dir(&path);
                return Err("speech temporary storage permissions failed".into());
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

fn materialize_assets(model: &SpeechModel) -> Result<PrivateWorkerTemp, String> {
    let assets = PrivateWorkerTemp::create()?;
    for (name, bytes) in [
        (
            "tts_worker.py",
            include_bytes!("../scripts/tts_worker.py").as_slice(),
        ),
        ("tts_model_files.json", model.manifest.as_bytes()),
    ] {
        std::fs::write(assets.0.join(name), bytes)
            .map_err(|_| "speech runtime materialization failed")?;
    }
    Ok(assets)
}

fn blocked_worker_env(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("DYLD_")
        || name.starts_with("__XPC_DYLD_")
        || name.starts_with("LD_")
        || name.starts_with("PYTHON")
        || name.starts_with("MLX_")
        || matches!(name.as_str(), "BASH_ENV" | "ENV" | "GCONV_PATH")
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
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    for (name, _) in std::env::vars_os() {
        if blocked_worker_env(&name.to_string_lossy()) {
            command.env_remove(name);
        }
    }
    let mut child = match command.spawn() {
        Ok(child) => OwnedChild(child),
        Err(_) => {
            let _ = ready.send(Err("speech worker spawn failed".into()));
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
            if !valid_frame || frames_tx.try_send(frame).is_err() {
                h.store(false, Ordering::Release);
                break;
            }
        }
        open.store(false, Ordering::Release);
    });
    let startup = receive_frame(&frames_rx, config.startup_timeout, &stopped, None)
        .and_then(|frame| validate_ready(&frame, config.model));
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
            let _ = ready.send(Err("invalid speech readiness".into()));
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
        let response = match run_job(
            &job,
            &writes_tx,
            &frames_rx,
            &stopped,
            config.request_timeout,
        ) {
            Ok(output) => Ok(output),
            Err(WorkerFailure::InvalidInput) => Err("speech_input_invalid".into()),
            Err(WorkerFailure::Fatal(_reason)) => {
                healthy.store(false, Ordering::Release);
                Err("speech_runtime_failed".into())
            }
        };
        let _ = job.reply.send(response);
    }
    healthy.store(false, Ordering::Release);
    drop(writes_tx);
    drop(child);
    let _ = reader.join();
    let _ = writer.join();
}

fn receive_frame(
    frames: &mpsc::Receiver<Vec<u8>>,
    timeout: Duration,
    stopped: &AtomicBool,
    reply: Option<&tokio::sync::oneshot::Sender<Result<SpeechOutput, String>>>,
) -> Result<Vec<u8>, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if stopped.load(Ordering::Acquire) || reply.is_some_and(|reply| reply.is_closed()) {
            return Err("speech request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("speech worker timed out".into());
        }
        match frames.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(frame) => return Ok(frame),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => return Err("speech worker closed output".into()),
        }
    }
}

fn validate_ready(frame: &[u8], model: &SpeechModel) -> Result<Capabilities, String> {
    let ready: Ready = serde_json::from_slice(frame).map_err(|_| "invalid speech readiness")?;
    let voices_match = ready.voices.len() == model.voices.len()
        && ready.voices.iter().zip(model.voices).all(|(a, b)| a == b);
    if ready.frame_type != "ready"
        || ready.protocol != PROTOCOL
        || ready.runtime != RUNTIME
        || ready.runtime_revision != RUNTIME_REVISION
        || !model.pins_match(&ready.model_id, &ready.model_revision, &ready.model_sha256)
        || !voices_match
        || ready.output_format != OUTPUT_FORMAT
        || ready.max_input_bytes != MAX_INPUT_BYTES
        || ready.max_audio_seconds != MAX_AUDIO_SECONDS
        || !(0.8..=10.0).contains(&ready.smoke_seconds)
    {
        return Err("speech worker identity or smoke mismatch".into());
    }
    Ok(Capabilities {
        kind: "speech".into(),
        speech_ready: true,
        speech_protocol: PROTOCOL.into(),
        speech_models: vec![model.id.into()],
        speech_runtime: RUNTIME.into(),
        runtime_revision: ready.runtime_revision,
        model_revision: ready.model_revision,
        model_sha256: ready.model_sha256,
        model_repository: model.repository.into(),
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        model_license: model.license.into(),
        voices: model.voices.iter().map(|v| (*v).to_string()).collect(),
        max_input_bytes: MAX_INPUT_BYTES,
        max_audio_seconds: MAX_AUDIO_SECONDS,
        output_format: OUTPUT_FORMAT.into(),
        input_envelope_encodings: vec!["base64".into()],
        media_transport: vec!["inline-base64".into()],
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
) -> Result<SpeechOutput, WorkerFailure> {
    use base64::Engine;
    let text_sha256 = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(&job.request.text))
    };
    let request = serde_json::json!({
        "type": "speak",
        "request_id": job.id,
        "protocol": PROTOCOL,
        "text_b64": base64::engine::general_purpose::STANDARD.encode(&job.request.text),
        "text_sha256": text_sha256,
        "voice": job.request.voice,
        "speed": job.request.speed,
    });
    let mut bytes = serde_json::to_vec(&request).map_err(|_| "invalid speech request")?;
    if bytes.len() > MAX_WORKER_INPUT_FRAME {
        return Err("speech request exceeds frame limit".into());
    }
    bytes.push(b'\n');
    let deadline = Instant::now() + timeout;
    let (ack_tx, ack_rx) = mpsc::sync_channel(1);
    writes
        .try_send((bytes, ack_tx))
        .map_err(|_| "speech worker input unavailable")?;
    loop {
        if stopped.load(Ordering::Acquire) || job.reply.is_closed() {
            return Err("speech request cancelled".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("speech worker input timed out".into());
        }
        match ack_rx.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(true) => break,
            Ok(false) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("speech worker input closed".into())
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
        serde_json::from_slice(&frame).map_err(|_| "invalid speech worker frame")?;
    if value.get("type").and_then(|v| v.as_str()) == Some("request_error") {
        let error: RequestError =
            serde_json::from_value(value).map_err(|_| "invalid speech request-error frame")?;
        if error.frame_type != "request_error"
            || error.request_id != job.id
            || error.code != "invalid_input"
        {
            return Err("speech request-error mismatch".into());
        }
        return Err(WorkerFailure::InvalidInput);
    }
    let result: Response =
        serde_json::from_slice(&frame).map_err(|_| "invalid speech worker output")?;
    if result.frame_type != "result" || result.request_id != job.id {
        return Err("speech worker response mismatch".into());
    }
    validate_output(result).map_err(WorkerFailure::Fatal)
}

fn validate_output(result: Response) -> Result<SpeechOutput, String> {
    if !result.duration_seconds.is_finite()
        || result.duration_seconds <= 0.0
        || result.duration_seconds > MAX_AUDIO_SECONDS as f64
    {
        return Err("speech duration out of bounds".into());
    }
    if result.bytes == 0 || result.bytes > MAX_AUDIO_BYTES {
        return Err("speech audio size out of bounds".into());
    }
    use base64::Engine;
    let audio = base64::engine::general_purpose::STANDARD
        .decode(&result.audio_b64)
        .map_err(|_| "speech audio base64 invalid")?;
    let digest = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(&audio))
    };
    if audio.len() != result.bytes || digest != result.sha256 {
        return Err("speech audio contradicts its digest".into());
    }
    // An m4a file starts with an `ftyp` box.
    if audio.len() < 12 || &audio[4..8] != b"ftyp" {
        return Err("speech audio is not an MP4 container".into());
    }
    Ok(SpeechOutput {
        audio,
        duration_seconds: result.duration_seconds,
        sha256: digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RID: &str = "3f1c2a54-9b7e-4d21-8c3a-0f6e5d4c3b2a";

    fn canonical(text: &str, voice: &str, speed: &str) -> serde_json::Value {
        use base64::Engine;
        serde_json::json!({
            "protocol": PROTOCOL, "request_id": RID, "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "voice": voice, "speed": speed,
            "input": {"encoding": "base64", "bytes": text.len(),
                      "data": base64::engine::general_purpose::STANDARD.encode(text)},
        })
    }

    fn binding(bytes: u64) -> ReservationBinding {
        ReservationBinding {
            protocol: PROTOCOL.into(),
            request_id: RID.into(),
            model: MODEL_ID.into(),
            model_revision: MODEL_REVISION.into(),
            model_sha256: MODEL_SHA256.into(),
            voice: "af_heart".into(),
            speed: "1.00".into(),
            input_bytes: bytes,
        }
    }

    #[test]
    fn embedded_manifest_matches_pins() {
        let manifest = embedded_manifest(&MODELS[0]).expect("manifest matches its pins");
        assert_eq!(manifest.files.len(), 2 + VOICES.len());
        for voice in VOICES {
            assert!(manifest
                .files
                .iter()
                .any(|f| f.path == format!("voices/{voice}.safetensors")));
        }
    }

    fn chatterbox_dispatch(voice: &str, speed: &str, sha: &str) -> serde_json::Value {
        let m = &MODELS[1];
        serde_json::json!({"enc": {}, "speech_reservation": {
            "protocol": PROTOCOL, "request_id": RID, "model": m.id,
            "model_revision": m.revision, "model_sha256": sha,
            "voice": voice, "speed": speed, "input_bytes": 12}})
    }

    #[test]
    fn registry_keeps_kokoro_first_and_pins_every_manifest() {
        assert_eq!(MODELS[0].id, MODEL_ID);
        assert_eq!(MODELS[0].sha256, MODEL_SHA256);
        assert_eq!(MODELS[0].voices, VOICES);
        for m in &MODELS {
            let manifest = embedded_manifest(m).expect("manifest matches its pins");
            assert!(!manifest.files.is_empty());
        }
        let turbo = embedded_manifest(&MODELS[1]).unwrap();
        assert_eq!(turbo.files.len(), 9);
        assert!(turbo.files.iter().any(|f| f.path == "conds.safetensors"));
        assert_eq!(
            model_for_name("Chatterbox-Turbo").map(|m| m.id),
            Some("chatterbox-turbo")
        );
        assert!(is_speech_model("kokoro-82m") && !is_speech_model("whisper-1"));
    }

    #[test]
    fn chatterbox_binding_is_its_own_pins_voice_and_fixed_speed() {
        let sha = MODELS[1].sha256;
        assert!(
            ReservationBinding::parse_dispatch(&chatterbox_dispatch("aria", "1.00", sha)).is_ok()
        );
        assert!(
            ReservationBinding::parse_dispatch(&chatterbox_dispatch("aria", "1.25", sha)).is_err()
        );
        assert!(
            ReservationBinding::parse_dispatch(&chatterbox_dispatch("af_heart", "1.00", sha))
                .is_err()
        );
        // Mixed pins (chatterbox id with Kokoro's manifest digest) never pass.
        assert!(ReservationBinding::parse_dispatch(&chatterbox_dispatch(
            "aria",
            "1.00",
            MODEL_SHA256
        ))
        .is_err());
        // Kokoro voices stay Kokoro-only and keep their speed range.
        let mut kokoro = chatterbox_dispatch("aria", "1.50", MODEL_SHA256);
        kokoro["speech_reservation"]["model"] = MODEL_ID.into();
        kokoro["speech_reservation"]["model_revision"] = MODEL_REVISION.into();
        assert!(ReservationBinding::parse_dispatch(&kokoro).is_err());
        kokoro["speech_reservation"]["voice"] = "af_heart".into();
        assert!(ReservationBinding::parse_dispatch(&kokoro).is_ok());
    }

    #[test]
    fn result_header_names_the_model_that_spoke() {
        // Regression: v1.10.5 stamped every result with Kokoro's pins, so clients rejected
        // chatterbox-turbo audio as a binding mismatch (caught by the live private canary).
        for m in &MODELS {
            let mut wire = canonical("Hi there.", m.voices[0], "1.00");
            wire["model"] = m.id.into();
            wire["model_revision"] = m.revision.into();
            wire["model_sha256"] = m.sha256.into();
            let request = SpeechRequest::parse(&wire).unwrap();
            let output = SpeechOutput {
                audio: vec![1, 2, 3],
                duration_seconds: 1.0,
                sha256: "ab".repeat(32),
            };
            let plain = result_plaintext("job-1", &request, &output);
            let line = plain.split(|b| *b == b'\n').next().unwrap();
            let header: serde_json::Value = serde_json::from_slice(line).unwrap();
            assert_eq!(header["model"], m.id);
            assert_eq!(header["model_revision"], m.revision);
            assert_eq!(header["model_sha256"], m.sha256);
            assert_eq!(header["voice"], m.voices[0]);
        }
    }

    #[test]
    fn readiness_must_name_the_configured_model() {
        let m = &MODELS[1];
        let ready = serde_json::json!({
            "type": "ready", "protocol": PROTOCOL, "runtime": RUNTIME,
            "runtime_revision": RUNTIME_REVISION, "model_id": m.id,
            "model_revision": m.revision, "model_sha256": m.sha256,
            "voices": m.voices, "output_format": OUTPUT_FORMAT,
            "max_input_bytes": MAX_INPUT_BYTES, "max_audio_seconds": MAX_AUDIO_SECONDS,
            "smoke_seconds": 2.0});
        let bytes = serde_json::to_vec(&ready).unwrap();
        let cap = validate_ready(&bytes, m).unwrap();
        assert_eq!(cap.speech_models, vec!["chatterbox-turbo".to_string()]);
        assert_eq!(cap.voices, vec!["aria".to_string()]);
        assert_eq!(cap.model_license, "MIT");
        // A worker reporting another model than the one configured is refused.
        assert!(validate_ready(&bytes, &MODELS[0]).is_err());
    }

    #[test]
    fn speed_format_is_strict() {
        for ok in ["0.50", "1.00", "1.25", "2.00"] {
            assert!(valid_speed(ok), "{ok}");
        }
        for bad in ["0.49", "2.01", "1", "1.0", "1.000", "a.bc", "-1.0", " 1.0"] {
            assert!(!valid_speed(bad), "{bad}");
        }
    }

    #[test]
    fn request_parse_enforces_bounds_and_binding() {
        let request = SpeechRequest::parse(&canonical("Hello there.", "af_heart", "1.00")).unwrap();
        assert!(binding(12).matches(&request));
        assert!(
            !binding(11).matches(&request),
            "byte count must bind exactly"
        );
        let mut other_voice = binding(12);
        other_voice.voice = "am_puck".into();
        assert!(!other_voice.matches(&request));

        let long = "a".repeat(MAX_INPUT_BYTES + 1);
        assert!(SpeechRequest::parse(&canonical(&long, "af_heart", "1.00")).is_err());
        assert!(SpeechRequest::parse(&canonical("   ", "af_heart", "1.00")).is_err());
        assert!(SpeechRequest::parse(&canonical("hi\0", "af_heart", "1.00")).is_err());
        assert!(SpeechRequest::parse(&canonical("hi", "alloy", "1.00")).is_err());
        assert!(SpeechRequest::parse(&canonical("hi", "af_heart", "3.00")).is_err());

        let mut lying = canonical("Hello", "af_heart", "1.00");
        lying["input"]["bytes"] = serde_json::json!(4);
        assert!(
            SpeechRequest::parse(&lying).is_err(),
            "declared bytes must equal decoded bytes"
        );
        let mut extra = canonical("Hello", "af_heart", "1.00");
        extra["instructions"] = serde_json::json!("x");
        assert!(
            SpeechRequest::parse(&extra).is_err(),
            "unknown fields rejected"
        );
        let mut wrong_model = canonical("Hello", "af_heart", "1.00");
        wrong_model["model_sha256"] = serde_json::json!("0".repeat(64));
        assert!(SpeechRequest::parse(&wrong_model).is_err());
        // Multi-byte UTF-8 counts bytes, not characters.
        let accented = SpeechRequest::parse(&canonical("café", "af_heart", "1.00")).unwrap();
        assert_eq!(accented.text.len(), 5);
    }

    #[test]
    fn dispatch_requires_exact_shape() {
        let good = serde_json::json!({"enc": {}, "speech_reservation": {
            "protocol": PROTOCOL, "request_id": RID, "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "voice": "af_heart", "speed": "1.00", "input_bytes": 12}});
        assert_eq!(
            ReservationBinding::parse_dispatch(&good).unwrap(),
            binding(12)
        );
        let mut extra = good.clone();
        extra["transcription_reservation"] = serde_json::json!({});
        assert!(ReservationBinding::parse_dispatch(&extra).is_err());
        let mut zero = good.clone();
        zero["speech_reservation"]["input_bytes"] = serde_json::json!(0);
        assert!(ReservationBinding::parse_dispatch(&zero).is_err());
        let mut over = good.clone();
        over["speech_reservation"]["input_bytes"] = serde_json::json!(MAX_INPUT_BYTES + 1);
        assert!(ReservationBinding::parse_dispatch(&over).is_err());
        let mut upper = good;
        upper["speech_reservation"]["request_id"] = serde_json::json!(RID.to_ascii_uppercase());
        assert!(ReservationBinding::parse_dispatch(&upper).is_err());
    }

    #[test]
    fn result_plaintext_is_header_line_then_audio() {
        let request = SpeechRequest::parse(&canonical("Hello there.", "af_heart", "1.00")).unwrap();
        let audio = b"\0\0\0\x18ftypM4A \0\0\0\0rest".to_vec();
        let output = SpeechOutput {
            sha256: {
                use sha2::{Digest, Sha256};
                hex::encode(Sha256::digest(&audio))
            },
            audio: audio.clone(),
            duration_seconds: 1.5,
        };
        let bytes = result_plaintext("job-1", &request, &output);
        let split = bytes.iter().position(|b| *b == b'\n').unwrap();
        let header: serde_json::Value = serde_json::from_slice(&bytes[..split]).unwrap();
        assert_eq!(&bytes[split + 1..], audio.as_slice());
        assert_eq!(header["object"], "speech");
        assert_eq!(header["job_id"], "job-1");
        assert_eq!(header["input_bytes"], 12);
        assert_eq!(header["audio"]["bytes"], audio.len());
        let text = std::str::from_utf8(&bytes[..split]).unwrap();
        assert!(text.starts_with(r#"{"object":"speech","protocol":"speech-v1","request_id":"#));
    }

    #[test]
    fn output_validation_rejects_lies() {
        use base64::Engine;
        let audio = b"\0\0\0\x18ftypM4A \0\0\0\0".to_vec();
        let sha = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(&audio))
        };
        let make = |bytes: usize, sha: &str, data: &[u8], seconds: f64| Response {
            frame_type: "result".into(),
            request_id: 1,
            duration_seconds: seconds,
            bytes,
            sha256: sha.into(),
            audio_b64: base64::engine::general_purpose::STANDARD.encode(data),
        };
        assert!(validate_output(make(audio.len(), &sha, &audio, 1.0)).is_ok());
        assert!(validate_output(make(audio.len(), &"0".repeat(64), &audio, 1.0)).is_err());
        assert!(validate_output(make(audio.len() + 1, &sha, &audio, 1.0)).is_err());
        assert!(validate_output(make(audio.len(), &sha, &audio, 91.0)).is_err());
        assert!(validate_output(make(audio.len(), &sha, &audio, f64::NAN)).is_err());
        let not_mp4 = b"RIFF0000WAVEfmt ".to_vec();
        let not_mp4_sha = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(&not_mp4))
        };
        assert!(validate_output(make(not_mp4.len(), &not_mp4_sha, &not_mp4, 1.0)).is_err());
    }

    #[test]
    fn worker_env_is_scrubbed() {
        for name in [
            "DYLD_INSERT_LIBRARIES",
            "PYTHONPATH",
            "LD_PRELOAD",
            "MLX_METAL_PATH",
        ] {
            assert!(blocked_worker_env(name), "{name}");
        }
        assert!(!blocked_worker_env("HOME"));
    }

    fn ready_frame() -> serde_json::Value {
        serde_json::json!({
            "type": "ready", "protocol": PROTOCOL, "runtime": RUNTIME,
            "runtime_revision": RUNTIME_REVISION, "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
            "voices": VOICES, "output_format": OUTPUT_FORMAT,
            "max_input_bytes": MAX_INPUT_BYTES, "max_audio_seconds": MAX_AUDIO_SECONDS,
            "smoke_seconds": 2.35,
        })
    }

    #[test]
    fn readiness_must_match_every_pin() {
        let ok = serde_json::to_vec(&ready_frame()).unwrap();
        let cap = validate_ready(&ok, &MODELS[0]).unwrap();
        assert!(cap.speech_ready);
        assert_eq!(cap.kind, "speech");
        let unbound = cap.clone().bind_heartbeat(1, "pk", None, None);
        assert!(
            !unbound.speech_ready && unbound.free_slots == 0,
            "no key signature, no routing"
        );
        let bound = cap.bind_heartbeat(4, "pk", Some("sig"), Some(1));
        assert!(bound.speech_ready && bound.free_slots == 1);
        for (field, value) in [
            ("model_sha256", serde_json::json!("0".repeat(64))),
            ("runtime_revision", serde_json::json!("mlx-audio==0.6.0")),
            ("voices", serde_json::json!(["af_heart"])),
            ("smoke_seconds", serde_json::json!(0.1)),
        ] {
            let mut bad = ready_frame();
            bad[field] = value;
            assert!(
                validate_ready(&serde_json::to_vec(&bad).unwrap(), &MODELS[0]).is_err(),
                "{field}"
            );
        }
    }

    /// A fake worker exercises the supervisor end to end without the real model.
    fn fake_worker(script: &str) -> WorkerConfig {
        WorkerConfig {
            model: &MODELS[0],
            program: "python3".into(),
            args: vec!["-I".into(), "-u".into(), "-c".into(), script.into()],
            startup_timeout: Duration::from_secs(20),
            request_timeout: Duration::from_secs(20),
        }
    }

    fn fake_script(mode: &str) -> String {
        format!(
            r#"
import base64, hashlib, json, sys
ready = {ready}
print(json.dumps(ready), flush=True)
audio = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00" + b"x" * 100
for line in sys.stdin:
    f = json.loads(line)
    rid = f["request_id"]
    if "{mode}" == "invalid":
        print(json.dumps({{"type":"request_error","request_id":rid,"code":"invalid_input"}}), flush=True)
    elif "{mode}" == "lie":
        print(json.dumps({{"type":"result","request_id":rid,"duration_seconds":1.0,"bytes":5,"sha256":"0"*64,"audio_b64":base64.b64encode(audio).decode()}}), flush=True)
    else:
        print(json.dumps({{"type":"result","request_id":rid,"duration_seconds":1.0,"bytes":len(audio),"sha256":hashlib.sha256(audio).hexdigest(),"audio_b64":base64.b64encode(audio).decode()}}), flush=True)
"#,
            ready = ready_frame()
        )
    }

    #[tokio::test]
    async fn supervisor_speaks_rejects_and_fails_closed() {
        let request = SpeechRequest::parse(&canonical("Hello there.", "af_heart", "1.00")).unwrap();

        let engine = TtsEngine::start(fake_worker(&fake_script("ok")))
            .await
            .unwrap();
        let out = engine.speak(request.clone()).await.unwrap();
        assert_eq!(&out.audio[4..8], b"ftyp");
        // Same request id again is a replay.
        assert_eq!(
            engine.speak(request.clone()).await.unwrap_err(),
            "speech_input_invalid"
        );
        engine.stop();

        let engine = TtsEngine::start(fake_worker(&fake_script("invalid")))
            .await
            .unwrap();
        assert_eq!(
            engine.speak(request.clone()).await.unwrap_err(),
            "speech_input_invalid"
        );
        assert!(engine.is_healthy(), "client mistakes keep the node ready");
        engine.stop();

        let engine = TtsEngine::start(fake_worker(&fake_script("lie")))
            .await
            .unwrap();
        assert_eq!(
            engine.speak(request).await.unwrap_err(),
            "speech_runtime_failed"
        );
        assert!(!engine.is_healthy(), "a lying worker is de-advertised");
        assert!(engine.capabilities().is_none());
    }

    #[tokio::test]
    async fn bad_readiness_never_starts() {
        let engine = TtsEngine::start(fake_worker(
            "import json; print(json.dumps({'type':'ready'}), flush=True)",
        ))
        .await;
        assert!(engine.is_err());
    }

    /// Real model on this Mac. Run with:
    /// SGL_TTS_CANARY_PYTHON=... SGL_TTS_CANARY_MODEL_DIR=... [SGL_TTS_CANARY_MODEL=chatterbox-turbo] cargo test --lib tts::tests::real_canary -- --ignored
    #[tokio::test]
    #[ignore]
    async fn real_canary() {
        let python = std::env::var("SGL_TTS_CANARY_PYTHON").expect("SGL_TTS_CANARY_PYTHON");
        let model = std::env::var("SGL_TTS_CANARY_MODEL_DIR").expect("SGL_TTS_CANARY_MODEL_DIR");
        // SGL_TTS_CANARY_MODEL=chatterbox-turbo picks another registry entry (default Kokoro).
        let spec = std::env::var("SGL_TTS_CANARY_MODEL")
            .ok()
            .and_then(|id| model_by_id(&id))
            .unwrap_or(&MODELS[0]);
        let engine = TtsEngine::production(
            spec,
            std::path::Path::new(&model),
            std::path::Path::new(&python),
        )
        .await
        .expect("production worker starts");
        let mut wire = canonical(
            if spec.fixed_speed {
                "This sentence was spoken by a node on the grid. [laugh] Really, it was."
            } else {
                "This sentence was spoken by a node on the Singularity Layer grid."
            },
            if spec.id == MODEL_ID {
                "am_michael"
            } else {
                spec.voices[0]
            },
            "1.00",
        );
        wire["model"] = spec.id.into();
        wire["model_revision"] = spec.revision.into();
        wire["model_sha256"] = spec.sha256.into();
        let request = SpeechRequest::parse(&wire).unwrap();
        let out = engine.speak(request.clone()).await.unwrap();
        assert!(out.duration_seconds > 2.0 && out.audio.len() > 10_000);
        if let Ok(path) = std::env::var("SGL_TTS_CANARY_OUT") {
            std::fs::write(path, &out.audio).unwrap();
        }
        let plaintext = result_plaintext("job-canary", &request, &out);
        assert!(plaintext.len() > out.audio.len());
        engine.stop();
    }
}
