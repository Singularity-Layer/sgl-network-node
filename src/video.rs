//! Grid `video-v1` engine supervisor: text-to-video with the pinned stable-diffusion.cpp `sd-cli`
//! native binary (Wan 2.1 T2V 1.3B or Wan 2.2 TI2V 5B, one model per node) and a pinned ffmpeg
//! run as a separate program for the MP4 encode. Spec: x402Studio
//! `documentation/product/sgl-network/GRID_VIDEO_V1.md` (sections 2 and 5).
//!
//! Shape copied from `image.rs`: everything is checked once at start (engine and encoder sha256,
//! every model file against its pin, the NVIDIA GPU and its memory), then a smoke clip must come
//! out as a valid, labelled MP4 before the node advertises `capabilities.video`. One job at a
//! time. A crash or a timeout de-advertises the engine; a relaunch repeats the startup checks and
//! is bounded by rolling budgets. sd-cli is a one-shot program, so there is no long-lived worker:
//! each job is one supervised child process, killed with its whole process group on deadline or
//! cancel.
//!
//! Self-contained on purpose (no orchestrator/node imports) so the lib target can build it. The
//! job glue (unseal, bindings, progress, upload, signing, completion) lives in `video_job.rs`.
//!
//! FAKE mode (tests only, set through `Runtime::fake`, never from the environment or the CLI)
//! runs a small `/bin/sh` script in place of sd-cli and writes a synthetic MP4 in place of
//! ffmpeg, so `cargo test` never needs a GPU.
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const PROTOCOL: &str = "video-v1";
/// `mode_label()` of the engine and the node-level `capabilities.engine`.
pub const ENGINE_LABEL: &str = "video-worker";
/// Heartbeat telemetry runtime label.
pub const RUNTIME: &str = "sdcpp";
/// `capabilities.video.engine`.
pub const ENGINE_NAME: &str = "sdcpp-cuda";
pub const ENGINE_REVISION: &str = "master-1043-3be2378";
pub const ENGINE_COMMIT: &str = "3be23788fc0c9f35a2c24b1211caae11d4a54dbc";
pub const ENCODER_NAME: &str = "ffmpeg";
/// sha256 of the published Linux x86_64 CUDA `sd-cli` build. None today: no CI build has been
/// published yet, so the operator must pass `--video-engine-sha256` (see `resolve_pin`). When a
/// build is published, its sha256 goes here and an operator value must then match it.
pub const PINNED_ENGINE_SHA256: Option<&str> = None;
/// sha256 of the approved ffmpeg executable. None until the licence choice (Phase 0) picks a
/// build; until then the operator must pass `--video-encoder-sha256`.
pub const PINNED_ENCODER_SHA256: Option<&str> = None;

pub const MAX_PROMPT_BYTES: usize = 2000;
pub const MIN_SECONDS: u32 = 2;
pub const MAX_SECONDS: u32 = 5;
pub const FRAME_MULTIPLE: u32 = 4;
pub const SAMPLER: &str = "euler";
pub const FLOW_SHIFT: f64 = 5.0;
/// The Wan default negative prompt (same text as the desktop catalogue).
pub const NEGATIVE_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
/// Node-side hard caps; the reservation may ask for less, never more (spec 5.4).
pub const MAX_VIDEO_BYTES: u64 = 60 * 1024 * 1024;
pub const MAX_THUMBNAIL_BYTES: u64 = 512 * 1024;
/// Bounds on the reservation's `deadline_seconds`.
pub const MIN_DEADLINE_SECONDS: u64 = 120;
pub const MAX_DEADLINE_SECONDS: u64 = 3600;
/// Busy/no-progress watchdog threshold for a video node. Must exceed the longest node deadline,
/// or the watchdog would restart an engine that is legitimately generating or uploading.
pub const WEDGE_MS: u64 = (MAX_DEADLINE_SECONDS + 600) * 1000;
/// Same tolerance the orchestrator applies to `mvhd` duration (spec 5.4).
pub const DURATION_TOLERANCE_MS: u64 = 500;
pub const CONTENT_TYPE_VIDEO: &str = "video/mp4";
pub const CONTENT_TYPE_THUMBNAIL: &str = "image/webp";

const ENCODE_TIMEOUT: Duration = Duration::from_secs(180);
const THUMBNAIL_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_SMOKE_TIMEOUT: Duration = Duration::from_secs(900);
/// The smoke clip: Wan latents are 16 px (2.1) or 32 px (2.2) per side, so 256x160 is valid for
/// both models and is the smallest landscape size that keeps a real aspect ratio.
pub const SMOKE_WIDTH: u32 = 256;
pub const SMOKE_HEIGHT: u32 = 160;
pub const SMOKE_FRAMES: u32 = 9;
pub const SMOKE_STEPS: u32 = 4;
const SMOKE_SEED: u32 = 1;
const SMOKE_PROMPT: &str = "a red ball rolling slowly across a wooden table, soft daylight";
const SMOKE_JOB_ID: &str = "smoke";
/// Raw sd-cli output (what this sd.cpp revision writes for `-o *.webm`: VP8 in WebM).
const RAW_FILE: &str = "raw.webm";
const MAX_RAW_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const VIDEO_FILE: &str = "video.mp4";
const THUMB_FILE: &str = "thumb.webp";
const VERIFIED_MARKER: &str = ".sgl-video-verified.json";
const MAX_MOOV_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SEGMENT_BYTES: usize = 4096;
const TAIL_BYTES: usize = 8192;

/// Crash-loop guard: at most this many non-timeout relaunches in any rolling hour.
const MAX_RESTARTS_PER_WINDOW: usize = 3;
/// Timeouts have their own budget so a few slow jobs never strand the node de-advertised...
const MAX_TIMEOUT_RELAUNCHES_PER_WINDOW: usize = 6;
/// ...but an engine that passes its smoke clip and then times out on every real job is parked.
const MAX_CONSECUTIVE_TIMEOUTS: u32 = 3;
const RESTART_WINDOW: Duration = Duration::from_secs(60 * 60);

const _: () = assert!(WEDGE_MS > MAX_DEADLINE_SECONDS * 1000);
const _: () = assert!(MIN_DEADLINE_SECONDS < MAX_DEADLINE_SECONDS);

// ── Model table (spec section 2) ─────────────────────────────────────────────────────────────

/// One pinned file: Hugging Face `resolve/<revision>/<path>`, verified by size and sha256.
#[derive(Debug)]
pub struct PinnedFile {
    pub role: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

impl PinnedFile {
    /// Files sit flat in `--video-model-dir` under their base name.
    pub fn file_name(&self) -> &'static str {
        self.path.rsplit('/').next().unwrap_or(self.path)
    }

    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.path
        )
    }
}

#[derive(Debug)]
pub struct VideoModel {
    pub id: &'static str,
    pub fps: u32,
    /// Fixed per model (spec section 2, "tune from Phase 0").
    pub steps: u32,
    pub cfg_scale: f64,
    pub sizes: &'static [(u32, u32)],
    pub min_vram_mb: u64,
    /// Pass `--vae-tiling` when the card has less memory than this (MiB). Phase 0: the 5B
    /// decode peaks at 30 to 33 GB untiled and 9.6 GB tiled; the 1.3B never needs it.
    pub tile_below_vram_mb: Option<u64>,
    pub diffusion: PinnedFile,
    pub vae: PinnedFile,
    pub t5: PinnedFile,
}

impl VideoModel {
    pub fn files(&self) -> [&PinnedFile; 3] {
        [&self.diffusion, &self.t5, &self.vae]
    }

    pub fn size_allowed(&self, width: u32, height: u32) -> bool {
        self.sizes.contains(&(width, height))
    }

    pub fn needs_vae_tiling(&self, vram_mb: u64) -> bool {
        self.tile_below_vram_mb.is_some_and(|limit| vram_mb < limit)
    }
}

const T5: PinnedFile = PinnedFile {
    role: "t5",
    repo: "city96/umt5-xxl-encoder-gguf",
    revision: "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7",
    path: "umt5-xxl-encoder-Q8_0.gguf",
    bytes: 6_043_068_256,
    sha256: "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc",
};

pub static MODELS: [VideoModel; 2] = [
    VideoModel {
        id: "wan2.1-t2v-1.3b",
        fps: 16,
        steps: 20,
        cfg_scale: 5.0,
        sizes: &[(832, 480), (480, 832)],
        min_vram_mb: 15 * 1024,
        tile_below_vram_mb: None,
        diffusion: PinnedFile {
            role: "diffusion",
            repo: "samuelchristlie/Wan2.1-T2V-1.3B-GGUF",
            revision: "5a512b15fc35d1b67a074cfe55a591be9e9ef9b5",
            path: "Wan2.1-T2V-1.3B-Q8_0.gguf",
            bytes: 1_535_768_800,
            sha256: "30a44f695b4275a915810120360d6fd26152ec303c2226b5152ec33a93c380e4",
        },
        vae: PinnedFile {
            role: "vae",
            repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            revision: "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
            path: "split_files/vae/wan_2.1_vae.safetensors",
            bytes: 253_815_318,
            sha256: "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        },
        t5: T5,
    },
    VideoModel {
        id: "wan2.2-ti2v-5b",
        fps: 24,
        steps: 30,
        cfg_scale: 5.0,
        sizes: &[(1280, 704), (704, 1280)],
        min_vram_mb: 22 * 1024,
        tile_below_vram_mb: Some(34 * 1024),
        diffusion: PinnedFile {
            role: "diffusion",
            repo: "QuantStack/Wan2.2-TI2V-5B-GGUF",
            revision: "57437632ddd08bdcbd1508c866aa22e126ed51d2",
            path: "Wan2.2-TI2V-5B-Q8_0.gguf",
            bytes: 5_400_179_040,
            sha256: "57bece983817ab2f957546683bb670f13be7d99022d45674840cd999a050ea8f",
        },
        vae: PinnedFile {
            role: "vae",
            repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            revision: "ee6f4a40737a995bf5818954cfce6d59443b0f04",
            path: "split_files/vae/wan2.2_vae.safetensors",
            bytes: 1_409_400_960,
            sha256: "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
        },
        t5: T5,
    },
];

pub fn model_by_id(id: &str) -> Option<&'static VideoModel> {
    MODELS.iter().find(|m| m.id == id)
}

/// Operator-facing lookup (case-insensitive, like the rest of the CLI).
pub fn is_video_model(name: &str) -> bool {
    MODELS.iter().any(|m| m.id.eq_ignore_ascii_case(name))
}

/// Wan needs 4n+1 frames: `round(seconds*fps/4)*4 + 1` (spec section 2), half rounding up.
pub fn frames_for(seconds: u32, fps: u32) -> u32 {
    let n = (seconds * fps + FRAME_MULTIPLE / 2) / FRAME_MULTIPLE;
    n.max(1) * FRAME_MULTIPLE + 1
}

/// The machine-readable label every delivered file carries (spec 5.6).
pub fn ai_label(model_id: &str, job_id: &str) -> String {
    format!("AI-generated video. Made with {model_id} on the Singularity Layer Grid. Job {job_id}.")
}

fn platform_approved() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
}

pub const PLATFORM_REFUSAL: &str =
    "the video engine runs on Linux x86_64 with an NVIDIA GPU only (Windows support comes later)";

// ── Node-side floor (spec 5.8, ported from the desktop's media.rs) ──────────────────────────

/// A narrow floor: a prompt that clearly pairs a minor with sexual content is refused. It is not
/// a content policy. Adult content, style and themes, and non-sexual depictions of children are
/// never touched. Same word lists and rules as the desktop app's `asks_for_sexual_minor`.
pub fn asks_for_sexual_minor(prompt: &str) -> bool {
    let p = format!(" {} ", prompt.to_lowercase());
    let has = |words: &[&str]| {
        words
            .iter()
            .any(|w| p.contains(&format!(" {w} ")) || p.contains(&format!(" {w}s ")))
    };
    let sexual = has(&[
        "nude",
        "naked",
        "nsfw",
        "sex",
        "sexual",
        "sexy",
        "porn",
        "porno",
        "erotic",
        "topless",
        "explicit",
        "genitals",
        "genitalia",
        "breasts",
        "boobs",
        "nipples",
        "penis",
        "vagina",
        "pussy",
        "fuck",
        "fucking",
        "cum",
        "undressed",
        "undressing",
        "lingerie",
        "fetish",
        "hentai",
        "bondage",
    ]);
    if !sexual {
        return false;
    }
    let minor_word = has(&[
        "child",
        "children",
        "kid",
        "kids",
        "minor",
        "minors",
        "underage",
        "under-age",
        "toddler",
        "infant",
        "baby",
        "preteen",
        "pre-teen",
        "schoolgirl",
        "schoolboy",
        "loli",
        "lolita",
        "shota",
        "cp",
        "tween",
        "prepubescent",
        "juvenile",
    ]) || p.contains("under age")
        || p.contains("little girl")
        || p.contains("little boy");
    // "13 year old", "15yo", "16-year-old" and similar: any age under 18.
    let mut under_18 = false;
    let b = p.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            let n: u32 = p[start..i].parse().unwrap_or(99);
            let rest = p[i..].trim_start_matches(['-', ' ']);
            if n < 18
                && (rest.starts_with("yo") || rest.starts_with("y/o") || rest.starts_with("year"))
            {
                under_18 = true;
            }
        } else {
            i += 1;
        }
    }
    minor_word || under_18
}

// ── Configuration ────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Libx264,
    Libopenh264,
}

impl Codec {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "libx264" => Ok(Self::Libx264),
            "libopenh264" => Ok(Self::Libopenh264),
            _ => Err("--video-encoder-codec must be libx264 or libopenh264".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Libx264 => "libx264",
            Self::Libopenh264 => "libopenh264",
        }
    }
}

/// Everything `sgl start --video-*` (or `sgl video doctor`) passes in.
#[derive(Clone, Debug)]
pub struct VideoConfig {
    pub model: String,
    pub engine: PathBuf,
    pub engine_sha256: Option<String>,
    pub encoder: PathBuf,
    pub encoder_sha256: Option<String>,
    pub model_dir: PathBuf,
    pub codec: Codec,
    /// Private per-job scratch space under the node's config directory (0700).
    pub work_root: PathBuf,
}

/// The engine (or encoder) sha256 that must match the file. A constant pinned in this binary
/// wins; an operator value must then equal it. With no constant, the operator value is required,
/// so no unpinned executable ever runs.
pub fn resolve_pin(
    pinned: Option<&str>,
    operator: Option<&str>,
    flag: &str,
) -> Result<String, String> {
    let operator = operator.map(str::to_ascii_lowercase);
    if let Some(value) = &operator {
        if !is_sha256_hex(value) {
            return Err(format!("{flag} must be 64 hexadecimal characters"));
        }
    }
    match (pinned, operator) {
        (Some(pin), None) => Ok(pin.to_string()),
        (Some(pin), Some(value)) if value == pin => Ok(value),
        (Some(_), Some(_)) => Err(format!(
            "{flag} does not match the sha256 pinned in this sgl build"
        )),
        (None, Some(value)) => Ok(value),
        (None, None) => Err(format!(
            "{flag} is required: this sgl build has no published pin for it, so pass the sha256 of the exact file you run"
        )),
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

// ── GPU ──────────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub driver: String,
    pub cuda: Option<String>,
}

/// First line of `nvidia-smi --query-gpu=name,memory.total,driver_version
/// --format=csv,noheader,nounits` (device 0, the card sd-cli uses): `(name, MiB, driver)`.
pub fn parse_gpu_query(out: &str) -> Option<(String, u64, String)> {
    let line = out.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut parts = line.rsplitn(3, ',');
    let driver = parts.next()?.trim().to_string();
    let vram = parts.next()?.trim().parse::<u64>().ok()?;
    let name = parts.next()?.trim().to_string();
    (!name.is_empty() && !driver.is_empty() && vram > 0).then_some((name, vram, driver))
}

/// `CUDA Version: 12.4` from the plain `nvidia-smi` header.
pub fn parse_cuda_version(out: &str) -> Option<String> {
    let at = out.find("CUDA Version:")?;
    let version: String = out[at + "CUDA Version:".len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    (!version.is_empty()).then_some(version)
}

fn run_with_timeout(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut out = String::new();
        let _ = stdout.read_to_string(&mut out);
        let _ = tx.send(out);
    });
    let out = rx.recv_timeout(timeout).ok();
    let _ = child.kill();
    let _ = child.wait();
    out
}

/// GPU 0 as nvidia-smi reports it, or an operator-readable reason.
pub fn detect_gpu() -> Result<GpuInfo, String> {
    let query = run_with_timeout(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_secs(20),
    )
    .ok_or("no NVIDIA GPU found (nvidia-smi did not answer)")?;
    let (name, vram_mb, driver) =
        parse_gpu_query(&query).ok_or("nvidia-smi output could not be read")?;
    let cuda = run_with_timeout("nvidia-smi", &[], Duration::from_secs(20))
        .and_then(|header| parse_cuda_version(&header));
    Ok(GpuInfo {
        name,
        vram_mb,
        driver,
        cuda,
    })
}

// ── Capability (heartbeat `capabilities.video`, spec 5.1) ──────────────────────────────────

/// Field order is the wire order.
#[derive(Clone, Debug, Serialize)]
pub struct ModelCapability {
    pub id: String,
    pub model_sha256: String,
    pub t5_sha256: String,
    pub vae_sha256: String,
    pub sizes: Vec<[u32; 2]>,
    pub max_seconds: u32,
}

/// Field order is the wire order.
#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub kind: String,
    pub protocol: String,
    pub engine: String,
    pub engine_revision: String,
    pub engine_sha256: Option<String>,
    pub encoder: String,
    pub encoder_sha256: Option<String>,
    pub models: Vec<ModelCapability>,
    pub gpu: GpuInfo,
    pub video_ready: bool,
    pub free_slots: u32,
}

impl Capabilities {
    /// One job at a time: 1 when idle, 0 while a job runs.
    pub fn bind_heartbeat(mut self, free_slots: u32) -> Self {
        self.free_slots = free_slots.min(1);
        self
    }
}

pub fn base_capabilities(
    model: &VideoModel,
    engine_sha256: &str,
    encoder_sha256: &str,
    gpu: GpuInfo,
) -> Capabilities {
    Capabilities {
        kind: "video".into(),
        protocol: PROTOCOL.into(),
        engine: ENGINE_NAME.into(),
        engine_revision: ENGINE_REVISION.into(),
        engine_sha256: Some(engine_sha256.into()),
        encoder: ENCODER_NAME.into(),
        encoder_sha256: Some(encoder_sha256.into()),
        models: vec![ModelCapability {
            id: model.id.into(),
            model_sha256: model.diffusion.sha256.into(),
            t5_sha256: model.t5.sha256.into(),
            vae_sha256: model.vae.sha256.into(),
            sizes: model.sizes.iter().map(|&(w, h)| [w, h]).collect(),
            max_seconds: MAX_SECONDS,
        }],
        gpu,
        video_ready: true,
        free_slots: 1,
    }
}

pub struct ReadinessSnapshot {
    pub available_models: Vec<String>,
    pub capabilities: Option<Capabilities>,
}

// ── Model file verification ──────────────────────────────────────────────────────────────────

/// Streaming sha256 of a file (never the whole file in memory).
pub fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    use sha2::Digest;
    let mut file = std::fs::File::open(path).map_err(|_| "file unavailable".to_string())?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| "file could not be hashed".to_string())?;
        if read == 0 {
            break;
        }
        total += read as u64;
        hasher.update(&buffer[..read]);
    }
    Ok((hex::encode(hasher.finalize()), total))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
struct VerifiedEntry {
    bytes: u64,
    modified_ns: u128,
    sha256: String,
}

fn modified_ns(meta: &std::fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Local paths of the three pinned files.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPaths {
    pub diffusion: PathBuf,
    pub t5: PathBuf,
    pub vae: PathBuf,
}

/// The state of one pinned file, for `sgl video doctor` and startup.
#[derive(Debug, PartialEq)]
pub enum FileState {
    Verified,
    Missing,
    Mismatch(&'static str),
}

/// Check one pinned file: a regular file (never a symlink, so the bytes hashed are the bytes
/// loaded), exact size, exact sha256. A verified (size, mtime) is remembered in
/// `.sgl-video-verified.json` in the model folder so a restart does not re-hash ~13 GB.
fn check_file(
    dir: &Path,
    pin: &PinnedFile,
    cache: &mut std::collections::BTreeMap<String, VerifiedEntry>,
) -> FileState {
    let path = dir.join(pin.file_name());
    let Ok(meta) = std::fs::symlink_metadata(&path) else {
        return FileState::Missing;
    };
    if !meta.is_file() {
        return FileState::Mismatch("not a regular file (symlinks are refused)");
    }
    if meta.len() != pin.bytes {
        return FileState::Mismatch("size does not match its pin");
    }
    let stamp = modified_ns(&meta);
    if cache.get(pin.file_name()).is_some_and(|e| {
        e.bytes == pin.bytes && e.modified_ns == stamp && stamp != 0 && e.sha256 == pin.sha256
    }) {
        return FileState::Verified;
    }
    match sha256_file(&path) {
        Ok((sha, bytes)) if sha == pin.sha256 && bytes == pin.bytes => {
            cache.insert(
                pin.file_name().to_string(),
                VerifiedEntry {
                    bytes,
                    modified_ns: stamp,
                    sha256: sha,
                },
            );
            FileState::Verified
        }
        Ok(_) => FileState::Mismatch("sha256 does not match its pin"),
        Err(_) => FileState::Mismatch("could not be read"),
    }
}

/// Verify every pinned file of `model` in `dir`. Returns each file's state (for the doctor) and
/// the local paths when all are verified.
pub fn verify_model_files(
    dir: &Path,
    model: &VideoModel,
) -> (Vec<(&'static str, FileState)>, Option<ModelPaths>) {
    let marker = dir.join(VERIFIED_MARKER);
    let mut cache: std::collections::BTreeMap<String, VerifiedEntry> = std::fs::read(&marker)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let before = cache.clone();
    let states: Vec<(&'static str, FileState)> = model
        .files()
        .iter()
        .map(|pin| (pin.file_name(), check_file(dir, pin, &mut cache)))
        .collect();
    if cache != before {
        if let Ok(bytes) = serde_json::to_vec(&cache) {
            let tmp = dir.join(format!("{VERIFIED_MARKER}.tmp"));
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &marker);
            }
        }
    }
    let all_ok = states.iter().all(|(_, s)| *s == FileState::Verified);
    let paths = all_ok.then(|| ModelPaths {
        diffusion: dir.join(model.diffusion.file_name()),
        t5: dir.join(model.t5.file_name()),
        vae: dir.join(model.vae.file_name()),
    });
    (states, paths)
}

/// Resolve an executable to its canonical path and check its sha256 against the pin.
pub fn verify_executable(path: &Path, expected: &str, what: &str) -> Result<PathBuf, String> {
    let canonical = std::fs::canonicalize(path).map_err(|_| format!("{what} not found"))?;
    if !std::fs::metadata(&canonical).is_ok_and(|m| m.is_file()) {
        return Err(format!("{what} is not a file"));
    }
    let (sha, _) = sha256_file(&canonical).map_err(|e| format!("{what}: {e}"))?;
    if sha != expected {
        return Err(format!("{what} sha256 does not match its pin"));
    }
    Ok(canonical)
}

// ── Command lines (pure, so they are tested without running anything) ───────────────────────

/// sd-cli arguments. Mirrors the desktop's `engine_args` for a Wan text-to-video row. Every
/// value is its own argv element (no shell), so the prompt can never become a flag.
#[allow(clippy::too_many_arguments)]
pub fn engine_args(
    model: &VideoModel,
    files: &ModelPaths,
    prompt: &str,
    width: u32,
    height: u32,
    frames: u32,
    steps: u32,
    seed: u32,
    vae_tiling: bool,
    out: &Path,
) -> Vec<String> {
    let s = |p: &Path| p.to_string_lossy().to_string();
    let mut a: Vec<String> = vec![
        "-M".into(),
        "vid_gen".into(),
        "--diffusion-model".into(),
        s(&files.diffusion),
        "--t5xxl".into(),
        s(&files.t5),
        "--vae".into(),
        s(&files.vae),
        "-p".into(),
        prompt.to_string(),
        "-n".into(),
        NEGATIVE_PROMPT.into(),
        "--cfg-scale".into(),
        format!("{}", model.cfg_scale),
        "--sampling-method".into(),
        SAMPLER.into(),
        "--steps".into(),
        steps.to_string(),
        "--flow-shift".into(),
        format!("{FLOW_SHIFT}"),
        "-W".into(),
        width.to_string(),
        "-H".into(),
        height.to_string(),
        "--video-frames".into(),
        frames.to_string(),
        "--fps".into(),
        model.fps.to_string(),
        "-s".into(),
        seed.to_string(),
        "--params-backend".into(),
        "te=disk".into(),
        "--flash-attn".into(),
        "on".into(),
        "-o".into(),
        s(out),
    ];
    if vae_tiling {
        a.push("--vae-tiling".into());
    }
    a
}

/// ffmpeg: sd-cli's WebM to MP4 (H.264, yuv420p, faststart) with the AI label as `comment` and
/// `description`. Input metadata is dropped first so only our label is written.
pub fn encode_args(codec: Codec, input: &Path, output: &Path, label: &str) -> Vec<String> {
    let s = |p: &Path| p.to_string_lossy().to_string();
    let mut a: Vec<String> = vec![
        "-hide_banner".into(),
        "-nostdin".into(),
        "-loglevel".into(),
        "error".into(),
        "-y".into(),
        "-i".into(),
        s(input),
        "-map".into(),
        "0:v:0".into(),
        "-an".into(),
        "-map_metadata".into(),
        "-1".into(),
        "-c:v".into(),
        codec.as_str().into(),
    ];
    match codec {
        Codec::Libx264 => a.extend(["-preset", "medium", "-crf", "20"].map(String::from)),
        Codec::Libopenh264 => a.extend(["-b:v", "8M"].map(String::from)),
    }
    a.extend([
        "-pix_fmt".into(),
        "yuv420p".into(),
        "-movflags".into(),
        "+faststart".into(),
        "-metadata".into(),
        format!("comment={label}"),
        "-metadata".into(),
        format!("description={label}"),
        "-f".into(),
        "mp4".into(),
        s(output),
    ]);
    a
}

/// ffmpeg: the middle frame as WebP, at most 512 px wide.
pub fn thumbnail_args(input: &Path, middle_ms: u64, output: &Path) -> Vec<String> {
    let s = |p: &Path| p.to_string_lossy().to_string();
    vec![
        "-hide_banner".into(),
        "-nostdin".into(),
        "-loglevel".into(),
        "error".into(),
        "-y".into(),
        "-ss".into(),
        format!("{}.{:03}", middle_ms / 1000, middle_ms % 1000),
        "-i".into(),
        s(input),
        "-frames:v".into(),
        "1".into(),
        "-vf".into(),
        "scale=w='min(512,iw)':h=-2".into(),
        "-c:v".into(),
        "libwebp".into(),
        "-quality".into(),
        "80".into(),
        "-f".into(),
        "webp".into(),
        s(output),
    ]
}

// ── Progress ─────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Loading,
    Denoising,
    Decoding,
    Encoding,
    Uploading,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Denoising => "denoising",
            Self::Decoding => "decoding",
            Self::Encoding => "encoding",
            Self::Uploading => "uploading",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub stage: Stage,
    pub step: u32,
    pub total: u32,
}

/// Reads sd-cli's output one `\r`/`\n` segment at a time. Phases come from its log lines
/// (`generate_video ...`, `sampling completed`); steps come from the sampler's progress bar
/// (`|====>   |  3/12 - 75.20s/it`), counted only while denoising and only when the total is
/// the job's step count, so model-loading and decoder-tile bars never look like steps.
pub struct ProgressParser {
    in_denoise: bool,
    total_steps: u32,
}

impl ProgressParser {
    pub fn new(total_steps: u32) -> Self {
        Self {
            in_denoise: false,
            total_steps,
        }
    }

    pub fn feed(&mut self, segment: &str) -> Option<Progress> {
        let s: String = segment.chars().filter(|c| !c.is_control()).collect();
        let s = s.trim();
        if s.contains("generate_video ") {
            self.in_denoise = true;
            return Some(self.at(Stage::Denoising, 0));
        }
        if s.contains("sampling completed") {
            self.in_denoise = false;
            return Some(self.at(Stage::Decoding, self.total_steps));
        }
        if !self.in_denoise || !(s.contains("s/it") || s.contains("it/s")) {
            return None;
        }
        for part in s.split('|').flat_map(|p| p.split(" - ")) {
            if let Some((n, t)) = part.trim().split_once('/') {
                if let (Ok(n), Ok(t)) = (n.trim().parse::<u32>(), t.trim().parse::<u32>()) {
                    if t == self.total_steps && n <= t {
                        return Some(self.at(Stage::Denoising, n));
                    }
                }
            }
        }
        None
    }

    fn at(&self, stage: Stage, step: u32) -> Progress {
        Progress {
            stage,
            step,
            total: self.total_steps,
        }
    }
}

/// The engine said the decoder failed: the clip was written but its last frames are noise.
pub fn decoder_failed(log: &str) -> bool {
    log.contains("vae decode compute failed") || log.contains("decode_first_stage failed")
}

// ── MP4 box check (mirrors the orchestrator's check, spec 5.4 and 5.6) ──────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Mp4Info {
    pub major_brand: [u8; 4],
    pub duration_ms: u64,
    pub width: u32,
    pub height: u32,
    /// `moov/udta/meta/ilst/©cmt` (or `moov/udta/©cmt`), lossy UTF-8.
    pub comment: Option<String>,
}

const CMT: [u8; 4] = [0xA9, b'c', b'm', b't'];

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// Child boxes of an in-memory box body: `(type, body)`.
fn boxes(body: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= body.len() {
        let size32 = be32(body, pos)? as u64;
        let kind: [u8; 4] = body[pos + 4..pos + 8].try_into().ok()?;
        let (size, header) = match size32 {
            0 => ((body.len() - pos) as u64, 8usize),
            1 => (be64(body, pos + 8)?, 16usize),
            n => (n, 8usize),
        };
        let end = pos.checked_add(usize::try_from(size).ok()?)?;
        if size < header as u64 || end > body.len() {
            return None;
        }
        out.push((kind, &body[pos + header..end]));
        pos = end;
    }
    (pos == body.len()).then_some(out)
}

fn child<'a>(list: &[([u8; 4], &'a [u8])], kind: &[u8; 4]) -> Option<&'a [u8]> {
    list.iter().find(|(k, _)| k == kind).map(|(_, b)| *b)
}

fn ilst_comment(meta: &[u8]) -> Option<String> {
    // ISO `meta` is a full box (4 bytes of version/flags); QuickTime's is not.
    let inner = if meta.get(..4) == Some(&[0, 0, 0, 0]) {
        &meta[4..]
    } else {
        meta
    };
    let ilst = child(&boxes(inner)?, b"ilst")?;
    let cmt = child(&boxes(ilst)?, &CMT)?;
    let data = child(&boxes(cmt)?, b"data")?;
    Some(String::from_utf8_lossy(data.get(8..)?).into_owned())
}

fn parse_moov(moov: &[u8]) -> Result<(u64, u32, u32, Option<String>), String> {
    let children = boxes(moov).ok_or("moov is malformed")?;
    let mvhd = child(&children, b"mvhd").ok_or("mvhd missing")?;
    let (timescale, duration) = match mvhd.first() {
        Some(0) => (be32(mvhd, 12), be32(mvhd, 16).map(u64::from)),
        Some(1) => (be32(mvhd, 20), be64(mvhd, 24)),
        _ => (None, None),
    };
    let (timescale, duration) = timescale
        .zip(duration)
        .filter(|(t, _)| *t > 0)
        .ok_or("mvhd is malformed")?;
    let duration_ms = duration.saturating_mul(1000) / u64::from(timescale);
    let mut size = None;
    for (_, trak) in children.iter().filter(|(k, _)| k == b"trak") {
        let tkhd = boxes(trak)
            .and_then(|b| child(&b, b"tkhd"))
            .ok_or("tkhd missing")?;
        let base = match tkhd.first() {
            Some(0) => 76,
            Some(1) => 88,
            _ => return Err("tkhd is malformed".into()),
        };
        let w = be32(tkhd, base).ok_or("tkhd is malformed")? >> 16;
        let h = be32(tkhd, base + 4).ok_or("tkhd is malformed")? >> 16;
        if w > 0 && h > 0 {
            if size.is_some() {
                return Err("more than one video track".into());
            }
            size = Some((w, h));
        }
    }
    let (width, height) = size.ok_or("no video track")?;
    let comment = child(&children, b"udta").and_then(|udta| {
        let list = boxes(udta)?;
        child(&list, b"meta").and_then(ilst_comment).or_else(|| {
            // QuickTime style: 2-byte length, 2-byte language, then the text.
            child(&list, &CMT)
                .map(|b| String::from_utf8_lossy(b.get(4..).unwrap_or(&[])).into_owned())
        })
    });
    Ok((duration_ms, width, height, comment))
}

/// Read the top-level boxes of an MP4 file (only `moov` is loaded, capped at 16 MiB).
pub fn inspect_mp4(path: &Path) -> Result<Mp4Info, String> {
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::File::open(path).map_err(|_| "mp4 unavailable")?;
    let len = file.metadata().map_err(|_| "mp4 unavailable")?.len();
    let mut pos = 0u64;
    let mut brand = None;
    let mut moov = None;
    while pos + 8 <= len {
        file.seek(SeekFrom::Start(pos))
            .map_err(|_| "mp4 unreadable")?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header[..8])
            .map_err(|_| "mp4 unreadable")?;
        let size32 = u64::from(u32::from_be_bytes(header[..4].try_into().unwrap()));
        let kind: [u8; 4] = header[4..8].try_into().unwrap();
        let (size, header_len) = match size32 {
            0 => (len - pos, 8u64),
            1 => {
                file.read_exact(&mut header[8..16])
                    .map_err(|_| "mp4 truncated")?;
                (u64::from_be_bytes(header[8..16].try_into().unwrap()), 16u64)
            }
            n => (n, 8u64),
        };
        if size < header_len || pos.checked_add(size).is_none_or(|end| end > len) {
            return Err("mp4 box is truncated".into());
        }
        if pos == 0 && kind != *b"ftyp" {
            return Err("mp4 does not start with ftyp".into());
        }
        match &kind {
            b"ftyp" if pos == 0 => {
                let mut b = [0u8; 4];
                file.read_exact(&mut b).map_err(|_| "ftyp truncated")?;
                brand = Some(b);
            }
            b"moov" => {
                if moov.is_some() {
                    return Err("more than one moov".into());
                }
                let body_len = size - header_len;
                if body_len > MAX_MOOV_BYTES {
                    return Err("moov too large".into());
                }
                let mut body = vec![0u8; body_len as usize];
                file.read_exact(&mut body).map_err(|_| "moov truncated")?;
                moov = Some(body);
            }
            _ => {}
        }
        pos += size;
    }
    if pos != len {
        return Err("mp4 has trailing bytes".into());
    }
    let major_brand = brand.ok_or("ftyp missing")?;
    let (duration_ms, width, height, comment) = parse_moov(&moov.ok_or("moov missing")?)?;
    Ok(Mp4Info {
        major_brand,
        duration_ms,
        width,
        height,
        comment,
    })
}

fn brand_ok(brand: &[u8; 4]) -> bool {
    matches!(brand, b"isom" | b"mp42" | b"avc1") || brand.starts_with(b"iso")
}

/// What a delivered file must be. `duration_ms` is the buyer's `seconds` for a job (the same
/// comparison the orchestrator makes) and `frames/fps` for the smoke clip.
pub struct Expected<'a> {
    pub width: u32,
    pub height: u32,
    pub duration_ms: u64,
    pub job_id: &'a str,
}

pub fn check_mp4(info: &Mp4Info, expected: &Expected) -> Result<(), String> {
    if !brand_ok(&info.major_brand) {
        return Err("mp4 brand not accepted".into());
    }
    if info.duration_ms.abs_diff(expected.duration_ms) > DURATION_TOLERANCE_MS {
        return Err("mp4 duration does not match the request".into());
    }
    if (info.width, info.height) != (expected.width, expected.height) {
        return Err("mp4 size does not match the request".into());
    }
    let label_ok = info
        .comment
        .as_deref()
        .is_some_and(|c| c.contains("AI-generated") && c.contains(expected.job_id));
    if !label_ok {
        return Err("mp4 is missing the AI-generated label".into());
    }
    Ok(())
}

fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// A minimal but structurally valid MP4 (ftyp, moov/mvhd, one video trak/tkhd, optional
/// udta/meta/ilst/©cmt, mdat). Used by FAKE mode in place of ffmpeg and by tests.
pub fn synthetic_mp4(width: u32, height: u32, duration_ms: u32, comment: Option<&str>) -> Vec<u8> {
    let matrix: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
    let matrix: Vec<u8> = matrix.iter().flat_map(|v| v.to_be_bytes()).collect();
    let mut mvhd = vec![0u8; 4];
    for v in [0u32, 0, 1000, duration_ms, 0x0001_0000] {
        mvhd.extend_from_slice(&v.to_be_bytes());
    }
    mvhd.extend_from_slice(&0x0100u16.to_be_bytes());
    mvhd.extend_from_slice(&[0u8; 10]);
    mvhd.extend_from_slice(&matrix);
    mvhd.extend_from_slice(&[0u8; 24]);
    mvhd.extend_from_slice(&2u32.to_be_bytes());
    let mut tkhd = vec![0, 0, 0, 3];
    for v in [0u32, 0, 1, 0, duration_ms] {
        tkhd.extend_from_slice(&v.to_be_bytes());
    }
    tkhd.extend_from_slice(&[0u8; 16]);
    tkhd.extend_from_slice(&matrix);
    tkhd.extend_from_slice(&(width << 16).to_be_bytes());
    tkhd.extend_from_slice(&(height << 16).to_be_bytes());
    let mut moov = mp4_box(b"mvhd", &mvhd);
    moov.extend(mp4_box(b"trak", &mp4_box(b"tkhd", &tkhd)));
    if let Some(text) = comment {
        let mut data = vec![0, 0, 0, 1, 0, 0, 0, 0];
        data.extend_from_slice(text.as_bytes());
        let ilst = mp4_box(b"ilst", &mp4_box(&CMT, &mp4_box(b"data", &data)));
        let mut hdlr = vec![0u8; 8];
        hdlr.extend_from_slice(b"mdirappl");
        hdlr.extend_from_slice(&[0u8; 9]);
        let mut meta = vec![0u8; 4];
        meta.extend(mp4_box(b"hdlr", &hdlr));
        meta.extend(ilst);
        moov.extend(mp4_box(b"udta", &mp4_box(b"meta", &meta)));
    }
    let mut out = mp4_box(b"ftyp", b"isom\0\0\x02\0isomavc1");
    out.extend(mp4_box(b"moov", &moov));
    out.extend(mp4_box(b"mdat", &[0x42u8; 64]));
    out
}

/// RIFF/WEBP magic.
pub fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

fn synthetic_webp() -> Vec<u8> {
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&24u32.to_le_bytes());
    out.extend_from_slice(b"WEBPVP8 ");
    out.extend_from_slice(&[0u8; 20]);
    out
}

// ── FAKE mode (tests) ────────────────────────────────────────────────────────────────────────

// Constructed only by tests (the binary never builds a fake runtime).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FakeMode {
    /// A valid, labelled clip.
    Ok,
    /// The "encoder" writes bytes that are not an MP4.
    Invalid,
    /// A valid MP4 of the wrong size and duration.
    Lie,
    /// sd-cli exits non-zero mid-run.
    Crash,
    /// sd-cli (and a child it started) hang until killed.
    Hang,
    /// A valid MP4 without the AI-generated label.
    NoLabel,
}

#[derive(Clone, Copy, Debug)]
pub struct Fake {
    pub job: FakeMode,
    pub smoke: FakeMode,
}

/// The `/bin/sh` stand-in for sd-cli. It gets the real argument vector, prints progress lines
/// the way sd.cpp does, and writes the `-o` file. `SGL_VIDEO_FAKE_PID_FILE` (test env only)
/// records the pid of the child a hanging run starts, so a test can prove the group was killed.
fn fake_script(mode: FakeMode) -> String {
    let body = match mode {
        FakeMode::Crash => "bar 1; bar 2; echo '[E] ggml_cuda_error: out of memory' >&2; exit 3",
        FakeMode::Hang => {
            "bar 1; sleep 600 & if [ -n \"$SGL_VIDEO_FAKE_PID_FILE\" ]; then echo $! > \"$SGL_VIDEO_FAKE_PID_FILE\"; fi; wait"
        }
        _ => {
            "i=1; while [ $i -le \"$steps\" ]; do bar $i; i=$((i+1)); done; echo '[I] sampling completed, taking 0.01s'; printf 'fake-webm' > \"$out\""
        }
    };
    format!(
        "out=''; steps=1; prev=''; for a in \"$@\"; do \
           if [ \"$prev\" = '-o' ]; then out=\"$a\"; fi; \
           if [ \"$prev\" = '--steps' ]; then steps=\"$a\"; fi; prev=\"$a\"; done; \
         echo '  |##########| 27/27 - 400.1MB/s'; \
         echo '[I] generate_video 256x160x9 --- video.cpp:1679' >&2; \
         bar() {{ printf '  |====>     | %s/%s - 0.01s/it\\r' \"$1\" \"$steps\"; }}; \
         {body}"
    )
}

// ── Engine ───────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
pub enum VideoError {
    /// The request contradicts the model contract (client side; not this node's fault).
    InputInvalid(String),
    /// Stopped on request (`continue:false`); nothing is reported.
    Cancelled,
    /// This node failed.
    RuntimeFailed(String),
}

/// One validated generation, built by `video_job.rs` from the bound reservation.
#[derive(Clone, Debug)]
pub struct VideoRequest {
    pub job_id: String,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub seconds: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub seed: u32,
    pub max_video_bytes: u64,
    pub max_thumbnail_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct Artifact {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct VideoOutput {
    pub video: Artifact,
    pub thumbnail: Option<Artifact>,
    pub duration_ms: u64,
    pub gen_ms: u64,
    pub encode_ms: u64,
}

/// Everything the engine runs with, fixed at start.
#[derive(Clone, Debug)]
pub struct Runtime {
    pub model: &'static VideoModel,
    pub engine: PathBuf,
    pub engine_sha256: String,
    pub encoder: PathBuf,
    pub encoder_sha256: String,
    pub codec: Codec,
    pub files: ModelPaths,
    pub gpu: GpuInfo,
    pub work_root: PathBuf,
    pub smoke_timeout: Duration,
    /// Tests only. Production (`prepare_runtime`) always sets None.
    pub fake: Option<Fake>,
    /// Tests only: extra environment for the fake engine.
    pub extra_env: Vec<(String, String)>,
}

impl Runtime {
    /// A runtime that runs the FAKE engine and encoder. Never used by `sgl start`.
    #[allow(dead_code)]
    pub fn fake(model: &'static VideoModel, work_root: PathBuf, fake: Fake) -> Self {
        let script_sha = {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(fake_script(fake.job).as_bytes()))
        };
        Self {
            model,
            engine: PathBuf::from("/bin/sh"),
            engine_sha256: script_sha.clone(),
            encoder: PathBuf::from("/bin/sh"),
            encoder_sha256: script_sha,
            codec: Codec::Libx264,
            files: ModelPaths {
                diffusion: PathBuf::from("/models").join(model.diffusion.file_name()),
                t5: PathBuf::from("/models").join(model.t5.file_name()),
                vae: PathBuf::from("/models").join(model.vae.file_name()),
            },
            gpu: GpuInfo {
                name: "NVIDIA GeForce RTX 4090".into(),
                vram_mb: 24564,
                driver: "550.54.15".into(),
                cuda: Some("12.4".into()),
            },
            work_root,
            smoke_timeout: Duration::from_secs(30),
            fake: Some(fake),
            extra_env: vec![],
        }
    }
}

/// Every production startup check except the smoke clip: platform, pins, GPU, model files.
pub fn prepare_runtime(cfg: &VideoConfig) -> Result<Runtime, String> {
    if !platform_approved() {
        return Err(PLATFORM_REFUSAL.into());
    }
    let model = model_by_id(&cfg.model).ok_or_else(|| {
        format!(
            "unknown video model {:?}; use one of: {}",
            cfg.model,
            MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")
        )
    })?;
    let engine_sha256 = resolve_pin(
        PINNED_ENGINE_SHA256,
        cfg.engine_sha256.as_deref(),
        "--video-engine-sha256",
    )?;
    let encoder_sha256 = resolve_pin(
        PINNED_ENCODER_SHA256,
        cfg.encoder_sha256.as_deref(),
        "--video-encoder-sha256",
    )?;
    let engine = verify_executable(&cfg.engine, &engine_sha256, "--video-engine")?;
    let encoder = verify_executable(&cfg.encoder, &encoder_sha256, "--video-encoder")?;
    let gpu = detect_gpu()?;
    if gpu.vram_mb < model.min_vram_mb {
        return Err(format!(
            "{} needs at least {} MiB of GPU memory; {} has {} MiB",
            model.id, model.min_vram_mb, gpu.name, gpu.vram_mb
        ));
    }
    if !cfg.model_dir.is_dir() {
        return Err("--video-model-dir is not a folder".into());
    }
    let (states, files) = verify_model_files(&cfg.model_dir, model);
    let files = files.ok_or_else(|| {
        let bad: Vec<String> = states
            .iter()
            .filter(|(_, s)| *s != FileState::Verified)
            .map(|(name, s)| format!("{name}: {s:?}"))
            .collect();
        format!("video model files not verified ({})", bad.join("; "))
    })?;
    Ok(Runtime {
        model,
        engine,
        engine_sha256,
        encoder,
        encoder_sha256,
        codec: cfg.codec,
        files,
        gpu,
        work_root: cfg.work_root.clone(),
        smoke_timeout: DEFAULT_SMOKE_TIMEOUT,
        fake: None,
        extra_env: vec![],
    })
}

/// Private per-job scratch folder (0700), removed with everything in it when dropped.
pub struct JobDir(PathBuf);

impl JobDir {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for JobDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn create_private_dir(path: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Names the engine itself ever creates under the work root: job ids (canonical UUIDs) and
/// `smoke-<hex>`. Nothing else there is touched.
fn owned_entry(name: &str) -> bool {
    crate::image::canonical_uuid_v4(name)
        || name
            .strip_prefix("smoke-")
            .is_some_and(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Rolling-window restart budget. Records the restart when it is allowed.
fn restart_allowed(history: &mut VecDeque<Instant>, now: Instant, max: usize) -> bool {
    while history
        .front()
        .is_some_and(|at| now.duration_since(*at) >= RESTART_WINDOW)
    {
        history.pop_front();
    }
    if history.len() >= max {
        return false;
    }
    history.push_back(now);
    true
}

enum RunError {
    Cancelled,
    TimedOut,
    Failed(String),
}

/// Kills the child's whole process group (sd-cli leads its own) when dropped, so an abandoned
/// job never leaves a GPU process behind.
struct GroupKill(Option<u32>);

impl GroupKill {
    fn kill(&self) {
        #[cfg(unix)]
        if let Some(pid) = self.0 {
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }
}

impl Drop for GroupKill {
    fn drop(&mut self) {
        self.kill();
    }
}

fn blocked_env(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("LD_")
        || name.starts_with("DYLD_")
        || name.starts_with("SGL_VIDEO_FAKE")
        || matches!(name.as_str(), "BASH_ENV" | "ENV" | "GCONV_PATH")
}

fn base_command(program: &Path, args: &[String], cwd: &Path) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    for (name, _) in std::env::vars_os() {
        if blocked_env(&name.to_string_lossy()) {
            cmd.env_remove(name);
        }
    }
    cmd.env("TMPDIR", cwd).env("TMP", cwd).env("TEMP", cwd);
    cmd
}

/// Run one child to completion under a deadline and a cancel signal, feeding every `\r`/`\n`
/// segment of its stdout and stderr to `on_segment`. Returns (exit success, output tail).
async fn run_child(
    mut cmd: tokio::process::Command,
    deadline: Instant,
    cancel: &mut tokio::sync::watch::Receiver<bool>,
    mut on_segment: impl FnMut(&str),
) -> Result<(bool, String), RunError> {
    use tokio::io::AsyncReadExt;
    if *cancel.borrow() {
        return Err(RunError::Cancelled);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| RunError::Failed(format!("could not start: {}", e.kind())))?;
    let group = GroupKill(child.id());
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    let pipes: Vec<Box<dyn tokio::io::AsyncRead + Send + Unpin>> = [
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn tokio::io::AsyncRead + Send + Unpin>),
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn tokio::io::AsyncRead + Send + Unpin>),
    ]
    .into_iter()
    .flatten()
    .collect();
    for mut pipe in pipes {
        let tx = tx.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 8192];
            while let Ok(n) = pipe.read(&mut buf).await {
                if n == 0 || tx.send(buf[..n].to_vec()).await.is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);
    let mut tail = String::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut feed = |bytes: &[u8], tail: &mut String| {
        for &b in bytes {
            if b == b'\r' || b == b'\n' {
                if !pending.is_empty() {
                    let segment = String::from_utf8_lossy(&pending).into_owned();
                    on_segment(&segment);
                    tail.push_str(&segment);
                    tail.push('\n');
                    if tail.len() > TAIL_BYTES * 2 {
                        let cut = tail.len() - TAIL_BYTES;
                        let at = (cut..tail.len())
                            .find(|i| tail.is_char_boundary(*i))
                            .unwrap_or(tail.len());
                        tail.drain(..at);
                    }
                    pending.clear();
                }
            } else if pending.len() < MAX_SEGMENT_BYTES {
                pending.push(b);
            }
        }
    };
    let deadline = tokio::time::Instant::from_std(deadline);
    let mut rx_open = true;
    let mut cancel_open = true;
    let status = loop {
        tokio::select! {
            status = child.wait() => break status,
            chunk = rx.recv(), if rx_open => match chunk {
                Some(bytes) => feed(&bytes, &mut tail),
                None => rx_open = false,
            },
            changed = cancel.changed(), if cancel_open => match changed {
                Ok(()) if *cancel.borrow() => {
                    group.kill();
                    let _ = child.wait().await;
                    return Err(RunError::Cancelled);
                }
                Ok(()) => {}
                Err(_) => cancel_open = false,
            },
            _ = tokio::time::sleep_until(deadline) => {
                group.kill();
                let _ = child.wait().await;
                return Err(RunError::TimedOut);
            }
        }
    };
    // The leader is gone; take anything it left in its group with it, then drain the pipes.
    group.kill();
    while rx_open {
        match tokio::time::timeout(Duration::from_millis(500), rx.recv()).await {
            Ok(Some(bytes)) => feed(&bytes, &mut tail),
            _ => rx_open = false,
        }
    }
    feed(b"\n", &mut tail);
    let status = status.map_err(|_| RunError::Failed("could not wait".into()))?;
    Ok((status.success(), tail))
}

fn file_size(path: &Path) -> Option<u64> {
    std::fs::symlink_metadata(path)
        .ok()
        .filter(|m| m.is_file())
        .map(|m| m.len())
}

/// Releases the one-job slot even if the generating future is dropped.
struct BusyGuard<'a>(&'a AtomicBool);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub struct VideoEngine {
    runtime: Runtime,
    capabilities: Capabilities,
    healthy: AtomicBool,
    stopped: AtomicBool,
    busy: AtomicBool,
    parked: AtomicBool,
    /// Set when the last fault was a deadline timeout.
    timed_out: AtomicBool,
    consecutive_timeouts: AtomicU32,
    restart_lock: tokio::sync::Mutex<()>,
    restart_history: Mutex<VecDeque<Instant>>,
    timeout_history: Mutex<VecDeque<Instant>>,
}

impl VideoEngine {
    /// `sgl start --video-model ...`: every startup check, then the smoke clip.
    pub async fn production(cfg: VideoConfig) -> Result<Self, String> {
        let prepared = tokio::task::spawn_blocking(move || prepare_runtime(&cfg))
            .await
            .map_err(|_| "video startup checks failed".to_string())??;
        Self::start(prepared).await
    }

    /// Start from a prepared runtime: private work folder, smoke clip, then ready.
    pub async fn start(runtime: Runtime) -> Result<Self, String> {
        create_private_dir(&runtime.work_root)
            .map_err(|_| "video work folder could not be created")?;
        if let Ok(entries) = std::fs::read_dir(&runtime.work_root) {
            for entry in entries.flatten() {
                if owned_entry(&entry.file_name().to_string_lossy()) {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
        let capabilities = base_capabilities(
            runtime.model,
            &runtime.engine_sha256,
            &runtime.encoder_sha256,
            runtime.gpu.clone(),
        );
        let engine = Self {
            runtime,
            capabilities,
            healthy: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            parked: AtomicBool::new(false),
            timed_out: AtomicBool::new(false),
            consecutive_timeouts: AtomicU32::new(0),
            restart_lock: tokio::sync::Mutex::new(()),
            restart_history: Mutex::new(VecDeque::new()),
            timeout_history: Mutex::new(VecDeque::new()),
        };
        engine.smoke().await?;
        engine.healthy.store(true, Ordering::Release);
        Ok(engine)
    }

    pub fn model(&self) -> &'static VideoModel {
        self.runtime.model
    }

    pub fn gpu(&self) -> &GpuInfo {
        &self.runtime.gpu
    }

    /// The smoke clip: 256x160, 9 frames, 4 steps, through the real engine and encoder, and it
    /// must pass the same MP4 check a job does.
    async fn smoke(&self) -> Result<(), String> {
        let dir = self
            .job_dir(&format!("smoke-{:016x}", rand::random::<u64>()))
            .map_err(|_| "video smoke folder unavailable")?;
        let request = VideoRequest {
            job_id: SMOKE_JOB_ID.into(),
            prompt: SMOKE_PROMPT.into(),
            width: SMOKE_WIDTH,
            height: SMOKE_HEIGHT,
            seconds: 0,
            frames: SMOKE_FRAMES,
            fps: self.runtime.model.fps,
            steps: SMOKE_STEPS,
            seed: SMOKE_SEED,
            max_video_bytes: MAX_VIDEO_BYTES,
            max_thumbnail_bytes: MAX_THUMBNAIL_BYTES,
        };
        let (progress, _) = tokio::sync::watch::channel(Progress {
            stage: Stage::Loading,
            step: 0,
            total: SMOKE_STEPS,
        });
        let (_cancel_tx, cancel) = tokio::sync::watch::channel(false);
        let mode = self.runtime.fake.map(|f| f.smoke);
        let deadline = Instant::now() + self.runtime.smoke_timeout;
        match self
            .run(
                &request,
                dir.path(),
                &progress,
                cancel,
                deadline,
                mode,
                true,
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("video smoke clip failed: {e:?}")),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && !self.stopped.load(Ordering::Acquire)
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    pub fn capabilities(&self) -> Option<Capabilities> {
        self.is_healthy().then(|| self.capabilities.clone())
    }

    /// `available_models` carries the model id only while the engine is healthy.
    pub fn readiness_snapshot(&self, models: &[String]) -> ReadinessSnapshot {
        let capabilities = self.capabilities();
        ReadinessSnapshot {
            available_models: if capabilities.is_some() {
                models.to_vec()
            } else {
                vec![]
            },
            capabilities,
        }
    }

    /// A fresh private folder for one job under the work root.
    pub fn job_dir(&self, name: &str) -> Result<JobDir, VideoError> {
        if !owned_entry(name) {
            return Err(VideoError::InputInvalid("job id not accepted".into()));
        }
        let path = self.runtime.work_root.join(name);
        let _ = std::fs::remove_dir_all(&path);
        create_private_dir(&path)
            .map_err(|_| VideoError::RuntimeFailed("job folder could not be created".into()))?;
        Ok(JobDir(path))
    }

    fn validate(&self, request: &VideoRequest) -> Result<(), VideoError> {
        let model = self.runtime.model;
        let prompt = &request.prompt;
        let ok = model.size_allowed(request.width, request.height)
            && (MIN_SECONDS..=MAX_SECONDS).contains(&request.seconds)
            && request.fps == model.fps
            && request.frames == frames_for(request.seconds, model.fps)
            && request.steps == model.steps
            && !prompt.trim().is_empty()
            && prompt.len() <= MAX_PROMPT_BYTES
            && !prompt.contains('\0')
            && !prompt.to_ascii_lowercase().contains("<lora:")
            && request.max_video_bytes > 0
            && request.max_video_bytes <= MAX_VIDEO_BYTES
            && request.max_thumbnail_bytes <= MAX_THUMBNAIL_BYTES;
        if ok {
            Ok(())
        } else {
            Err(VideoError::InputInvalid(
                "request contradicts the model contract".into(),
            ))
        }
    }

    /// One clip, one at a time. `deadline` bounds generation and encoding together; the caller
    /// keeps its own margin for the upload. `cancel` flipping to true kills the engine at once.
    pub async fn generate(
        &self,
        request: &VideoRequest,
        dir: &Path,
        progress: &tokio::sync::watch::Sender<Progress>,
        cancel: tokio::sync::watch::Receiver<bool>,
        deadline: Instant,
    ) -> Result<VideoOutput, VideoError> {
        if !self.is_healthy() {
            return Err(VideoError::RuntimeFailed("video engine not ready".into()));
        }
        self.validate(request)?;
        let mode = self.runtime.fake.map(|f| f.job);
        let result = self
            .run(request, dir, progress, cancel, deadline, mode, false)
            .await;
        match &result {
            Ok(_) => {
                self.consecutive_timeouts.store(0, Ordering::Release);
            }
            Err(VideoError::RuntimeFailed(reason)) => {
                if reason == "timed out" {
                    self.timed_out.store(true, Ordering::Release);
                }
                self.healthy.store(false, Ordering::Release);
            }
            Err(_) => {}
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        request: &VideoRequest,
        dir: &Path,
        progress: &tokio::sync::watch::Sender<Progress>,
        mut cancel: tokio::sync::watch::Receiver<bool>,
        deadline: Instant,
        fake: Option<FakeMode>,
        smoke: bool,
    ) -> Result<VideoOutput, VideoError> {
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(VideoError::RuntimeFailed("video engine busy".into()));
        }
        let _busy = BusyGuard(&self.busy);
        let runtime = &self.runtime;
        let failed = |reason: &str| VideoError::RuntimeFailed(reason.to_string());
        let map_run = |e: RunError| match e {
            RunError::Cancelled => VideoError::Cancelled,
            RunError::TimedOut => failed("timed out"),
            RunError::Failed(reason) => VideoError::RuntimeFailed(reason),
        };
        let raw = dir.join(RAW_FILE);
        let mp4 = dir.join(VIDEO_FILE);
        let thumb = dir.join(THUMB_FILE);
        let total = request.steps;
        progress.send_replace(Progress {
            stage: Stage::Loading,
            step: 0,
            total,
        });

        // 1. sd-cli
        let args = engine_args(
            runtime.model,
            &runtime.files,
            &request.prompt,
            request.width,
            request.height,
            request.frames,
            request.steps,
            request.seed,
            runtime.model.needs_vae_tiling(runtime.gpu.vram_mb),
            &raw,
        );
        let mut cmd = match fake {
            Some(mode) => {
                let mut full = vec!["-c".to_string(), fake_script(mode), "sd-cli".into()];
                full.extend(args);
                base_command(Path::new("/bin/sh"), &full, dir)
            }
            None => base_command(&runtime.engine, &args, dir),
        };
        if fake.is_none() {
            if let Some(engine_dir) = runtime.engine.parent() {
                // The CUDA build ships its shared libraries next to sd-cli (as the desktop runs it).
                cmd.env("LD_LIBRARY_PATH", engine_dir);
            }
        }
        for (k, v) in &runtime.extra_env {
            cmd.env(k, v);
        }
        let started = Instant::now();
        let mut parser = ProgressParser::new(total);
        let (ok, tail) = run_child(cmd, deadline, &mut cancel, |segment| {
            if let Some(p) = parser.feed(segment) {
                progress.send_replace(p);
            }
        })
        .await
        .map_err(map_run)?;
        if !ok {
            return Err(failed("engine exited with an error"));
        }
        if decoder_failed(&tail) {
            return Err(failed("engine decoder failed"));
        }
        if !file_size(&raw).is_some_and(|n| n > 0 && n <= MAX_RAW_BYTES) {
            return Err(failed("engine wrote no clip"));
        }
        let gen_ms = started.elapsed().as_millis() as u64;

        // 2. ffmpeg: MP4 with the AI label.
        progress.send_replace(Progress {
            stage: Stage::Encoding,
            step: total,
            total,
        });
        let label = ai_label(runtime.model.id, &request.job_id);
        let encode_started = Instant::now();
        let encode_deadline = deadline.min(Instant::now() + ENCODE_TIMEOUT);
        match fake {
            Some(mode) => {
                let (w, h, ms) = (request.width, request.height, self.clip_ms(request));
                let bytes = match mode {
                    FakeMode::Invalid => b"this is not an mp4 file at all".to_vec(),
                    FakeMode::Lie => synthetic_mp4(h, w, ms as u32 + 3000, Some(&label)),
                    FakeMode::NoLabel => synthetic_mp4(w, h, ms as u32, None),
                    _ => synthetic_mp4(w, h, ms as u32, Some(&label)),
                };
                std::fs::write(&mp4, bytes).map_err(|_| failed("encoder could not write"))?;
            }
            None => {
                let args = encode_args(runtime.codec, &raw, &mp4, &label);
                let cmd = base_command(&runtime.encoder, &args, dir);
                let (ok, _) = run_child(cmd, encode_deadline, &mut cancel, |_| {})
                    .await
                    .map_err(map_run)?;
                if !ok {
                    return Err(failed("encoder exited with an error"));
                }
            }
        }
        let _ = std::fs::remove_file(&raw);
        let encode_ms = encode_started.elapsed().as_millis() as u64;

        // 3. Our own MP4 check, the same one the orchestrator makes, before anything leaves.
        let bytes = file_size(&mp4).ok_or_else(|| failed("encoder wrote no file"))?;
        if bytes == 0 || bytes > request.max_video_bytes {
            return Err(failed("encoded clip exceeds the upload cap"));
        }
        let info = inspect_mp4(&mp4).map_err(VideoError::RuntimeFailed)?;
        let expected_ms = if smoke {
            u64::from(request.frames) * 1000 / u64::from(request.fps)
        } else {
            u64::from(request.seconds) * 1000
        };
        check_mp4(
            &info,
            &Expected {
                width: request.width,
                height: request.height,
                duration_ms: expected_ms,
                job_id: &request.job_id,
            },
        )
        .map_err(VideoError::RuntimeFailed)?;
        let (sha256, hashed) = sha256_file(&mp4).map_err(VideoError::RuntimeFailed)?;
        if hashed != bytes {
            return Err(failed("encoded clip changed while hashing"));
        }

        // 4. Thumbnail (best effort: a clip without one is still delivered).
        let thumbnail = if smoke {
            None
        } else {
            self.thumbnail(
                &mp4,
                &thumb,
                info.duration_ms / 2,
                request,
                &mut cancel,
                deadline,
                fake,
            )
            .await
        };
        if *cancel.borrow() {
            return Err(VideoError::Cancelled);
        }
        Ok(VideoOutput {
            video: Artifact {
                path: mp4,
                bytes,
                sha256,
            },
            thumbnail,
            duration_ms: info.duration_ms,
            gen_ms,
            encode_ms,
        })
    }

    fn clip_ms(&self, request: &VideoRequest) -> u64 {
        u64::from(request.frames) * 1000 / u64::from(request.fps.max(1))
    }

    #[allow(clippy::too_many_arguments)]
    async fn thumbnail(
        &self,
        mp4: &Path,
        thumb: &Path,
        middle_ms: u64,
        request: &VideoRequest,
        cancel: &mut tokio::sync::watch::Receiver<bool>,
        deadline: Instant,
        fake: Option<FakeMode>,
    ) -> Option<Artifact> {
        if fake.is_some() {
            std::fs::write(thumb, synthetic_webp()).ok()?;
        } else {
            let args = thumbnail_args(mp4, middle_ms, thumb);
            let cmd = base_command(&self.runtime.encoder, &args, thumb.parent()?);
            let deadline = deadline.min(Instant::now() + THUMBNAIL_TIMEOUT);
            match run_child(cmd, deadline, cancel, |_| {}).await {
                Ok((true, _)) => {}
                _ => return None,
            }
        }
        let bytes = file_size(thumb)?;
        if bytes == 0 || bytes > request.max_thumbnail_bytes {
            return None;
        }
        let mut head = [0u8; 12];
        std::fs::File::open(thumb)
            .ok()?
            .read_exact(&mut head)
            .ok()?;
        if !is_webp(&head) {
            return None;
        }
        let (sha256, hashed) = sha256_file(thumb).ok()?;
        (hashed == bytes).then(|| Artifact {
            path: thumb.to_path_buf(),
            bytes,
            sha256,
        })
    }

    /// Relaunch after a fault: repeats the executable pin checks and the smoke clip. A deadline
    /// timeout spends the timeout budget (6 per rolling hour) and parks the engine after 3 in a
    /// row with no good clip in between; any other fault spends the crash budget (3 per hour).
    pub async fn restart(&self) -> Result<(), String> {
        let _guard = self.restart_lock.lock().await;
        if self.parked.load(Ordering::Acquire) {
            return Err("video engine parked: it timed out on every job".into());
        }
        if self.is_busy() {
            return Err("video engine busy".into());
        }
        let after_timeout = self.timed_out.swap(false, Ordering::AcqRel);
        let (history, max) = if after_timeout {
            let consecutive = self.consecutive_timeouts.fetch_add(1, Ordering::AcqRel) + 1;
            if consecutive >= MAX_CONSECUTIVE_TIMEOUTS {
                self.parked.store(true, Ordering::Release);
                self.stop();
                return Err("video engine parked: it timed out on every job".into());
            }
            (&self.timeout_history, MAX_TIMEOUT_RELAUNCHES_PER_WINDOW)
        } else {
            (&self.restart_history, MAX_RESTARTS_PER_WINDOW)
        };
        if !restart_allowed(
            &mut history.lock().unwrap_or_else(|p| p.into_inner()),
            Instant::now(),
            max,
        ) {
            return Err("video engine restart budget exhausted for this hour".into());
        }
        self.healthy.store(false, Ordering::Release);
        if self.runtime.fake.is_none() {
            let runtime = self.runtime.clone();
            tokio::task::spawn_blocking(move || {
                verify_executable(&runtime.engine, &runtime.engine_sha256, "--video-engine")?;
                verify_executable(&runtime.encoder, &runtime.encoder_sha256, "--video-encoder")
            })
            .await
            .map_err(|_| "video restart checks failed".to_string())??;
        }
        self.smoke().await?;
        self.stopped.store(false, Ordering::Release);
        self.healthy.store(true, Ordering::Release);
        Ok(())
    }

    pub fn stop(&self) {
        self.healthy.store(false, Ordering::Release);
        self.stopped.store(true, Ordering::Release);
    }
}

// ── `sgl video doctor` ───────────────────────────────────────────────────────────────────────

/// Every check `sgl start --video-*` makes, printed one line each, plus the smoke clip when
/// everything before it passed. Returns (lines, ready).
pub async fn doctor(cfg: &VideoConfig) -> (Vec<String>, bool) {
    let mut lines = Vec::new();
    let mut ready = true;
    let mut note = |text: String, good: bool| {
        ready &= good;
        lines.push(text);
    };
    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    if platform_approved() {
        note(format!("platform      {platform}: supported"), true);
    } else {
        note(
            format!("platform      {platform}: NOT supported ({PLATFORM_REFUSAL})"),
            false,
        );
    }
    let model = model_by_id(&cfg.model);
    match model {
        Some(m) => note(
            format!(
                "model         {} (needs {} MiB GPU memory, {} fps, {} steps)",
                m.id, m.min_vram_mb, m.fps, m.steps
            ),
            true,
        ),
        None => note(format!("model         {:?}: unknown", cfg.model), false),
    }
    let gpu = tokio::task::spawn_blocking(detect_gpu)
        .await
        .unwrap_or_else(|_| Err("gpu check failed".into()));
    match gpu {
        Ok(gpu) => {
            note(format!("gpu           {}", gpu.name), true);
            note(
                format!(
                    "driver        {}   CUDA {}",
                    gpu.driver,
                    gpu.cuda.as_deref().unwrap_or("unknown")
                ),
                true,
            );
            let enough = model.is_none_or(|m| gpu.vram_mb >= m.min_vram_mb);
            note(
                format!(
                    "vram          {} MiB{}",
                    gpu.vram_mb,
                    if enough {
                        ""
                    } else {
                        " (NOT enough for this model)"
                    }
                ),
                enough,
            );
        }
        Err(e) => note(format!("gpu           {e}"), false),
    }
    note(
        format!("engine build  {ENGINE_NAME} {ENGINE_REVISION} (commit {ENGINE_COMMIT})"),
        true,
    );
    let executables = [
        (
            &cfg.engine,
            cfg.engine_sha256.as_deref(),
            PINNED_ENGINE_SHA256,
            "--video-engine-sha256",
            "engine ",
        ),
        (
            &cfg.encoder,
            cfg.encoder_sha256.as_deref(),
            PINNED_ENCODER_SHA256,
            "--video-encoder-sha256",
            "encoder",
        ),
    ];
    for (path, operator, pinned, flag, what) in executables {
        let actual = std::fs::canonicalize(path)
            .ok()
            .and_then(|p| sha256_file(&p).ok().map(|(sha, _)| (p, sha)));
        match (actual, resolve_pin(pinned, operator, flag)) {
            (Some((p, sha)), Ok(pin)) if sha == pin => note(
                format!("{what}       {} sha256 {sha} (pinned)", p.display()),
                true,
            ),
            (Some((p, sha)), Ok(_)) => note(
                format!(
                    "{what}       {} sha256 {sha} does NOT match its pin",
                    p.display()
                ),
                false,
            ),
            (Some((p, sha)), Err(e)) => note(
                format!("{what}       {} sha256 {sha}: {e}", p.display()),
                false,
            ),
            (None, _) => note(format!("{what}       {}: missing", path.display()), false),
        }
    }
    if let Some(m) = model {
        let dir = cfg.model_dir.clone();
        let states = tokio::task::spawn_blocking(move || verify_model_files(&dir, m).0)
            .await
            .unwrap_or_default();
        for (pin, (name, state)) in m.files().into_iter().zip(states) {
            let good = state == FileState::Verified;
            let fix = if good {
                String::new()
            } else {
                format!(
                    " ({} {} at {}, get it from {})",
                    pin.role,
                    pin.repo,
                    pin.revision,
                    pin.url()
                )
            };
            note(format!("model file    {name}: {state:?}{fix}"), good);
        }
    }
    if !ready {
        lines.push("smoke clip    not run (fix the lines above first)".into());
        return (lines, false);
    }
    let started = Instant::now();
    match VideoEngine::production(cfg.clone()).await {
        Ok(engine) => {
            lines.push(format!(
                "smoke clip    passed in {:.0}s: ready to serve",
                started.elapsed().as_secs_f64()
            ));
            engine.stop();
            (lines, true)
        }
        Err(e) => {
            lines.push(format!("smoke clip    FAILED: {e}"));
            (lines, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RID: &str = "3f1c2a54-9b7e-4d21-8c3a-0f6e5d4c3b2a";

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sgl-video-test-{tag}-{}-{:x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn model_5b() -> &'static VideoModel {
        model_by_id("wan2.2-ti2v-5b").unwrap()
    }

    fn model_1b() -> &'static VideoModel {
        model_by_id("wan2.1-t2v-1.3b").unwrap()
    }

    pub(crate) fn request(model: &VideoModel, seconds: u32) -> VideoRequest {
        let (w, h) = model.sizes[0];
        VideoRequest {
            job_id: RID.into(),
            prompt: "a lighthouse at dusk, waves".into(),
            width: w,
            height: h,
            seconds,
            frames: frames_for(seconds, model.fps),
            fps: model.fps,
            steps: model.steps,
            seed: 42,
            max_video_bytes: MAX_VIDEO_BYTES,
            max_thumbnail_bytes: MAX_THUMBNAIL_BYTES,
        }
    }

    #[test]
    fn frames_formula_matches_the_spec() {
        assert_eq!(frames_for(5, 16), 81);
        assert_eq!(frames_for(5, 24), 121);
        assert_eq!(frames_for(2, 16), 33);
        assert_eq!(frames_for(2, 24), 49);
        assert_eq!(frames_for(3, 16), 49);
        assert_eq!(frames_for(4, 24), 97);
        for s in MIN_SECONDS..=MAX_SECONDS {
            for fps in [16, 24] {
                assert_eq!(frames_for(s, fps) % 4, 1, "4n+1");
            }
        }
        // Rounding: 1 s at 10 fps is 2.5 groups of 4, rounded half up to 3.
        assert_eq!(frames_for(1, 10), 13);
    }

    #[test]
    fn model_table_carries_the_spec_pins() {
        let small = model_1b();
        assert_eq!(small.fps, 16);
        assert_eq!(small.sizes, &[(832, 480), (480, 832)]);
        assert_eq!(
            small.min_vram_mb, 15360,
            "16 GB cards report about 16376 MiB"
        );
        assert_eq!(
            small.diffusion.sha256,
            "30a44f695b4275a915810120360d6fd26152ec303c2226b5152ec33a93c380e4"
        );
        assert_eq!(small.diffusion.bytes, 1_535_768_800);
        assert_eq!(small.vae.file_name(), "wan_2.1_vae.safetensors");
        let big = model_5b();
        assert_eq!(big.fps, 24);
        assert_eq!(big.sizes, &[(1280, 704), (704, 1280)]);
        assert_eq!(big.min_vram_mb, 22528);
        assert_eq!(big.vae.bytes, 1_409_400_960);
        assert_eq!(big.t5.sha256, small.t5.sha256);
        assert_eq!(
            big.diffusion.url(),
            "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/57437632ddd08bdcbd1508c866aa22e126ed51d2/Wan2.2-TI2V-5B-Q8_0.gguf"
        );
        for m in &MODELS {
            for f in m.files() {
                assert!(is_sha256_hex(f.sha256), "{}", f.path);
                assert!(!f.file_name().contains('/'));
            }
            // Wan 2.2 latents are 32 px; every size and the smoke size must fit both models.
            for &(w, h) in m.sizes {
                assert_eq!((w % 32, h % 32), (0, 0));
            }
        }
        assert_eq!((SMOKE_WIDTH % 32, SMOKE_HEIGHT % 32), (0, 0));
        assert!(is_video_model("WAN2.2-TI2V-5B") && !is_video_model("flux2-klein-4b-q4"));
    }

    #[test]
    fn pins_resolve_fail_closed() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        assert_eq!(resolve_pin(None, Some(&a), "--x").unwrap(), a);
        assert_eq!(
            resolve_pin(None, Some(&a.to_ascii_uppercase()), "--x").unwrap(),
            a
        );
        assert!(resolve_pin(None, None, "--x")
            .unwrap_err()
            .contains("required"));
        assert_eq!(resolve_pin(Some(&a), None, "--x").unwrap(), a);
        assert_eq!(resolve_pin(Some(&a), Some(&a), "--x").unwrap(), a);
        assert!(resolve_pin(Some(&a), Some(&b), "--x").is_err());
        assert!(resolve_pin(None, Some("abc"), "--x").is_err());
        assert!(resolve_pin(None, Some(&"g".repeat(64)), "--x").is_err());
        // Today nothing is pinned in the binary: both flags are required.
        assert!(PINNED_ENGINE_SHA256.is_none() && PINNED_ENCODER_SHA256.is_none());
    }

    #[test]
    fn executable_hash_must_match() {
        let dir = tmpdir("exe");
        let path = dir.join("sd-cli");
        std::fs::write(&path, b"binary").unwrap();
        let (sha, _) = sha256_file(&path).unwrap();
        assert_eq!(
            verify_executable(&path, &sha, "e").unwrap(),
            std::fs::canonicalize(&path).unwrap()
        );
        assert!(verify_executable(&path, &"0".repeat(64), "e").is_err());
        assert!(verify_executable(&dir.join("missing"), &sha, "e").is_err());
        assert!(
            verify_executable(&dir, &sha, "e").is_err(),
            "a folder is not an executable"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gpu_parsing() {
        assert_eq!(
            parse_gpu_query("NVIDIA GeForce RTX 4090, 24564, 550.54.15\n"),
            Some(("NVIDIA GeForce RTX 4090".into(), 24564, "550.54.15".into()))
        );
        // Two cards: device 0 counts (sd-cli runs on it).
        assert_eq!(
            parse_gpu_query(
                "\nNVIDIA L40S, 46068, 550.90.07\nNVIDIA GeForce RTX 3060, 12288, 550.90.07\n"
            )
            .unwrap()
            .1,
            46068
        );
        assert_eq!(parse_gpu_query(""), None);
        assert_eq!(parse_gpu_query("NVIDIA-SMI has failed"), None);
        assert_eq!(parse_gpu_query("X, [N/A], 1"), None);
        let header = "+---------------------------------------------------------------------------------------+\n| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |\n";
        assert_eq!(parse_cuda_version(header).as_deref(), Some("12.4"));
        assert_eq!(parse_cuda_version("no cuda here"), None);
    }

    #[test]
    fn engine_args_snapshot() {
        let m = model_5b();
        let files = ModelPaths {
            diffusion: "/m/Wan2.2-TI2V-5B-Q8_0.gguf".into(),
            t5: "/m/umt5-xxl-encoder-Q8_0.gguf".into(),
            vae: "/m/wan2.2_vae.safetensors".into(),
        };
        let prompt = "-o /etc/passwd --steps 999 \"; rm -rf /";
        let args = engine_args(
            m,
            &files,
            prompt,
            1280,
            704,
            121,
            30,
            42,
            true,
            Path::new("/w/raw.webm"),
        );
        let expected: Vec<String> = [
            "-M",
            "vid_gen",
            "--diffusion-model",
            "/m/Wan2.2-TI2V-5B-Q8_0.gguf",
            "--t5xxl",
            "/m/umt5-xxl-encoder-Q8_0.gguf",
            "--vae",
            "/m/wan2.2_vae.safetensors",
            "-p",
            prompt,
            "-n",
            NEGATIVE_PROMPT,
            "--cfg-scale",
            "5",
            "--sampling-method",
            "euler",
            "--steps",
            "30",
            "--flow-shift",
            "5",
            "-W",
            "1280",
            "-H",
            "704",
            "--video-frames",
            "121",
            "--fps",
            "24",
            "-s",
            "42",
            "--params-backend",
            "te=disk",
            "--flash-attn",
            "on",
            "-o",
            "/w/raw.webm",
            "--vae-tiling",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(args, expected);
        // The prompt is exactly one argv element, right after -p.
        assert_eq!(args.iter().filter(|a| a.as_str() == prompt).count(), 1);
        assert_eq!(args.iter().filter(|a| a.as_str() == "-o").count(), 1);
        let small = engine_args(
            model_1b(),
            &files,
            "x",
            832,
            480,
            81,
            20,
            7,
            false,
            Path::new("/o"),
        );
        assert!(!small.contains(&"--vae-tiling".to_string()));
        // Phase 0: tile the 5B decode under 34 GB of GPU memory; never the 1.3B.
        assert!(m.needs_vae_tiling(24564) && !m.needs_vae_tiling(46068));
        assert!(!model_1b().needs_vae_tiling(8192));
        assert!(small.windows(2).any(|w| w == ["--fps", "16"]));
    }

    #[test]
    fn encoder_args_snapshot() {
        let label = ai_label("wan2.1-t2v-1.3b", RID);
        assert_eq!(
            label,
            format!("AI-generated video. Made with wan2.1-t2v-1.3b on the Singularity Layer Grid. Job {RID}.")
        );
        let args = encode_args(
            Codec::Libx264,
            Path::new("/w/raw.webm"),
            Path::new("/w/video.mp4"),
            &label,
        );
        let comment = format!("comment={label}");
        let description = format!("description={label}");
        let expected: Vec<&str> = vec![
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "/w/raw.webm",
            "-map",
            "0:v:0",
            "-an",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-metadata",
            &comment,
            "-metadata",
            &description,
            "-f",
            "mp4",
            "/w/video.mp4",
        ];
        assert_eq!(args, expected);
        let open = encode_args(Codec::Libopenh264, Path::new("/i"), Path::new("/o"), &label);
        assert!(open.windows(2).any(|w| w == ["-c:v", "libopenh264"]));
        assert!(open.windows(2).any(|w| w == ["-b:v", "8M"]));
        assert!(!open.contains(&"-crf".to_string()));
        assert_eq!(Codec::parse("libopenh264").unwrap(), Codec::Libopenh264);
        assert!(Codec::parse("libx265").is_err());
        let thumb = thumbnail_args(Path::new("/w/video.mp4"), 2531, Path::new("/w/thumb.webp"));
        assert!(thumb.windows(2).any(|w| w == ["-ss", "2.531"]));
        assert!(thumb.windows(2).any(|w| w == ["-c:v", "libwebp"]));
        assert_eq!(thumb.last().unwrap(), "/w/thumb.webp");
    }

    #[test]
    fn progress_parser_reads_sdcpp_output() {
        let mut p = ProgressParser::new(12);
        // Captured shapes from sd.cpp master-1043 (desktop logs).
        assert_eq!(
            p.feed("  |##################################################| 27/27 - 400.1MB/s"),
            None,
            "loading bar"
        );
        assert_eq!(
            p.feed("[INFO ] stable-diffusion.cpp:300 - loading model"),
            None
        );
        assert_eq!(
            p.feed("[I] generate_video 832x480x33 --- video.cpp:1679"),
            Some(Progress {
                stage: Stage::Denoising,
                step: 0,
                total: 12
            })
        );
        assert_eq!(
            p.feed("  |##########| 27/27 - 400.1MB/s"),
            None,
            "not a sampler step"
        );
        assert_eq!(
            p.feed("  |====>                                             |  3/12 - 75.20s/it - ETA 10:00"),
            Some(Progress { stage: Stage::Denoising, step: 3, total: 12 })
        );
        assert_eq!(
            p.feed("\x1b[K  |==========| 12/12 - 1.52it/s"),
            Some(Progress {
                stage: Stage::Denoising,
                step: 12,
                total: 12
            })
        );
        assert_eq!(p.feed("  |===| 13/12 - 1.0s/it"), None, "step past total");
        assert_eq!(p.feed("  |===| 3/30 - 1.0s/it"), None, "foreign total");
        assert_eq!(p.feed("|=| -1/12 - 1.0s/it"), None);
        assert_eq!(p.feed("garbage / with / slashes s/it"), None);
        assert_eq!(p.feed(&"x".repeat(10_000)), None);
        assert_eq!(
            p.feed("[I] sampling completed, taking 2963.50s --- video.cpp:1776"),
            Some(Progress {
                stage: Stage::Decoding,
                step: 12,
                total: 12
            })
        );
        assert_eq!(
            p.feed("  |=> | 1/10 - 946.38s/it"),
            None,
            "decoder tiles are not steps"
        );
        assert!(decoder_failed(
            "[E] vae decode compute failed --- vae.hpp:377"
        ));
        assert!(!decoder_failed(
            "[I] decode_first_stage completed, taking 55.12s"
        ));
        for (s, name) in [
            (Stage::Loading, "loading"),
            (Stage::Denoising, "denoising"),
            (Stage::Decoding, "decoding"),
            (Stage::Encoding, "encoding"),
            (Stage::Uploading, "uploading"),
        ] {
            assert_eq!(s.as_str(), name);
        }
    }

    #[test]
    fn mp4_check_accepts_good_and_rejects_lies() {
        let dir = tmpdir("mp4");
        let path = dir.join("v.mp4");
        let label = ai_label("wan2.2-ti2v-5b", RID);
        let expected = Expected {
            width: 1280,
            height: 704,
            duration_ms: 5000,
            job_id: RID,
        };
        std::fs::write(&path, synthetic_mp4(1280, 704, 5042, Some(&label))).unwrap();
        let info = inspect_mp4(&path).unwrap();
        assert_eq!(&info.major_brand, b"isom");
        assert_eq!(
            (info.width, info.height, info.duration_ms),
            (1280, 704, 5042)
        );
        assert!(info.comment.as_deref().unwrap().contains("AI-generated"));
        check_mp4(&info, &expected).unwrap();
        // Wrong size, wrong duration, no label, label for another job.
        for (bytes, why) in [
            (synthetic_mp4(704, 1280, 5000, Some(&label)), "size"),
            (synthetic_mp4(1280, 704, 5600, Some(&label)), "duration"),
            (synthetic_mp4(1280, 704, 4400, Some(&label)), "duration"),
            (synthetic_mp4(1280, 704, 5000, None), "label"),
            (
                synthetic_mp4(1280, 704, 5000, Some("AI-generated video. Job other.")),
                "label",
            ),
            (
                synthetic_mp4(1280, 704, 5000, Some(&format!("made by a person {RID}"))),
                "label",
            ),
        ] {
            std::fs::write(&path, bytes).unwrap();
            let info = inspect_mp4(&path).unwrap();
            assert!(
                check_mp4(&info, &expected).unwrap_err().contains(why),
                "{why}"
            );
        }
        // Not an MP4, truncated, trailing garbage, bad brand.
        std::fs::write(&path, b"this is not an mp4 file at all").unwrap();
        assert!(inspect_mp4(&path).is_err());
        let good = synthetic_mp4(1280, 704, 5000, Some(&label));
        std::fs::write(&path, &good[..good.len() - 10]).unwrap();
        assert!(inspect_mp4(&path).is_err());
        let mut trailing = good.clone();
        trailing.extend_from_slice(b"xyz");
        std::fs::write(&path, &trailing).unwrap();
        assert!(inspect_mp4(&path).is_err());
        let mut brand = good.clone();
        brand[8..12].copy_from_slice(b"qt  ");
        std::fs::write(&path, &brand).unwrap();
        let info = inspect_mp4(&path).unwrap();
        assert!(check_mp4(&info, &expected).unwrap_err().contains("brand"));
        assert!(is_webp(&synthetic_webp()) && !is_webp(b"RIFF0000WAVE"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn floor_blocks_only_sexual_plus_minor() {
        for p in [
            "a naked 15 year old",
            "nsfw photo of a child",
            "erotic loli art",
            "a 16-year-old topless",
            "sexy schoolgirl",
            "NSFW CHILD",
        ] {
            assert!(asks_for_sexual_minor(p), "{p}");
        }
        for p in [
            "a child eating ice cream at a birthday party",
            "a baby sleeping in a crib, soft light",
            "a naked 25 year old woman on a beach, photo",
            "nude marble statue in a museum",
            "two adults, explicit scene, 30 year old",
            "a kid's birthday cake, 7 year old boy smiling",
            "a woman wearing lingerie, studio fashion shoot",
            "",
        ] {
            assert!(!asks_for_sexual_minor(p), "{p}");
        }
    }

    #[test]
    fn model_files_verify_cache_and_refuse_symlinks() {
        // A tiny fake model so the real hashing path runs (the real pins are GBs).
        static TINY: VideoModel = VideoModel {
            id: "tiny",
            fps: 16,
            steps: 2,
            cfg_scale: 5.0,
            sizes: &[(256, 160)],
            min_vram_mb: 1,
            tile_below_vram_mb: None,
            diffusion: PinnedFile {
                role: "diffusion",
                repo: "r",
                revision: "v",
                path: "d.gguf",
                bytes: 3,
                // sha256("abc")
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            },
            vae: PinnedFile {
                role: "vae",
                repo: "r",
                revision: "v",
                path: "sub/v.safetensors",
                bytes: 0,
                sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
            t5: PinnedFile {
                role: "t5",
                repo: "r",
                revision: "v",
                path: "t.gguf",
                bytes: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            },
        };
        let dir = tmpdir("files");
        let (states, paths) = verify_model_files(&dir, &TINY);
        assert!(paths.is_none());
        assert!(states.iter().all(|(_, s)| *s == FileState::Missing));
        std::fs::write(dir.join("d.gguf"), b"abc").unwrap();
        std::fs::write(dir.join("t.gguf"), b"abd").unwrap();
        std::fs::write(dir.join("v.safetensors"), b"").unwrap();
        let (states, paths) = verify_model_files(&dir, &TINY);
        assert!(paths.is_none());
        assert_eq!(
            states[1],
            (
                "t.gguf",
                FileState::Mismatch("sha256 does not match its pin")
            )
        );
        std::fs::write(dir.join("t.gguf"), b"abc").unwrap();
        let (_, paths) = verify_model_files(&dir, &TINY);
        assert_eq!(paths.unwrap().t5, dir.join("t.gguf"));
        assert!(
            dir.join(VERIFIED_MARKER).is_file(),
            "verified files are remembered"
        );
        // A changed file (new size) is re-checked, not trusted from the marker.
        std::fs::write(dir.join("d.gguf"), b"abcd").unwrap();
        assert!(verify_model_files(&dir, &TINY).1.is_none());
        #[cfg(unix)]
        {
            std::fs::remove_file(dir.join("d.gguf")).unwrap();
            let other = tmpdir("files-other");
            std::fs::write(other.join("real"), b"abc").unwrap();
            std::os::unix::fs::symlink(other.join("real"), dir.join("d.gguf")).unwrap();
            let (states, paths) = verify_model_files(&dir, &TINY);
            assert!(paths.is_none());
            assert!(matches!(states[0].1, FileState::Mismatch(_)));
            let _ = std::fs::remove_dir_all(other);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn capability_json_is_the_exact_wire_shape() {
        let rt = Runtime::fake(
            model_5b(),
            PathBuf::from("/tmp/x"),
            Fake {
                job: FakeMode::Ok,
                smoke: FakeMode::Ok,
            },
        );
        let cap = base_capabilities(rt.model, &"e".repeat(64), &"f".repeat(64), rt.gpu.clone())
            .bind_heartbeat(5);
        let wire = serde_json::to_string(&cap).unwrap();
        let expected = format!(
            concat!(
                r#"{{"kind":"video","protocol":"video-v1","engine":"sdcpp-cuda","engine_revision":"master-1043-3be2378","#,
                r#""engine_sha256":"{e}","encoder":"ffmpeg","encoder_sha256":"{f}","#,
                r#""models":[{{"id":"wan2.2-ti2v-5b","model_sha256":"57bece983817ab2f957546683bb670f13be7d99022d45674840cd999a050ea8f","#,
                r#""t5_sha256":"2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc","#,
                r#""vae_sha256":"e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156","#,
                r#""sizes":[[1280,704],[704,1280]],"max_seconds":5}}],"#,
                r#""gpu":{{"name":"NVIDIA GeForce RTX 4090","vram_mb":24564,"driver":"550.54.15","cuda":"12.4"}},"#,
                r#""video_ready":true,"free_slots":1}}"#
            ),
            e = "e".repeat(64),
            f = "f".repeat(64)
        );
        assert_eq!(wire, expected);
        assert_eq!(cap.bind_heartbeat(0).free_slots, 0);
    }

    fn fake_engine_runtime(job: FakeMode, smoke: FakeMode) -> (Runtime, PathBuf) {
        let root = tmpdir("engine");
        let rt = Runtime::fake(model_1b(), root.join("work"), Fake { job, smoke });
        (rt, root)
    }

    async fn run_job(
        engine: &VideoEngine,
        req: &VideoRequest,
    ) -> (Result<VideoOutput, VideoError>, Progress) {
        let dir = engine.job_dir(&req.job_id).unwrap();
        let (tx, rx) = tokio::sync::watch::channel(Progress {
            stage: Stage::Loading,
            step: 0,
            total: req.steps,
        });
        let (_cancel_tx, cancel) = tokio::sync::watch::channel(false);
        let out = engine
            .generate(
                req,
                dir.path(),
                &tx,
                cancel,
                Instant::now() + Duration::from_secs(20),
            )
            .await;
        // Keep the job folder for the caller's checks: leak the guard's path, clean up later.
        std::mem::forget(dir);
        let last = *rx.borrow();
        (out, last)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_engine_happy_path_and_failure_modes() {
        let (rt, root) = fake_engine_runtime(FakeMode::Ok, FakeMode::Ok);
        let engine = VideoEngine::start(rt).await.unwrap();
        assert!(engine.is_healthy());
        let models = vec![model_1b().id.to_string()];
        assert_eq!(engine.readiness_snapshot(&models).available_models, models);
        let req = request(model_1b(), 5);
        let (out, last) = run_job(&engine, &req).await;
        let out = out.unwrap();
        assert_eq!(
            out.video.bytes,
            std::fs::metadata(&out.video.path).unwrap().len()
        );
        assert_eq!(out.video.sha256, sha256_file(&out.video.path).unwrap().0);
        assert!(out.thumbnail.is_some());
        assert_eq!(out.duration_ms, 81 * 1000 / 16);
        assert_eq!(
            last,
            Progress {
                stage: Stage::Encoding,
                step: 20,
                total: 20
            }
        );
        assert!(!engine.is_busy());

        // Bad requests never reach the engine and keep it healthy.
        for bad in [
            VideoRequest {
                prompt: "a\0b".into(),
                ..req.clone()
            },
            VideoRequest {
                prompt: " ".into(),
                ..req.clone()
            },
            VideoRequest {
                prompt: "x".repeat(MAX_PROMPT_BYTES + 1),
                ..req.clone()
            },
            VideoRequest {
                prompt: "a cat <LoRA:../../x:1>".into(),
                ..req.clone()
            },
            VideoRequest {
                width: 1280,
                height: 704,
                ..req.clone()
            },
            VideoRequest {
                steps: 4,
                ..req.clone()
            },
            VideoRequest {
                frames: 80,
                ..req.clone()
            },
            VideoRequest {
                seconds: 6,
                frames: frames_for(6, 16),
                ..req.clone()
            },
            VideoRequest {
                fps: 24,
                ..req.clone()
            },
            VideoRequest {
                max_video_bytes: MAX_VIDEO_BYTES + 1,
                ..req.clone()
            },
        ] {
            let (out, _) = run_job(&engine, &bad).await;
            assert!(matches!(out, Err(VideoError::InputInvalid(_))), "{bad:?}");
        }
        assert!(engine.is_healthy());
        assert!(
            engine.job_dir("../etc").is_err(),
            "path traversal via an id"
        );
        assert!(engine.job_dir("smoke-../../x").is_err());
        let _ = std::fs::remove_dir_all(root);

        for mode in [
            FakeMode::Invalid,
            FakeMode::Lie,
            FakeMode::NoLabel,
            FakeMode::Crash,
        ] {
            let (rt, root) = fake_engine_runtime(mode, FakeMode::Ok);
            let engine = VideoEngine::start(rt).await.unwrap();
            let (out, _) = run_job(&engine, &req).await;
            assert!(
                matches!(out, Err(VideoError::RuntimeFailed(_))),
                "{mode:?}: {out:?}"
            );
            assert!(
                !engine.is_healthy(),
                "{mode:?}: a broken engine is de-advertised"
            );
            assert!(engine
                .readiness_snapshot(&models)
                .available_models
                .is_empty());
            assert!(engine.capabilities().is_none());
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn smoke_failure_never_starts_and_restart_budgets_hold() {
        for mode in [
            FakeMode::Crash,
            FakeMode::Lie,
            FakeMode::NoLabel,
            FakeMode::Invalid,
        ] {
            let (rt, root) = fake_engine_runtime(FakeMode::Ok, mode);
            assert!(VideoEngine::start(rt).await.is_err(), "{mode:?}");
            let _ = std::fs::remove_dir_all(root);
        }
        let (rt, root) = fake_engine_runtime(FakeMode::Crash, FakeMode::Ok);
        let engine = VideoEngine::start(rt).await.unwrap();
        for _ in 0..MAX_RESTARTS_PER_WINDOW {
            engine.restart().await.unwrap();
            assert!(engine.is_healthy());
        }
        assert!(engine.restart().await.unwrap_err().contains("budget"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hang_is_killed_with_its_whole_group_on_deadline_and_cancel() {
        let (mut rt, root) = fake_engine_runtime(FakeMode::Hang, FakeMode::Ok);
        let pid_file = root.join("child.pid");
        rt.extra_env = vec![(
            "SGL_VIDEO_FAKE_PID_FILE".into(),
            pid_file.to_string_lossy().into(),
        )];
        let engine = VideoEngine::start(rt).await.unwrap();
        let req = request(model_1b(), 2);

        // Deadline.
        let dir = engine.job_dir(RID).unwrap();
        let (tx, _rx) = tokio::sync::watch::channel(Progress {
            stage: Stage::Loading,
            step: 0,
            total: 20,
        });
        let (_c, cancel) = tokio::sync::watch::channel(false);
        let started = Instant::now();
        let out = engine
            .generate(
                &req,
                dir.path(),
                &tx,
                cancel,
                Instant::now() + Duration::from_millis(800),
            )
            .await;
        assert_eq!(
            out.unwrap_err(),
            VideoError::RuntimeFailed("timed out".into())
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_grandchild_dead(&pid_file).await;
        assert!(!engine.is_healthy());
        drop(dir);
        assert!(
            !engine.runtime.work_root.join(RID).exists(),
            "job folder removed"
        );

        // Cancel (continue:false) kills it at once and keeps the engine healthy.
        engine.restart().await.unwrap();
        let _ = std::fs::remove_file(&pid_file);
        let dir = engine.job_dir(RID).unwrap();
        let (cancel_tx, cancel) = tokio::sync::watch::channel(false);
        let flip = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            let _ = cancel_tx.send(true);
            cancel_tx
        });
        let started = Instant::now();
        let out = engine
            .generate(
                &req,
                dir.path(),
                &tx,
                cancel,
                Instant::now() + Duration::from_secs(60),
            )
            .await;
        let _ = flip.await;
        assert_eq!(out.unwrap_err(), VideoError::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_grandchild_dead(&pid_file).await;
        assert!(engine.is_healthy(), "a cancel is not a fault");
        drop(dir);

        // Timeouts park the engine after three in a row with no good clip in between (the
        // first one above already counted).
        for round in 0..2 {
            let dir = engine.job_dir(RID).unwrap();
            let (_c, cancel) = tokio::sync::watch::channel(false);
            let out = engine
                .generate(
                    &req,
                    dir.path(),
                    &tx,
                    cancel,
                    Instant::now() + Duration::from_millis(300),
                )
                .await;
            assert_eq!(
                out.unwrap_err(),
                VideoError::RuntimeFailed("timed out".into())
            );
            let restarted = engine.restart().await;
            if round == 0 {
                restarted.unwrap();
            } else {
                assert!(restarted.unwrap_err().contains("parked"));
            }
        }
        assert!(!engine.is_healthy());
        assert!(engine.restart().await.unwrap_err().contains("parked"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    async fn assert_grandchild_dead(pid_file: &Path) {
        let mut pid = None;
        for _ in 0..50 {
            if let Ok(text) = std::fs::read_to_string(pid_file) {
                if let Ok(p) = text.trim().parse::<i32>() {
                    pid = Some(p);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let pid = pid.expect("fake engine recorded its child");
        for _ in 0..50 {
            // 0 = still exists (maybe a zombie of an exited shell); ESRCH = gone.
            let alive = unsafe { libc::kill(pid, 0) } == 0;
            let zombie = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .map(|s| s.contains(") Z "))
                .unwrap_or(false);
            if !alive || zombie {
                return;
            }
            // macOS: an orphan reparented to launchd is reaped quickly; poll briefly.
            let mut status = 0;
            unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the hanging engine's child {pid} survived the group kill");
    }

    #[test]
    fn work_root_cleanup_only_touches_engine_folders() {
        assert!(owned_entry(RID));
        assert!(owned_entry("smoke-0123456789abcdef"));
        for name in [
            "models",
            "node.json",
            "..",
            "smoke-",
            "smoke-xyz",
            "smoke-0123456789abcdef0",
            "3F1C2A54-9B7E-4D21-8C3A-0F6E5D4C3B2A",
        ] {
            assert!(!owned_entry(name), "{name}");
        }
    }

    #[test]
    fn doctor_refuses_unsupported_platforms() {
        if platform_approved() {
            return;
        }
        let cfg = VideoConfig {
            model: "wan2.1-t2v-1.3b".into(),
            engine: "/nonexistent/sd-cli".into(),
            engine_sha256: None,
            encoder: "/nonexistent/ffmpeg".into(),
            encoder_sha256: None,
            model_dir: "/nonexistent".into(),
            codec: Codec::Libx264,
            work_root: std::env::temp_dir().join("sgl-video-doctor-unused"),
        };
        assert_eq!(prepare_runtime(&cfg).unwrap_err(), PLATFORM_REFUSAL);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (lines, ready) = rt.block_on(doctor(&cfg));
        assert!(!ready);
        assert!(lines.iter().any(|l| l.contains("NOT supported")));
        assert!(lines.iter().any(|l| l.contains("smoke clip    not run")));
    }
}
