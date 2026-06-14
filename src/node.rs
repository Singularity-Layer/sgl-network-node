use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::config::{self, NodeConfig};
use crate::crypto::NodeKeypair;
use crate::inference::{ChatMessage, InferenceEngine, InferenceEngineConfig};
use crate::orchestrator::{OrchestratorClient, PendingJob};
use crate::tee;

pub struct ResourceConfig {
    pub threads: u32,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub max_jobs: u32,
    pub batch_size: u32,
    pub heartbeat_interval: u64,
    pub resource_percent: u8,
    /// Confidential token streaming. ALWAYS enabled: the node always advertises
    /// the `streaming` capability and serves stream jobs. Whether a given request
    /// streams is the CALLER's choice (`stream: true`), not the operator's — a
    /// provider can't silently disable it. The legacy `--enable-streaming` flag is
    /// now a deprecated no-op kept only so old service definitions still parse.
    pub streaming_enabled: bool,
}

impl ResourceConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn from_args(
        resource_percent: u8,
        threads: Option<u32>,
        gpu_layers: Option<u32>,
        context_size: u32,
        max_jobs: u32,
        batch_size: u32,
        heartbeat_interval: u64,
        _streaming_enabled: bool, // deprecated no-op: streaming is always on (see below)
    ) -> Self {
        let total_cpus = std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(4);

        let computed_threads =
            ((total_cpus as f64 * resource_percent as f64 / 100.0).ceil() as u32).max(1);
        let computed_gpu_layers = if resource_percent >= 50 {
            99
        } else {
            (99.0 * resource_percent as f64 / 100.0).round() as u32
        };

        Self {
            threads: threads.unwrap_or(computed_threads),
            gpu_layers: gpu_layers.unwrap_or(computed_gpu_layers),
            context_size,
            max_jobs,
            batch_size,
            heartbeat_interval,
            resource_percent,
            // Always on — providers always stream; the caller opts in per request.
            streaming_enabled: true,
        }
    }

    pub fn load_factor(&self) -> f64 {
        1.0 - (self.resource_percent as f64 / 100.0)
    }
}

/// Bounded de-dup set of job ids the node has already handled. A job can arrive
/// via WS push AND a REST heartbeat poll during transitions — this ensures each
/// runs exactly once. Capped so it can't grow without bound.
struct SeenJobs {
    set: HashSet<String>,
    order: VecDeque<String>,
    cap: usize,
}

impl SeenJobs {
    fn new() -> Self {
        Self {
            set: HashSet::new(),
            order: VecDeque::new(),
            cap: 1024,
        }
    }

    /// Returns true if the id is new (caller should handle it); false if a duplicate.
    fn check_and_insert(&mut self, id: &str) -> bool {
        if self.set.contains(id) {
            return false;
        }
        if self.order.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        self.set.insert(id.to_string());
        self.order.push_back(id.to_string());
        true
    }
}

/// De-dup + capacity check, then spawn job processing. Shared by the REST poll
/// loop and the WS push callback so both transports are equivalent and safe.
#[allow(clippy::too_many_arguments)]
fn maybe_spawn_job(
    job: PendingJob,
    client: Arc<OrchestratorClient>,
    engine: Option<Arc<InferenceEngine>>,
    node_secret: [u8; 32],
    streaming_enabled: bool,
    active_jobs: Arc<AtomicU32>,
    seen: Arc<Mutex<SeenJobs>>,
    max_jobs: u32,
) {
    // Atomically reserve a slot (capacity check + increment in one CAS) so
    // concurrent WS-push + REST-poll arrivals can't exceed max_jobs.
    if active_jobs
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
            if c < max_jobs {
                Some(c + 1)
            } else {
                None
            }
        })
        .is_err()
    {
        // At capacity. Not marked seen, so the next REST poll retries it.
        tracing::warn!("At max concurrent jobs ({max_jobs}), deferring job {}", job.id);
        return;
    }
    // De-dup; roll back the reserved slot if this id was already handled.
    {
        let mut s = seen.lock().unwrap();
        if !s.check_and_insert(&job.id) {
            active_jobs.fetch_sub(1, Ordering::Relaxed);
            tracing::debug!("Duplicate job {} ignored", job.id);
            return;
        }
    }
    tracing::info!("Accepted job {} (type: {})", job.id, job.job_type);
    tokio::spawn(async move {
        process_job(&client, &engine, &job, &node_secret, streaming_enabled).await;
        active_jobs.fetch_sub(1, Ordering::Relaxed);
    });
}

pub async fn init(
    config_dir: &Path,
    orchestrator_url: &str,
    wallet: &str,
    tee_type: &str,
    models: &[String],
) -> Result<(), String> {
    let cfg_path = config::config_path(config_dir);
    if cfg_path.exists() {
        return Err(format!(
            "Node already initialized. Config at: {}\nTo reinitialize, delete the config directory first.",
            cfg_path.display()
        ));
    }

    let caps = tee::detect();
    tee::print_capabilities(&caps);
    println!();

    tracing::info!("Generating ed25519 keypair...");
    let keypair = NodeKeypair::generate();
    let kp_path = config::keypair_path(config_dir);
    keypair.save(&kp_path)?;
    tracing::info!("Keypair saved to {}", kp_path.display());

    let public_key = keypair.public_key_bs58();
    tracing::info!("Public key: {public_key}");

    tracing::info!("Registering with orchestrator at {orchestrator_url}...");
    let client = OrchestratorClient::new(orchestrator_url, None);
    let registration = client
        .register(wallet, None, tee_type, models, &public_key, &caps)
        .await?;

    tracing::info!("Registered! Node ID: {}", registration.node_id);

    let node_config = NodeConfig {
        node_id: registration.node_id,
        auth_token: registration.auth_token,
        wallet_address: wallet.to_string(),
        tee_type: tee_type.to_string(),
        orchestrator_url: orchestrator_url.to_string(),
        keypair_path: kp_path.to_string_lossy().to_string(),
    };

    config::save_config(config_dir, &node_config)?;
    tracing::info!("Config saved to {}", cfg_path.display());
    tracing::info!("Node initialized. Run `sgl start` to begin processing jobs.");
    tracing::info!("Run `sgl attest` to verify identity before receiving jobs.");

    Ok(())
}

/// `sgl login` — browser device-authorization flow.
pub async fn login(
    config_dir: &Path,
    orchestrator_url: &str,
    tee_type: &str,
    models: &[String],
) -> Result<(), String> {
    let cfg_path = config::config_path(config_dir);
    if cfg_path.exists() {
        return Err(format!(
            "Node already initialized. Config at: {}",
            cfg_path.display()
        ));
    }

    let caps = tee::detect();
    tee::print_capabilities(&caps);
    println!();

    let keypair = NodeKeypair::generate();
    let kp_path = config::keypair_path(config_dir);
    keypair.save(&kp_path)?;
    let public_key = keypair.public_key_bs58();

    let client = OrchestratorClient::new(orchestrator_url, None);
    let session = client.device_start().await?;

    println!("\n  Open to link this node:\n      {}\n  Approve with your staked Solana wallet (code: {}).\n", session.verify_url, session.user_code);
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(opener)
        .arg(&session.verify_url)
        .spawn();

    let interval = session.interval.max(2);
    let max_polls = if session.expires_in > 0 {
        (session.expires_in / interval) + 2
    } else {
        200
    };
    tracing::info!("Waiting for approval in the browser...");

    let mut reg_code: Option<String> = None;
    let mut wallet: Option<String> = None;
    for _ in 0..max_polls {
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        match client.device_poll(&session.device_code).await {
            Ok(p) if p.status == "approved" => {
                reg_code = p.registration_code;
                wallet = p.wallet_address;
                break;
            }
            Ok(p) if p.status == "expired" => {
                return Err("Login session expired. Run `sgl login` again.".to_string())
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("poll error (retrying): {e}"),
        }
    }

    let reg_code = reg_code.ok_or("Login timed out waiting for approval.")?;
    let wallet = wallet.unwrap_or_default();
    tracing::info!("Approved by wallet {wallet}. Registering node...");

    let registration = client
        .register(
            &wallet,
            Some(&reg_code),
            tee_type,
            models,
            &public_key,
            &caps,
        )
        .await?;

    let node_config = NodeConfig {
        node_id: registration.node_id,
        auth_token: registration.auth_token,
        wallet_address: wallet,
        tee_type: tee_type.to_string(),
        orchestrator_url: orchestrator_url.to_string(),
        keypair_path: kp_path.to_string_lossy().to_string(),
    };
    config::save_config(config_dir, &node_config)?;
    tracing::info!(
        "Linked! Node ID: {}. Run `sgl attest`, then `sgl start --model-path <model.gguf>`.",
        node_config.node_id
    );
    Ok(())
}

pub async fn start(
    config_dir: &Path,
    orchestrator_url: &str,
    model_path: Option<&str>,
    model_name: Option<&str>,
    inference_port: u16,
    rc: &ResourceConfig,
) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let keypair = NodeKeypair::load(&config::keypair_path(config_dir))?;

    let client = Arc::new(OrchestratorClient::new(
        orchestrator_url,
        Some(cfg.auth_token.clone()),
    ));

    let total_cpus = std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4);

    tracing::info!(
        "Starting node {} (wallet: {})",
        cfg.node_id,
        cfg.wallet_address
    );
    tracing::info!("Public key: {}", keypair.public_key_bs58());
    tracing::info!("Resource config:");
    tracing::info!("  Preset:       {}%", rc.resource_percent);
    tracing::info!("  Threads:      {}/{}", rc.threads, total_cpus);
    tracing::info!("  GPU layers:   {}", rc.gpu_layers);
    tracing::info!("  Context:      {} tokens", rc.context_size);
    tracing::info!("  Batch size:   {}", rc.batch_size);
    tracing::info!("  Max jobs:     {}", rc.max_jobs);
    tracing::info!("  Streaming:    {}", if rc.streaming_enabled { "enabled" } else { "disabled" });

    let mut engine: Option<Arc<InferenceEngine>> = None;
    let mut models: Vec<String> = vec![];

    if let Some(path) = model_path {
        let name = model_name.map(|s| s.to_string()).unwrap_or_else(|| {
            Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        });

        tracing::info!("Loading model: {name} from {path}");
        let eng_config = InferenceEngineConfig {
            model_path: PathBuf::from(path),
            model_name: name.clone(),
            port: inference_port,
            threads: rc.threads,
            gpu_layers: rc.gpu_layers,
            context_size: rc.context_size,
            batch_size: rc.batch_size,
            parallel_slots: rc.max_jobs,
        };
        let mut eng = InferenceEngine::new(eng_config);
        eng.start().await?;
        models.push(name);
        engine = Some(Arc::new(eng));
        tracing::info!("Inference engine ready on port {inference_port}");
    } else {
        tracing::warn!("No model specified — node will register but cannot process inference jobs");
        tracing::warn!("Use --model-path <path.gguf> --model-name <name> to enable inference");
    }

    tracing::info!("Heartbeat interval: {}s", rc.heartbeat_interval);

    // Node's X25519 encryption key (derived from its ed25519 seed). Published on
    // every REST heartbeat so the orchestrator can seal prompts to it (E2E).
    let node_secret = keypair.signing_key.to_bytes();
    let node_enc_pubkey =
        crate::encryption::EncryptionKeypair::from_ed25519_seed(&node_secret).public_key_bs58();
    tracing::info!("X25519 encryption key: {node_enc_pubkey}");

    let active_jobs = Arc::new(AtomicU32::new(0));
    let seen_jobs = Arc::new(Mutex::new(SeenJobs::new()));

    // ── WebSocket push-dispatch (additive fast-path) ──────────────────
    // Connects to the orchestrator and processes jobs the instant they're pushed,
    // removing the heartbeat pickup delay. If the socket is down the REST loop
    // below keeps serving (fallback). Jobs are de-duplicated by id across both.
    let ws_state = Arc::new(crate::ws::WsState::new());
    {
        let base = orchestrator_url.to_string();
        let node_id = cfg.node_id.clone();
        let client_ws = Arc::clone(&client);
        let client_job = Arc::clone(&client);
        let client_tok = Arc::clone(&client);
        let engine_ws = engine.clone();
        let secret = node_secret;
        let se = rc.streaming_enabled;
        let aj = Arc::clone(&active_jobs);
        let sj = Arc::clone(&seen_jobs);
        let mj = rc.max_jobs;
        let st = Arc::clone(&ws_state);
        let cfg_tok = cfg.clone();
        let config_dir_buf = config_dir.to_path_buf();
        tokio::spawn(async move {
            crate::ws::run(
                base,
                node_id,
                client_ws,
                st,
                move |job| {
                    maybe_spawn_job(
                        job,
                        Arc::clone(&client_job),
                        engine_ws.clone(),
                        secret,
                        se,
                        Arc::clone(&aj),
                        Arc::clone(&sj),
                        mj,
                    );
                },
                move |new_tok, _exp| {
                    client_tok.update_auth_token(new_tok.clone());
                    let mut updated = cfg_tok.clone();
                    updated.auth_token = new_tok;
                    if let Err(e) = config::save_config(&config_dir_buf, &updated) {
                        tracing::error!("Failed to save WS-rotated token: {e}");
                    } else {
                        tracing::info!("Auth token rotated over WS");
                    }
                },
            )
            .await;
        });
    }

    loop {
        match client
            .heartbeat(
                &cfg.node_id,
                &models,
                rc.load_factor(),
                Some(&node_enc_pubkey),
                rc.streaming_enabled,
            )
            .await
        {
            Ok(resp) => {
                tracing::debug!("Heartbeat OK — status: {}", resp.status);

                // Handle token rotation
                if let Some(new_token) = &resp.new_auth_token {
                    tracing::info!("Auth token rotated by orchestrator, saving new token...");
                    let mut updated_cfg = cfg.clone();
                    updated_cfg.auth_token = new_token.clone();
                    if let Err(e) = config::save_config(config_dir, &updated_cfg) {
                        tracing::error!("Failed to save rotated token: {e}");
                    } else {
                        client.update_auth_token(new_token.clone());
                        tracing::info!(
                            "New token saved (expires: {})",
                            resp.token_expires_at.as_deref().unwrap_or("unknown")
                        );
                    }
                }

                // Process jobs concurrently (REST fallback path; de-duped against
                // anything already picked up via WS push).
                for job in resp.pending_jobs {
                    maybe_spawn_job(
                        job,
                        Arc::clone(&client),
                        engine.clone(),
                        node_secret,
                        rc.streaming_enabled,
                        Arc::clone(&active_jobs),
                        Arc::clone(&seen_jobs),
                        rc.max_jobs,
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Heartbeat failed: {e}");
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(rc.heartbeat_interval)).await;
    }
}

pub async fn status(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let keypair = NodeKeypair::load(&config::keypair_path(config_dir))?;

    println!("=== SGL Node Status ===");
    println!("Node ID:    {}", cfg.node_id);
    println!("Wallet:     {}", cfg.wallet_address);
    println!("TEE type:   {}", cfg.tee_type);
    println!("Public key: {}", keypair.public_key_bs58());
    println!("Config:     {}", config::config_path(config_dir).display());
    println!();

    let caps = tee::detect();
    tee::print_capabilities(&caps);
    println!();

    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    match client.get_node_status(&cfg.node_id).await {
        Ok(info) => {
            println!("--- Orchestrator ---");
            println!("Status:       {}", info.status);
            println!("Attested:     {}", info.attestation_verified);
            if let Some(score) = info.reputation_score {
                println!("Reputation:   {:.1}", score);
            }
            if let Some(completed) = info.jobs_completed {
                println!("Jobs done:    {completed}");
            }
            if let Some(failed) = info.jobs_failed {
                println!("Jobs failed:  {failed}");
            }
        }
        Err(e) => {
            println!("Could not reach orchestrator: {e}");
        }
    }

    Ok(())
}

/// Toggle off-grid (maintenance) mode. Off-grid removes the node from job
/// dispatch for planned downtime — no jobs are routed to it and it isn't
/// penalized for being offline. Tamper slashing is unaffected.
pub async fn set_off_grid(config_dir: &Path, orchestrator_url: &str, off_grid: bool) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.set_off_grid(&cfg.node_id, off_grid).await?;
    if off_grid {
        println!("🔌 Node is now OFF-GRID (maintenance).");
        println!("   It won't receive new jobs and won't be penalized for being offline.");
        println!("   Run `sgl on-grid` when you're ready to serve again.");
    } else {
        println!("✅ Node is back ON-GRID — eligible to receive jobs again.");
    }
    Ok(())
}

/// Show this node's per-model pricing (custom vs platform suggested, + the band).
pub async fn show_prices(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    let data = client.get_prices(&cfg.node_id).await?;
    let prices = data.get("prices").and_then(|p| p.as_array()).cloned().unwrap_or_default();
    if prices.is_empty() {
        println!("This node isn't serving any models yet.");
        return Ok(());
    }
    println!("Per-model pricing (USD per 1M tokens):\n");
    for p in prices {
        let model = p.get("model").and_then(|m| m.as_str()).unwrap_or("?");
        let eff = p.get("effective");
        let custom = p.get("custom").map(|c| !c.is_null()).unwrap_or(false);
        let r = p.get("reference");
        let g = |v: Option<&serde_json::Value>, k: &str| v.and_then(|o| o.get(k)).and_then(|n| n.as_f64()).unwrap_or(0.0);
        println!(
            "  {model:<20} in ${:.6} / out ${:.6}  [{}]   (suggested in ${:.6} / out ${:.6})",
            g(eff, "inputPerM"), g(eff, "outputPerM"),
            if custom { "custom" } else { "suggested" },
            g(r, "inputPerM"), g(r, "outputPerM"),
        );
    }
    Ok(())
}

/// Set a custom per-token price for a model (USD per 1M tokens). Band-enforced server-side.
pub async fn set_price(config_dir: &Path, orchestrator_url: &str, model: &str, input_per_m: f64, output_per_m: f64) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.set_price(&cfg.node_id, model, input_per_m, output_per_m).await?;
    println!("✅ Price set for {model}: in ${input_per_m}/1M · out ${output_per_m}/1M. You earn 80% of what you charge.");
    Ok(())
}

/// Reset a model's price back to the platform suggested rate.
pub async fn reset_price(config_dir: &Path, orchestrator_url: &str, model: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.reset_price(&cfg.node_id, model).await?;
    println!("✅ {model} reset to the platform suggested price.");
    Ok(())
}

pub async fn attest(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let keypair = NodeKeypair::load(&config::keypair_path(config_dir))?;

    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));

    tracing::info!("Requesting attestation challenge...");
    let challenge = client.request_challenge(&cfg.node_id).await?;
    let expiry_owned;
    let expiry = match challenge.expires_at.as_deref() {
        Some(at) => at,
        None => {
            expiry_owned = challenge
                .expires_in_seconds
                .map(|s| format!("{s}s"))
                .unwrap_or_else(|| "unknown".to_string());
            &expiry_owned
        }
    };
    tracing::info!("Challenge received (expires: {expiry})");

    // Build the hardware report (TEE type, SIP status, binary self-hash). The
    // orchestrator gates on SIP + binary-hash allowlist before activating.
    let report = crate::tee::generate_attestation_report();
    let report_hash = report.report_hash.clone();
    tracing::info!(
        "Hardware report: sip_enabled={}, binary_hash={}…",
        report.sip_enabled,
        &report.binary_hash[..report.binary_hash.len().min(12)]
    );

    // Sign the plain challenge (proves key ownership). The hardware report is
    // delivered over the authenticated node session and gated server-side.
    let _ = report_hash;
    let signature = keypair.sign_message(challenge.challenge.as_bytes());
    tracing::info!("Challenge signed, submitting with hardware report...");

    // Derive the node's X25519 encryption key (for E2E-encrypted prompts) from
    // the same ed25519 seed and publish it during attestation.
    let enc_keypair =
        crate::encryption::EncryptionKeypair::from_ed25519_seed(&keypair.signing_key.to_bytes());
    let encryption_public_key = enc_keypair.public_key_bs58();
    tracing::info!("Publishing X25519 encryption key: {encryption_public_key}");

    let report_json = serde_json::to_value(&report).ok();
    let result = client
        .verify_attestation(
            &cfg.node_id,
            &signature,
            Some(encryption_public_key),
            report_json,
        )
        .await?;

    if result.verified {
        println!("✅ Attestation verified — node status: {}", result.status);
    } else {
        return Err("Attestation verification failed".to_string());
    }

    Ok(())
}

async fn process_job(
    client: &OrchestratorClient,
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    node_secret: &[u8; 32],
    streaming_enabled: bool,
) {
    tracing::info!("Processing job {} (type: {})", job.id, job.job_type);

    // If the prompt is sealed (E2E), decrypt it with the node's X25519 key and
    // remember the caller's response key so we can seal the reply back.
    let mut response_pubkey: Option<[u8; 32]> = None;
    let mut enc_version = crate::encryption::EncVersion::V1;
    let mut effective_job = job.clone();
    if let Some(payload) = &job.input_payload {
        match crate::encryption::unseal_input(payload, node_secret) {
            Ok((inner, resp, version)) => {
                response_pubkey = resp;
                enc_version = version;
                if resp.is_some() {
                    effective_job.input_payload = Some(inner);
                }
            }
            Err(e) => {
                tracing::error!("Failed to unseal job {}: {e}", job.id);
                let _ = client
                    .fail_job(&job.id, &format!("decrypt failed: {e}"))
                    .await;
                return;
            }
        }
    }

    // Streaming path requires three independent agreements, so a single party
    // can't force it:
    //   1. this node has streaming enabled locally (`streaming_enabled`)
    //   2. the orchestrator set the cleartext dispatch marker (it set up the SSE
    //      relay) — read from the ORIGINAL job payload, alongside `enc`
    //   3. the client asked for it in the AUTHENTICATED, sealed payload — read from
    //      the DECRYPTED inner payload (a relay can't flip this)
    let dispatch_stream = job
        .input_payload
        .as_ref()
        .and_then(|p| p.get("stream"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let sealed_stream = effective_job
        .input_payload
        .as_ref()
        .and_then(|p| p.get("stream"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if streaming_enabled
        && dispatch_stream
        && sealed_stream
        && enc_version == crate::encryption::EncVersion::V2
        && effective_job.job_type == "inference"
    {
        if let Some(resp_pub) = response_pubkey {
            process_inference_stream(client, engine, &effective_job, node_secret, &resp_pub).await;
            return;
        }
    }

    let result = match effective_job.job_type.as_str() {
        "inference" => execute_inference(engine, &effective_job).await,
        _ => {
            tracing::warn!("Unsupported job type: {}", effective_job.job_type);
            Err(format!("Unsupported job type: {}", effective_job.job_type))
        }
    };

    match result {
        Ok(output) => {
            if let Some(resp_pub) = response_pubkey {
                // ── E2E: seal the result to the caller's response key ──
                let result_bytes = output.to_string();
                let usage = output.get("usage").cloned();
                // Reply in the SAME version the caller used (v2 = HKDF + AAD).
                let sealed = if enc_version == crate::encryption::EncVersion::V2 {
                    crate::encryption::encrypt_for_recipient_v2(&resp_pub, result_bytes.as_bytes())
                } else {
                    crate::encryption::encrypt_for_recipient(&resp_pub, result_bytes.as_bytes())
                };
                let algo = if enc_version == crate::encryption::EncVersion::V2 {
                    crate::encryption::ALGO_V2
                } else {
                    "x25519-xchacha20poly1305"
                };
                match sealed {
                    Ok((sealed, ephemeral_pub)) => {
                        let ciphertext_b58 = bs58::encode(&sealed).into_string();
                        // Sign an envelope over the *public* ciphertext + job id so the
                        // orchestrator can prove which node produced this result for this
                        // job (anti-replay) without ever seeing the plaintext.
                        let env_sig = crate::crypto::sign_result_envelope(
                            node_secret,
                            &job.id,
                            "sealed",
                            ciphertext_b58.as_bytes(),
                        );
                        let sealed_result = serde_json::json!({
                            "ciphertext": ciphertext_b58,
                            "ephemeral_public_key": bs58::encode(ephemeral_pub).into_string(),
                            "algorithm": algo,
                        });
                        if let Err(e) = client
                            .complete_job_sealed(&job.id, sealed_result, usage, Some(env_sig))
                            .await
                        {
                            tracing::error!("Failed to report sealed completion: {e}");
                        } else {
                            tracing::info!("Job {} completed (E2E sealed, REST)", job.id);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to seal result for job {}: {e}", job.id);
                        let _ = client.fail_job(&job.id, &format!("seal failed: {e}")).await;
                    }
                }
            } else {
                let result_str = output.to_string();
                let env_sig = crate::crypto::sign_result_envelope(
                    node_secret,
                    &job.id,
                    "plain",
                    result_str.as_bytes(),
                );
                if let Err(e) = client.complete_job(&job.id, &output, Some(env_sig)).await {
                    tracing::error!("Failed to report job completion: {e}");
                } else {
                    tracing::info!("Job {} completed", job.id);
                }
            }
        }
        Err(reason) => {
            if let Err(e) = client.fail_job(&job.id, &reason).await {
                tracing::error!("Failed to report job failure: {e}");
            } else {
                tracing::warn!("Job {} failed: {reason}", job.id);
            }
        }
    }
}

/// Parse + bound the inference parameters from a (decrypted) job payload. Shared
/// by the non-streaming and streaming paths so both apply identical validation.
fn parse_inference_params(
    payload: Option<&serde_json::Value>,
) -> Result<(Vec<ChatMessage>, f64, i32), String> {
    let payload = payload.ok_or("Job has no input payload")?;

    let messages: Vec<ChatMessage> = if let Some(msgs) = payload.get("messages") {
        serde_json::from_value(msgs.clone()).map_err(|e| format!("Invalid messages format: {e}"))?
    } else if let Some(prompt) = payload.get("prompt").and_then(|p| p.as_str()) {
        vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }]
    } else {
        return Err("Payload must contain 'messages' array or 'prompt' string".to_string());
    };

    // Bound untrusted input before handing it to the inference server.
    const MAX_MESSAGES: usize = 256;
    const MAX_TOTAL_PROMPT_BYTES: usize = 4 * 1024 * 1024; // 4 MiB
    if messages.len() > MAX_MESSAGES {
        return Err(format!(
            "too many messages ({} > {MAX_MESSAGES})",
            messages.len()
        ));
    }
    let total_bytes: usize = messages
        .iter()
        .map(|m| m.content.len() + m.role.len())
        .sum();
    if total_bytes > MAX_TOTAL_PROMPT_BYTES {
        return Err(format!(
            "prompt too large ({total_bytes} bytes > {MAX_TOTAL_PROMPT_BYTES})"
        ));
    }

    let temperature = payload
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.7)
        .clamp(0.0, 2.0);

    let max_tokens = payload
        .get("max_tokens")
        .and_then(|t| t.as_i64())
        .unwrap_or(2048)
        .clamp(1, 8192) as i32;

    Ok((messages, temperature, max_tokens))
}

async fn execute_inference(
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
) -> Result<serde_json::Value, String> {
    let engine = engine
        .as_ref()
        .ok_or("No inference engine configured — start with --model-path")?;

    let (messages, temperature, max_tokens) = parse_inference_params(job.input_payload.as_ref())?;

    let result = engine
        .chat_completion(messages, temperature, max_tokens)
        .await?;

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

/// Streaming inference: run llama-server with streaming, seal each token batch as
/// an ordered chunk, and POST it to the orchestrator (which relays it to the
/// client over SSE). Each chunk's AAD binds its seq + final flag; the node also
/// signs an envelope per chunk so the orchestrator can attribute it. Billing
/// happens on the final chunk's usage; if the client aborts, chunk POSTs start
/// failing and we stop early (no final → no charge).
/// Seal one stream chunk, sign its envelope, and POST it. Returns `Ok(true)` if
/// the orchestrator reports the client is gone (stop early), `Ok(false)` to keep
/// going, `Err` on a hard failure.
#[allow(clippy::too_many_arguments)]
async fn seal_post_chunk(
    client: &OrchestratorClient,
    sealer: &crate::encryption::StreamSealer,
    node_secret: &[u8; 32],
    job_id: &str,
    eph_b58: &str,
    seq: u64,
    is_final: bool,
    plaintext: &[u8],
    usage: Option<serde_json::Value>,
) -> Result<bool, String> {
    let ct = sealer.seal_chunk(plaintext, seq, is_final)?;
    let kind = format!("stream:{seq}:{}", if is_final { 1 } else { 0 });
    let sig = crate::crypto::sign_result_envelope(node_secret, job_id, &kind, ct.as_bytes());
    let eph = if seq == 0 { Some(eph_b58) } else { None };
    client
        .post_chunk(job_id, seq, is_final, eph, &ct, usage, Some(sig))
        .await
}

async fn process_inference_stream(
    client: &OrchestratorClient,
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    node_secret: &[u8; 32],
    resp_pub: &[u8; 32],
) {
    let engine = match engine {
        Some(e) => e.clone(),
        None => {
            let _ = client
                .fail_job(&job.id, "No inference engine configured")
                .await;
            return;
        }
    };

    let (messages, temperature, max_tokens) =
        match parse_inference_params(job.input_payload.as_ref()) {
            Ok(v) => v,
            Err(e) => {
                let _ = client.fail_job(&job.id, &e).await;
                return;
            }
        };

    // Per-request nonce chosen by the client (inside the sealed prompt) — bound
    // into every chunk's AAD so a stream can't be spliced into another request.
    let req_nonce = job
        .input_payload
        .as_ref()
        .and_then(|p| p.get("nonce"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let sealer = match crate::encryption::StreamSealer::new(resp_pub, req_nonce) {
        Ok(s) => s,
        Err(e) => {
            let _ = client
                .fail_job(&job.id, &format!("stream seal init failed: {e}"))
                .await;
            return;
        }
    };
    let eph_b58 = sealer.ephemeral_public_b58().to_string();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::inference::StreamEvent>(64);
    let engine2 = engine.clone();
    let inf = tokio::spawn(async move {
        engine2
            .chat_completion_stream(messages, temperature, max_tokens, tx)
            .await
    });

    let mut seq: u64 = 0;
    let mut emitted_tokens: u32 = 0;
    let mut final_sent = false;
    let mut client_gone = false;

    while let Some(ev) = rx.recv().await {
        match ev {
            crate::inference::StreamEvent::Delta { text, tokens } => {
                emitted_tokens = emitted_tokens.saturating_add(tokens);
                match seal_post_chunk(
                    client, &sealer, node_secret, &job.id, &eph_b58, seq, false,
                    text.as_bytes(), None,
                )
                .await
                {
                    Ok(false) => seq += 1,
                    Ok(true) => {
                        client_gone = true;
                        break;
                    }
                    Err(e) => {
                        tracing::warn!("Job {} chunk {seq} post failed: {e}", job.id);
                        client_gone = true;
                        break;
                    }
                }
            }
            crate::inference::StreamEvent::Done {
                prompt_tokens,
                completion_tokens,
            } => {
                let usage = serde_json::json!({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                });
                // Only treat as success if the final chunk was actually accepted;
                // otherwise fall through to the failure path below.
                match seal_post_chunk(
                    client, &sealer, node_secret, &job.id, &eph_b58, seq, true, b"", Some(usage),
                )
                .await
                {
                    Ok(_) => final_sent = true,
                    Err(e) => tracing::warn!("Job {} final chunk post failed: {e}", job.id),
                }
                break;
            }
        }
    }

    if final_sent {
        inf.abort(); // generation already finished; ensure the task is reaped
        tracing::info!("Job {} completed (E2E stream, {} chunk(s))", job.id, seq);
        return;
    }

    if client_gone {
        // Client disconnected mid-stream. Stop the generator FIRST (drop the
        // receiver so llama-server reads stop unblocking the task, then abort it),
        // then settle the partial so the generated tokens aren't free. Prompt
        // tokens are unknown without [DONE]; bill completion tokens only
        // (conservative, favors the user).
        drop(rx);
        inf.abort();
        let usage = serde_json::json!({
            "prompt_tokens": 0,
            "completion_tokens": emitted_tokens,
            "total_tokens": emitted_tokens,
        });
        let _ = seal_post_chunk(
            client, &sealer, node_secret, &job.id, &eph_b58, seq, true, b"", Some(usage),
        )
        .await;
        tracing::warn!(
            "Job {} client aborted after {} chunk(s); settled partial ({} tokens)",
            job.id,
            seq,
            emitted_tokens
        );
        return;
    }

    // Generation failure (upstream EOF without [DONE], or the final post failed) —
    // abort the client stream and fail the job. NO billing.
    let inf_res = inf
        .await
        .unwrap_or_else(|_| Err("inference task panicked".to_string()));
    let reason = inf_res.err().unwrap_or_else(|| "stream ended without completion".to_string());
    if let Err(e) = client.report_stream_error(&job.id).await {
        tracing::warn!("stream error report failed for {}: {e}", job.id);
        let _ = client.fail_job(&job.id, &format!("stream failed: {reason}")).await;
    }
    tracing::warn!("Job {} stream failed: {reason}", job.id);
}
