mod config;
mod crypto;
mod encryption;
mod inference;
mod node;
mod orchestrator;
mod runtime_hardening;
mod service;
mod tee;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "sgl",
    version,
    about = "SGL Network node agent — earn $SGL by providing confidential AI compute"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Orchestrator URL
    #[arg(long, default_value = "https://grid.x402compute.cc", global = true)]
    orchestrator_url: String,

    /// Config directory
    #[arg(long, global = true)]
    config_dir: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize the node: generate keys and register with the orchestrator
    Init {
        /// Staked Solana wallet address this node operates under
        #[arg(long)]
        wallet: String,

        /// TEE type on this machine
        #[arg(long, default_value = "apple_se")]
        tee_type: String,

        /// Available models (comma-separated)
        #[arg(long)]
        models: Option<String>,
    },

    /// Log in via browser and register this node (recommended)
    Login {
        /// TEE type on this machine
        #[arg(long, default_value = "apple_se")]
        tee_type: String,

        /// Available models (comma-separated)
        #[arg(long)]
        models: Option<String>,
    },

    /// Start the node: begin heartbeating and processing jobs
    Start {
        /// Path to GGUF model file for inference
        #[arg(long)]
        model_path: Option<String>,

        /// Model name to advertise (e.g. "llama-3.2-3b")
        #[arg(long)]
        model_name: Option<String>,

        /// Port for local llama-server (default: 8081)
        #[arg(long, default_value = "8081")]
        inference_port: u16,

        /// Quick preset: percentage of system resources to dedicate (1-100).
        /// Sets threads, GPU layers, and concurrent jobs proportionally.
        /// Individual flags below override the preset values.
        #[arg(long, default_value = "100", value_parser = clap::value_parser!(u8).range(1..=100))]
        resource_percent: u8,

        /// CPU threads for inference (overrides --resource-percent calculation)
        #[arg(long)]
        threads: Option<u32>,

        /// GPU layers to offload to Metal (0 = CPU only, 99 = all layers)
        #[arg(long)]
        gpu_layers: Option<u32>,

        /// Context window size in tokens
        #[arg(long, default_value = "4096")]
        context_size: u32,

        /// Max concurrent jobs this node will accept
        #[arg(long, default_value = "1")]
        max_jobs: u32,

        /// Prompt batch size for processing
        #[arg(long, default_value = "512")]
        batch_size: u32,

        /// Heartbeat interval in seconds (lower = faster job pickup, more network traffic)
        #[arg(long, default_value = "5")]
        heartbeat_interval: u64,
    },

    /// Show node status, hardware capabilities, and orchestrator info
    Status,

    /// Verify attestation: sign the challenge from the orchestrator
    Attest,

    /// Detect hardware capabilities (TEE, GPU, memory)
    Detect,

    /// Install/manage the node as a background OS service (launchd/systemd).
    /// Keeps the node serving across reboots, logout, crashes, and idle sleep.
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Subcommand)]
enum ServiceAction {
    /// Install and start the node as a managed background service.
    Install {
        /// Path to GGUF model file for inference
        #[arg(long)]
        model_path: Option<String>,

        /// Model name to advertise (e.g. "llama-3.2-3b")
        #[arg(long)]
        model_name: Option<String>,

        /// Percentage of system resources to dedicate (1-100)
        #[arg(long, default_value = "50", value_parser = clap::value_parser!(u8).range(1..=100))]
        resource_percent: u8,

        /// Port for local llama-server
        #[arg(long, default_value = "8081")]
        inference_port: u16,

        /// Max concurrent jobs this node will accept
        #[arg(long, default_value = "1")]
        max_jobs: u32,

        /// Heartbeat interval in seconds
        #[arg(long, default_value = "5")]
        heartbeat_interval: u64,
    },

    /// Stop and remove the background service.
    Uninstall,

    /// Show whether the background service is installed and running.
    Status,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("sgl_node=info".parse().unwrap()),
        )
        .init();

    runtime_hardening::deny_debugger_attach();

    let cli = Cli::parse();
    let config_dir = config::resolve_config_dir(cli.config_dir.as_deref());

    match cli.command {
        Commands::Init {
            wallet,
            tee_type,
            models,
        } => {
            let models_vec: Vec<String> = models
                .map(|m| m.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            if let Err(e) = node::init(
                &config_dir,
                &cli.orchestrator_url,
                &wallet,
                &tee_type,
                &models_vec,
            )
            .await
            {
                tracing::error!("Init failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Login { tee_type, models } => {
            let models_vec: Vec<String> = models
                .map(|m| m.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            if let Err(e) =
                node::login(&config_dir, &cli.orchestrator_url, &tee_type, &models_vec).await
            {
                tracing::error!("Login failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Start {
            model_path,
            model_name,
            inference_port,
            resource_percent,
            threads,
            gpu_layers,
            context_size,
            max_jobs,
            batch_size,
            heartbeat_interval,
        } => {
            let rc = node::ResourceConfig::from_args(
                resource_percent,
                threads,
                gpu_layers,
                context_size,
                max_jobs,
                batch_size,
                heartbeat_interval,
            );
            if let Err(e) = node::start(
                &config_dir,
                &cli.orchestrator_url,
                model_path.as_deref(),
                model_name.as_deref(),
                inference_port,
                &rc,
            )
            .await
            {
                tracing::error!("Node stopped: {e}");
                std::process::exit(1);
            }
        }
        Commands::Status => {
            if let Err(e) = node::status(&config_dir, &cli.orchestrator_url).await {
                tracing::error!("Status check failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Attest => {
            if let Err(e) = node::attest(&config_dir, &cli.orchestrator_url).await {
                tracing::error!("Attestation failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Detect => {
            let caps = tee::detect();
            tee::print_capabilities(&caps);
        }
        Commands::Service { action } => {
            let result = match action {
                ServiceAction::Install {
                    model_path,
                    model_name,
                    resource_percent,
                    inference_port,
                    max_jobs,
                    heartbeat_interval,
                } => {
                    let opts = service::ServiceStartOptions {
                        model_path,
                        model_name,
                        orchestrator_url: cli.orchestrator_url.clone(),
                        resource_percent,
                        inference_port,
                        max_jobs,
                        heartbeat_interval,
                    };
                    service::install(&opts)
                }
                ServiceAction::Uninstall => service::uninstall(),
                ServiceAction::Status => service::status(),
            };
            if let Err(e) = result {
                tracing::error!("Service command failed: {e}");
                std::process::exit(1);
            }
        }
    }
}
