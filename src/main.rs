mod config;
mod crypto;
mod inference;
mod node;
mod orchestrator;
mod tee;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "sgl-node", about = "SGL Network node agent — earn $SGL by providing TEE compute")]
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
        /// Solana wallet address (provider's payout wallet)
        #[arg(long)]
        wallet: String,

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

        /// Percentage of system resources to dedicate (1-100, default: 100)
        #[arg(long, default_value = "100", value_parser = clap::value_parser!(u8).range(1..=100))]
        resource_percent: u8,
    },

    /// Show node status, hardware capabilities, and orchestrator info
    Status,

    /// Verify attestation: sign the challenge from the orchestrator
    Attest,

    /// Detect hardware capabilities (TEE, GPU, memory)
    Detect,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("sgl_node=info".parse().unwrap()))
        .init();

    let cli = Cli::parse();
    let config_dir = config::resolve_config_dir(cli.config_dir.as_deref());

    match cli.command {
        Commands::Init { wallet, tee_type, models } => {
            let models_vec: Vec<String> = models
                .map(|m| m.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            if let Err(e) = node::init(&config_dir, &cli.orchestrator_url, &wallet, &tee_type, &models_vec).await {
                tracing::error!("Init failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Start { model_path, model_name, inference_port, resource_percent } => {
            if let Err(e) = node::start(
                &config_dir,
                &cli.orchestrator_url,
                model_path.as_deref(),
                model_name.as_deref(),
                inference_port,
                resource_percent,
            ).await {
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
    }
}
