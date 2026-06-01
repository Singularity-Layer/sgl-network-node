use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone)]
pub struct NodeConfig {
    pub node_id: String,
    pub auth_token: String,
    pub wallet_address: String,
    pub tee_type: String,
    pub orchestrator_url: String,
    pub keypair_path: String,
}

pub fn resolve_config_dir(custom: Option<&str>) -> PathBuf {
    if let Some(dir) = custom {
        return PathBuf::from(dir);
    }

    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("sgl-node")
}

pub fn config_path(config_dir: &Path) -> PathBuf {
    config_dir.join("node.json")
}

pub fn keypair_path(config_dir: &Path) -> PathBuf {
    config_dir.join("keypair.json")
}

pub fn load_config(config_dir: &Path) -> Result<NodeConfig, String> {
    let path = config_path(config_dir);
    if !path.exists() {
        return Err(format!(
            "Node not initialized. Run `sgl init` first.\nExpected config at: {}",
            path.display()
        ));
    }

    // node.json holds the auth_token — keep it owner-only (tighten if loosened).
    crate::crypto::check_file_permissions(&path)?;

    let contents =
        std::fs::read_to_string(&path).map_err(|e| format!("Failed to read config: {e}"))?;

    serde_json::from_str(&contents).map_err(|e| format!("Failed to parse config: {e}"))
}

pub fn save_config(config_dir: &Path, config: &NodeConfig) -> Result<(), String> {
    crate::crypto::create_dir_0700(config_dir)?;

    let contents = serde_json::to_string_pretty(config)
        .map_err(|e| format!("Failed to serialize config: {e}"))?;

    crate::crypto::write_file_0600(&config_path(config_dir), contents.as_bytes())
}
