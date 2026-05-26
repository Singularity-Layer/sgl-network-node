use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use std::path::Path;

pub struct NodeKeypair {
    pub signing_key: SigningKey,
    pub public_key: VerifyingKey,
}

impl NodeKeypair {
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let public_key = signing_key.verifying_key();
        Self { signing_key, public_key }
    }

    pub fn public_key_bs58(&self) -> String {
        bs58::encode(self.public_key.as_bytes()).into_string()
    }

    pub fn sign_message(&self, message: &[u8]) -> String {
        let signature = self.signing_key.sign(message);
        bs58::encode(signature.to_bytes()).into_string()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes: Vec<u8> = self.signing_key.to_bytes()
            .iter()
            .chain(self.public_key.as_bytes().iter())
            .copied()
            .collect();

        let json = serde_json::to_string(&bytes)
            .map_err(|e| format!("Failed to serialize keypair: {e}"))?;

        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write keypair: {e}"))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("Keypair not found at {}", path.display()));
        }

        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read keypair: {e}"))?;

        let bytes: Vec<u8> = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse keypair: {e}"))?;

        if bytes.len() != 64 {
            return Err(format!("Invalid keypair length: expected 64, got {}", bytes.len()));
        }

        let secret_bytes: [u8; 32] = bytes[..32].try_into().unwrap();
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let public_key = signing_key.verifying_key();

        Ok(Self { signing_key, public_key })
    }
}
