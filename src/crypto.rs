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
        if let Some(parent) = path.parent() {
            create_secure_dir(parent)?;
        }

        let bytes: Vec<u8> = self.signing_key.to_bytes()
            .iter()
            .chain(self.public_key.as_bytes().iter())
            .copied()
            .collect();

        let json = serde_json::to_string(&bytes)
            .map_err(|e| format!("Failed to serialize keypair: {e}"))?;

        write_secure_file(path, json.as_bytes())?;

        // Zeroize intermediate buffer
        let mut bytes = bytes;
        bytes.iter_mut().for_each(|b| *b = 0);

        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("Keypair not found at {}", path.display()));
        }

        check_file_permissions(path)?;

        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read keypair: {e}"))?;

        let bytes: Vec<u8> = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse keypair: {e}"))?;

        if bytes.len() != 64 {
            return Err(format!("Invalid keypair length: expected 64, got {}", bytes.len()));
        }

        let mut secret_bytes: [u8; 32] = bytes[..32].try_into().unwrap();
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let public_key = signing_key.verifying_key();

        // Zeroize secret material from stack
        secret_bytes.iter_mut().for_each(|b| *b = 0);
        let mut bytes = bytes;
        bytes.iter_mut().for_each(|b| *b = 0);

        Ok(Self { signing_key, public_key })
    }
}

#[cfg(unix)]
pub fn write_secure_file(path: &Path, data: &[u8]) -> Result<(), String> {
    use std::os::unix::fs::OpenOptionsExt;
    use std::io::Write;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("Failed to create secure file: {e}"))?;

    file.write_all(data)
        .map_err(|e| format!("Failed to write secure file: {e}"))
}

#[cfg(not(unix))]
pub fn write_secure_file(path: &Path, data: &[u8]) -> Result<(), String> {
    std::fs::write(path, data)
        .map_err(|e| format!("Failed to write file: {e}"))
}

#[cfg(unix)]
pub fn create_secure_dir(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;

    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .map_err(|e| format!("Failed to create config directory: {e}"))
}

#[cfg(not(unix))]
pub fn create_secure_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path)
        .map_err(|e| format!("Failed to create config directory: {e}"))
}

#[cfg(unix)]
fn check_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let perms = std::fs::metadata(path)
        .map_err(|e| format!("Failed to read file metadata: {e}"))?
        .permissions();

    let mode = perms.mode() & 0o777;
    if mode & 0o077 != 0 {
        tracing::warn!(
            "WARNING: {} has permissions {:o} — should be 600. Other users may read your private key. \
             Fix with: chmod 600 {}",
            path.display(), mode, path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_file_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

pub use write_secure_file as write_file_0600;
pub use create_secure_dir as create_dir_0700;
