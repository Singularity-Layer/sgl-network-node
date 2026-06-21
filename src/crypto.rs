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
        Self {
            signing_key,
            public_key,
        }
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

        let bytes: Vec<u8> = self
            .signing_key
            .to_bytes()
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

        let json =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read keypair: {e}"))?;

        let bytes: Vec<u8> =
            serde_json::from_str(&json).map_err(|e| format!("Failed to parse keypair: {e}"))?;

        if bytes.len() != 64 {
            return Err(format!(
                "Invalid keypair length: expected 64, got {}",
                bytes.len()
            ));
        }

        let mut secret_bytes: [u8; 32] = bytes[..32].try_into().unwrap();
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let public_key = signing_key.verifying_key();

        // Zeroize secret material from stack
        secret_bytes.iter_mut().for_each(|b| *b = 0);
        let mut bytes = bytes;
        bytes.iter_mut().for_each(|b| *b = 0);

        Ok(Self {
            signing_key,
            public_key,
        })
    }
}

#[cfg(unix)]
pub fn write_secure_file(path: &Path, data: &[u8]) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    use std::os::unix::fs::PermissionsExt;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("Failed to create secure file: {e}"))?;

    file.write_all(data)
        .map_err(|e| format!("Failed to write secure file: {e}"))?;

    // `.mode(0o600)` only applies when the file is newly created; if it already
    // existed with looser perms, tighten it now so secrets are never group/world
    // readable.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| format!("Failed to set secure permissions: {e}"))
}

#[cfg(not(unix))]
pub fn write_secure_file(path: &Path, data: &[u8]) -> Result<(), String> {
    std::fs::write(path, data).map_err(|e| format!("Failed to write file: {e}"))
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
    std::fs::create_dir_all(path).map_err(|e| format!("Failed to create config directory: {e}"))
}

#[cfg(unix)]
pub fn check_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let perms = std::fs::metadata(path)
        .map_err(|e| format!("Failed to read file metadata: {e}"))?
        .permissions();

    let mode = perms.mode() & 0o777;
    if mode & 0o077 != 0 {
        // Don't just warn — repair it. A group/world-readable key/config is a real
        // exposure, so tighten to 0600 and tell the operator we did.
        tracing::warn!(
            "{} had permissions {:o} (group/other-readable) — tightening to 600.",
            path.display(),
            mode
        );
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|e| {
            format!(
                "{} is {:o} and could not be tightened to 600: {e}. \
                 Refusing to load secrets from a world-readable file.",
                path.display(),
                mode
            )
        })?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn check_file_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Sign a canonical "result envelope" that binds the node's identity to a specific
/// job and result content (so a result can't be replayed or reassigned to another
/// job). The orchestrator verifies the same string against the node's public key —
/// it works for sealed results too because it signs the *public* ciphertext, not
/// the plaintext. Message: "sgl-result-v1\n{job_id}\n{kind}\n{sha256_hex(payload)}".
pub fn sign_result_envelope(secret: &[u8; 32], job_id: &str, kind: &str, payload: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = hex::encode(Sha256::digest(payload));
    let msg = format!("sgl-result-v1\n{job_id}\n{kind}\n{hash}");
    let sk = SigningKey::from_bytes(secret);
    bs58::encode(sk.sign(msg.as_bytes()).to_bytes()).into_string()
}

/// #94 — sign the canonical "keybind" blob binding the node's X25519 encryption key
/// to its ed25519 identity (+ node_id + key_version). A client (#105) verifies this
/// against the node's ed25519 identity and on-chain stake before sealing its prompt,
/// so a malicious/compromised orchestrator can't substitute its own X25519 key and
/// MITM the "E2E" handoff. The orchestrator relays the signature + key_version
/// untouched (it never needs the node's secret).
///
/// Blob = b"SGL-NODE-KEYBIND-v1" || 0x00 || node_id(16) || ed25519_pub(32)
///        || x25519_pub(32) || key_version(u32 LE)
/// Returns the bs58 signature, or None if node_id isn't a parseable UUID (caller then
/// simply omits the signature — fully backward-compatible).
pub fn sign_keybind_v1(
    secret: &[u8; 32],
    node_id: &str,
    x25519_pub: &[u8; 32],
    key_version: u32,
) -> Option<String> {
    let node_id_bytes = uuid_to_bytes(node_id)?;
    let sk = SigningKey::from_bytes(secret);
    let ed25519_pub = sk.verifying_key().to_bytes();
    let mut blob = Vec::with_capacity(19 + 1 + 16 + 32 + 32 + 4);
    blob.extend_from_slice(b"SGL-NODE-KEYBIND-v1");
    blob.push(0x00);
    blob.extend_from_slice(&node_id_bytes);
    blob.extend_from_slice(&ed25519_pub);
    blob.extend_from_slice(x25519_pub);
    blob.extend_from_slice(&key_version.to_le_bytes());
    Some(bs58::encode(sk.sign(&blob).to_bytes()).into_string())
}

/// Parse a UUID string (dashed or not) into its 16 raw bytes.
fn uuid_to_bytes(s: &str) -> Option<[u8; 16]> {
    let hex: String = s.chars().filter(|c| *c != '-').collect();
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

pub use create_secure_dir as create_dir_0700;
pub use write_secure_file as write_file_0600;

#[cfg(test)]
mod keybind_tests {
    use super::*;
    use ed25519_dalek::{Signature, Verifier};

    #[test]
    fn keybind_sig_verifies_against_identity() {
        let secret = [7u8; 32];
        let x25519 = [9u8; 32];
        let node_id = "f617ee39-17e3-4904-9474-112446c9326a";
        let sig_b58 = sign_keybind_v1(&secret, node_id, &x25519, 1).expect("sig");

        // Rebuild the exact blob a verifier (the client, #105) would and check it.
        let sk = SigningKey::from_bytes(&secret);
        let vk = sk.verifying_key();
        let mut blob = Vec::new();
        blob.extend_from_slice(b"SGL-NODE-KEYBIND-v1");
        blob.push(0x00);
        blob.extend_from_slice(&uuid_to_bytes(node_id).unwrap());
        blob.extend_from_slice(&vk.to_bytes());
        blob.extend_from_slice(&x25519);
        blob.extend_from_slice(&1u32.to_le_bytes());

        let sig = Signature::from_slice(&bs58::decode(&sig_b58).into_vec().unwrap()).unwrap();
        assert!(vk.verify(&blob, &sig).is_ok());

        // A tampered x25519 key must NOT verify (the MITM case).
        let mut bad = blob.clone();
        bad[19 + 1 + 16 + 32] ^= 0xFF; // flip a byte in the x25519 section
        assert!(vk.verify(&bad, &sig).is_err());
    }

    #[test]
    fn bad_uuid_returns_none() {
        assert!(sign_keybind_v1(&[0u8; 32], "not-a-uuid", &[0u8; 32], 1).is_none());
    }
}
