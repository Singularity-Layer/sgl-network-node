use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    XChaCha20Poly1305, XNonce,
};
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};
use sha2::{Sha256, Digest};
use rand::RngCore;

fn bs58_to_32(s: &str) -> Result<[u8; 32], String> {
    let v = bs58::decode(s).into_vec().map_err(|e| format!("bad base58: {e}"))?;
    if v.len() != 32 {
        return Err(format!("expected 32 bytes, got {}", v.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    Ok(out)
}

/// If the job payload is sealed (`enc` object present), decrypt it with the
/// node's X25519 key and return (plaintext_payload, response_pubkey). Otherwise
/// return the payload unchanged with no response key (the plaintext path).
/// Shared by both the WebSocket push path and the REST poll path.
pub fn unseal_input(
    payload: &serde_json::Value,
    node_ed25519_secret: &[u8; 32],
) -> Result<(serde_json::Value, Option<[u8; 32]>), String> {
    let enc = match payload.get("enc") {
        Some(e) if e.is_object() => e,
        _ => return Ok((payload.clone(), None)),
    };

    let ciphertext_b58 = enc.get("ciphertext").and_then(|v| v.as_str())
        .ok_or("enc.ciphertext missing")?;
    let ephemeral_b58 = enc.get("client_ephemeral_pubkey").and_then(|v| v.as_str())
        .ok_or("enc.client_ephemeral_pubkey missing")?;
    let response_b58 = enc.get("client_response_pubkey").and_then(|v| v.as_str())
        .ok_or("enc.client_response_pubkey missing")?;

    let ciphertext = bs58::decode(ciphertext_b58).into_vec()
        .map_err(|e| format!("bad ciphertext base58: {e}"))?;
    let ephemeral = bs58_to_32(ephemeral_b58)?;
    let response_pub = bs58_to_32(response_b58)?;

    let kp = EncryptionKeypair::from_ed25519_seed(node_ed25519_secret);
    let plaintext = kp.decrypt(&ephemeral, &ciphertext)?;
    let inner: serde_json::Value = serde_json::from_slice(&plaintext)
        .map_err(|e| format!("decrypted payload is not valid JSON: {e}"))?;

    Ok((inner, Some(response_pub)))
}

pub struct EncryptionKeypair {
    secret: StaticSecret,
    public: PublicKey,
}

impl EncryptionKeypair {
    pub fn from_ed25519_seed(ed25519_secret: &[u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"sgl-x25519-derive:");
        hasher.update(ed25519_secret);
        let derived: [u8; 32] = hasher.finalize().into();

        let secret = StaticSecret::from(derived);
        let public = PublicKey::from(&secret);
        Self { secret, public }
    }

    pub fn public_key_bs58(&self) -> String {
        bs58::encode(self.public.as_bytes()).into_string()
    }

    pub fn decrypt(&self, sender_public: &[u8; 32], ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        if ciphertext.len() < 24 {
            return Err("Ciphertext too short".to_string());
        }

        let sender_pk = PublicKey::from(*sender_public);
        let shared = self.secret.diffie_hellman(&sender_pk);

        let mut key_hasher = Sha256::new();
        key_hasher.update(shared.as_bytes());
        let symmetric_key: [u8; 32] = key_hasher.finalize().into();

        let nonce = XNonce::from_slice(&ciphertext[..24]);
        let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
            .map_err(|e| format!("Cipher init failed: {e}"))?;

        cipher.decrypt(nonce, &ciphertext[24..])
            .map_err(|e| format!("Decryption failed: {e}"))
    }
}

pub fn encrypt_for_recipient(
    recipient_public: &[u8; 32],
    plaintext: &[u8],
) -> Result<(Vec<u8>, [u8; 32]), String> {
    let ephemeral_secret = EphemeralSecret::random_from_rng(OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);

    let recipient_pk = PublicKey::from(*recipient_public);
    let shared = ephemeral_secret.diffie_hellman(&recipient_pk);

    let mut key_hasher = Sha256::new();
    key_hasher.update(shared.as_bytes());
    let symmetric_key: [u8; 32] = key_hasher.finalize().into();

    let mut nonce_bytes = [0u8; 24];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = XNonce::from_slice(&nonce_bytes);

    let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
        .map_err(|e| format!("Cipher init failed: {e}"))?;

    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| format!("Encryption failed: {e}"))?;

    let mut output = Vec::with_capacity(24 + ciphertext.len());
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&ciphertext);

    Ok((output, *ephemeral_public.as_bytes()))
}

pub fn encrypt_result(
    node_secret: &[u8; 32],
    plaintext: &[u8],
) -> Result<EncryptedPayload, String> {
    let keypair = EncryptionKeypair::from_ed25519_seed(node_secret);

    let ephemeral_secret = EphemeralSecret::random_from_rng(OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);

    let shared = ephemeral_secret.diffie_hellman(&keypair.public);

    let mut key_hasher = Sha256::new();
    key_hasher.update(shared.as_bytes());
    let symmetric_key: [u8; 32] = key_hasher.finalize().into();

    let mut nonce_bytes = [0u8; 24];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = XNonce::from_slice(&nonce_bytes);

    let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
        .map_err(|e| format!("Cipher init failed: {e}"))?;

    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| format!("Encryption failed: {e}"))?;

    let mut sealed = Vec::with_capacity(24 + ciphertext.len());
    sealed.extend_from_slice(&nonce_bytes);
    sealed.extend_from_slice(&ciphertext);

    Ok(EncryptedPayload {
        ciphertext: bs58::encode(&sealed).into_string(),
        ephemeral_public_key: bs58::encode(ephemeral_public.as_bytes()).into_string(),
        algorithm: "x25519-xchacha20poly1305".to_string(),
    })
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct EncryptedPayload {
    pub ciphertext: String,
    pub ephemeral_public_key: String,
    pub algorithm: String,
}
