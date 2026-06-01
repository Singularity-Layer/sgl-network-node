use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng, Payload},
    XChaCha20Poly1305, XNonce,
};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::{Digest, Sha256};
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

/// Hard cap on a sealed prompt blob (base58 chars) from an untrusted orchestrator,
/// so a malicious/huge payload can't force an unbounded base58-decode + allocation
/// before we ever look at it. 8 MiB of base58 is far more than any real prompt.
const MAX_SEALED_B58_LEN: usize = 8 * 1024 * 1024;

/// Wire algorithm tags.
const ALGO_V1: &str = "x25519-xchacha20poly1305";
pub const ALGO_V2: &str = "x25519-xchacha20poly1305-hkdf-v2";

// v2 HKDF domain separation. Must match the orchestrator + cloud e2e.ts byte-for-byte.
const HKDF_SALT: &[u8] = b"sgl-e2e-v2-salt";
const HKDF_INFO_INPUT: &[u8] = b"sgl-e2e-v2-input";
const HKDF_INFO_OUTPUT: &[u8] = b"sgl-e2e-v2-output";

/// Which E2E scheme a payload used. The node replies in the same version it received.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncVersion {
    V1,
    V2,
}

/// v2 key derivation: HKDF-SHA256 over the raw X25519 shared secret with a
/// domain-separating `info`. Replaces v1's bare SHA256(shared).
fn hkdf_key(shared: &[u8; 32], info: &[u8]) -> [u8; 32] {
    let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT), shared);
    let mut okm = [0u8; 32];
    hk.expand(info, &mut okm)
        .expect("hkdf expand 32 bytes is valid");
    okm
}

/// AAD bound into a v2 *input* (client → node): the recipient node key, the
/// sender's ephemeral key, and — critically — the response key. Binding the
/// response key means a relay (orchestrator) that swaps it to redirect the reply
/// breaks input decryption instead of silently succeeding.
fn aad_input(node_b58: &str, eph_b58: &str, resp_b58: &str) -> Vec<u8> {
    format!("sgl-aad/v2/input|node={node_b58}|eph={eph_b58}|resp={resp_b58}").into_bytes()
}

/// AAD bound into a v2 *output* (node → caller): the recipient response key and
/// the node's ephemeral key.
fn aad_output(resp_b58: &str, eph_b58: &str) -> Vec<u8> {
    format!("sgl-aad/v2/output|resp={resp_b58}|eph={eph_b58}").into_bytes()
}

fn bs58_to_32(s: &str) -> Result<[u8; 32], String> {
    let v = bs58::decode(s)
        .into_vec()
        .map_err(|e| format!("bad base58: {e}"))?;
    if v.len() != 32 {
        return Err(format!("expected 32 bytes, got {}", v.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    Ok(out)
}

/// If the job payload is sealed (`enc` object present), decrypt it with the node's
/// X25519 key and return (plaintext_payload, response_pubkey, version). Otherwise
/// return the payload unchanged (the plaintext path). Supports both v1 (legacy
/// SHA256 KDF, no AAD) and v2 (HKDF + AAD) by the `enc.algorithm` tag.
pub fn unseal_input(
    payload: &serde_json::Value,
    node_ed25519_secret: &[u8; 32],
) -> Result<(serde_json::Value, Option<[u8; 32]>, EncVersion), String> {
    let enc = match payload.get("enc") {
        Some(e) if e.is_object() => e,
        _ => return Ok((payload.clone(), None, EncVersion::V1)),
    };

    let ciphertext_b58 = enc
        .get("ciphertext")
        .and_then(|v| v.as_str())
        .ok_or("enc.ciphertext missing")?;
    if ciphertext_b58.len() > MAX_SEALED_B58_LEN {
        return Err("sealed ciphertext exceeds maximum size".to_string());
    }
    let ephemeral_b58 = enc
        .get("client_ephemeral_pubkey")
        .and_then(|v| v.as_str())
        .ok_or("enc.client_ephemeral_pubkey missing")?;
    let response_b58 = enc
        .get("client_response_pubkey")
        .and_then(|v| v.as_str())
        .ok_or("enc.client_response_pubkey missing")?;

    let ciphertext = bs58::decode(ciphertext_b58)
        .into_vec()
        .map_err(|e| format!("bad ciphertext base58: {e}"))?;
    let ephemeral = bs58_to_32(ephemeral_b58)?;
    let response_pub = bs58_to_32(response_b58)?;

    let kp = EncryptionKeypair::from_ed25519_seed(node_ed25519_secret);

    // Fail-closed algorithm negotiation: v2 (preferred), explicit/absent v1 (legacy,
    // accepted during rollout), and ANYTHING ELSE is rejected rather than silently
    // downgraded to v1.
    let (plaintext, version) = match enc.get("algorithm").and_then(|v| v.as_str()) {
        Some(a) if a == ALGO_V2 => {
            let aad = aad_input(&kp.public_key_bs58(), ephemeral_b58, response_b58);
            (
                kp.decrypt_v2(&ephemeral, &ciphertext, &aad)?,
                EncVersion::V2,
            )
        }
        Some(a) if a == ALGO_V1 => (kp.decrypt(&ephemeral, &ciphertext)?, EncVersion::V1),
        None => (kp.decrypt(&ephemeral, &ciphertext)?, EncVersion::V1),
        Some(other) => return Err(format!("unsupported enc.algorithm: {other}")),
    };

    let inner: serde_json::Value = serde_json::from_slice(&plaintext)
        .map_err(|e| format!("decrypted payload is not valid JSON: {e}"))?;

    Ok((inner, Some(response_pub), version))
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

    fn shared_with(&self, sender_public: &[u8; 32]) -> Result<[u8; 32], String> {
        let sender_pk = PublicKey::from(*sender_public);
        let shared = self.secret.diffie_hellman(&sender_pk);
        // Reject low-order/contributory-failure points (all-zero shared secret).
        if !shared.was_contributory() {
            return Err("invalid peer key (non-contributory shared secret)".to_string());
        }
        Ok(*shared.as_bytes())
    }

    /// v1 decrypt: bare SHA256(shared) key, no AAD.
    pub fn decrypt(&self, sender_public: &[u8; 32], ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        if ciphertext.len() < 24 {
            return Err("Ciphertext too short".to_string());
        }
        let shared = self.shared_with(sender_public)?;
        let symmetric_key: [u8; 32] = Sha256::digest(shared).into();

        let nonce = XNonce::from_slice(&ciphertext[..24]);
        let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
            .map_err(|e| format!("Cipher init failed: {e}"))?;
        cipher
            .decrypt(nonce, &ciphertext[24..])
            .map_err(|e| format!("Decryption failed: {e}"))
    }

    /// v2 decrypt: HKDF-SHA256 key + AEAD AAD.
    pub fn decrypt_v2(
        &self,
        sender_public: &[u8; 32],
        ciphertext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, String> {
        if ciphertext.len() < 24 {
            return Err("Ciphertext too short".to_string());
        }
        let shared = self.shared_with(sender_public)?;
        let symmetric_key = hkdf_key(&shared, HKDF_INFO_INPUT);

        let nonce = XNonce::from_slice(&ciphertext[..24]);
        let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
            .map_err(|e| format!("Cipher init failed: {e}"))?;
        cipher
            .decrypt(
                nonce,
                Payload {
                    msg: &ciphertext[24..],
                    aad,
                },
            )
            .map_err(|e| format!("Decryption failed: {e}"))
    }
}

/// v1 seal to a recipient X25519 key: bare SHA256(shared) key, no AAD.
pub fn encrypt_for_recipient(
    recipient_public: &[u8; 32],
    plaintext: &[u8],
) -> Result<(Vec<u8>, [u8; 32]), String> {
    seal_common(recipient_public, plaintext, None)
}

/// v2 seal to a recipient X25519 key: HKDF-SHA256 key + AEAD AAD binding the
/// recipient (response) key and the node ephemeral key.
pub fn encrypt_for_recipient_v2(
    recipient_public: &[u8; 32],
    plaintext: &[u8],
) -> Result<(Vec<u8>, [u8; 32]), String> {
    let recipient_b58 = bs58::encode(recipient_public).into_string();
    seal_common(recipient_public, plaintext, Some(recipient_b58))
}

/// Shared seal path. `v2_recipient_b58 = Some(..)` selects v2 (HKDF + AAD); `None`
/// is v1 (SHA256, no AAD).
fn seal_common(
    recipient_public: &[u8; 32],
    plaintext: &[u8],
    v2_recipient_b58: Option<String>,
) -> Result<(Vec<u8>, [u8; 32]), String> {
    let ephemeral_secret = EphemeralSecret::random_from_rng(OsRng);
    let ephemeral_public = PublicKey::from(&ephemeral_secret);

    let recipient_pk = PublicKey::from(*recipient_public);
    let shared = ephemeral_secret.diffie_hellman(&recipient_pk);
    if !shared.was_contributory() {
        return Err("invalid recipient key (non-contributory shared secret)".to_string());
    }

    let mut nonce_bytes = [0u8; 24];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = XNonce::from_slice(&nonce_bytes);

    let (symmetric_key, aad): ([u8; 32], Option<Vec<u8>>) = match &v2_recipient_b58 {
        Some(resp_b58) => {
            let eph_b58 = bs58::encode(ephemeral_public.as_bytes()).into_string();
            (
                hkdf_key(shared.as_bytes(), HKDF_INFO_OUTPUT),
                Some(aad_output(resp_b58, &eph_b58)),
            )
        }
        None => (Sha256::digest(shared.as_bytes()).into(), None),
    };

    let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
        .map_err(|e| format!("Cipher init failed: {e}"))?;
    let ciphertext = match &aad {
        Some(a) => cipher.encrypt(
            nonce,
            Payload {
                msg: plaintext,
                aad: a,
            },
        ),
        None => cipher.encrypt(nonce, plaintext),
    }
    .map_err(|e| format!("Encryption failed: {e}"))?;

    let mut output = Vec::with_capacity(24 + ciphertext.len());
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&ciphertext);
    Ok((output, *ephemeral_public.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vector produced by the cloud/orchestrator TS (sgl-network-cloud/src/lib/e2e.ts)
    // sealInputV2 with node ed25519 secret = [0x42;32]. Proves browser-seal →
    // node-decrypt is byte-compatible across Rust + TS (HKDF + AAD + base58).
    #[test]
    fn ts_v2_input_vector_decrypts() {
        let node_secret = [0x42u8; 32];
        let payload = serde_json::json!({
            "enc": {
                "ciphertext": "gbEA6dFFVPxdar6e8QsjPKWj7xHcBo32nAqweQC5arnt4M5LmHhjREKoUTdVZsU6mmkxKu1XmvEo4oG8EUySndq2ytTyzDgyfMjyBSmPE2fqjdPDzKYtdrC2kZbAfCXv227GczHgmtQBqchA5qMB5ydxgxYnk9V8jb8sifTjHM61iEQkisdwYCqna",
                "client_ephemeral_pubkey": "DQFdwcBsqukJEBn9UNfQruaTHKHxHFMVRA2B5qZuFdfB",
                "client_response_pubkey": "2L54SXdEHm5mraF2X2GPid3m4PSkwVehEvhk487mWTx8",
                "algorithm": "x25519-xchacha20poly1305-hkdf-v2"
            }
        });
        let (inner, resp, ver) = unseal_input(&payload, &node_secret).expect("v2 decrypt");
        assert_eq!(ver, EncVersion::V2);
        assert!(resp.is_some());
        let expected: serde_json::Value = serde_json::from_str(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"cross-lang v2 test\"}],\"temperature\":0.7,\"max_tokens\":512}"
        ).unwrap();
        assert_eq!(inner, expected);
    }

    // Tampering with the response key (as a malicious relay would) must break
    // input decryption, because it's bound into the AAD.
    #[test]
    fn v2_response_key_swap_is_rejected() {
        let node_secret = [0x42u8; 32];
        let payload = serde_json::json!({
            "enc": {
                "ciphertext": "gbEA6dFFVPxdar6e8QsjPKWj7xHcBo32nAqweQC5arnt4M5LmHhjREKoUTdVZsU6mmkxKu1XmvEo4oG8EUySndq2ytTyzDgyfMjyBSmPE2fqjdPDzKYtdrC2kZbAfCXv227GczHgmtQBqchA5qMB5ydxgxYnk9V8jb8sifTjHM61iEQkisdwYCqna",
                "client_ephemeral_pubkey": "DQFdwcBsqukJEBn9UNfQruaTHKHxHFMVRA2B5qZuFdfB",
                "client_response_pubkey": "11111111111111111111111111111111", // swapped
                "algorithm": "x25519-xchacha20poly1305-hkdf-v2"
            }
        });
        assert!(unseal_input(&payload, &node_secret).is_err());
    }

    // Rust v2 seal → Rust v2 open round-trip (output direction self-consistency).
    #[test]
    fn v2_output_roundtrip() {
        let resp_secret = StaticSecret::from([0x07u8; 32]);
        let resp_pub = PublicKey::from(&resp_secret);
        let msg = b"hello v2 output";
        let (blob, node_eph) = encrypt_for_recipient_v2(resp_pub.as_bytes(), msg).unwrap();
        // Recompute the way a recipient would: shared = resp_secret * node_eph.
        let shared = resp_secret.diffie_hellman(&PublicKey::from(node_eph));
        let key = hkdf_key(shared.as_bytes(), HKDF_INFO_OUTPUT);
        let aad = aad_output(
            &bs58::encode(resp_pub.as_bytes()).into_string(),
            &bs58::encode(node_eph).into_string(),
        );
        let cipher = XChaCha20Poly1305::new_from_slice(&key).unwrap();
        let nonce = XNonce::from_slice(&blob[..24]);
        let pt = cipher
            .decrypt(
                nonce,
                Payload {
                    msg: &blob[24..],
                    aad: &aad,
                },
            )
            .unwrap();
        assert_eq!(pt, msg);
    }
}
