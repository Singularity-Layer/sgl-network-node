//! Mac step 1 (2026-10-30): a Secure Enclave identity key, reported on the heartbeat.
//!
//! RECORD-ONLY. The orchestrator verifies and stores it; nothing routes or gates on it yet, so a
//! node without it (Linux, Windows, an Intel Mac, a VM, `SGL_SE_IDENTITY=0`) serves exactly as
//! before. Every failure here is a warning and `None`, never an error that could stop a node.
//!
//! The key is generated inside the enclave (src/enclave/se.swift). The file we keep is an
//! enclave-encrypted blob that only this Mac can use; copying it elsewhere is useless.

use std::path::Path;

/// What the heartbeat carries. All bs58, matching the node's other keys and signatures.
#[derive(Debug, Clone)]
pub struct SeIdentity {
    /// 65-byte X9.63 uncompressed P-256 public key.
    pub public_key: String,
    /// Enclave (P-256, raw r||s) signature over the sebind blob: proves possession.
    pub bind_signature: String,
    /// ed25519 identity signature over the same blob: "this enclave key is mine".
    pub bind_ed25519_signature: String,
}

const BLOB_FILE: &str = "se_identity.key";

/// Runtime kill switch: `SGL_SE_IDENTITY=0` (or `false`/`off`) disables it entirely.
fn disabled_by_env() -> bool {
    matches!(
        std::env::var("SGL_SE_IDENTITY").map(|v| v.to_ascii_lowercase()).as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Load this node's enclave key (creating it on first run) and sign the sebind blob.
pub fn load_or_create(config_dir: &Path, ed25519_secret: &[u8; 32], node_id: &str) -> Option<SeIdentity> {
    if disabled_by_env() {
        tracing::info!("Secure Enclave identity disabled (SGL_SE_IDENTITY)");
        return None;
    }
    if !enclave::available() {
        tracing::debug!("Secure Enclave not available on this machine; no enclave identity");
        return None;
    }
    let path = config_dir.join(BLOB_FILE);

    // Reuse the existing key when it still loads; otherwise (first run, a blob from another
    // Mac, an enclave reset) make a new one. A new key is fine: step 1 only records it.
    let existing = std::fs::read(&path).ok().and_then(|blob| enclave::public_key(&blob).map(|p| (blob, p)));
    let (blob, se_pub) = match existing {
        Some(v) => v,
        None => {
            let (blob, se_pub) = match enclave::create() {
                Some(v) => v,
                None => {
                    tracing::warn!("Secure Enclave key creation failed; continuing without it");
                    return None;
                }
            };
            if let Err(e) = crate::crypto::write_file_0600(&path, &blob) {
                tracing::warn!("could not save the Secure Enclave key blob ({e}); continuing without it");
                return None;
            }
            (blob, se_pub)
        }
    };

    let (bind_blob, ed_sig) = crate::crypto::sebind_v1(ed25519_secret, node_id, &se_pub)?;
    let se_sig = match enclave::sign(&blob, &bind_blob) {
        Some(s) => s,
        None => {
            tracing::warn!("Secure Enclave signing failed; continuing without the enclave identity");
            return None;
        }
    };
    Some(SeIdentity {
        public_key: bs58::encode(se_pub).into_string(),
        bind_signature: bs58::encode(se_sig).into_string(),
        bind_ed25519_signature: ed_sig,
    })
}

#[cfg(sgl_se)]
mod enclave {
    extern "C" {
        fn sgl_se_available() -> i32;
        fn sgl_se_create(out_blob: *mut u8, blob_cap: usize, out_blob_len: *mut usize, out_pub: *mut u8) -> i32;
        fn sgl_se_public_key(blob: *const u8, blob_len: usize, out_pub: *mut u8) -> i32;
        fn sgl_se_sign(blob: *const u8, blob_len: usize, msg: *const u8, msg_len: usize, out_sig: *mut u8) -> i32;
    }

    const BLOB_CAP: usize = 4096;

    pub fn available() -> bool {
        unsafe { sgl_se_available() == 1 }
    }

    pub fn create() -> Option<(Vec<u8>, Vec<u8>)> {
        let mut blob = vec![0u8; BLOB_CAP];
        let mut len = 0usize;
        let mut pubk = vec![0u8; 65];
        let rc = unsafe { sgl_se_create(blob.as_mut_ptr(), blob.len(), &mut len, pubk.as_mut_ptr()) };
        if rc != 0 || len == 0 || len > BLOB_CAP {
            return None;
        }
        blob.truncate(len);
        Some((blob, pubk))
    }

    pub fn public_key(blob: &[u8]) -> Option<Vec<u8>> {
        if blob.is_empty() {
            return None;
        }
        let mut pubk = vec![0u8; 65];
        let rc = unsafe { sgl_se_public_key(blob.as_ptr(), blob.len(), pubk.as_mut_ptr()) };
        (rc == 0).then_some(pubk)
    }

    pub fn sign(blob: &[u8], msg: &[u8]) -> Option<Vec<u8>> {
        let mut sig = vec![0u8; 64];
        let rc = unsafe { sgl_se_sign(blob.as_ptr(), blob.len(), msg.as_ptr(), msg.len(), sig.as_mut_ptr()) };
        (rc == 0).then_some(sig)
    }
}

/// Every non-macOS build (and macOS with SGL_NO_SWIFT=1): no enclave, the node runs as before.
#[cfg(not(sgl_se))]
mod enclave {
    pub fn available() -> bool { false }
    pub fn create() -> Option<(Vec<u8>, Vec<u8>)> { None }
    pub fn public_key(_blob: &[u8]) -> Option<Vec<u8>> { None }
    pub fn sign(_blob: &[u8], _msg: &[u8]) -> Option<Vec<u8>> { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NODE_ID: &str = "3f8a1c2e-4b5d-4e6f-8a9b-0c1d2e3f4a5b";
    // The two load_or_create tests share the SGL_SE_IDENTITY env var; serialise them.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn sebind_blob_layout_is_fixed() {
        let secret = [7u8; 32];
        let mut se_pub = vec![0x04u8];
        se_pub.extend_from_slice(&[9u8; 64]);
        let (blob, _sig) = crate::crypto::sebind_v1(&secret, NODE_ID, &se_pub).unwrap();
        assert!(blob.starts_with(b"SGL-NODE-SEBIND-v1\0"));
        assert_eq!(blob.len(), 19 + 16 + 32 + 65);
        assert_eq!(&blob[blob.len() - 65..], &se_pub[..]);
    }

    #[test]
    fn sebind_rejects_compressed_or_short_keys() {
        assert!(crate::crypto::sebind_v1(&[7u8; 32], NODE_ID, &[0x02u8; 33]).is_none());
        assert!(crate::crypto::sebind_v1(&[7u8; 32], NODE_ID, &[0x04u8; 64]).is_none());
    }

    #[test]
    fn kill_switch_returns_none() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("SGL_SE_IDENTITY", "0");
        let dir = std::env::temp_dir();
        assert!(load_or_create(&dir, &[7u8; 32], NODE_ID).is_none());
        std::env::remove_var("SGL_SE_IDENTITY");
    }

    /// Real enclave round trip. Skips quietly where there is no enclave (CI VMs, Linux).
    #[test]
    fn enclave_round_trip_when_available() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !enclave::available() {
            eprintln!("enclave_round_trip: no Secure Enclave here, skipped");
            return;
        }
        let dir = std::env::temp_dir().join(format!("sgl-se-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = load_or_create(&dir, &[7u8; 32], NODE_ID).expect("enclave identity");
        let b = load_or_create(&dir, &[7u8; 32], NODE_ID).expect("reloaded identity");
        assert_eq!(a.public_key, b.public_key, "the saved blob must reload the same key");
        eprintln!("enclave_round_trip: real enclave key {}", a.public_key);
        // SGL_SE_FIXTURE=1 prints a real signed sample for the orchestrator's contract test.
        if std::env::var("SGL_SE_FIXTURE").as_deref() == Ok("1") {
            use ed25519_dalek::SigningKey;
            let ed_pub = SigningKey::from_bytes(&[7u8; 32]).verifying_key().to_bytes();
            eprintln!(
                "SE_FIXTURE {{\"node_id\":\"{NODE_ID}\",\"node_public_key\":\"{}\",\"se_public_key\":\"{}\",\"se_bind_signature\":\"{}\",\"se_bind_ed25519_signature\":\"{}\"}}",
                bs58::encode(ed_pub).into_string(), a.public_key, a.bind_signature, a.bind_ed25519_signature
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
