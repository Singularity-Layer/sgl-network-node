//! Building the per-request stream sealer, and the one rule that must not be relaxed.
//!
//! Extracted from `node.rs` so the nonce rule is testable without a live job, engine
//! and orchestrator — and so the hot file does not carry it.

use crate::encryption::StreamSealer;
use serde_json::Value;

/// Reason reported to the orchestrator when the sealed payload has no usable nonce.
pub const MISSING_NONCE: &str =
    "sealed stream request missing required nonce (must be a non-empty string inside \
     the sealed payload)";

/// The per-request stream nonce, or `None` when the sealed payload has no usable one.
///
/// Required, never defaulted. This used to fall back to `""`, which made every
/// chunk's AAD predictable and silently removed the nonce's forgery protection.
fn require_nonce(payload: Option<&Value>) -> Option<&str> {
    payload
        .and_then(|p| p.get("nonce"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|n| !n.is_empty())
}

/// Validate the nonce and build the sealer, or return the reason to fail the job with.
pub fn init(payload: Option<&Value>, resp_pub: &[u8; 32]) -> Result<StreamSealer, String> {
    let nonce = require_nonce(payload).ok_or(MISSING_NONCE)?;
    StreamSealer::new(resp_pub, nonce).map_err(|e| format!("stream seal init failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::require_nonce;
    use serde_json::json;

    #[test]
    fn accepts_a_real_nonce() {
        let payload = json!({ "nonce": "7Yc2Kq1mFbA9", "stream": true });
        assert_eq!(require_nonce(Some(&payload)), Some("7Yc2Kq1mFbA9"));
    }

    #[test]
    fn rejects_a_missing_nonce() {
        assert_eq!(require_nonce(Some(&json!({ "stream": true }))), None);
    }

    #[test]
    fn rejects_the_misnamed_field() {
        let payload = json!({ "stream_nonce": "7Yc2Kq1mFbA9", "stream": true });
        assert_eq!(require_nonce(Some(&payload)), None);
    }

    #[test]
    fn rejects_empty_and_whitespace() {
        assert_eq!(require_nonce(Some(&json!({ "nonce": "" }))), None);
        assert_eq!(require_nonce(Some(&json!({ "nonce": "   " }))), None);
    }

    #[test]
    fn rejects_a_non_string_nonce() {
        assert_eq!(require_nonce(Some(&json!({ "nonce": 12345 }))), None);
        assert_eq!(require_nonce(Some(&json!({ "nonce": null }))), None);
        assert_eq!(require_nonce(Some(&json!({ "nonce": ["a"] }))), None);
    }

    #[test]
    fn rejects_an_absent_payload() {
        assert_eq!(require_nonce(None), None);
    }
}
