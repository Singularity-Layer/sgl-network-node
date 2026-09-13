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
/// REQUIRED, never defaulted. This used to fall back to `""`, which looks harmless and
/// is not: an empty nonce is a known constant, so every chunk's AAD becomes predictable
/// and the forgery protection the nonce exists to provide silently disappears. Nothing
/// reported it — the stream just worked, weakly.
///
/// Not hypothetical: a first-party client shipped exactly that bug, sending the field
/// as `stream_nonce` while this reads `nonce`. Caught in review, not by any system.
fn require_nonce(payload: Option<&Value>) -> Option<&str> {
    payload
        .and_then(|p| p.get("nonce"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|n| !n.is_empty())
}

/// Validate the nonce and build the sealer, or return the reason to fail the job with.
///
/// Fails CLOSED on a missing nonce rather than substituting a default: a stream that
/// silently loses its forgery protection is worse than one that refuses to start.
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
        let p = json!({ "nonce": "7Yc2Kq1mFbA9", "stream": true });
        assert_eq!(require_nonce(Some(&p)), Some("7Yc2Kq1mFbA9"));
    }

    #[test]
    fn rejects_a_missing_nonce() {
        // The whole point: no nonce must FAIL, not quietly become "".
        assert_eq!(require_nonce(Some(&json!({ "stream": true }))), None);
    }

    #[test]
    fn rejects_the_misnamed_field() {
        // The exact bug a first-party client shipped.
        let p = json!({ "stream_nonce": "7Yc2Kq1mFbA9", "stream": true });
        assert_eq!(require_nonce(Some(&p)), None);
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
