//! Library surface for the sgl-node crate. The node itself runs as the `sgl` binary
//! (src/main.rs); this lib exposes the inference engines so examples + integration tests
//! can drive them directly (e.g. examples/inproc_batch_test.rs). The binary keeps its own
//! `mod` declarations and does not depend on this lib.

#[cfg(feature = "inprocess")]
pub mod embed;
pub mod embed_catalog;
// Exposed for examples/seal_b64_check.rs (cross-language sealed-encoding proof).
pub mod encryption;
pub mod inference;
// Engine installer — inference.rs drives the crash-loop auto-swap through it.
pub mod setup;
#[cfg(feature = "inprocess")]
pub mod inprocess;
