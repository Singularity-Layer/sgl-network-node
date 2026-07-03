//! Library surface for the sgl-node crate. The node itself runs as the `sgl` binary
//! (src/main.rs); this lib exposes the inference engines so examples + integration tests
//! can drive them directly (e.g. examples/inproc_batch_test.rs). The binary keeps its own
//! `mod` declarations and does not depend on this lib.

pub mod inference;
#[cfg(feature = "inprocess")]
pub mod inprocess;
