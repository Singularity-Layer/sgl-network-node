//! In-process inference PoC (v1.5.0 spike).
//!
//! Validates the hard parts before the full InProcessEngine: the llama-cpp-2 vendored
//! cmake build, Metal load, model load, and tokenizer parity (prompt_tokens). Generation
//! is added once this builds + the crate's sampler API is confirmed for the resolved
//! version.
//!
//! Run: cargo run --release --example inproc_poc --features inprocess,metal -- <model.gguf>

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inproc_poc <model.gguf>");

    let backend = LlamaBackend::init().expect("llama backend init");

    // n_gpu_layers high → offload to Metal/GPU on Apple Silicon.
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
    let model = LlamaModel::load_from_file(&backend, &path, &model_params).expect("load model");

    let prompt = "Say hi in three words.";
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .expect("tokenize prompt");

    println!("OK: loaded {path}");
    println!("prompt = {prompt:?}");
    println!("prompt_tokens = {}", tokens.len());
}
