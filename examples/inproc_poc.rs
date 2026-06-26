//! In-process inference PoC (v1.5.0 spike) — full load → generate → token counts.
//!
//! Validates the whole in-process path before building InProcessEngine: llama-cpp-2
//! vendored build, Metal load, tokenizer (prompt_tokens), decode, sampling, detokenize,
//! EOG stop, and completion_tokens accounting (the billing-critical numbers).
//!
//! Run: cargo run --release --example inproc_poc --features inprocess,metal -- <model.gguf>

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inproc_poc <model.gguf>");

    let backend = LlamaBackend::init().expect("llama backend init");
    let model_params = LlamaModelParams::default().with_n_gpu_layers(999); // offload to Metal
    let model = LlamaModel::load_from_file(&backend, &path, &model_params).expect("load model");
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .expect("create context");

    let prompt = "Say hi in three words.";
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .expect("tokenize prompt");
    let prompt_tokens = tokens.len();

    // Decode the prompt (only the last token needs logits).
    let mut batch = LlamaBatch::new(512, 1);
    let last = tokens.len() - 1;
    for (i, tok) in tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], i == last).expect("add prompt token");
    }
    ctx.decode(&mut batch).expect("decode prompt");

    // Greedy generation loop.
    let mut sampler = LlamaSampler::greedy();
    let mut n_cur = batch.n_tokens();
    let mut completion_tokens = 0i32;
    let mut out = String::new();
    for _ in 0..64 {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        out.push_str(&model.token_to_str(token, Special::Plaintext).unwrap_or_default());
        completion_tokens += 1;
        batch.clear();
        batch.add(token, n_cur, &[0], true).expect("add gen token");
        n_cur += 1;
        ctx.decode(&mut batch).expect("decode gen");
    }

    println!("---");
    println!("prompt        = {prompt:?}");
    println!("prompt_tokens = {prompt_tokens}");
    println!("completion_tokens = {completion_tokens}");
    println!("output        = {out:?}");
}
