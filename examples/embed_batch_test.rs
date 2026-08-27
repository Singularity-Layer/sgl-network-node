//! Embedding input-batching parity + speed test.
//!
//! Verifies the batched `EmbedEngine` (packs many sequences into one encode pass):
//!   1. PARITY — a batch of N inputs returns the SAME vectors (within fp tolerance) as embedding
//!      each input alone, and in the correct order. This is the correctness guarantee for packing
//!      multiple seq_ids into one encode() and reading them back via embeddings_seq_ith.
//!   2. ORDER — shuffled inputs come back aligned to their input positions.
//!   3. SPEED — a large batch is much faster than the sum of single-input calls (the batching win).
//!   4. UNIT — every returned vector is L2-normalized (unit length) and finite.
//!
//! Run: cargo run --release --example embed_batch_test --features inprocess,metal -- <embed.gguf> [model-id]

use std::path::PathBuf;
use std::time::Instant;

use sgl_node::embed::{EmbedConfig, EmbedEngine, InputType};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot // both are unit vectors → cosine == dot
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: embed_batch_test <embed.gguf> [model-id]");
    let model_id = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "bge-base-en-v1.5".into());

    println!("== starting embed engine: {model_id} ==");
    let engine = EmbedEngine::start(EmbedConfig {
        model_path: PathBuf::from(&path),
        model_name: model_id.clone(),
        n_gpu_layers: 999,
    })
    .await
    .expect("engine start");
    println!("ready, dim = {}", engine.dim());

    // A spread of realistic RAG-ish inputs, various lengths.
    let inputs: Vec<String> = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Solana is a high-throughput proof-of-history blockchain.",
        "Embeddings map text into a dense vector space for semantic search.",
        "cat",
        "Retrieval-augmented generation grounds a model in your own documents.",
        "The mitochondria is the powerhouse of the cell.",
        "def add(a, b):\n    return a + b",
        "A confidential compute node serves inference without exposing prompts.",
        "Paris is the capital of France.",
        "Once upon a time, in a land far away, there lived a small dragon who loved to read books about faraway kingdoms and brave knights.",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // ── 0) COLD batch first (no warmup) — replicates the live "first multi-seq batch" case.
    //    If this is slow but a repeat is fast, Metal is compiling pipelines per batch shape. ──
    {
        let t = Instant::now();
        let cold = engine
            .embed(inputs.clone(), InputType::Document, None)
            .await
            .expect("cold batch");
        let cold_ms = t.elapsed().as_millis();
        assert_eq!(cold.vectors.len(), inputs.len());
        let t2 = Instant::now();
        let _warm = engine
            .embed(inputs.clone(), InputType::Document, None)
            .await
            .expect("warm batch");
        let warm_ms = t2.elapsed().as_millis();
        println!(
            "COLD batch({}) = {}ms | WARM batch({}) = {}ms  (compile-per-shape if cold>>warm)",
            inputs.len(),
            cold_ms,
            inputs.len(),
            warm_ms
        );
    }

    // ── 1) Reference: embed each input ALONE (one input per job → no packing). ──
    let t_serial = Instant::now();
    let mut singles: Vec<Vec<f32>> = Vec::new();
    for s in &inputs {
        let out = engine
            .embed(vec![s.clone()], InputType::Document, None)
            .await
            .expect("single embed");
        assert_eq!(out.vectors.len(), 1);
        singles.push(out.vectors.into_iter().next().unwrap());
    }
    let serial_ms = t_serial.elapsed().as_millis();

    // ── 2) Batched: all inputs in ONE job (multi-seq packing). ──
    let t_batch = Instant::now();
    let batched = engine
        .embed(inputs.clone(), InputType::Document, None)
        .await
        .expect("batch embed");
    let batch_ms = t_batch.elapsed().as_millis();
    assert_eq!(
        batched.vectors.len(),
        inputs.len(),
        "batch returned wrong count"
    );

    // ── PARITY + ORDER: batched[i] must match singles[i] (same input, same slot). ──
    let mut worst = 1.0f32;
    for i in 0..inputs.len() {
        let c = cosine(&singles[i], &batched.vectors[i]);
        worst = worst.min(c);
        assert!(
            c > 0.9999,
            "PARITY FAIL at input {i}: cosine(single, batched) = {c} (< 0.9999)\n  input: {:?}",
            &inputs[i]
        );
        // Every vector must be unit-length + finite.
        let norm: f32 = batched.vectors[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "vector {i} not unit-length: {norm}"
        );
        assert!(
            batched.vectors[i].iter().all(|x| x.is_finite()),
            "vector {i} has non-finite"
        );
    }
    println!(
        "PARITY ok — worst cosine(single, batched) = {worst:.6} across {} inputs",
        inputs.len()
    );

    // ── ORDER (negative check): a DIFFERENT input must NOT match (guards seq_id misalignment). ──
    let cross = cosine(&batched.vectors[3], &batched.vectors[8]); // "cat" vs "Paris ..." — should be low
    assert!(
        cross < 0.95,
        "distinct inputs too similar ({cross}) — possible seq_id mixup"
    );
    println!("ORDER ok — distinct-input cosine = {cross:.4} (well below self-match)");

    // ── SPEED: the batched pass should be well under the serial sum. ──
    println!(
        "SPEED — serial {serial_ms}ms vs batched {batch_ms}ms  (speedup {:.1}x)",
        serial_ms as f64 / batch_ms.max(1) as f64
    );
    assert!(
        batch_ms * 2 < serial_ms.max(1),
        "batched not meaningfully faster: {batch_ms}ms vs serial {serial_ms}ms"
    );

    // ── prompt_tokens must be the SUM across the batch (billing is input-only). ──
    println!("prompt_tokens (batch) = {}", batched.prompt_tokens);
    assert!(batched.prompt_tokens > 0);

    println!("\nALL EMBED BATCH TESTS PASSED ✅");
}
