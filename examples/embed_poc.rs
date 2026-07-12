//! Embedding engine end-to-end validation (#231). Loads a real embedding GGUF through the
//! ACTUAL `EmbedEngine` (not a hand-rolled path) and checks the things `cargo check` can't:
//!   1. Vectors have the catalog dim.
//!   2. Vectors are unit-norm (L2-normalized) and non-zero.
//!   3. SEMANTIC sanity: paraphrases score higher cosine than unrelated sentences. This is the
//!      real proof that pooling + prefix + normalization are correct (a wrong pooling produces
//!      valid-but-meaningless vectors that only a similarity test catches).
//!   4. Matryoshka truncation (if the model supports it) still yields a unit vector.
//!   5. query vs document input_type both work.
//!
//! Run: cargo run --release --example embed_poc --features inprocess -- <model.gguf> <catalog-id>
//! e.g. cargo run --release --example embed_poc --features inprocess -- \
//!         /tmp/nomic-embed-text-v1.5.Q8_0.gguf nomic-embed-text-v1.5

use sgl_node::embed::{embed_model_spec, EmbedConfig, EmbedEngine, InputType};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Vectors are already L2-normalized, so cosine == dot product.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: embed_poc <model.gguf> <catalog-id>");
    let model_id = args
        .next()
        .expect("usage: embed_poc <model.gguf> <catalog-id>");

    let spec = embed_model_spec(&model_id).unwrap_or_else(|| {
        panic!("'{model_id}' is not in EMBED_CATALOG — pass a known catalog id")
    });
    println!(
        "== model {} | dim {} | pooling {:?} | normalize {} | max_input {} ==",
        spec.id, spec.dim, spec.pooling, spec.normalize, spec.max_input_tokens
    );

    let engine = EmbedEngine::start(EmbedConfig {
        model_path: model_path.into(),
        model_name: model_id.clone(),
        n_gpu_layers: 0, // CPU — this is the "plain CPU box joins the grid" story
    })
    .await
    .expect("engine failed to start");
    println!("engine started OK (advertised dim = {})\n", engine.dim());

    let mut failures: Vec<String> = Vec::new();
    let mut check = |cond: bool, msg: String| {
        println!("[{}] {msg}", if cond { "PASS" } else { "FAIL" });
        if !cond {
            failures.push(msg);
        }
    };

    // ── 1-3: shape + norm + semantics ──
    let sentences = vec![
        "The cat sat quietly on the warm windowsill.".to_string(), // 0
        "A feline rested calmly on the sunny ledge.".to_string(),  // 1  (paraphrase of 0)
        "Quantum entanglement links particles across vast distances.".to_string(), // 2 (unrelated)
        "The central bank raised interest rates on Tuesday.".to_string(), // 3 (unrelated)
    ];
    let out = engine
        .embed(sentences.clone(), InputType::Document, None)
        .await
        .expect("embed failed");

    check(
        out.vectors.len() == sentences.len(),
        format!("returned {} vectors for {} inputs", out.vectors.len(), sentences.len()),
    );
    check(
        out.prompt_tokens > 0,
        format!("prompt_tokens reported = {}", out.prompt_tokens),
    );
    for (i, v) in out.vectors.iter().enumerate() {
        check(v.len() == spec.dim as usize, format!("vec[{i}] dim {} == {}", v.len(), spec.dim));
        let n = l2(v);
        check((n - 1.0).abs() < 1e-3, format!("vec[{i}] L2 norm {n:.6} ≈ 1.0"));
        let all_zero = v.iter().all(|x| *x == 0.0);
        let any_nan = v.iter().any(|x| !x.is_finite());
        check(!all_zero && !any_nan, format!("vec[{i}] non-zero + finite"));
    }

    // The load-bearing correctness check: paraphrase pair must beat every unrelated pair.
    let sim_paraphrase = cosine(&out.vectors[0], &out.vectors[1]);
    let sim_unrelated_a = cosine(&out.vectors[0], &out.vectors[2]);
    let sim_unrelated_b = cosine(&out.vectors[0], &out.vectors[3]);
    let sim_unrelated_c = cosine(&out.vectors[2], &out.vectors[3]);
    println!(
        "\ncosine: paraphrase(0,1)={sim_paraphrase:.3}  unrelated(0,2)={sim_unrelated_a:.3}  \
         unrelated(0,3)={sim_unrelated_b:.3}  unrelated(2,3)={sim_unrelated_c:.3}"
    );
    check(
        sim_paraphrase > sim_unrelated_a + 0.05 && sim_paraphrase > sim_unrelated_b + 0.05,
        format!("paraphrase cosine ({sim_paraphrase:.3}) clearly > unrelated — pooling is correct"),
    );

    // ── 4: Matryoshka truncation (only if the model allows it) ──
    if let Some(&small) = spec.allowed_dimensions.iter().find(|&&d| d < spec.dim) {
        let t = engine
            .embed(vec![sentences[0].clone()], InputType::Document, Some(small))
            .await
            .expect("matryoshka embed failed");
        check(
            t.vectors[0].len() == small as usize,
            format!("matryoshka: requested {small}-dim → got {}", t.vectors[0].len()),
        );
        check(
            (l2(&t.vectors[0]) - 1.0).abs() < 1e-3,
            format!("matryoshka {small}-dim vector renormalized to unit length"),
        );
    } else {
        println!("[skip] model has no Matryoshka dims");
    }

    // ── 5: query vs document input_type both produce valid vectors ──
    let q = engine
        .embed(vec!["what is the capital of France?".to_string()], InputType::Query, None)
        .await
        .expect("query embed failed");
    check(
        q.vectors[0].len() == spec.dim as usize && (l2(&q.vectors[0]) - 1.0).abs() < 1e-3,
        "query input_type produced a valid unit vector".to_string(),
    );

    // ── 6: reject an unsupported dimensions request ──
    let bad = engine
        .embed(vec!["hello".to_string()], InputType::Document, Some(spec.dim + 7))
        .await;
    check(bad.is_err(), "unsupported `dimensions` is rejected (not silently wrong)".to_string());

    println!(
        "\n================ {} ================",
        if failures.is_empty() { "ALL CHECKS PASSED" } else { "FAILURES PRESENT" }
    );
    if !failures.is_empty() {
        for f in &failures {
            println!("  FAILED: {f}");
        }
        std::process::exit(1);
    }
}
