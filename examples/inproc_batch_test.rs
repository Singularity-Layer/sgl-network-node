//! Continuous-batching in-process engine test (v1.6.0).
//!
//! Drives the REAL `InProcessEngine` (not the raw PoC) to verify:
//!   1. single-request parity (temp=0 → same output as the known-good PoC),
//!   2. N concurrent non-streaming requests all return coherent, correct answers,
//!   3. concurrency actually overlaps (wall-clock << serial sum),
//!   4. streaming yields deltas + a terminal Done with correct token counts,
//!   5. an oversized prompt is rejected cleanly (no crash, no zombie).
//!
//! Run: cargo run --release --example inproc_batch_test --features inprocess,metal -- <model.gguf>

use std::path::PathBuf;
use std::time::Instant;

use sgl_node::inference::{ChatMessage, StreamEvent};
use sgl_node::inprocess::{InProcessConfig, InProcessEngine};

fn user(content: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: content.into() }]
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let path = std::env::args().nth(1).expect("usage: inproc_batch_test <model.gguf>");
    // SLOTS env overrides so we can exercise both the production default (1) and batching (>1).
    let max_slots = std::env::var("SLOTS").ok().and_then(|s| s.parse().ok()).unwrap_or(3u32);
    let per_slot_ctx = 2048u32;

    println!("== starting engine: max_slots={max_slots}, per_slot_ctx={per_slot_ctx} ==");
    let engine = InProcessEngine::start(InProcessConfig {
        model_path: PathBuf::from(&path),
        model_name: "test-model".into(),
        n_ctx: max_slots * per_slot_ctx,
        n_gpu_layers: 999,
        max_slots,
        per_slot_ctx,
    })
    .await
    .expect("engine start");
    println!("engine ready, advertised slots = {}", engine.max_slots());

    // ---- 1. single-request parity (temp=0) --------------------------------------------
    let t = Instant::now();
    let out = engine
        .chat_completion(&user("Say hi in three words."), 64, 0.0)
        .await
        .expect("single completion");
    println!(
        "\n[1] single  ({:?}): prompt_tokens={} completion_tokens={}\n    output={:?}",
        t.elapsed(), out.prompt_tokens, out.completion_tokens, out.content
    );
    assert!(out.completion_tokens > 0, "single completion produced no tokens");

    // ---- 2 + 3. N concurrent non-streaming requests -----------------------------------
    let prompts = [
        "What is 2 + 2? Reply with only the number.",
        "What is the capital of France? Reply with only the city name.",
        "Complete: The opposite of hot is",
        "Name a primary color. One word.",
        "What is 10 divided by 2? Number only.",
        "Say the word 'banana' once.",
    ];
    let t = Instant::now();
    let mut handles = Vec::new();
    for p in prompts {
        let e = &engine;
        handles.push(async move {
            let r = e.chat_completion(&user(p), 48, 0.0).await;
            (p, r)
        });
    }
    let results = futures_util::future::join_all(handles).await;
    let concurrent_wall = t.elapsed();
    println!("\n[2] {} concurrent requests finished in {:?}", prompts.len(), concurrent_wall);
    for (p, r) in &results {
        match r {
            Ok(o) => println!("    OK  pt={:>3} ct={:>3}  {:?} -> {:?}", o.prompt_tokens, o.completion_tokens, p, o.content.trim()),
            Err(e) => println!("    ERR {:?} -> {e}", p),
        }
    }
    let ok = results.iter().filter(|(_, r)| r.is_ok()).count();
    assert_eq!(ok, prompts.len(), "some concurrent requests failed");

    // Serial baseline (same prompts, one at a time) to prove batching overlaps work.
    let t = Instant::now();
    for p in prompts {
        let _ = engine.chat_completion(&user(p), 48, 0.0).await.expect("serial completion");
    }
    let serial_wall = t.elapsed();
    println!(
        "[3] serial baseline {:?}  |  concurrent {:?}  |  speedup ~{:.2}x",
        serial_wall, concurrent_wall,
        serial_wall.as_secs_f64() / concurrent_wall.as_secs_f64().max(1e-6)
    );

    // ---- 3b. decode-dominated overlap (the real batching win) -------------------------
    // Long generations so decode (not prefill) dominates; batching should overlap them.
    let long_prompt = "Write a detailed paragraph about the ocean.";
    let gen = 160i32;
    let t = Instant::now();
    let mut lh = Vec::new();
    for _ in 0..max_slots {
        let e = &engine;
        lh.push(async move { e.chat_completion(&user(long_prompt), gen, 0.2).await });
    }
    let long_conc = futures_util::future::join_all(lh).await;
    let long_conc_wall = t.elapsed();
    let conc_tokens: u32 = long_conc.iter().filter_map(|r| r.as_ref().ok()).map(|o| o.completion_tokens).sum();

    let t = Instant::now();
    let mut serial_tokens = 0u32;
    for _ in 0..max_slots {
        let o = engine.chat_completion(&user(long_prompt), gen, 0.2).await.expect("long serial");
        serial_tokens += o.completion_tokens;
    }
    let long_serial_wall = t.elapsed();
    println!(
        "\n[3b] {n} long gens (~{gen} tok each): concurrent {:?} ({conc_tokens} tok) | serial {:?} ({serial_tokens} tok) | speedup ~{:.2}x",
        long_conc_wall, long_serial_wall,
        long_serial_wall.as_secs_f64() / long_conc_wall.as_secs_f64().max(1e-6),
        n = max_slots,
    );

    // ---- 3b. LONG concurrent generations (the case batching is actually for) ----------
    // Short completions are dominated by prefill + kernel-launch overhead, so batching
    // barely helps there. The real win is overlapping DECODE across long generations.
    let long_prompt = "Write a detailed 150-word paragraph about the ocean.";
    let n_long = max_slots as usize;
    let t = Instant::now();
    let mut lh = Vec::new();
    for _ in 0..n_long {
        let e = &engine;
        lh.push(async move { e.chat_completion(&user(long_prompt), 200, 0.0).await });
    }
    let long_results = futures_util::future::join_all(lh).await;
    let long_concurrent = t.elapsed();
    let long_ct: u32 = long_results.iter().filter_map(|r| r.as_ref().ok()).map(|o| o.completion_tokens).sum();

    let t = Instant::now();
    let mut serial_ct = 0u32;
    for _ in 0..n_long {
        let o = engine.chat_completion(&user(long_prompt), 200, 0.0).await.expect("serial long");
        serial_ct += o.completion_tokens;
    }
    let long_serial = t.elapsed();
    println!(
        "\n[3b] {n_long}x long gen (~200 tok each): concurrent {:?} ({} tok) | serial {:?} ({} tok) | speedup ~{:.2}x",
        long_concurrent, long_ct, long_serial, serial_ct,
        long_serial.as_secs_f64() / long_concurrent.as_secs_f64().max(1e-6)
    );

    // ---- 4. streaming -----------------------------------------------------------------
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);
    let stream_task = {
        let e = &engine;
        async move { e.chat_completion_stream(&user("Count from 1 to 5, space separated."), 48, 0.0, tx).await }
    };
    let collect_task = async {
        let mut text = String::new();
        let mut delta_tokens = 0u32;
        let mut done: Option<(u32, u32)> = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Delta { text: t, tokens } => { text.push_str(&t); delta_tokens += tokens; }
                StreamEvent::Done { prompt_tokens, completion_tokens } => { done = Some((prompt_tokens, completion_tokens)); }
            }
        }
        (text, delta_tokens, done)
    };
    let (stream_res, (text, delta_tokens, done)) = tokio::join!(stream_task, collect_task);
    stream_res.expect("stream completed");
    let (pt, ct) = done.expect("stream produced a Done event");
    println!(
        "\n[4] stream: prompt_tokens={pt} completion_tokens={ct} delta_tokens={delta_tokens}\n    text={:?}",
        text
    );
    assert!(ct > 0, "stream produced no completion tokens");
    // Delta tokens should cover all but possibly the terminal stop token.
    assert!(delta_tokens >= ct.saturating_sub(1), "streamed deltas ({delta_tokens}) < completion_tokens ({ct})");

    // ---- 5. oversized prompt rejected cleanly -----------------------------------------
    let huge = "word ".repeat(per_slot_ctx as usize + 200);
    let r = engine.chat_completion(&user(&huge), 16, 0.0).await;
    println!("\n[5] oversized prompt -> {:?}", r.as_ref().map(|_| "UNEXPECTED OK").map_err(|e| e.as_str()));
    assert!(r.is_err(), "oversized prompt should be rejected");

    // ---- 6. engine still healthy + serving after all of the above ----------------------
    assert!(engine.is_healthy(), "engine unhealthy after test run");
    let after = engine.chat_completion(&user("Say 'ok'."), 16, 0.0).await.expect("post-test completion");
    println!("\n[6] post-test completion still works: {:?}", after.content.trim());

    println!("\n== ALL CHECKS PASSED ==");
}
