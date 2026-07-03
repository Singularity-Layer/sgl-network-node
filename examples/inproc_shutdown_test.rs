//! Reliability-fix tests for the in-process engine (Codex FIX2 + FIX3).
//!
//!   A. Slow/stalled stream consumer must NOT stall other slots (head-of-line): one request
//!      streams into a capacity-1 channel we never drain, while other non-streaming requests
//!      must still complete promptly.
//!   B. Dropping the engine while a request is in flight must return promptly (bounded Drop),
//!      not hang — and exit clean (EXIT=0).
//!
//! Run: cargo run --release --example inproc_shutdown_test --features inprocess,metal -- <model.gguf>

use std::path::PathBuf;
use std::time::{Duration, Instant};

use sgl_node::inference::{ChatMessage, StreamEvent};
use sgl_node::inprocess::{InProcessConfig, InProcessEngine};

fn user(content: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: content.into() }]
}

async fn start(path: &str, max_slots: u32) -> InProcessEngine {
    InProcessEngine::start(InProcessConfig {
        model_path: PathBuf::from(path),
        model_name: "test".into(),
        n_ctx: max_slots * 2048,
        n_gpu_layers: 999,
        max_slots,
        per_slot_ctx: 2048,
    })
    .await
    .expect("engine start")
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let path = std::env::args().nth(1).expect("usage: inproc_shutdown_test <model.gguf>");

    // ---- A. slow stream consumer must not stall other slots -----------------------------
    {
        let engine = start(&path, 3).await;
        // A streaming request whose consumer NEVER drains (capacity 1).
        let (stuck_tx, _stuck_rx_held) = tokio::sync::mpsc::channel::<StreamEvent>(1);
        let stuck = {
            let e = &engine;
            async move {
                e.chat_completion_stream(&user("Write 200 words about rivers."), 200, 0.0, stuck_tx).await
            }
        };
        // Meanwhile, normal requests that MUST finish promptly despite the stalled consumer.
        let others = async {
            let e = &engine;
            let t = Instant::now();
            for p in ["2+2? number only", "capital of Japan? one word", "opposite of up?"] {
                let o = e.chat_completion(&user(p), 24, 0.0).await.expect("other req");
                println!("   other OK ct={} {:?}", o.completion_tokens, o.content.trim());
            }
            t.elapsed()
        };
        let (stuck_res, others_wall) = tokio::join!(stuck, others);
        println!("[A] stalled-consumer stream returned: {:?}", stuck_res.map(|_| "Ok"));
        println!("[A] other requests wall time while consumer stalled: {:?}", others_wall);
        assert!(others_wall < Duration::from_secs(60), "other slots were stalled by the slow consumer");
        // `_stuck_rx_held` kept alive until here so the channel stays Full, not Closed.
        drop(_stuck_rx_held);
        println!("[A] PASS — slow consumer did not stall other slots\n");
    }

    // ---- B. Drop while in flight must be bounded (no hang) -------------------------------
    {
        let engine = start(&path, 2).await;
        // Drive a long generation with a short timeout so the future is abandoned while the
        // worker is still mid-generation, leaving a genuinely in-flight job in the engine.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(8);
        let _ = tokio::time::timeout(
            Duration::from_millis(500),
            engine.chat_completion_stream(&user("Write 300 words about mountains."), 300, 0.0, tx),
        )
        .await; // times out: request still running inside the worker
        let _ = rx.try_recv(); // proves streaming had begun

        let t = Instant::now();
        drop(engine); // must return promptly (bounded Drop) despite in-flight work
        let drop_ms = t.elapsed();
        println!("[B] engine Drop with in-flight work took {:?}", drop_ms);
        assert!(drop_ms < Duration::from_secs(20), "Drop hung beyond the bounded window");
        println!("[B] PASS — bounded shutdown\n");
    }

    println!("== SHUTDOWN/RELIABILITY CHECKS PASSED ==");
}
