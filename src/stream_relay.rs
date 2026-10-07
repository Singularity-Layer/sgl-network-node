//! Sealed-stream relay helpers: seal + sign + POST one chunk, and coalesce engine
//! text deltas that queued up while the previous chunk POST was in flight.
//!
//! Why coalesce: the stream loop awaits one node→orchestrator POST per chunk, and each
//! POST costs the orchestrator a job read, a signing-key read and a JobStream push
//! before it acks. With one chunk per engine delta, delivery was capped at about one
//! token per round trip once the engine outran the network. Coalescing only drains
//! what is ALREADY queued (`try_recv`, never a timer), so an engine slower than the
//! network sees exactly the old one-delta-per-chunk behaviour and no added latency.
//!
//! What does not change: every chunk is still sealed with its own seq + final flag in
//! the AAD and signed per chunk; seq stays contiguous from 0; tool-call deltas and
//! Done are never merged (they end a batch and are processed next, in order); the
//! final chunk and its usage are built exactly as before.

use crate::inference::StreamEvent;
use crate::orchestrator::OrchestratorClient;

/// Cap on one coalesced chunk's plaintext. A batch stops draining once it reaches this,
/// so a chunk is at most this plus one delta. Far below the orchestrator's 256,000-char
/// base58 ciphertext limit, and keeps base58 encoding (quadratic) cheap.
pub const COALESCE_MAX_BYTES: usize = 8 * 1024;

/// Kill switch: `SGL_STREAM_COALESCE=0` (or `false`/`off`) restores one chunk per
/// engine delta, byte-for-byte the pre-coalescing behaviour. Default ON.
pub fn coalesce_max_bytes() -> usize {
    coalesce_max_bytes_from(std::env::var("SGL_STREAM_COALESCE").ok().as_deref())
}

fn coalesce_max_bytes_from(v: Option<&str>) -> usize {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s == "0" || s == "false" || s == "off" => 0,
        _ => COALESCE_MAX_BYTES,
    }
}

/// One or more consecutive text deltas destined for a single sealed chunk.
pub struct TextBatch {
    /// Cleaned output to seal as ONE chunk. Empty when every delta cleaned to nothing
    /// (the caller then posts nothing, as the old loop skipped empty output).
    pub text: String,
    /// Engine tokens across every delta in the batch. Billed once the chunk is accepted.
    pub tokens: u32,
    /// Tokens up to and including the first delta that produced output. If the chunk
    /// POST fails, this is exactly what the old one-delta-per-chunk loop had counted when
    /// its (identical) first POST failed. The rest of the batch would still have been
    /// sitting in the channel, unbilled, so billing it now would bill undelivered output.
    pub lead_tokens: u32,
    /// Engine deltas merged into this batch (diagnostics / tests).
    pub deltas: u32,
    /// A non-text event pulled while draining. It must be processed next, before
    /// receiving anything else, so event order is preserved.
    pub held: Option<StreamEvent>,
}

/// Build a batch from `first` plus any text deltas `try_next` can return WITHOUT
/// waiting. `clean` is applied to each delta in order (the stateful HOMURA cleaner is
/// fed exactly the sequence it would have seen before). `max_bytes == 0` disables
/// draining, so the batch is exactly `first`.
pub fn coalesce_deltas(
    first_text: String,
    first_tokens: u32,
    max_bytes: usize,
    mut clean: impl FnMut(String) -> String,
    mut try_next: impl FnMut() -> Option<StreamEvent>,
) -> TextBatch {
    let mut b = TextBatch {
        text: String::new(),
        tokens: 0,
        lead_tokens: 0,
        deltas: 0,
        held: None,
    };
    let mut add = |b: &mut TextBatch, text: String, tokens: u32| {
        b.tokens = b.tokens.saturating_add(tokens);
        b.deltas += 1;
        if b.text.is_empty() {
            // No output yet, so this delta is still part of the lead.
            b.lead_tokens = b.tokens;
        }
        b.text.push_str(&clean(text));
    };
    add(&mut b, first_text, first_tokens);
    while b.text.len() < max_bytes {
        match try_next() {
            Some(StreamEvent::Delta { text, tokens }) => add(&mut b, text, tokens),
            Some(other) => {
                b.held = Some(other);
                break;
            }
            None => break,
        }
    }
    b
}

/// Seal one stream chunk, sign its envelope, and POST it. Returns `Ok(true)` if
/// the orchestrator reports the client is gone (stop early), `Ok(false)` to keep
/// going, `Err` on a hard failure.
#[allow(clippy::too_many_arguments)]
pub async fn seal_post_chunk(
    client: &OrchestratorClient,
    sealer: &crate::encryption::StreamSealer,
    node_secret: &[u8; 32],
    job_id: &str,
    eph_b58: &str,
    seq: u64,
    is_final: bool,
    plaintext: &[u8],
    usage: Option<serde_json::Value>,
    fmt: Option<&str>,
) -> Result<bool, String> {
    let ct = sealer.seal_chunk(plaintext, seq, is_final)?;
    let kind = format!("stream:{seq}:{}", if is_final { 1 } else { 0 });
    let sig = crate::crypto::sign_result_envelope(node_secret, job_id, &kind, ct.as_bytes());
    let eph = if seq == 0 { Some(eph_b58) } else { None };
    client
        .post_chunk(job_id, seq, is_final, eph, &ct, usage, Some(sig), fmt)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn delta(t: &str, n: u32) -> StreamEvent {
        StreamEvent::Delta {
            text: t.to_string(),
            tokens: n,
        }
    }

    fn done() -> StreamEvent {
        StreamEvent::Done {
            prompt_tokens: 5,
            completion_tokens: 9,
        }
    }

    /// Drive `coalesce_deltas` the way the stream loop does: a blocking recv for the
    /// first event of each batch, a held event processed before the next recv.
    /// Returns (chunks as (seq, text), tokens billed on accept, terminal event seen).
    fn run(events: Vec<StreamEvent>, max_bytes: usize) -> (Vec<(u64, String)>, u32, bool) {
        let mut q: VecDeque<StreamEvent> = events.into();
        let mut held: Option<StreamEvent> = None;
        let mut chunks = Vec::new();
        let mut seq = 0u64;
        let mut billed = 0u32;
        loop {
            let ev = match held.take().or_else(|| q.pop_front()) {
                Some(ev) => ev,
                None => return (chunks, billed, false),
            };
            match ev {
                StreamEvent::Delta { text, tokens } => {
                    let b = coalesce_deltas(text, tokens, max_bytes, |t| t, || q.pop_front());
                    held = b.held;
                    billed += b.tokens;
                    if b.text.is_empty() {
                        continue;
                    }
                    chunks.push((seq, b.text));
                    seq += 1;
                }
                StreamEvent::ToolCalls { delta } => {
                    chunks.push((seq, format!("tc:{delta}")));
                    seq += 1;
                }
                StreamEvent::Done { .. } => {
                    chunks.push((seq, "<final>".to_string()));
                    return (chunks, billed, true);
                }
            }
        }
    }

    #[test]
    fn kill_switch_values() {
        assert_eq!(coalesce_max_bytes_from(None), COALESCE_MAX_BYTES);
        assert_eq!(coalesce_max_bytes_from(Some("1")), COALESCE_MAX_BYTES);
        assert_eq!(coalesce_max_bytes_from(Some("0")), 0);
        assert_eq!(coalesce_max_bytes_from(Some(" FALSE ")), 0);
        assert_eq!(coalesce_max_bytes_from(Some("off")), 0);
    }

    #[test]
    fn queued_deltas_merge_into_one_chunk_and_final_stays_last() {
        let evs = vec![delta("Hel", 1), delta("lo", 1), delta(" world", 1), done()];
        let (chunks, billed, done_seen) = run(evs, COALESCE_MAX_BYTES);
        assert!(done_seen);
        assert_eq!(
            chunks,
            vec![(0, "Hello world".to_string()), (1, "<final>".to_string())]
        );
        assert_eq!(billed, 3);
    }

    #[test]
    fn disabled_is_one_chunk_per_delta() {
        let evs = vec![delta("a", 1), delta("b", 2), delta("c", 3), done()];
        let (chunks, billed, _) = run(evs, 0);
        assert_eq!(
            chunks,
            vec![
                (0, "a".to_string()),
                (1, "b".to_string()),
                (2, "c".to_string()),
                (3, "<final>".to_string())
            ]
        );
        assert_eq!(billed, 6);
    }

    #[test]
    fn plaintext_and_tokens_are_identical_to_uncoalesced() {
        let evs = || {
            vec![
                delta("The ", 1),
                delta("", 1),
                delta("quick ", 1),
                StreamEvent::ToolCalls {
                    delta: serde_json::json!([{"index":0}]),
                },
                delta("brown ", 2),
                delta("fox", 1),
                done(),
            ]
        };
        let (on, billed_on, _) = run(evs(), COALESCE_MAX_BYTES);
        let (off, billed_off, _) = run(evs(), 0);
        let join = |c: &[(u64, String)]| c.iter().map(|(_, t)| t.as_str()).collect::<String>();
        assert_eq!(join(&on), join(&off), "concatenated stream must not change");
        assert_eq!(billed_on, billed_off, "billing must not change");
        // Tool call is a hard boundary: text before it and after it are separate chunks,
        // and it keeps its position.
        assert_eq!(
            on,
            vec![
                (0, "The quick ".to_string()),
                (1, "tc:[{\"index\":0}]".to_string()),
                (2, "brown fox".to_string()),
                (3, "<final>".to_string())
            ]
        );
    }

    #[test]
    fn seq_is_contiguous_from_zero() {
        let mut evs: Vec<StreamEvent> = (0..500).map(|i| delta(&format!("t{i} "), 1)).collect();
        evs.push(done());
        let (chunks, billed, done_seen) = run(evs, 64);
        assert!(done_seen);
        assert_eq!(billed, 500);
        for (i, (seq, _)) in chunks.iter().enumerate() {
            assert_eq!(*seq, i as u64);
        }
        assert!(chunks.len() > 2 && chunks.len() < 500);
    }

    #[test]
    fn batch_stops_at_byte_cap_without_splitting_a_delta() {
        let mut q: VecDeque<StreamEvent> = (0..10).map(|_| delta("abcd", 1)).collect();
        let b = coalesce_deltas("abcd".into(), 1, 10, |t| t, || q.pop_front());
        // 4 → 8 → 12 (>= 10, stop): whole deltas only, cap exceeded by < one delta.
        assert_eq!(b.text, "abcdabcdabcd");
        assert_eq!(b.deltas, 3);
        assert_eq!(b.tokens, 3);
        assert_eq!(q.len(), 8, "undrained deltas stay queued, in order");
    }

    #[test]
    fn non_text_event_is_held_not_dropped() {
        let mut q: VecDeque<StreamEvent> = vec![delta("b", 1), done(), delta("late", 1)].into();
        let b = coalesce_deltas("a".into(), 1, COALESCE_MAX_BYTES, |t| t, || q.pop_front());
        assert_eq!(b.text, "ab");
        assert!(matches!(b.held, Some(StreamEvent::Done { .. })));
        assert_eq!(q.len(), 1, "nothing after the held event is consumed");
    }

    #[test]
    fn lead_tokens_bill_only_through_the_first_output_delta() {
        // Two deltas clean to nothing, then output, then more output. If this chunk's
        // POST fails, the old loop had counted 1+1+2 = 4 (the empties, then the delta
        // whose POST failed) and never pulled the rest from the channel.
        let mut q: VecDeque<StreamEvent> = vec![delta("", 1), delta("x", 2), delta("y", 7)].into();
        let b = coalesce_deltas("".into(), 1, COALESCE_MAX_BYTES, |t| t, || q.pop_front());
        assert_eq!(b.text, "xy");
        assert_eq!(b.tokens, 11);
        assert_eq!(b.lead_tokens, 4);
    }

    #[test]
    fn single_delta_lead_equals_tokens() {
        let b = coalesce_deltas("x".into(), 3, 0, |t| t, || None);
        assert_eq!((b.tokens, b.lead_tokens, b.deltas), (3, 3, 1));
    }

    #[test]
    fn stateful_cleaner_sees_every_delta_in_order() {
        // A cleaner that withholds text until it sees '>' — stands in for the HOMURA
        // cleaner, whose output depends on everything it was fed before.
        let mut buf = String::new();
        let mut seen = Vec::new();
        let clean = |t: String| {
            seen.push(t.clone());
            buf.push_str(&t);
            if buf.contains('>') {
                std::mem::take(&mut buf)
            } else {
                String::new()
            }
        };
        let mut q: VecDeque<StreamEvent> = vec![delta("b", 1), delta(">c", 1)].into();
        let b = coalesce_deltas("<a".into(), 1, COALESCE_MAX_BYTES, clean, || q.pop_front());
        assert_eq!(b.text, "<ab>c");
        assert_eq!(seen, vec!["<a", "b", ">c"]);
        assert_eq!(b.lead_tokens, 3);
    }

    /// Chunk-count / throughput model for the handoff (not a gate). A producer emits
    /// one token every `tok_ms`; each chunk POST takes `rtt_ms`; channel capacity 64
    /// (the real `mpsc::channel(64)`). Returns (chunks, elapsed_ms) for `n` tokens.
    fn simulate(n: u32, tok_ms: f64, rtt_ms: f64, coalesce: bool) -> (u32, f64) {
        let cap = 64usize;
        let mut produced = 0u32; // tokens handed to the channel
        let mut queued = 0usize;
        let mut next_tok_at = tok_ms;
        let mut t = 0.0f64;
        let mut delivered = 0u32;
        let mut chunks = 0u32;
        while delivered < n {
            // Advance producer to time t (it blocks while the channel is full).
            while produced < n && next_tok_at <= t && queued < cap {
                produced += 1;
                queued += 1;
                next_tok_at += tok_ms;
            }
            if queued == 0 {
                t = next_tok_at; // blocking recv waits for the next token
                continue;
            }
            let take = if coalesce { queued } else { 1 };
            queued -= take;
            // Producer blocked on a full channel resumes as soon as there is room.
            if next_tok_at < t {
                next_tok_at = t;
            }
            t += rtt_ms;
            delivered += take as u32;
            chunks += 1;
        }
        (chunks, t)
    }

    #[test]
    fn model_coalescing_removes_the_per_token_rtt_cap() {
        // 50 tok/s engine, 150 ms chunk round trip, 256 tokens.
        let (c_off, t_off) = simulate(256, 20.0, 150.0, false);
        let (c_on, t_on) = simulate(256, 20.0, 150.0, true);
        assert_eq!(c_off, 256);
        assert!(c_on < 50, "chunks with coalescing: {c_on}");
        assert!(t_on < t_off / 5.0, "on {t_on}ms vs off {t_off}ms");
        // Engine slower than the network: identical chunking, identical timing.
        let (c_slow_off, t_slow_off) = simulate(64, 200.0, 50.0, false);
        let (c_slow_on, t_slow_on) = simulate(64, 200.0, 50.0, true);
        assert_eq!(c_slow_off, c_slow_on);
        assert_eq!(t_slow_off, t_slow_on);
    }

    #[test]
    #[ignore = "prints the latency model table for the handoff"]
    fn print_latency_model() {
        for (tok_ms, rtt) in [(20.0, 60.0), (20.0, 150.0), (10.0, 150.0), (25.0, 300.0)] {
            let (c0, t0) = simulate(512, tok_ms, rtt, false);
            let (c1, t1) = simulate(512, tok_ms, rtt, true);
            println!(
                "engine {:>4.0} tok/s rtt {rtt:>4}ms: off {c0} chunks {:.1} tok/s | on {c1} chunks {:.1} tok/s",
                1000.0 / tok_ms,
                512.0 / (t0 / 1000.0),
                512.0 / (t1 / 1000.0)
            );
        }
    }
}
