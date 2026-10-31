//! Grid `image-v1` job glue: the seam between the node loop (`node.rs`) and the image worker
//! (`image.rs`). Kept out of node.rs on purpose (CLAUDE.md rule 6: the hot file must not grow).
//!
//! Flow for one dispatched job:
//!   1. parse the cleartext reservation binding (`input_payload.image_reservation`);
//!   2. require the negotiated sealed v2 envelope with a client response key, and unseal it;
//!   3. bind the decrypted canonical request to the job id and the reservation, generate;
//!   4. seal `image-result-v1` (JSON header line + raw PNG) to the client's response key,
//!      sign it in the `image` result domain;
//!   5. complete over REST, retrying with exponential backoff until the node deadline minus an
//!      upload margin. The server completion is one idempotent transaction (CAS on the job
//!      row), so a retry after a lost response is safe and never double-bills.
//!
//! Failure codes reported to the orchestrator are only `image_input_invalid` (anything the
//! client controls) and `image_runtime_failed` (this node).
use crate::image::{self, ReservationBinding};
use crate::inference::InferenceEngine;
use crate::orchestrator::{CompletionAttemptError, OrchestratorClient, PendingJob};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const INPUT_INVALID: &str = "image_input_invalid";
pub const RUNTIME_FAILED: &str = "image_runtime_failed";

/// Cleartext reservation binding from the dispatch, checked before anything is decrypted.
pub fn parse_binding(job: &PendingJob) -> Result<ReservationBinding, &'static str> {
    if job.model.as_deref() != Some(image::MODEL_ID) {
        return Err(INPUT_INVALID);
    }
    job.input_payload
        .as_ref()
        .ok_or(INPUT_INVALID)
        .and_then(|payload| ReservationBinding::parse_dispatch(payload).map_err(|_| INPUT_INVALID))
}

/// Unseal the dispatch. Only the negotiated v2 envelope with a response key is accepted: a
/// relay can neither downgrade it nor strip the key the result is sealed to.
pub fn unseal(
    job: &PendingJob,
    node_secret: &[u8; 32],
) -> Result<(PendingJob, [u8; 32]), &'static str> {
    let payload = job.input_payload.as_ref().ok_or(INPUT_INVALID)?;
    let (inner, response, version) =
        crate::encryption::unseal_input(payload, node_secret).map_err(|_| INPUT_INVALID)?;
    match (response, version) {
        (Some(response), crate::encryption::EncVersion::V2) => {
            let mut effective = job.clone();
            effective.input_payload = Some(inner);
            Ok((effective, response))
        }
        _ => Err(INPUT_INVALID),
    }
}

/// Generate one image for an unsealed job. Returns the `image-result-v1` plaintext and the
/// cleartext usage.
pub async fn execute(
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    reservation: &ReservationBinding,
) -> Result<(Vec<u8>, serde_json::Value), String> {
    let engine = engine
        .as_ref()
        .filter(|e| e.mode_label() == image::ENGINE_LABEL)
        .ok_or(RUNTIME_FAILED)?;
    if job.model.as_deref() != Some(image::MODEL_ID) {
        return Err(INPUT_INVALID.into());
    }
    let request = job
        .input_payload
        .as_ref()
        .ok_or(INPUT_INVALID)
        .and_then(|p| image::ImageRequest::parse(p).map_err(|_| INPUT_INVALID))?;
    if request.request_id != job.id || !reservation.matches(&request) {
        return Err(INPUT_INVALID.into());
    }
    let output = engine.generate_image(request.clone()).await?;
    Ok((
        image::result_plaintext(&job.id, &request, &output),
        image::usage(&output),
    ))
}

/// Seal the result to the client's response key and build the REST completion body, signed in
/// the `image` domain over the base64 ciphertext exactly as it travels.
pub fn completion_body(
    node_secret: &[u8; 32],
    job_id: &str,
    response_pub: &[u8; 32],
    plaintext: &[u8],
    usage: serde_json::Value,
) -> Result<serde_json::Value, &'static str> {
    use base64::Engine as _;
    let (sealed, ephemeral_pub) =
        crate::encryption::encrypt_for_recipient_v2(response_pub, plaintext)
            .map_err(|_| RUNTIME_FAILED)?;
    let ciphertext = base64::engine::general_purpose::STANDARD.encode(&sealed);
    if ciphertext.len() > image::MAX_SEALED_RESULT_B64 {
        return Err(RUNTIME_FAILED);
    }
    let signature = crate::node::sign_sealed_result_envelope(
        node_secret,
        "image",
        job_id,
        ciphertext.as_bytes(),
    );
    Ok(serde_json::json!({
        "sealed_result": {
            "algorithm": crate::encryption::ALGO_V2,
            "encoding": "base64",
            "ciphertext": ciphertext,
            "ephemeral_public_key": bs58::encode(ephemeral_pub).into_string(),
        },
        "usage": usage,
        "result_envelope_signature": signature,
        "result_envelope_version": "v1",
    }))
}

/// REST completion with exponential backoff, bounded by `image::NODE_DEADLINE` from `received`.
/// Gives up on a permanent refusal or when no attempt fits before the deadline; the
/// orchestrator's reaper then fails the job and releases the client's credit hold, so an
/// abandoned completion is never billed.
pub async fn complete_with_retry(
    client: &OrchestratorClient,
    job_id: &str,
    body: &serde_json::Value,
    received: Instant,
) -> Result<u32, String> {
    let mut attempt = 0u32;
    let mut last = String::from("no completion attempt fit before the deadline");
    while let Some((wait, timeout)) = image::plan_completion_attempt(received.elapsed(), attempt) {
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
        attempt += 1;
        match client.complete_image_attempt(job_id, body, timeout).await {
            Ok(()) => return Ok(attempt),
            Err(CompletionAttemptError::Permanent(e)) => return Err(e),
            Err(CompletionAttemptError::Retryable(e)) => {
                tracing::warn!("image completion attempt {attempt} for {job_id} failed: {e}");
                last = e;
            }
        }
    }
    Err(last)
}

/// The whole image job. Called from `process_job` before the generic unseal path.
pub async fn process(
    client: &OrchestratorClient,
    engine: &Option<Arc<InferenceEngine>>,
    job: &PendingJob,
    node_secret: &[u8; 32],
    telemetry: &crate::telemetry::Telemetry,
) {
    let received = Instant::now();
    let prepared = parse_binding(job).and_then(|binding| {
        unseal(job, node_secret).map(|(effective, response)| (binding, effective, response))
    });
    let (binding, effective, response) = match prepared {
        Ok(prepared) => prepared,
        Err(reason) => {
            let _ = client.fail_job(&job.id, reason).await;
            return;
        }
    };
    let outcome = execute(engine, &effective, &binding).await;
    let summary = outcome
        .as_ref()
        .map(|(_, usage)| usage.clone())
        .map_err(|e| e.clone());
    telemetry.record_buffered_result("image", received.elapsed(), &summary);
    let body = outcome.and_then(|(plaintext, usage)| {
        completion_body(node_secret, &job.id, &response, &plaintext, usage).map_err(str::to_string)
    });
    match body {
        Ok(body) => match complete_with_retry(client, &job.id, &body, received).await {
            Ok(attempts) => tracing::info!(
                "Job {} completed (E2E sealed image, {attempts} attempt(s))",
                job.id
            ),
            Err(e) => tracing::error!("Image completion for {} abandoned: {e}", job.id),
        },
        Err(reason) => {
            let reason = if reason == INPUT_INVALID {
                INPUT_INVALID
            } else {
                RUNTIME_FAILED
            };
            let _ = client.fail_job(&job.id, reason).await;
        }
    }
}

/// WebSocket heartbeat for an image node: capabilities nested at `capabilities.image`, and
/// `available_models` only while the worker is healthy.
pub fn ws_heartbeat(
    engine: Arc<InferenceEngine>,
    models: Vec<String>,
    encryption_public_key: String,
    signature: Option<String>,
    key_version: Option<u32>,
    current_load: f64,
    active: Arc<AtomicU32>,
) -> Arc<dyn Fn() -> serde_json::Value + Send + Sync> {
    Arc::new(move || {
        let snapshot = engine
            .image_snapshot(&models)
            .expect("dedicated image engine");
        let cap = snapshot.capabilities.map(|c| {
            c.bind_heartbeat(
                1u32.saturating_sub(active.load(Ordering::Relaxed)),
                &encryption_public_key,
                signature.as_deref(),
                key_version,
            )
        });
        serde_json::json!({"type":"heartbeat", "current_load":current_load,
            "available_models":snapshot.available_models,
            "capabilities":crate::orchestrator::image_capability_wire(cap),
            "encryption_public_key":encryption_public_key,
            "encryption_public_key_signature":signature, "key_version":key_version,
            "max_concurrent_jobs":1})
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RID: &str = "6f1d2c3b-4a59-4e8f-9a7b-1c2d3e4f5a6b";

    fn fake_engine_config() -> (crate::image::WorkerConfig, std::path::PathBuf) {
        // The real image_worker.py in FAKE mode, materialized into a private folder.
        let dir = std::env::temp_dir().join(format!(
            "sgl-image-job-test-{}-{:x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("image_worker.py"),
            include_bytes!("../scripts/image_worker.py"),
        )
        .unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            include_bytes!("../scripts/image_model_files_flux2_klein_4b_q4.json"),
        )
        .unwrap();
        let config = crate::image::WorkerConfig {
            program: "python3".into(),
            args: vec![
                "-I".into(),
                "-u".into(),
                dir.join("image_worker.py").to_str().unwrap().into(),
                "--model-dir".into(),
                dir.to_str().unwrap().into(),
                "--manifest".into(),
                dir.join("manifest.json").to_str().unwrap().into(),
            ],
            startup_timeout: std::time::Duration::from_secs(20),
            request_timeout: std::time::Duration::from_secs(20),
            extra_env: vec![(crate::image::FAKE_ENV.into(), "ok".into())],
            allow_fake: true,
        };
        (config, dir)
    }

    fn canonical_plaintext(prompt: &str, seed: &str) -> String {
        use base64::Engine;
        // Exact key order from the spec: protocol, request_id, model, model_revision,
        // model_sha256, width, height, seed, prompt{encoding, bytes, data}.
        format!(
            r#"{{"protocol":"{}","request_id":"{}","model":"{}","model_revision":"{}","model_sha256":"{}","width":1024,"height":1024,"seed":{},"prompt":{{"encoding":"base64","bytes":{},"data":"{}"}}}}"#,
            image::PROTOCOL,
            RID,
            image::MODEL_ID,
            image::MODEL_REVISION,
            image::MODEL_SHA256,
            seed,
            prompt.len(),
            base64::engine::general_purpose::STANDARD.encode(prompt)
        )
    }

    fn dispatch(enc: serde_json::Value, prompt_bytes: usize, seed: Option<u64>) -> PendingJob {
        PendingJob {
            id: RID.into(),
            job_type: "image".into(),
            model: Some(image::MODEL_ID.into()),
            input_payload: Some(serde_json::json!({"enc": enc, "image_reservation": {
                "protocol": image::PROTOCOL, "request_id": RID, "model": image::MODEL_ID,
                "model_revision": image::MODEL_REVISION, "model_sha256": image::MODEL_SHA256,
                "width": 1024, "height": 1024, "seed": seed, "prompt_bytes": prompt_bytes}})),
        }
    }

    #[test]
    fn image_results_sign_in_their_own_domain_and_verify() {
        use ed25519_dalek::{Signature, Signer, SigningKey, Verifier};
        let secret = [7u8; 32];
        let ciphertext = b"Y2lwaGVydGV4dA==";
        let signature = crate::node::sign_sealed_result_envelope(&secret, "image", RID, ciphertext);
        assert_eq!(
            signature,
            crate::crypto::sign_result_envelope(&secret, RID, "image", ciphertext)
        );
        for other in ["speech", "transcription", "inference"] {
            assert_ne!(
                signature,
                crate::node::sign_sealed_result_envelope(&secret, other, RID, ciphertext)
            );
        }
        // Verify exactly as the orchestrator does: Ed25519 over
        // "sgl-result-v1\n<jobId>\nimage\n<sha256hex(ciphertext)>".
        use sha2::{Digest, Sha256};
        let message = format!(
            "sgl-result-v1\n{RID}\nimage\n{}",
            hex::encode(Sha256::digest(ciphertext))
        );
        let key = SigningKey::from_bytes(&secret);
        let bytes: [u8; 64] = bs58::decode(&signature)
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap();
        let parsed = Signature::from_bytes(&bytes);
        assert!(key
            .verifying_key()
            .verify(message.as_bytes(), &parsed)
            .is_ok());
        assert_eq!(key.sign(message.as_bytes()), parsed);
        assert!(key
            .verifying_key()
            .verify(
                message.replace("\nimage\n", "\nspeech\n").as_bytes(),
                &parsed
            )
            .is_err());
    }

    #[test]
    fn binding_and_unseal_fail_closed() {
        let node_seed = [0x55u8; 32];
        let node_pub =
            crate::encryption::EncryptionKeypair::from_ed25519_seed(&node_seed).public_key_bs58();
        let response_pub =
            *x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from([0x66u8; 32]))
                .as_bytes();
        let plaintext = canonical_plaintext("a fox", "null");
        let enc = crate::encryption::test_client::seal_input_v2(
            &node_pub,
            [0x77u8; 32],
            &response_pub,
            plaintext.as_bytes(),
        );
        let job = dispatch(enc.clone(), 5, None);
        assert!(parse_binding(&job).is_ok());
        let (effective, response) = unseal(&job, &node_seed).unwrap();
        assert_eq!(response, response_pub);
        assert_eq!(
            effective.input_payload.unwrap()["prompt"]["encoding"],
            "base64"
        );

        let mut wrong_model = job.clone();
        wrong_model.model = Some("kokoro-82m".into());
        assert_eq!(parse_binding(&wrong_model).unwrap_err(), INPUT_INVALID);
        // v1 / unknown algorithm, extra envelope keys, a foreign node key: all rejected.
        let mut v1 = enc.clone();
        v1["algorithm"] = serde_json::json!("x25519-xchacha20poly1305-v1");
        assert_eq!(
            unseal(&dispatch(v1, 5, None), &node_seed).unwrap_err(),
            INPUT_INVALID
        );
        let mut extra = enc.clone();
        extra["embedding_protocol"] = serde_json::json!("x");
        assert!(unseal(&dispatch(extra, 5, None), &node_seed).is_err());
        assert!(unseal(&job, &[0x56u8; 32]).is_err());
        // A plaintext (unsealed) dispatch is never served.
        let mut plain = job.clone();
        plain.input_payload = Some(serde_json::json!({"image_reservation": {}}));
        assert!(unseal(&plain, &node_seed).is_err());
    }

    #[tokio::test]
    async fn sealed_round_trip_with_fake_worker() {
        use base64::Engine;
        let (config, dir) = fake_engine_config();
        let image_engine = crate::image::ImageEngine::start(config).await.unwrap();
        let engine = Some(Arc::new(InferenceEngine::Image(Box::new(image_engine))));

        let node_seed = [0x55u8; 32];
        let node_pub =
            crate::encryption::EncryptionKeypair::from_ed25519_seed(&node_seed).public_key_bs58();
        let response_secret = [0x66u8; 32];
        let response_pub =
            *x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from(response_secret))
                .as_bytes();
        let prompt = "a lighthouse at dusk, \"cinematic\"";
        let plaintext = canonical_plaintext(prompt, "1234");
        let enc = crate::encryption::test_client::seal_input_v2(
            &node_pub,
            [0x77u8; 32],
            &response_pub,
            plaintext.as_bytes(),
        );
        // The orchestrator's billing invariant: the exact ciphertext length proves prompt_bytes
        // (canonical template without the data, plus ceil(bytes/3)*4 base64 chars, plus the
        // 24-byte nonce and 16-byte tag).
        let data = base64::engine::general_purpose::STANDARD.encode(prompt);
        let template_len = plaintext.replace(&data, "").len();
        let sealed_len = base64::engine::general_purpose::STANDARD
            .decode(enc["ciphertext"].as_str().unwrap())
            .unwrap()
            .len();
        assert_eq!(sealed_len, template_len + prompt.len().div_ceil(3) * 4 + 40);

        let job = dispatch(enc, prompt.len(), Some(1234));
        let binding = parse_binding(&job).unwrap();
        let (effective, response) = unseal(&job, &node_seed).unwrap();

        // request_id must equal the job id; the reservation must bind every field.
        let mut other_id = effective.clone();
        other_id.id = "7f1d2c3b-4a59-4e8f-9a7b-1c2d3e4f5a6b".into();
        assert_eq!(
            execute(&engine, &other_id, &binding).await.unwrap_err(),
            INPUT_INVALID
        );
        let mut wrong_seed = binding.clone();
        wrong_seed.seed = Some(1235);
        assert_eq!(
            execute(&engine, &effective, &wrong_seed).await.unwrap_err(),
            INPUT_INVALID
        );
        let mut wrong_bytes = binding.clone();
        wrong_bytes.prompt_bytes += 1;
        assert_eq!(
            execute(&engine, &effective, &wrong_bytes)
                .await
                .unwrap_err(),
            INPUT_INVALID
        );
        // A non-image engine never serves an image job.
        assert_eq!(
            execute(&None, &effective, &binding).await.unwrap_err(),
            RUNTIME_FAILED
        );

        let (result, usage) = execute(&engine, &effective, &binding).await.unwrap();
        assert_eq!(usage["pixels"], 1024 * 1024);
        let body = completion_body(&node_seed, &job.id, &response, &result, usage).unwrap();
        let sealed = &body["sealed_result"];
        let mut keys: Vec<&str> = sealed
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "algorithm",
                "ciphertext",
                "encoding",
                "ephemeral_public_key"
            ]
        );
        assert_eq!(sealed["algorithm"], crate::encryption::ALGO_V2);
        assert_eq!(body["result_envelope_version"], "v1");
        assert_eq!(
            body["result_envelope_signature"],
            crate::crypto::sign_result_envelope(
                &node_seed,
                &job.id,
                "image",
                sealed["ciphertext"].as_str().unwrap().as_bytes()
            )
        );

        // The client opens it: header line, then a PNG of the requested size.
        let ciphertext = base64::engine::general_purpose::STANDARD
            .decode(sealed["ciphertext"].as_str().unwrap())
            .unwrap();
        let eph: [u8; 32] = bs58::decode(sealed["ephemeral_public_key"].as_str().unwrap())
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap();
        let opened =
            crate::encryption::test_client::open_result_v2(response_secret, &eph, &ciphertext);
        assert_eq!(opened, result);
        let split = opened.iter().position(|b| *b == b'\n').unwrap();
        let header: serde_json::Value = serde_json::from_slice(&opened[..split]).unwrap();
        let png = &opened[split + 1..];
        assert_eq!(header["job_id"], RID);
        assert_eq!(header["request_id"], RID);
        assert_eq!(header["seed"], 1234);
        assert_eq!(header["image"]["bytes"], png.len());
        assert_eq!(crate::image::png_dimensions(png), Some((1024, 1024)));
        // The same request id is never served twice by this node.
        assert_eq!(
            execute(&engine, &effective, &binding).await.unwrap_err(),
            INPUT_INVALID
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn oversized_results_are_a_runtime_failure() {
        let response_pub =
            *x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from([0x66u8; 32]))
                .as_bytes();
        let ok = vec![0u8; 6_000_640];
        assert!(
            completion_body(&[1u8; 32], RID, &response_pub, &ok, serde_json::json!({})).is_ok()
        );
        let huge = vec![0u8; 6_400_000];
        assert_eq!(
            completion_body(&[1u8; 32], RID, &response_pub, &huge, serde_json::json!({}))
                .unwrap_err(),
            RUNTIME_FAILED
        );
    }
}
