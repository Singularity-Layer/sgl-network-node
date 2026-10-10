use serde_json::json;
use sgl_node::stt::{
    AudioRequest, SttEngine, WorkerConfig, MAX_SAMPLES, MODEL_BYTES, MODEL_ID, MODEL_REVISION,
    MODEL_SHA256, PROTOCOL, RUNTIME, RUNTIME_REVISION,
};
use std::time::Duration;

fn worker(body: &str) -> WorkerConfig {
    let ready = json!({
        "type":"ready","protocol":PROTOCOL,"runtime":RUNTIME,
        "runtime_revision":RUNTIME_REVISION,"model_revision":MODEL_REVISION,
        "model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES,
        "runtime_binary_sha256": "11".repeat(32), "runtime_binary_bytes": 123,
        "model_id":MODEL_ID,"audio_format":"pcm_s16le_16k_mono",
        "max_duration_seconds":60,
        "smoke_transcript":"the grid node is ready for service"
    });
    WorkerConfig {
        program: "python3".into(),
        args: vec![
            "-u".into(),
            "-c".into(),
            format!(
                "import sys,json,time\nprint({},flush=True)\n{}",
                serde_json::to_string(&ready.to_string()).unwrap(),
                body
            ),
        ],
        startup_timeout: Duration::from_secs(2),
        request_timeout: Duration::from_millis(150),
        expected_runtime_sha256: "11".repeat(32),
        expected_runtime_bytes: 123,
    }
}

fn audio(sample_count: u64) -> serde_json::Value {
    let pcm = vec![0u8; (sample_count * 2) as usize];
    canonical_audio(&pcm)
}

fn canonical_audio(pcm: &[u8]) -> serde_json::Value {
    use base64::Engine;
    json!({
        "protocol": PROTOCOL,
        "request_id": "00000000-0000-4000-8000-000000000001",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "language": "auto",
        "audio": {
            "format": "pcm_s16le",
            "data": base64::engine::general_purpose::STANDARD.encode(pcm),
            "sample_rate": 16000,
            "sample_count": pcm.len() / 2,
            "channels": 1,
            "bits_per_sample": 16,
        }
    })
}

#[tokio::test]
async fn validates_ready_round_trip_and_result() {
    let engine = SttEngine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'result','request_id':r['request_id'],'text':'hello world','language':'en','duration_seconds':1.0,'segments':[{'start':0.0,'end':1.0,'text':'hello world'}]}),flush=True)")).await.unwrap();
    assert!(engine.is_healthy());
    assert!(engine.capabilities().is_some());
    let caps = engine.capabilities().unwrap();
    assert_eq!(caps.transcription_protocol, PROTOCOL);
    assert_eq!(caps.max_duration_seconds, 60);
    let request = AudioRequest::parse(&audio(16000)).unwrap();
    let out = engine.transcribe(request).await.unwrap();
    assert_eq!(out.text, "hello world");
    assert_eq!(out.language.as_deref(), Some("en"));
    assert_eq!(out.segments.len(), 1);
    engine.stop();
}

#[tokio::test]
async fn eof_timeout_and_wrong_id_remove_capabilities() {
    for body in [
        "sys.stdin.readline(); sys.exit(0)",
        "time.sleep(5)",
        "for line in sys.stdin:\n print(json.dumps({'type':'result','request_id':999,'text':'x','language':'en','duration_seconds':1.0,'segments':[]}),flush=True)",
    ] {
        let engine = SttEngine::start(worker(body)).await.unwrap();
        let request = AudioRequest::parse(&audio(16000)).unwrap();
        assert!(engine.transcribe(request).await.is_err());
        assert!(!engine.is_healthy());
        assert!(engine.capabilities().is_none());
        engine.stop();
    }
}

#[tokio::test]
async fn startup_rejects_stdout_noise_and_bad_smoke() {
    let mut config = worker("time.sleep(5)");
    config.args = vec![
        "-u".into(),
        "-c".into(),
        "print('whisper log'); import time; time.sleep(5)".into(),
    ];
    assert!(SttEngine::start(config).await.is_err());

    // A smoke transcript that does not contain the expected words must not become ready.
    let mut wrong = worker("time.sleep(5)");
    let bad = json!({
        "type":"ready","protocol":PROTOCOL,"runtime":RUNTIME,
        "runtime_revision":RUNTIME_REVISION,"model_revision":MODEL_REVISION,
        "model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES,
        "runtime_binary_sha256": "11".repeat(32), "runtime_binary_bytes": 123,
        "model_id":MODEL_ID,"audio_format":"pcm_s16le_16k_mono",
        "max_duration_seconds":60,
        "smoke_transcript":"xyzzy completely unrelated words"
    });
    wrong.args = vec![
        "-u".into(),
        "-c".into(),
        format!(
            "import sys,json\nprint({},flush=True)\ntime=1\nimport time; time.sleep(5)",
            serde_json::to_string(&bad.to_string()).unwrap()
        ),
    ];
    assert!(SttEngine::start(wrong).await.is_err());
}

#[tokio::test]
async fn request_error_keeps_readiness_and_bad_frame_is_fatal() {
    let engine = SttEngine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'request_error','request_id':r['request_id'],'code':'invalid_input'}),flush=True)")).await.unwrap();
    let request = AudioRequest::parse(&audio(16000)).unwrap();
    assert_eq!(
        engine.transcribe(request).await.unwrap_err(),
        "transcription_input_invalid"
    );
    assert!(engine.is_healthy());
    engine.stop();

    let engine = SttEngine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'request_error','request_id':r['request_id'],'code':'other'}),flush=True)")).await.unwrap();
    let request = AudioRequest::parse(&audio(16000)).unwrap();
    assert!(engine.transcribe(request).await.is_err());
    assert!(!engine.is_healthy());
    engine.stop();
}

#[tokio::test]
async fn full_scale_audio_parses_and_oversized_is_rejected() {
    let max = audio(MAX_SAMPLES as u64);
    assert!(AudioRequest::parse(&max).is_ok());
    let over = audio(MAX_SAMPLES as u64 + 1);
    assert!(AudioRequest::parse(&over).is_err());
}

#[tokio::test]
async fn restart_repeats_readiness_then_budget_exhausts() {
    let engine = SttEngine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'result','request_id':r['request_id'],'text':'hi','language':'en','duration_seconds':1.0,'segments':[]}),flush=True)")).await.unwrap();
    assert!(engine.restart().await.is_ok());
    assert!(engine.is_healthy());
    assert!(engine.restart().await.is_ok());
    assert!(engine.restart().await.is_ok());
    assert!(engine.restart().await.is_err());
    engine.stop();
}

#[tokio::test]
async fn silence_result_is_valid_and_does_not_remove_readiness() {
    let engine = SttEngine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'result','request_id':r['request_id'],'text':'','language':None,'duration_seconds':1.0,'segments':[]}),flush=True)")).await.unwrap();
    let out = engine
        .transcribe(AudioRequest::parse(&audio(16000)).unwrap())
        .await
        .unwrap();
    assert_eq!(out.text, "");
    assert!(out.segments.is_empty());
    assert!(engine.is_healthy());
    engine.stop();
}

#[tokio::test]
async fn readiness_requires_exact_runtime_hash_and_size() {
    let mut wrong_size = worker("time.sleep(5)");
    wrong_size.expected_runtime_bytes = 124;
    assert!(SttEngine::start(wrong_size).await.is_err());
    let mut wrong_hash = worker("time.sleep(5)");
    wrong_hash.expected_runtime_sha256 = "22".repeat(32);
    assert!(SttEngine::start(wrong_hash).await.is_err());
}

/// Real whisper.cpp canary on the founder machine. Ignored by default; run with:
///   SGL_STT_CANARY_WHISPER=<whisper-cli> SGL_STT_CANARY_MODEL=<ggml-small.bin> \
///   SGL_STT_CANARY_SMOKE=<approved-fixture-directory> \
///   cargo test --release --test stt_supervisor -- --ignored production_worker_real_canary
#[tokio::test]
#[ignore]
async fn production_worker_real_canary() {
    let whisper = std::env::var("SGL_STT_CANARY_WHISPER").expect("SGL_STT_CANARY_WHISPER");
    let model = std::env::var("SGL_STT_CANARY_MODEL").expect("SGL_STT_CANARY_MODEL");
    let smoke = std::env::var("SGL_STT_CANARY_SMOKE").expect("SGL_STT_CANARY_SMOKE");
    let engine = SttEngine::production(
        std::path::Path::new(&model),
        std::path::Path::new(&whisper),
        &{
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(std::fs::read(&whisper).unwrap()))
        },
        std::path::Path::new(&smoke),
        None,
    )
    .await
    .expect("real worker startup");
    assert!(engine.is_healthy());
    let caps = engine.capabilities().unwrap();
    assert_eq!(caps.model_revision, MODEL_REVISION);
    assert_eq!(
        caps.runtime_binary_sha256,
        sgl_node::stt::MACOS_AARCH64_RUNTIME_SHA256
    );
    assert_eq!(
        caps.runtime_binary_bytes,
        sgl_node::stt::MACOS_AARCH64_RUNTIME_BYTES
    );
    assert_eq!(caps.os, "macos");
    assert_eq!(caps.architecture, "aarch64");
    // External approved known-audio fixture; no speech audio is bundled in the release.
    let wav = std::fs::read(std::path::Path::new(&smoke).join("audio.wav")).expect("smoke fixture");
    let pcm = &wav[44..];
    let value = canonical_audio(pcm);
    let request = AudioRequest::parse(&value).unwrap();
    let out = engine
        .transcribe(request)
        .await
        .expect("real transcription");
    let lowered = out.text.to_lowercase();
    assert!(
        lowered.contains("ready") && lowered.contains("service"),
        "unexpected transcript: {}",
        out.text
    );
    assert!(out.duration_seconds > 0.0);
    let mut silence = audio(16000);
    silence["request_id"] = json!("00000000-0000-4000-8000-000000000002");
    let silent = engine
        .transcribe(AudioRequest::parse(&silence).unwrap())
        .await
        .expect("silence transcription");
    assert!(silent.text.len() <= sgl_node::stt::MAX_TEXT_BYTES);
    assert!(silent.duration_seconds > 0.0 && silent.duration_seconds <= 2.0);
    assert!(engine.is_healthy());
    engine.stop();
}

#[tokio::test]
async fn replay_is_rejected_without_removing_capability() {
    let engine = SttEngine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'result','request_id':r['request_id'],'text':'hello','language':'en','duration_seconds':1.0,'segments':[]}),flush=True)")).await.unwrap();
    let request = AudioRequest::parse(&audio(16000)).unwrap();
    assert!(engine.transcribe(request.clone()).await.is_ok());
    assert_eq!(
        engine.transcribe(request).await.unwrap_err(),
        "transcription_input_invalid"
    );
    assert!(engine.is_healthy());
    engine.restart().await.unwrap();
    assert!(engine
        .transcribe(AudioRequest::parse(&audio(16000)).unwrap())
        .await
        .is_err());
    engine.stop();
}

#[cfg(unix)]
#[tokio::test]
async fn cancellation_terminates_descendants_and_removes_capability() {
    let path = std::env::temp_dir().join(format!("sgl-stt-child-{}", rand::random::<u64>()));
    let mut cfg = worker(&format!("import subprocess\np=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\nopen({},'w').write(str(p.pid))\nfor line in sys.stdin: time.sleep(30)", serde_json::to_string(&path.to_string_lossy()).unwrap()));
    cfg.request_timeout = Duration::from_secs(20);
    let engine = std::sync::Arc::new(SttEngine::start(cfg).await.unwrap());
    let worker_engine = engine.clone();
    let task = tokio::spawn(async move {
        worker_engine
            .transcribe(AudioRequest::parse(&audio(16000)).unwrap())
            .await
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    // Creation precedes write completion; wait for a complete PID rather than
    // racing an empty file on a busy native-build host.
    let pid: i32 = loop {
        if let Some(pid) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|value| value.parse().ok())
        {
            break pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "worker did not publish descendant PID"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    task.abort();
    let _ = task.await;
    while engine.is_healthy() && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    engine.stop();
    let mut gone = false;
    for _ in 0..100 {
        if unsafe { libc::kill(pid, 0) } != 0 {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(gone, "descendant survived cancellation");
    assert!(engine.capabilities().is_none());
    std::fs::remove_file(path).unwrap();
}
