use serde_json::json;
use sgl_node::eg2::{Eg2Engine, WorkerConfig, PROTOCOL};
use sgl_node::embedding_input::EmbeddingBatch;
use std::time::Duration;

fn worker(body: &str) -> WorkerConfig {
    let ready = json!({"type":"ready","protocol":PROTOCOL,"runtime":"mlx-vlm","processor_revision":sgl_node::eg2::MLX_VLM_REVISION,"model_revision":sgl_node::eg2::MLX_MODEL_REVISION,"modalities":["text","image","audio","video"],"dimensions":[768,512,256,128],"smoke_vectors":vec![vec![1.0 / (768.0f32).sqrt();768];5]});
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
    }
}

#[tokio::test]
async fn validates_ready_round_trip_and_mrl() {
    let engine = Eg2Engine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'result','request_id':r['request_id'],'vectors':[[1.0/(768**0.5)]*768 for _ in r['input']], 'usage':{'text':2,'image':0,'audio':0,'video':0},'item_usage':[{'text':1,'image':0,'audio':0,'video':0} for _ in r['input']]}),flush=True)")).await.unwrap();
    assert!(engine.is_healthy());
    assert!(engine.capabilities().is_some());
    let batch = EmbeddingBatch::parse(&json!(["hello", "world"]), true).unwrap();
    let out = engine
        .embed(batch, sgl_node::embed_catalog::InputType::Query, Some(128))
        .await
        .unwrap();
    assert_eq!(out.vectors.len(), 2);
    assert_eq!(out.vectors[0].len(), 128);
    assert!((out.vectors[0].iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 0.0001);
    engine.stop();
}

#[tokio::test]
async fn eof_timeout_and_wrong_id_remove_capabilities() {
    for body in ["time.sleep(0.03); sys.exit(0)", "time.sleep(5)", "for line in sys.stdin:\n print(json.dumps({'type':'result','request_id':999,'vectors':[[1.0/(768**0.5)]*768],'usage':{'text':1,'image':0,'audio':0,'video':0},'item_usage':[{'text':1,'image':0,'audio':0,'video':0}]}),flush=True)"] {
        let engine = Eg2Engine::start(worker(body)).await.unwrap();
        let batch = EmbeddingBatch::parse(&json!("hello"), true).unwrap();
        assert!(engine.embed(batch, sgl_node::embed_catalog::InputType::Unspecified, None).await.is_err());
        assert!(!engine.is_healthy());
        assert!(engine.capabilities().is_none());
        engine.stop();
    }
}

#[tokio::test]
async fn startup_rejects_stdout_and_unready_workers() {
    let mut config = worker("time.sleep(5)");
    config.args = vec![
        "-u".into(),
        "-c".into(),
        "print('model log'); import time; time.sleep(5)".into(),
    ];
    assert!(Eg2Engine::start(config).await.is_err());
}

#[tokio::test]
async fn bounded_pipe_write_timeout_and_cancellation_reap_worker() {
    use std::sync::Arc;
    for cancelled in [false, true] {
        let pid_file =
            std::env::temp_dir().join(format!("eg2-pid-{}-{}", std::process::id(), cancelled));
        let source = format!("import os\nopen({},'w').write(str(os.getpid()))\nopen(os.environ['TMPDIR']+'/sensitive','w').write('private')\nopen({}+'.tmp','w').write(os.environ['TMPDIR'])\ntime.sleep(5)",serde_json::to_string(pid_file.to_str().unwrap()).unwrap(),serde_json::to_string(pid_file.to_str().unwrap()).unwrap());
        let engine = Arc::new(Eg2Engine::start(worker(&source)).await.unwrap());
        // Much larger than the OS pipe: the supervisor must time out WRITE, not only READ.
        let batch = EmbeddingBatch::parse(&json!("x".repeat(1024 * 1024)), true).unwrap();
        let e = engine.clone();
        let task = tokio::spawn(async move {
            e.embed(batch, sgl_node::embed_catalog::InputType::Unspecified, None)
                .await
        });
        if cancelled {
            tokio::time::sleep(Duration::from_millis(50)).await;
            task.abort();
        } else {
            assert!(tokio::time::timeout(Duration::from_secs(3), task)
                .await
                .unwrap()
                .unwrap()
                .is_err());
        }
        for _ in 0..30 {
            if !engine.is_healthy() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        engine.stop();
        assert!(!engine.is_healthy());
        let pid: i32 = std::fs::read_to_string(&pid_file).unwrap().parse().unwrap();
        #[cfg(unix)]
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1, "worker was not reaped");
        let tmpfile = pid_file.with_extension("tmp");
        let tmpdir = std::fs::read_to_string(&tmpfile).unwrap();
        assert!(
            !std::path::Path::new(&tmpdir).exists(),
            "worker input files survived supervisor teardown"
        );
        std::fs::remove_file(tmpfile).unwrap();
        std::fs::remove_file(pid_file).unwrap();
    }
}

#[tokio::test]
async fn closed_stdout_and_oversized_frames_cannot_keep_ready() {
    for source in [
        "import os\ntime.sleep(0.03)\nos.close(1)\ntime.sleep(5)",
        "for line in sys.stdin:\n print('x'*(1024*1024+1),flush=True)",
    ] {
        let engine = Eg2Engine::start(worker(source)).await.unwrap();
        let batch = EmbeddingBatch::parse(&json!("hello"), true).unwrap();
        assert!(engine
            .embed(batch, sgl_node::embed_catalog::InputType::Unspecified, None)
            .await
            .is_err());
        assert!(engine.capabilities().is_none());
        engine.stop();
    }
}

#[tokio::test]
async fn mixed_modalities_preserve_parts_and_native_request_dimensions() {
    let engine=Eg2Engine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n assert r['dimensions']==768 and r['input_type']=='document'\n assert [p['type'] for p in r['input'][0]['content']]==['text','image','audio','video']\n print(json.dumps({'type':'result','request_id':r['request_id'],'vectors':[[1.0/(768**0.5)]*768],'usage':{'text':2,'image':280,'audio':25,'video':140},'item_usage':[{'text':2,'image':280,'audio':25,'video':140}]}),flush=True)")).await.unwrap();
    let media = |mime: &str| json!({"encoding":"base64","mime_type":mime,"data":"aGk=","sha256":"8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"});
    let input = json!([{"content":[{"type":"text","text":"caption"},{"type":"image","media":media("image/png")},{"type":"audio","media":media("audio/wav"),"duration_seconds":1},{"type":"video","media":media("video/mp4"),"duration_seconds":1}]}]);
    let output = engine
        .embed(
            EmbeddingBatch::parse(&input, true).unwrap(),
            sgl_node::embed_catalog::InputType::Document,
            Some(256),
        )
        .await
        .unwrap();
    assert_eq!(output.vectors[0].len(), 256);
    assert_eq!(output.usage.total().unwrap(), 447);
    engine.stop();
}

#[tokio::test]
async fn explicit_runtime_requirement_cannot_fall_through_to_chat_or_gguf() {
    use sgl_node::inference::{EngineMode, InferenceEngine, InferenceEngineConfig, ServerEngine};
    let config = InferenceEngineConfig {
        embedding_python: None,
        model_path: "/missing/eg2/snapshot".into(),
        model_name: "embeddinggemma-2".into(),
        port: 8081,
        threads: 1,
        gpu_layers: 0,
        context_size: 8192,
        batch_size: 1,
        parallel_slots: 1,
        mmproj_path: None,
        image_max_tokens: None,
    };
    for mode in [EngineMode::Server, EngineMode::InProcess] {
        match InferenceEngine::create(config.clone(), mode).await {
            Err(error) => assert!(error.contains("requires --embedding-python")),
            Ok(_) => panic!("runtime requirement bypassed"),
        }
    }
    assert!(ServerEngine::new(config)
        .start()
        .await
        .unwrap_err()
        .contains("cannot run in llama.cpp"));
}

#[tokio::test]
async fn restarts_repeat_readiness_and_stop_after_lifetime_budget() {
    let engine=Eg2Engine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n print(json.dumps({'type':'result','request_id':r['request_id'],'vectors':[[1.0/(768**0.5)]*768],'usage':{'text':1,'image':0,'audio':0,'video':0},'item_usage':[{'text':1,'image':0,'audio':0,'video':0}]}),flush=True)")).await.unwrap();
    for _ in 0..3 {
        engine.restart().await.unwrap();
        assert!(engine.capabilities().is_some());
        let output = engine
            .embed(
                EmbeddingBatch::parse(&json!("hello"), true).unwrap(),
                sgl_node::embed_catalog::InputType::Unspecified,
                None,
            )
            .await
            .unwrap();
        assert_eq!(output.vectors.len(), 1);
    }
    assert!(engine
        .restart()
        .await
        .unwrap_err()
        .contains("budget exhausted"));
    engine.stop();
    assert!(engine.capabilities().is_none());
}

#[tokio::test]
#[ignore = "requires an explicitly provisioned pinned M4 runtime and model snapshot"]
async fn production_factory_real_canary() {
    use base64::Engine as _;
    use sha2::{Digest, Sha256};
    let python = std::env::var("SGL_EG2_CANARY_PYTHON").expect("provisioned interpreter required");
    let model = std::env::var("SGL_EG2_CANARY_MODEL").expect("provisioned snapshot required");
    let prefix = format!("sgl-eg2-{}-", std::process::id());
    let owned_temp_paths = || {
        std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
            .map(|entry| entry.path())
            .collect::<std::collections::HashSet<_>>()
    };
    let before = owned_temp_paths();
    let engine = Eg2Engine::production(
        std::path::Path::new(&model),
        Some(std::path::Path::new(&python)),
    )
    .await
    .unwrap();
    assert!(engine.capabilities().is_some());
    let media = |mime: &str, bytes: &[u8]| json!({"encoding":"base64","mime_type":mime,"data":base64::engine::general_purpose::STANDARD.encode(bytes),"sha256":hex::encode(Sha256::digest(bytes))});
    let input = json!(["A quiet field",{"content":[{"type":"text","text":"A quiet field"},{"type":"image","media":media("image/png",include_bytes!("../assets/embeddinggemma2/smoke/image.png"))},{"type":"audio","duration_seconds":1.0,"media":media("audio/wav",include_bytes!("../assets/embeddinggemma2/smoke/audio.wav"))},{"type":"video","duration_seconds":1.0,"media":media("video/mp4",include_bytes!("../assets/embeddinggemma2/smoke/video.mp4"))}]}]);
    let bad_png = json!([{"content":[{"type":"image","media":media("image/png",b"not-a-png")}]}]);
    for _ in 0..5 {
        assert_eq!(
            engine
                .embed(
                    EmbeddingBatch::parse(&bad_png, true).unwrap(),
                    sgl_node::embed_catalog::InputType::Unspecified,
                    None
                )
                .await
                .err()
                .unwrap(),
            "embedding_input_invalid"
        );
        assert!(engine.capabilities().is_some());
    }
    for dim in [768, 512, 256, 128] {
        let output = engine
            .embed(
                EmbeddingBatch::parse(&input, true).unwrap(),
                sgl_node::embed_catalog::InputType::Document,
                Some(dim),
            )
            .await
            .unwrap();
        assert_eq!(output.vectors.len(), 2);
        for vector in output.vectors {
            assert_eq!(vector.len(), dim as usize);
            assert!(vector.iter().all(|v| v.is_finite()));
            assert!((vector.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 0.001);
        }
        assert!(output.usage.total().unwrap() > 0);
        println!(
            "production factory canary passed: {dim} dimensions, 2 rows, usage {}",
            output.usage.total().unwrap()
        );
    }
    engine.stop();
    assert!(engine.capabilities().is_none());
    drop(engine);
    assert!(
        owned_temp_paths().is_subset(&before),
        "private worker media/runtime directories survived shutdown"
    );
    assert!(std::path::Path::new(&model).is_dir());
    assert!(std::path::Path::new(&python).is_file());
}

#[tokio::test]
async fn bad_png_request_errors_leave_worker_and_snapshot_ready() {
    let engine=Eg2Engine::start(worker("for line in sys.stdin:\n r=json.loads(line)\n if r['input'][0]['content'][0]['type']=='image':\n  print(json.dumps({'type':'request_error','request_id':r['request_id'],'code':'invalid_input'}),flush=True)\n else:\n  print(json.dumps({'type':'result','request_id':r['request_id'],'vectors':[[1.0/(768**0.5)]*768],'usage':{'text':1,'image':0,'audio':0,'video':0},'item_usage':[{'text':1,'image':0,'audio':0,'video':0}]}),flush=True)")).await.unwrap();
    let input = json!([{"content":[{"type":"image","media":{"encoding":"base64","mime_type":"image/png","data":"aGk=","sha256":"8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"}}]}]);
    let models = vec!["embeddinggemma-2".into()];
    for _ in 0..5 {
        assert_eq!(
            engine
                .embed(
                    EmbeddingBatch::parse(&input, true).unwrap(),
                    sgl_node::embed_catalog::InputType::Unspecified,
                    None
                )
                .await
                .err()
                .unwrap(),
            "embedding_input_invalid"
        );
        let snapshot = engine.readiness_snapshot(&models);
        assert_eq!(snapshot.available_models, models);
        assert!(snapshot.capabilities.is_some());
    }
    assert!(engine
        .embed(
            EmbeddingBatch::parse(&json!("valid text"), true).unwrap(),
            sgl_node::embed_catalog::InputType::Unspecified,
            None
        )
        .await
        .is_ok());
    engine.stop();
    let dead = engine.readiness_snapshot(&models);
    assert!(dead.available_models.is_empty() && dead.capabilities.is_none());
}

#[tokio::test]
async fn corrupt_request_error_frame_is_a_runtime_failure() {
    let engine=Eg2Engine::start(worker("for line in sys.stdin:\n print(json.dumps({'type':'request_error','request_id':999,'code':'invalid_input'}),flush=True)")).await.unwrap();
    assert_eq!(
        engine
            .embed(
                EmbeddingBatch::parse(&json!("text"), true).unwrap(),
                sgl_node::embed_catalog::InputType::Unspecified,
                None
            )
            .await
            .err()
            .unwrap(),
        "embedding_runtime_failed"
    );
    assert!(engine
        .readiness_snapshot(&["embeddinggemma-2".into()])
        .available_models
        .is_empty());
    engine.stop();
}
