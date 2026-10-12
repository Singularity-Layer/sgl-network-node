//! Opt-in local Qwen vision smoke. It starts only a temporary llama-server child
//! on 127.0.0.1 and does not register a node, install a service, or contact Grid.

#[tokio::test]
#[ignore = "requires approved local Qwen GGUF, projector, and llama-server paths"]
async fn pinned_qwen_vision_answers_an_image() {
    use base64::Engine as _;
    use sgl_node::inference::{
        explicit_llama_server, EngineMode, InferenceEngine, InferenceEngineConfig, GPU_LAYERS_AUTO,
    };
    use std::path::PathBuf;
    use std::time::Duration;

    let model = std::env::var("SGL_VISION_CANARY_MODEL").expect("set SGL_VISION_CANARY_MODEL");
    let projector =
        std::env::var("SGL_VISION_CANARY_MMPROJ").expect("set SGL_VISION_CANARY_MMPROJ");
    let server = std::env::var("SGL_VISION_CANARY_SERVER").expect("set SGL_VISION_CANARY_SERVER");
    let sha256 = std::env::var("SGL_VISION_CANARY_SERVER_SHA256")
        .expect("set SGL_VISION_CANARY_SERVER_SHA256");
    let pinned = explicit_llama_server(Some(&server), Some(&sha256))
        .expect("approved pinned llama-server")
        .expect("explicit server must be present");
    let image = std::fs::read("assets/embeddinggemma2/smoke/image.png").unwrap();
    let encoded = base64::engine::general_purpose::STANDARD.encode(image);
    let messages = serde_json::json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image briefly."},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{encoded}")}}
        ]
    }]);
    let config = InferenceEngineConfig {
        llama_server: Some(pinned),
        embedding_python: None,
        stt_whisper: None,
        stt_whisper_sha256: None,
        stt_python: None,
        stt_smoke_dir: None,
        model_path: PathBuf::from(model),
        model_name: "qwen2.5-vl-3b".into(),
        port: 0,
        threads: 4,
        gpu_layers: GPU_LAYERS_AUTO,
        context_size: 4096,
        batch_size: 512,
        parallel_slots: 1,
        mmproj_path: Some(PathBuf::from(projector)),
        image_max_tokens: Some(1024),
    };
    let engine = tokio::time::timeout(Duration::from_secs(180), async {
        let engine = InferenceEngine::create(config, EngineMode::Server).await?;
        assert_eq!(engine.mode_label(), "server");
        assert!(engine.is_vision());
        let output = engine
            .chat_completion(messages, 0.0, 32, None, None)
            .await?;
        if output.content.trim().is_empty() || output.completion_tokens == 0 {
            return Err("Qwen returned no vision text".to_string());
        }
        Ok::<_, String>(output)
    })
    .await
    .expect("Qwen vision smoke timed out")
    .expect("Qwen vision smoke failed");
    assert!(engine.prompt_tokens > 0);
}
