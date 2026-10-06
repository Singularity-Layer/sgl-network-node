use serde_json::json;
use sgl_node::embedding_input::{EmbeddingBatch, Part};

#[test]
fn legacy_text_and_ordered_mixed_items_round_trip() {
    let legacy = EmbeddingBatch::parse(&json!("hello"), false).unwrap();
    assert_eq!(legacy.legacy_text().unwrap(), vec!["hello"]);
    let mixed = json!([{"content":[{"type":"text","text":"caption"},{"type":"image","media":{"encoding":"base64","mime_type":"image/png","data":"aGk=","sha256":"8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"}}]},"second"]);
    let batch = EmbeddingBatch::parse(&mixed, true).unwrap();
    assert_eq!(batch.items.len(), 2);
    assert!(matches!(batch.items[0].content[1], Part::Image { .. }));
    assert!(batch.legacy_text().is_err());
    assert!(EmbeddingBatch::parse(&mixed, false).is_err());
}

#[test]
fn media_declarations_and_transport_are_strict() {
    for media in [
        json!({"encoding":"url","mime_type":"image/png","data":"https://example.com"}),
        json!({"encoding":"base64","mime_type":"image/png","data":"!!"}),
        json!({"encoding":"base64","mime_type":"image/png","data":"aGk=","sha256":"8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4","path":"/tmp/x"}),
    ] {
        assert!(EmbeddingBatch::parse(
            &json!([{"content":[{"type":"image","media":media}]}]),
            true
        )
        .is_err());
    }
    for seconds in [0.0, 31.0] {
        assert!(EmbeddingBatch::parse(&json!([{"content":[{"type":"audio","duration_seconds":seconds,"media":{"encoding":"base64","mime_type":"audio/wav","data":"aGk=","sha256":"8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"}}]}]), true).is_err());
    }
}

#[test]
fn eg2_never_resolves_to_a_gguf_pooling_spec() {
    assert!(sgl_node::embed_catalog::is_embedding_model(
        "embeddinggemma-2"
    ));
    assert!(sgl_node::embed_catalog::embed_model_spec("embeddinggemma-2").is_none());
    assert!(sgl_node::embed_catalog::embed_model_spec("bge-base-en-v1.5").is_some());
}

#[test]
fn eg2_rejects_empty_text_parts_without_changing_gguf_legacy_input() {
    for text in ["", " \t\n"] {
        assert!(
            EmbeddingBatch::parse(&json!([{"content":[{"type":"text","text":text}]}]), true)
                .is_err()
        );
        assert!(EmbeddingBatch::parse(&json!(text), true).is_err());
        assert_eq!(
            EmbeddingBatch::parse(&json!(text), false)
                .unwrap()
                .legacy_text()
                .unwrap(),
            vec![text]
        );
    }
}
