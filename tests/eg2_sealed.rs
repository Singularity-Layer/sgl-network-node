use serde_json::Value;
use sgl_node::encryption::{unseal_input, EncVersion};

fn fixture() -> (Value, [u8; 32]) {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/eg2_sealed_input.json")).unwrap();
    let seed: [u8; 32] = hex::decode(fixture["node_ed25519_seed_hex"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    (fixture, seed)
}

#[test]
fn typescript_multimodal_envelope_decrypts_to_ordered_utf8_manifest() {
    let (fixture, seed) = fixture();
    let (inner, response, version) = unseal_input(&fixture["payload"], &seed).unwrap();
    assert_eq!(inner, fixture["expected_plaintext"]);
    assert!(response.is_some());
    assert_eq!(version, EncVersion::V2);
    let batch = sgl_node::embedding_input::EmbeddingBatch::parse(&inner["input"], true).unwrap();
    assert_eq!(batch.modalities(), vec!["text", "image"]);
}

#[test]
fn envelope_refuses_downgrade_and_response_key_swap() {
    let (fixture, seed) = fixture();
    for (field, value) in [
        ("encoding", serde_json::json!("base58")),
        ("payload_encoding", serde_json::json!("utf-16")),
        ("embedding_protocol", serde_json::json!("other")),
        ("algorithm", serde_json::json!("x25519-xchacha20poly1305")),
        (
            "client_response_pubkey",
            fixture["payload"]["enc"]["client_ephemeral_pubkey"].clone(),
        ),
    ] {
        let mut payload = fixture["payload"].clone();
        payload["enc"][field] = value;
        assert!(unseal_input(&payload, &seed).is_err());
    }
}
