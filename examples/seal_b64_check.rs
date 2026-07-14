//! Cross-language check for the base64 sealed-result encoding (node → orchestrator).
//!
//! Seals argv[2] to the X25519 response pubkey argv[1] exactly like the node's result
//! path (encrypt_for_recipient_v2 → base64 ciphertext + `encoding: "base64"`), printing
//! the envelope JSON. A JS harness generates the response keypair, invokes this, then
//! opens the envelope with the orchestrator's decoder logic — proving the two sides
//! agree byte-for-byte before the node release ships.
//!
//! Run: cargo run --release --example seal_b64_check -- <resp_pub_b58> <plaintext>

use base64::Engine as _;

fn main() {
    let mut args = std::env::args().skip(1);
    let resp_pub_b58 = args.next().expect("usage: seal_b64_check <resp_pub_b58> <plaintext>");
    let plaintext = args.next().expect("usage: seal_b64_check <resp_pub_b58> <plaintext>");

    let resp_pub_vec = bs58::decode(&resp_pub_b58).into_vec().expect("bad resp pub b58");
    let resp_pub: [u8; 32] = resp_pub_vec.as_slice().try_into().expect("resp pub must be 32 bytes");

    let (sealed, ephemeral_pub) =
        sgl_node::encryption::encrypt_for_recipient_v2(&resp_pub, plaintext.as_bytes())
            .expect("seal failed");

    let envelope = serde_json::json!({
        "ciphertext": base64::engine::general_purpose::STANDARD.encode(&sealed),
        "encoding": "base64",
        "ephemeral_public_key": bs58::encode(ephemeral_pub).into_string(),
        "algorithm": "x25519-xchacha20poly1305-hkdf-v2",
    });
    println!("{envelope}");
}
