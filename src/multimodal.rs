//! Parsing the OpenAI multimodal message shape into what the in-process engine needs.
//!
//! WHY THIS EXISTS (security, not features): user prompts are end-to-end encrypted and this
//! node is the only place they are decrypted. The `server` engine then hands that PLAINTEXT
//! to a separate `llama-server` child over unauthenticated HTTP on 127.0.0.1, where the
//! machine's operator can read it. Vision models were forced onto that engine because the
//! in-process one could not do images, so every Mac node serving vision exposed user prompts.
//! This module is the first step of closing that: turning `{"type":"image_url"}` parts into
//! (marker text + raw image bytes) that `mtmd` can consume INSIDE this process.
//!
//! Deliberately PURE and free of any llama/mtmd types so it compiles and is unit-testable
//! without the `vision` feature and without loading a model — the parity and abuse cases
//! below are exactly the ones that are painful to exercise against a real GPU.

use serde::Deserialize;

use crate::inference::ChatMessage;

/// The placeholder mtmd swaps for an image while tokenizing. Must equal the vendored
/// `mtmd_default_marker()`; pinned by a test under the `vision` feature so a crate bump
/// cannot silently desynchronise us (a wrong marker means images are dropped from the
/// prompt and the model answers about text it cannot see).
pub const MEDIA_MARKER: &str = "<__media__>";

/// Cheap DoS bound. The sealed payload is already capped (node.rs), but a payload full of
/// tiny images would still mean many vision-encoder passes on an operator's machine for one
/// request. Refuse rather than let one job monopolise a node.
pub const MAX_IMAGES: usize = 8;

#[derive(Deserialize)]
#[serde(untagged)]
enum RawContent {
    /// Plain `"content": "hello"` — every text request, unchanged.
    Text(String),
    /// `"content": [{"type":"text",...},{"type":"image_url",...}]`
    Parts(Vec<RawPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum RawPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: RawImageUrl },
}

#[derive(Deserialize)]
struct RawImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct RawMessage {
    role: String,
    content: RawContent,
}

/// Flatten OpenAI messages into (text-only messages, image bytes in marker order).
///
/// Each image part becomes [`MEDIA_MARKER`] in the message text and its decoded bytes are
/// appended to the returned vec. The Nth marker across the whole conversation corresponds to
/// the Nth image — mtmd matches them positionally, so ORDER IS LOAD-BEARING.
///
/// Text-only conversations must come out byte-identical to today's parse; that is what keeps
/// existing billing and output unchanged.
pub fn flatten(messages: &serde_json::Value) -> Result<(Vec<ChatMessage>, Vec<Vec<u8>>), String> {
    let raw: Vec<RawMessage> =
        serde_json::from_value(messages.clone()).map_err(|e| format!("Invalid messages format: {e}"))?;

    let mut out = Vec::with_capacity(raw.len());
    let mut images: Vec<Vec<u8>> = Vec::new();

    for m in raw {
        let content = match m.content {
            RawContent::Text(t) => t,
            RawContent::Parts(parts) => {
                let mut buf = String::new();
                for p in parts {
                    match p {
                        RawPart::Text { text } => buf.push_str(&text),
                        RawPart::ImageUrl { image_url } => {
                            if images.len() >= MAX_IMAGES {
                                return Err(format!("too many images (max {MAX_IMAGES})"));
                            }
                            images.push(decode_data_url(&image_url.url)?);
                            buf.push_str(MEDIA_MARKER);
                        }
                    }
                }
                buf
            }
        };
        out.push(ChatMessage {
            role: m.role,
            content,
        });
    }
    Ok((out, images))
}

/// True when any message carries array-style content. Callers use this to keep the opaque
/// forwarding path for the server engine instead of destructuring into `{role, content}`,
/// which silently rejects vision requests.
pub fn has_multimodal_content(messages: &serde_json::Value) -> bool {
    messages
        .as_array()
        .map(|ms| {
            ms.iter()
                .any(|m| m.get("content").map(|c| c.is_array()).unwrap_or(false))
        })
        .unwrap_or(false)
}

/// Decode a `data:image/...;base64,...` URL into raw file bytes.
///
/// Remote `http(s)` URLs are REFUSED on purpose. The prompt reaches us sealed precisely so
/// that nobody between the user and this process can see it; having the node then fetch an
/// attacker-chosen URL would leak that a request happened (and let the prompt author probe
/// the operator's network from inside their machine). Inline bytes only.
fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "image_url must be an inline data: URL (remote URLs are not fetched)".to_string())?;
    let comma = rest
        .find(',')
        .ok_or_else(|| "malformed data: URL (no comma)".to_string())?;
    let meta = &rest[..comma];
    if !meta.ends_with(";base64") {
        return Err("image data: URL must be base64-encoded".to_string());
    }
    let b64 = &rest[comma + 1..];
    let bytes = base64_decode(b64)?;
    if bytes.is_empty() {
        return Err("image data: URL decoded to zero bytes".to_string());
    }
    Ok(bytes)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    // Whitespace is legal inside a long data URL that has been line-wrapped.
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .map_err(|e| format!("invalid base64 image data: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn png_data_url() -> String {
        // 1x1 PNG, the smallest thing stb_image will actually accept.
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        format!("data:image/png;base64,{b64}")
    }

    /// Text-only must be byte-identical to the old parse, or every existing request changes.
    #[test]
    fn plain_text_is_unchanged_and_has_no_images() {
        let m = json!([{"role":"user","content":"hello"}]);
        let (msgs, imgs) = flatten(&m).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hello");
        assert!(imgs.is_empty());
        assert!(!has_multimodal_content(&m));
    }

    #[test]
    fn image_part_becomes_a_marker_and_bytes() {
        let m = json!([{"role":"user","content":[
            {"type":"text","text":"what is this? "},
            {"type":"image_url","image_url":{"url": png_data_url()}}
        ]}]);
        assert!(has_multimodal_content(&m));
        let (msgs, imgs) = flatten(&m).unwrap();
        assert_eq!(msgs[0].content, format!("what is this? {MEDIA_MARKER}"));
        assert_eq!(imgs.len(), 1);
        assert_eq!(&imgs[0][1..4], b"PNG");
    }

    /// mtmd pairs the Nth marker with the Nth bitmap, so a reordering bug here would make the
    /// model describe the wrong picture while everything still "works".
    #[test]
    fn multiple_images_keep_marker_order() {
        let a = png_data_url();
        let m = json!([{"role":"user","content":[
            {"type":"image_url","image_url":{"url": a}},
            {"type":"text","text":" and "},
            {"type":"image_url","image_url":{"url": "data:image/png;base64,QUJD"}}
        ]}]);
        let (msgs, imgs) = flatten(&m).unwrap();
        assert_eq!(msgs[0].content, format!("{MEDIA_MARKER} and {MEDIA_MARKER}"));
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[1], b"ABC");
    }

    #[test]
    fn remote_urls_are_refused_not_fetched() {
        let m = json!([{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}
        ]}]);
        let e = flatten(&m).unwrap_err();
        assert!(e.contains("remote URLs are not fetched"), "got: {e}");
    }

    #[test]
    fn rejects_non_base64_and_empty_images() {
        for url in ["data:image/png,notbase64", "data:image/png;base64,"] {
            let m = json!([{"role":"user","content":[
                {"type":"image_url","image_url":{"url": url}}
            ]}]);
            assert!(flatten(&m).is_err(), "should have rejected {url}");
        }
    }

    #[test]
    fn caps_the_number_of_images() {
        let parts: Vec<_> = (0..MAX_IMAGES + 1)
            .map(|_| json!({"type":"image_url","image_url":{"url": png_data_url()}}))
            .collect();
        let m = json!([{"role":"user","content": parts}]);
        assert!(flatten(&m).unwrap_err().contains("too many images"));
    }

    /// A user typing the marker themselves must not be able to smuggle in a phantom image
    /// slot. We keep their text verbatim; the marker/bitmap count mismatch is then caught by
    /// mtmd at tokenize time and fails the job rather than mis-billing it.
    #[test]
    fn user_typed_marker_is_preserved_verbatim() {
        let m = json!([{"role":"user","content": format!("look {MEDIA_MARKER} here")}]);
        let (msgs, imgs) = flatten(&m).unwrap();
        assert_eq!(msgs[0].content, format!("look {MEDIA_MARKER} here"));
        assert!(imgs.is_empty());
    }

    #[test]
    fn whitespace_wrapped_base64_still_decodes() {
        let m = json!([{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,QU\nJD"}}
        ]}]);
        let (_, imgs) = flatten(&m).unwrap();
        assert_eq!(imgs[0], b"ABC");
    }

    /// The marker we substitute MUST equal what the vendored mtmd expects, or images are
    /// silently dropped from the prompt.
    #[cfg(feature = "vision")]
    #[test]
    fn marker_matches_the_vendored_mtmd_default() {
        assert_eq!(MEDIA_MARKER, llama_cpp_2::mtmd::mtmd_default_marker());
    }
}
