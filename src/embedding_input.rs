//! Typed embedding wire input. Media remains inline and ordered; no paths or URL fetches.
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MAX_ENCODED_BODY: usize = 24 * 1024 * 1024;
const MIB: usize = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Media {
    pub encoding: String,
    pub mime_type: String,
    pub data: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Part {
    Text { text: String },
    Image { media: Media },
    Audio { media: Media, duration_seconds: f64 },
    Video { media: Media, duration_seconds: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingItem {
    pub content: Vec<Part>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EmbeddingBatch {
    pub items: Vec<EmbeddingItem>,
}

impl EmbeddingBatch {
    pub fn parse(input: &serde_json::Value, multimodal: bool) -> Result<Self, String> {
        // Serialization here is bounded before decoding media. The HTTP caller must also cap
        // the original body before JSON parsing (whitespace and duplicate keys disappear here).
        if serde_json::to_vec(input)
            .map_err(|_| "invalid embedding input")?
            .len()
            > MAX_ENCODED_BODY
        {
            return Err("embedding input exceeds encoded limit".into());
        }
        let values = match input {
            serde_json::Value::String(_) => vec![input],
            serde_json::Value::Array(items) if !items.is_empty() => items.iter().collect(),
            _ => return Err("input must be a string or a nonempty batch".into()),
        };
        let limit = if multimodal { 16 } else { 4096 };
        if values.len() > limit {
            return Err("embedding batch exceeds item limit".into());
        }
        let mut items = Vec::with_capacity(values.len());
        for value in values {
            let item = if let Some(text) = value.as_str() {
                EmbeddingItem {
                    content: vec![Part::Text { text: text.into() }],
                }
            } else if multimodal {
                // Do not include serde errors: some error variants echo input text/media.
                serde_json::from_value::<EmbeddingItem>(value.clone())
                    .map_err(|_| "invalid structured embedding item")?
            } else {
                return Err("text embedding models accept only strings".into());
            };
            items.push(item);
        }
        let batch = Self { items };
        if multimodal {
            batch.validate_media()?;
        }
        Ok(batch)
    }

    pub fn legacy_text(&self) -> Result<Vec<String>, String> {
        self.items
            .iter()
            .map(|item| match item.content.as_slice() {
                [Part::Text { text }] => Ok(text.clone()),
                _ => Err("text embedding models accept only strings".into()),
            })
            .collect()
    }

    pub fn modalities(&self) -> Vec<&'static str> {
        let mut found = Vec::new();
        for item in &self.items {
            for part in &item.content {
                let name = match part {
                    Part::Text { .. } => "text",
                    Part::Image { .. } => "image",
                    Part::Audio { .. } => "audio",
                    Part::Video { .. } => "video",
                };
                if !found.contains(&name) {
                    found.push(name);
                }
            }
        }
        found
    }

    fn validate_media(&self) -> Result<(), String> {
        let mut decoded_total = 0usize;
        for item in &self.items {
            if item.content.is_empty() || item.content.len() > 16 {
                return Err("invalid embedding part count".into());
            }
            let (mut images, mut audio, mut video, mut image_bytes) = (0, 0, 0, 0usize);
            for part in &item.content {
                let (media, allowed, cap) = match part {
                    Part::Text { text } => {
                        if text.trim().is_empty() {
                            return Err("embedding text parts must not be empty".into());
                        }
                        continue;
                    }
                    Part::Image { media } => {
                        images += 1;
                        (
                            media,
                            &["image/jpeg", "image/png", "image/webp"][..],
                            8 * MIB,
                        )
                    }
                    Part::Audio {
                        media,
                        duration_seconds,
                    } => {
                        audio += 1;
                        validate_duration(*duration_seconds, 30.0)?;
                        (
                            media,
                            &["audio/wav", "audio/flac", "audio/mpeg"][..],
                            8 * MIB,
                        )
                    }
                    Part::Video {
                        media,
                        duration_seconds,
                    } => {
                        video += 1;
                        validate_duration(*duration_seconds, 32.0)?;
                        (media, &["video/mp4"][..], 16 * MIB)
                    }
                };
                if images > 8 || audio > 1 || video > 1 {
                    return Err("embedding media count exceeds limit".into());
                }
                if media.encoding != "base64" || !allowed.contains(&media.mime_type.as_str()) {
                    return Err("unsupported embedding media transport or MIME".into());
                }
                if media.data.len() > cap.div_ceil(3) * 4 {
                    return Err("encoded embedding media exceeds limit".into());
                }
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(&media.data)
                    .map_err(|_| "invalid embedding media base64")?;
                if bytes.is_empty() || bytes.len() > cap {
                    return Err("decoded embedding media exceeds limit".into());
                }
                let hash = &media.sha256;
                if hash.len() != 64
                    || !hash
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                    || hex::encode(Sha256::digest(&bytes)) != *hash
                {
                    return Err("embedding media digest mismatch".into());
                }
                decoded_total += bytes.len();
                if matches!(part, Part::Image { .. }) {
                    image_bytes += bytes.len();
                }
                if decoded_total > 20 * MIB || image_bytes > 8 * MIB {
                    return Err("embedding media aggregate exceeds limit".into());
                }
            }
        }
        // Decoded magic, shape, actual durations, sample counts and processed token budget
        // are verified by the pinned processor before inference, not guessed from declarations.
        Ok(())
    }
}

fn validate_duration(seconds: f64, limit: f64) -> Result<(), String> {
    if !seconds.is_finite() || seconds <= 0.0 || seconds > limit {
        return Err("embedding media duration exceeds limit".into());
    }
    Ok(())
}
