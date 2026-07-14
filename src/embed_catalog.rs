//! Embedding model catalog — shared by BOTH embedding engines:
//!   * the in-process engine (`embed.rs`, `inprocess` feature — macOS/Linux), and
//!   * the llama-server-backed server engine (`inference.rs` — Windows, where the
//!     in-process engine isn't compiled).
//!
//! Lives OUTSIDE the `inprocess` feature gate so a server-engine build can resolve
//! pooling/prefixes/dims. MUST stay in lock-step with `src/lib/embeddings.ts` in the
//! orchestrator: the orchestrator structurally validates every vector (dim, finite,
//! non-zero) and rejects mismatches, so a wrong pooling or prefix here means failed
//! jobs — never silently-wrong embeddings.

/// Pooling strategy — how the per-token hidden states become one sentence vector. MUST match the
/// orchestrator catalog per model or the produced vector won't match what clients expect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // `Last` is a valid llama.cpp pooling type kept for future last-token models
pub enum Pooling {
    Cls,
    Mean,
    Last,
}

impl Pooling {
    /// The llama-server `--pooling` flag value (server engine).
    pub fn as_server_flag(self) -> &'static str {
        match self {
            Pooling::Cls => "cls",
            Pooling::Mean => "mean",
            Pooling::Last => "last",
        }
    }
}

/// Which retrieval prefix to apply to an input (OpenAI extension; catalog encodes the strings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputType {
    Query,
    Document,
    /// No explicit type given — use the document prefix (the common indexing default).
    Unspecified,
}

/// Result of embedding one batch. `vectors` is one row per input, in input order.
pub struct EmbedOut {
    pub vectors: Vec<Vec<f32>>,
    /// Total input tokens across the whole batch (the billable count — input only).
    pub prompt_tokens: u32,
}

/// One embedding model the node can serve. `id` is the grid model id (matches the orchestrator
/// catalog). `prefix_query`/`prefix_document` are the per-family retrieval prefixes applied
/// node-side by `input_type` (e5 needs `query:`/`passage:`; bge uses an instruction; nomic uses
/// `search_query:`/`search_document:`; some models want none).
#[derive(Clone, Copy, Debug)]
pub struct EmbedModelSpec {
    pub id: &'static str,
    pub pooling: Pooling,
    /// L2-normalize the output vector (so a valid embedding is never all-zero and cosine == dot).
    pub normalize: bool,
    /// Native output dimension (== `model.n_embd()`; used to size + validate).
    pub dim: u32,
    /// Extra Matryoshka dims this model can be truncated to (empty = fixed dim).
    pub allowed_dimensions: &'static [u32],
    /// Max input tokens per item (also sizes the context window).
    pub max_input_tokens: u32,
    pub prefix_query: &'static str,
    pub prefix_document: &'static str,
    /// Extra ids that resolve to this spec (aliases the orchestrator may route by).
    pub aliases: &'static [&'static str],
}

/// Authoritative node-side catalog. Keep in lock-step with `src/lib/embeddings.ts` in the
/// orchestrator. Adding a model here is what lets an operator serve it.
pub const EMBED_CATALOG: &[EmbedModelSpec] = &[
    EmbedModelSpec {
        id: "nomic-embed-text-v1.5",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 768,
        allowed_dimensions: &[768, 512, 256, 128, 64],
        max_input_tokens: 8192,
        prefix_query: "search_query: ",
        prefix_document: "search_document: ",
        aliases: &["nomic-embed-text", "text-embedding-3-small"],
    },
    EmbedModelSpec {
        id: "bge-small-en-v1.5",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 384,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "bge-base-en-v1.5",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 768,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "bge-large-en-v1.5",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "all-minilm-l6-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 384,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "",
        prefix_document: "",
        aliases: &["all-minilm", "minilm-l6"],
    },
    EmbedModelSpec {
        id: "e5-small-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 384,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "e5-base-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 768,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "e5-large-v2",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "mxbai-embed-large-v1",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[1024, 512, 256],
        max_input_tokens: 512,
        prefix_query: "Represent this sentence for searching relevant passages: ",
        prefix_document: "",
        aliases: &[],
    },
    EmbedModelSpec {
        id: "multilingual-e5-large",
        pooling: Pooling::Mean,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 512,
        prefix_query: "query: ",
        prefix_document: "passage: ",
        aliases: &["e5-multilingual"],
    },
    EmbedModelSpec {
        id: "bge-m3",
        pooling: Pooling::Cls,
        normalize: true,
        dim: 1024,
        allowed_dimensions: &[],
        max_input_tokens: 8192,
        prefix_query: "",
        prefix_document: "",
        aliases: &["bge-m3-multilingual"],
    },
];

/// Normalize a model id the way the orchestrator does (strip quant/`.gguf` suffixes), lowercased.
fn normalize_id(id: &str) -> String {
    let mut s = id.to_ascii_lowercase();
    if let Some(pos) = s.find("-q") {
        // Strip a trailing quant tag like "-q4_k_m" only if it looks like one (digit after -q).
        if s[pos + 2..].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            s.truncate(pos);
        }
    }
    if let Some(stripped) = s.strip_suffix(".gguf") {
        s = stripped.to_string();
    }
    s.trim().to_string()
}

/// Strict lookup — `None` for anything not in the catalog (NO default fallback, mirrors the
/// orchestrator: an unknown embedding model is rejected, never served with guessed params).
pub fn embed_model_spec(id: &str) -> Option<&'static EmbedModelSpec> {
    let want = id.to_ascii_lowercase();
    let want_norm = normalize_id(id);
    EMBED_CATALOG.iter().find(|m| {
        m.id.eq_ignore_ascii_case(&want)
            || m.id.eq_ignore_ascii_case(&want_norm)
            || m.aliases.iter().any(|a| a.eq_ignore_ascii_case(&want) || a.eq_ignore_ascii_case(&want_norm))
    })
}

/// True iff the given model id is a known embedding model — the node uses this to pick the
/// embedding engine over the chat engine at startup.
pub fn is_embedding_model(id: &str) -> bool {
    embed_model_spec(id).is_some()
}

/// L2-normalize in place (no-op for a zero vector). Guarantees a unit vector so cosine == dot and
/// the orchestrator's all-zero-vector rejection never trips on a legitimate embedding.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
