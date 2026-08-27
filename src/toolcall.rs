//! Tool calling for the IN-PROCESS engine: which syntax a model speaks, and parsing it back.
//!
//! WHY THIS EXISTS: the in-process engine keeps the decrypted prompt inside this hardened
//! process, which is the whole point of the confidentiality migration. But it could not do
//! tool calls — it errored on the stream path and silently dropped `tools` on the non-stream
//! path — so every node migrated to in-process lost agent support. Agents are the main use,
//! so that quietly blocked the migration entirely: we could never turn off the engine that
//! leaks prompts, because the safe one couldn't serve the traffic.
//!
//! llama-server does this in its own C++ layer (`server-chat.cpp`), which the `llama-cpp-2`
//! crate does not expose — we link the library, not the server. So it is reimplemented here.
//!
//! Kept PURE and free of llama types so every case below is testable without a model or a
//! GPU, which is exactly what makes tool-syntax edge cases affordable to cover.

/// The tool-call syntax a loaded model was taught. Detected from the chat template we
/// ACTUALLY rendered, because that template is what told the model how to speak this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>{"name":…,"arguments":{…}}</tool_call>` — Qwen (incl. our override
    /// template), Qwen3, Hermes/NousResearch.
    HermesXml,
    /// Llama 3.x: bare `{"name":…,"parameters":{…}}`, optionally after `<|python_tag|>`.
    Llama3Json,
    /// Mistral/Devstral: `[TOOL_CALLS][{"name":…,"arguments":{…}}]`.
    MistralBracket,
    /// HOMURA's trained protocol: bare `{"tool":…,"arguments":{…}}` emitted as plain text.
    Homura,
    /// The model has no way to express a tool call. Requests carrying `tools` must be
    /// REFUSED, never silently answered as prose — that was the original bug.
    None,
}

impl ToolFormat {
    pub fn supports_tools(self) -> bool {
        !matches!(self, ToolFormat::None)
    }
}

/// Decide which syntax the model speaks.
///
/// Sniffs the RENDERED template rather than the model name: the template is ground truth for
/// what the model was instructed to emit this request, and it already accounts for our Qwen
/// tools-template override. Name matching is only the fallback for models whose protocol is
/// trained in rather than templated (HOMURA).
pub fn detect(template: &str, model_name: &str, override_applied: bool) -> ToolFormat {
    // The override swaps in our Qwen2.5 tools template, which is Hermes-style.
    if override_applied {
        return ToolFormat::HermesXml;
    }
    if is_homura(model_name) {
        return ToolFormat::Homura;
    }
    if template.contains("<tool_call>") {
        return ToolFormat::HermesXml;
    }
    if template.contains("[TOOL_CALLS]") {
        return ToolFormat::MistralBracket;
    }
    if template.contains("<|python_tag|>") || template.contains("ipython") {
        return ToolFormat::Llama3Json;
    }
    // A template with no tools branch cannot express a call. Detecting this HERE is what
    // lets the engine refuse the job instead of returning prose to an agent.
    ToolFormat::None
}

/// Same predicate the node uses elsewhere for HOMURA's trained protocol.
fn is_homura(model_name: &str) -> bool {
    model_name.to_ascii_lowercase().contains("homura")
}

/// The opening/closing markers a scanner must watch for, if any.
pub fn markers(fmt: ToolFormat) -> Option<(&'static str, &'static str)> {
    match fmt {
        ToolFormat::HermesXml => Some(("<tool_call>", "</tool_call>")),
        ToolFormat::MistralBracket => Some(("[TOOL_CALLS]", "]")),
        // Llama3/HOMURA emit BARE JSON with no wrapper, so a marker scan cannot find them —
        // they need brace-balance detection instead. Kept explicit so a caller cannot assume
        // markers exist for every format.
        ToolFormat::Llama3Json | ToolFormat::Homura | ToolFormat::None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_TOOLS: &str = include_str!("templates/qwen2.5-tools.jinja");

    /// The override template is Hermes-style; a node running it must parse `<tool_call>`.
    #[test]
    fn override_template_is_hermes() {
        assert_eq!(detect("", "qwen-2.5-7b", true), ToolFormat::HermesXml);
        assert_eq!(detect(QWEN_TOOLS, "qwen-2.5-7b", false), ToolFormat::HermesXml);
    }

    #[test]
    fn homura_uses_its_trained_protocol() {
        assert_eq!(detect("", "homura-30b", false), ToolFormat::Homura);
        // Name wins over an unrelated template, since the protocol is trained in, not templated.
        assert_eq!(detect("plain", "HOMURA-30B", false), ToolFormat::Homura);
    }

    #[test]
    fn detects_families_from_their_templates() {
        assert_eq!(detect("… <|python_tag|> …", "llama-3.1-8b", false), ToolFormat::Llama3Json);
        assert_eq!(detect("… [TOOL_CALLS] …", "mistral-7b", false), ToolFormat::MistralBracket);
    }

    /// A template with NO tools branch must report None so the engine REFUSES the job.
    /// Answering as prose is the exact bug this module exists to kill.
    #[test]
    fn template_without_tools_support_is_none() {
        let plain = "{% for m in messages %}{{ m.content }}{% endfor %}";
        let f = detect(plain, "gemma-2-2b", false);
        assert_eq!(f, ToolFormat::None);
        assert!(!f.supports_tools());
    }

    #[test]
    fn bare_json_formats_have_no_markers() {
        assert!(markers(ToolFormat::Llama3Json).is_none());
        assert!(markers(ToolFormat::Homura).is_none());
        assert_eq!(markers(ToolFormat::HermesXml), Some(("<tool_call>", "</tool_call>")));
    }

    /// THE STAGE-0 BLOCKER: our tools template uses `| tojson`, which minijinja only provides
    /// under its `json` feature. That feature was OFF, so the first tools render would have
    /// failed with "unknown filter: tojson" — after the model was loaded and the operator was
    /// already serving. This test fails loudly at build time instead.
    #[test]
    fn tools_template_renders_with_tojson() {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("t", QWEN_TOOLS).expect("template parses");
        let tmpl = env.get_template("t").unwrap();
        let out = tmpl
            .render(minijinja::context! {
                messages => vec![minijinja::context! { role => "user", content => "weather in Paris?" }],
                tools => vec![minijinja::context! {
                    r#type => "function",
                    function => minijinja::context! {
                        name => "get_weather",
                        description => "Get weather",
                        parameters => minijinja::context! { r#type => "object" },
                    },
                }],
                add_generation_prompt => true,
            })
            .expect("tools render must succeed — if this fails, tojson/pycompat is missing");
        assert!(out.contains("get_weather"), "tool must reach the prompt: {out}");
        assert!(out.contains("<tool_call>"), "model must be told the call syntax: {out}");
    }

    /// Plain chat must render BYTE-IDENTICALLY whether or not `tools` is in the context.
    /// If it does not, every existing request's prompt_tokens moves and billing shifts under
    /// operators who changed nothing.
    #[test]
    fn absent_tools_renders_identically() {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("t", QWEN_TOOLS).unwrap();
        let tmpl = env.get_template("t").unwrap();
        let msgs = vec![minijinja::context! { role => "user", content => "hello" }];
        let without = tmpl
            .render(minijinja::context! { messages => msgs.clone(), add_generation_prompt => true })
            .unwrap();
        let with_empty = tmpl
            .render(minijinja::context! {
                messages => msgs,
                tools => Vec::<minijinja::Value>::new(),
                add_generation_prompt => true,
            })
            .unwrap();
        assert_eq!(without, with_empty, "an empty tools list must not change the prompt");
    }
}
