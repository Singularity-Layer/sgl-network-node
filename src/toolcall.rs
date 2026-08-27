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

/// The tool names an OpenAI `tools` array offers. Anything the model names outside this set
/// is returned as text rather than as an executable call.
pub fn allowed_names(tools: Option<&serde_json::Value>) -> Vec<String> {
    tools
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .or_else(|| t.get("name"))
                        .and_then(|n| n.as_str())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// One parsed tool call plus the prose that surrounded it.
#[derive(Debug, Default)]
pub struct ParsedOutput {
    /// Text the user should still see (everything that was not a tool call).
    pub text: String,
    /// OpenAI `tool_calls` array, or None when the model produced no valid call.
    pub tool_calls: Option<serde_json::Value>,
}

/// Extract tool calls from a COMPLETE generation (the non-streaming path).
///
/// `allowed` is the set of tool names the caller actually offered. A name outside it is left
/// as plain text rather than surfaced as a call: a hallucinated tool would otherwise be handed
/// to an agent as though the model had really asked for it. Matches the node's existing
/// server-engine extractor so both engines behave identically.
///
/// `id_seed` makes call ids deterministic per request (the node passes the job id), so a
/// retry of the same job cannot collide with a different call id.
pub fn parse_complete(
    fmt: ToolFormat,
    raw: &str,
    allowed: &[String],
    id_seed: &str,
) -> ParsedOutput {
    if !fmt.supports_tools() {
        return ParsedOutput { text: raw.to_string(), tool_calls: None };
    }
    let mut calls: Vec<serde_json::Value> = Vec::new();
    let mut text = String::new();
    let mut rest = raw;

    while !rest.is_empty() {
        let Some((start, body, consumed)) = next_candidate(fmt, rest) else {
            text.push_str(rest);
            break;
        };
        text.push_str(&rest[..start]);
        match to_openai_call(&body, allowed, id_seed, calls.len()) {
            Some(c) => calls.push(c),
            // Not a real call (bad JSON, or a tool we never offered): the model DID produce
            // this text, so the user must still receive it. Dropping it would bill tokens for
            // content never delivered.
            None => text.push_str(&rest[start..start + consumed]),
        }
        rest = &rest[start + consumed..];
    }

    ParsedOutput {
        text,
        tool_calls: (!calls.is_empty()).then(|| serde_json::Value::Array(calls)),
    }
}

/// Locate the next possible call: returns (offset, json body, bytes consumed).
fn next_candidate(fmt: ToolFormat, s: &str) -> Option<(usize, String, usize)> {
    if let Some((open, close)) = markers(fmt) {
        let start = s.find(open)?;
        let after = start + open.len();
        let end = s[after..].find(close)? + after;
        let body = s[after..end].trim().to_string();
        return Some((start, body, (end + close.len()) - start));
    }
    // Bare-JSON formats (Llama 3.x, HOMURA): no wrapper, so find a balanced object.
    let start = s.find('{')?;
    let len = balanced_object_len(&s[start..])?;
    Some((start, s[start..start + len].to_string(), len))
}

/// Length of the balanced `{...}` at the head of `s`, honouring strings and escapes so a brace
/// inside an argument value cannot end the object early.
fn balanced_object_len(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.first() != Some(&b'{') {
        return None;
    }
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for (i, &c) in b.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        match c {
            b'\\' if in_str => esc = true,
            b'"' => in_str = !in_str,
            b'{' if !in_str => depth += 1,
            b'}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None // truncated mid-object (e.g. hit max_tokens) — caller keeps it as text
}

/// Convert one candidate body into the OpenAI call shape, or None if it is not a valid call.
fn to_openai_call(
    body: &str,
    allowed: &[String],
    id_seed: &str,
    index: usize,
) -> Option<serde_json::Value> {
    let v: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    // Mistral wraps calls in an array.
    if let Some(arr) = v.as_array() {
        return arr
            .first()
            .and_then(|f| to_openai_call(&f.to_string(), allowed, id_seed, index));
    }
    // `name` for most families, `tool` for HOMURA's trained protocol.
    let name = v.get("name").or_else(|| v.get("tool"))?.as_str()?;
    if !allowed.iter().any(|a| a == name) {
        return None;
    }
    // `arguments` for most, `parameters` for Llama 3.x.
    let args = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    // OpenAI carries arguments as a STRING of JSON, not an object.
    let args_str = match args.as_str() {
        Some(s) => s.to_string(),
        None => args.to_string(),
    };
    Some(serde_json::json!({
        "id": format!("call_{id_seed}_{index}"),
        "type": "function",
        "function": { "name": name, "arguments": args_str },
    }))
}

/// Cap on withheld text. A model that emits an opening marker and never closes it (or plain
/// prose full of `{`) must not pin memory or swallow the whole answer — past this we give up
/// and flush what we held as ordinary text.
const MAX_HOLD_BYTES: usize = 8 * 1024;

/// What the scanner decided about a piece of generated text.
#[derive(Debug, PartialEq)]
pub enum ScanOut {
    /// Deliver to the user as ordinary content.
    Text(String),
    /// A complete, validated tool call, as one OpenAI streaming delta.
    Call(serde_json::Value),
}

/// Incremental tool-call scanner for the STREAMING path.
///
/// Emits a tool call as ONE complete delta rather than dribbling out partial `arguments`
/// fragments. That is legal OpenAI streaming (clients only require the concatenation to
/// parse) and it makes a malformed fragment impossible by construction: nothing leaves here
/// until the call is complete AND parses AND names a tool the caller offered.
///
/// The cost is that text is WITHHELD while a call might be forming. Withheld text is never
/// discarded — if it turns out not to be a call it is flushed as content, because those
/// tokens were counted at sampling time and billed.
pub struct ToolCallScanner {
    fmt: ToolFormat,
    allowed: Vec<String>,
    id_seed: String,
    /// Text held back because it might be (part of) a call.
    buf: String,
    /// True once an opening marker was seen and we are inside a candidate call.
    in_call: bool,
    /// The opening marker text we consumed to enter the call. Kept so that a candidate which
    /// turns out NOT to be a valid call can be delivered EXACTLY as the model wrote it —
    /// dropping the marker would bill tokens for content the user never received.
    open_marker: String,
    /// Next `index` for the OpenAI streaming tool_calls array (parallel calls increment).
    index: usize,
}

impl ToolCallScanner {
    pub fn new(fmt: ToolFormat, allowed: Vec<String>, id_seed: String) -> Self {
        Self {
            fmt,
            allowed,
            id_seed,
            buf: String::new(),
            in_call: false,
            open_marker: String::new(),
            index: 0,
        }
    }

    /// Feed one decoded piece; get back whatever is now safe to deliver.
    pub fn push(&mut self, piece: &str) -> Vec<ScanOut> {
        if !self.fmt.supports_tools() {
            return vec![ScanOut::Text(piece.to_string())];
        }
        self.buf.push_str(piece);
        self.drain(false)
    }

    /// Generation ended: flush everything still held. NEVER drops text.
    pub fn finish(&mut self) -> Vec<ScanOut> {
        let mut out = self.drain(true);
        if !self.buf.is_empty() || !self.open_marker.is_empty() {
            let held = format!(
                "{}{}",
                std::mem::take(&mut self.open_marker),
                std::mem::take(&mut self.buf)
            );
            out.push(ScanOut::Text(held));
        }
        out
    }

    fn drain(&mut self, final_pass: bool) -> Vec<ScanOut> {
        let mut out = Vec::new();
        loop {
            if self.in_call {
                match self.take_complete_call() {
                    Some(o) => out.push(o),
                    None => break,
                }
            } else {
                match self.take_until_call_start() {
                    Some(o) => out.push(o),
                    None => break,
                }
            }
        }
        // Runaway guard: held too much without completing a call → it was prose after all.
        if !final_pass && self.buf.len() > MAX_HOLD_BYTES {
            self.in_call = false;
            let held = format!(
                "{}{}",
                std::mem::take(&mut self.open_marker),
                std::mem::take(&mut self.buf)
            );
            out.push(ScanOut::Text(held));
        }
        out
    }

    /// Emit text up to the next possible call start; hold back anything that might be one.
    fn take_until_call_start(&mut self) -> Option<ScanOut> {
        let start = match markers(self.fmt) {
            Some((open, _)) => match self.buf.find(open) {
                Some(i) => {
                    self.in_call = true;
                    self.open_marker = open.to_string();
                    let text = self.buf[..i].to_string();
                    self.buf.drain(..i + open.len());
                    return (!text.is_empty()).then_some(ScanOut::Text(text));
                }
                None => {
                    // No marker yet. Emit everything except a tail that could still BECOME
                    // one ("<tool_c" waiting for "all>"), or we would leak half a marker.
                    let keep = partial_marker_tail(&self.buf, open);
                    let cut = self.buf.len() - keep;
                    if cut == 0 {
                        return None;
                    }
                    let text: String = self.buf.drain(..cut).collect();
                    return (!text.is_empty()).then_some(ScanOut::Text(text));
                }
            },
            // Bare JSON: a call starts at a `{`.
            None => self.buf.find('{'),
        };
        match start {
            Some(0) => {
                self.in_call = true;
                None
            }
            Some(i) => {
                let text: String = self.buf.drain(..i).collect();
                self.in_call = true;
                (!text.is_empty()).then_some(ScanOut::Text(text))
            }
            None => {
                let text = std::mem::take(&mut self.buf);
                (!text.is_empty()).then_some(ScanOut::Text(text))
            }
        }
    }

    /// Inside a candidate: emit it once it is complete, as a Call if valid else as Text.
    fn take_complete_call(&mut self) -> Option<ScanOut> {
        let (body, consumed) = match markers(self.fmt) {
            Some((_, close)) => {
                let end = self.buf.find(close)?;
                (self.buf[..end].trim().to_string(), end + close.len())
            }
            None => {
                let len = balanced_object_len(&self.buf)?;
                (self.buf[..len].to_string(), len)
            }
        };
        let raw: String = self.buf.drain(..consumed).collect();
        self.in_call = false;
        match to_openai_call(&body, &self.allowed, &self.id_seed, self.index) {
            Some(mut c) => {
                self.open_marker.clear();
                // Streaming deltas carry an `index`; parallel calls must not collide.
                c["index"] = serde_json::json!(self.index);
                self.index += 1;
                Some(ScanOut::Call(serde_json::Value::Array(vec![c])))
            }
            // Not a real call — the model produced it, so the user still gets it, INCLUDING
            // the opening marker we consumed to get here.
            None => Some(ScanOut::Text(format!("{}{raw}", std::mem::take(&mut self.open_marker)))),
        }
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `marker`.
fn partial_marker_tail(s: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(s.len());
    (1..=max)
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && marker.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
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


    fn allowed() -> Vec<String> { vec!["get_weather".into(), "search".into()] }

    fn call_name(o: &ParsedOutput, i: usize) -> String {
        o.tool_calls.as_ref().unwrap()[i]["function"]["name"].as_str().unwrap().into()
    }
    fn call_args(o: &ParsedOutput, i: usize) -> String {
        o.tool_calls.as_ref().unwrap()[i]["function"]["arguments"].as_str().unwrap().into()
    }

    #[test]
    fn hermes_call_is_extracted_and_prose_kept() {
        let raw = "Let me check. <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let o = parse_complete(ToolFormat::HermesXml, raw, &allowed(), "job1");
        assert_eq!(call_name(&o, 0), "get_weather");
        assert_eq!(call_args(&o, 0), r#"{"city":"Paris"}"#);
        assert_eq!(o.text, "Let me check. ");
    }

    /// OpenAI carries arguments as a STRING of JSON, not an object. An agent that gets an
    /// object here fails to parse the call.
    #[test]
    fn arguments_are_a_json_string() {
        let raw = "<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"a\"}}</tool_call>";
        let o = parse_complete(ToolFormat::HermesXml, raw, &allowed(), "j");
        let a = &o.tool_calls.as_ref().unwrap()[0]["function"]["arguments"];
        assert!(a.is_string(), "arguments must be a string, got {a}");
    }

    #[test]
    fn homura_uses_tool_key_and_llama_uses_parameters() {
        let h = parse_complete(ToolFormat::Homura,
            r#"{"tool":"search","arguments":{"q":"x"}}"#, &allowed(), "j");
        assert_eq!(call_name(&h, 0), "search");
        let l = parse_complete(ToolFormat::Llama3Json,
            r#"{"name":"search","parameters":{"q":"x"}}"#, &allowed(), "j");
        assert_eq!(call_args(&l, 0), r#"{"q":"x"}"#);
    }

    /// A tool we never offered must NOT reach the agent as a call — but the text the model
    /// produced must still be delivered, because those tokens were billed.
    #[test]
    fn hallucinated_tool_stays_as_text() {
        let raw = "<tool_call>{\"name\":\"rm_rf\",\"arguments\":{}}</tool_call>";
        let o = parse_complete(ToolFormat::HermesXml, raw, &allowed(), "j");
        assert!(o.tool_calls.is_none(), "must not surface an unoffered tool");
        assert!(o.text.contains("rm_rf"), "text must still be delivered: {:?}", o.text);
    }

    /// Truncated at max_tokens: unparseable, so it is prose. Never a partial call.
    #[test]
    fn truncated_call_is_delivered_as_text() {
        let raw = "<tool_call>{\"name\":\"get_weather\",\"argum";
        let o = parse_complete(ToolFormat::HermesXml, raw, &allowed(), "j");
        assert!(o.tool_calls.is_none());
        assert_eq!(o.text, raw, "withheld text must never be dropped");
    }

    /// THE BILLING INVARIANT (Fable + Codex both flagged it): every byte the model produced
    /// must come back either as a call or as text. Anything dropped is billed-but-undelivered.
    #[test]
    fn nothing_is_ever_dropped() {
        for raw in [
            "plain prose only",
            "<tool_call>{bad json}</tool_call>",
            "before <tool_call>{\"name\":\"nope\",\"arguments\":{}}</tool_call> after",
            "<tool_call>",
        ] {
            let o = parse_complete(ToolFormat::HermesXml, raw, &allowed(), "j");
            if o.tool_calls.is_none() {
                assert_eq!(o.text, raw, "no call parsed, so all text must survive: {raw:?}");
            }
        }
    }

    /// A brace inside an argument STRING must not end the object early.
    #[test]
    fn braces_inside_strings_do_not_break_parsing() {
        let raw = r#"{"tool":"search","arguments":{"q":"a } b {"}}"#;
        let o = parse_complete(ToolFormat::Homura, raw, &allowed(), "j");
        assert_eq!(call_name(&o, 0), "search");
        assert_eq!(call_args(&o, 0), r#"{"q":"a } b {"}"#);
    }

    #[test]
    fn parallel_calls_get_distinct_ids() {
        let raw = "<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>\
<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
        let o = parse_complete(ToolFormat::HermesXml, raw, &allowed(), "j");
        let c = o.tool_calls.as_ref().unwrap();
        assert_eq!(c.as_array().unwrap().len(), 2);
        assert_ne!(c[0]["id"], c[1]["id"], "ids must be distinct");
    }

    /// A model with no tools support must pass output through untouched — the engine refuses
    /// such requests earlier, so this must never invent a call.
    #[test]
    fn unsupported_format_passes_through() {
        let raw = r#"{"name":"get_weather","arguments":{}}"#;
        let o = parse_complete(ToolFormat::None, raw, &allowed(), "j");
        assert!(o.tool_calls.is_none());
        assert_eq!(o.text, raw);
    }


    fn scan_all(fmt: ToolFormat, pieces: &[&str]) -> Vec<ScanOut> {
        let mut sc = ToolCallScanner::new(fmt, allowed(), "j".into());
        let mut out = Vec::new();
        for p in pieces { out.extend(sc.push(p)); }
        out.extend(sc.finish());
        out
    }
    fn joined_text(o: &[ScanOut]) -> String {
        o.iter().filter_map(|x| match x { ScanOut::Text(t) => Some(t.as_str()), _ => None })
         .collect::<Vec<_>>().join("")
    }
    fn calls(o: &[ScanOut]) -> Vec<&serde_json::Value> {
        o.iter().filter_map(|x| match x { ScanOut::Call(c) => Some(c), _ => None }).collect()
    }

    /// THE REAL CONDITION: the model arrives one token at a time, so a marker is split across
    /// pushes. A scanner that emits eagerly would leak "<tool_c" to the user as content.
    #[test]
    fn call_split_across_many_pushes() {
        let whole = "Sure. <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let pieces: Vec<String> = whole.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = pieces.iter().map(|s| s.as_str()).collect();
        let out = scan_all(ToolFormat::HermesXml, &refs);
        assert_eq!(joined_text(&out), "Sure. ", "no marker fragment may leak as text");
        let c = calls(&out);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0][0]["function"]["name"], "get_weather");
        assert_eq!(c[0][0]["function"]["arguments"], r#"{"city":"Paris"}"#);
    }

    #[test]
    fn plain_text_streams_straight_through() {
        let out = scan_all(ToolFormat::HermesXml, &["Hello ", "there ", "friend"]);
        assert!(calls(&out).is_empty());
        assert_eq!(joined_text(&out), "Hello there friend");
    }

    /// A model with no tool support must never withhold anything.
    #[test]
    fn unsupported_format_never_withholds() {
        let out = scan_all(ToolFormat::None, &["{\"name\":", "\"x\"}"]);
        assert_eq!(joined_text(&out), r#"{"name":"x"}"#);
    }

    /// BILLING INVARIANT under streaming: every byte in must come out, as call or as text.
    #[test]
    fn streaming_never_drops_text() {
        for pieces in [
            vec!["<tool_call>", "{bad}", "</tool_call>"],
            vec!["<tool_call>", "{\"name\":\"nope\",\"arguments\":{}}", "</tool_call>"],
            vec!["<tool_call>", "{\"name\":\"get_weather\""],   // never closed
            vec!["prose ", "<tool_c"],                              // partial marker at EOF
        ] {
            let joined: String = pieces.concat();
            let out = scan_all(ToolFormat::HermesXml, &pieces);
            if calls(&out).is_empty() {
                assert_eq!(joined_text(&out), joined, "must survive: {joined:?}");
            }
        }
    }

    #[test]
    fn parallel_calls_get_incrementing_index() {
        let out = scan_all(ToolFormat::HermesXml, &[
            "<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>",
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>",
        ]);
        let c = calls(&out);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0][0]["index"], 0);
        assert_eq!(c[1][0]["index"], 1);
    }

    /// Bare-JSON families (HOMURA/Llama) have no wrapper, so the scanner balances braces.
    #[test]
    fn bare_json_call_streams() {
        let out = scan_all(ToolFormat::Homura, &["{\"tool\":\"sea", "rch\",\"arguments\":{\"q\":\"a\"}}"]);
        let c = calls(&out);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0][0]["function"]["name"], "search");
    }

    /// An unterminated marker must not swallow the answer forever.
    #[test]
    fn runaway_hold_is_flushed_as_text() {
        let big = "x".repeat(MAX_HOLD_BYTES + 64);
        let out = scan_all(ToolFormat::HermesXml, &["<tool_call>", &big]);
        assert!(calls(&out).is_empty());
        assert!(joined_text(&out).contains(&big), "held text must be delivered, not dropped");
    }

    /// Text BEFORE a call must reach the user before the call does.
    #[test]
    fn text_is_ordered_before_the_call() {
        let out = scan_all(ToolFormat::HermesXml,
            &["thinking... <tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>"]);
        match &out[0] { ScanOut::Text(t) => assert_eq!(t, "thinking... "), o => panic!("{o:?}") }
        assert!(matches!(&out[1], ScanOut::Call(_)));
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
