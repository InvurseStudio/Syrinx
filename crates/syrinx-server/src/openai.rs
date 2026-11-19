//! The OpenAI-compatible API (`/v1`), so other apps and scripts can use the loaded model
//! as they would OpenAI, LM Studio or llama.cpp's server: `GET /v1/models` and
//! `POST /v1/chat/completions`, whole or streamed. Who may use it is `developer.rs`'s.
//!
//! A request is a plain chat: no tools, skills or memories, and only the system prompt the
//! caller sends. Sampling settings (`temperature`, `top_p`, `top_k`, `min_p`,
//! `presence_penalty`) pass through, `max_tokens` (or `max_completion_tokens`) bounds the
//! reply, and up to four `stop` strings end it. A model that thinks sends its thoughts as
//! `reasoning_content`, apart from the answer, when the request turns thinking on
//! (`reasoning_effort` low, medium or high; or `"thinking": true`, or
//! `chat_template_kwargs.enable_thinking`).
//!
//! Not yet: tools in the request, images, `response_format`, more than one choice, and
//! `/v1/embeddings` and `/v1/responses`, which answer so in OpenAI's error shape.

use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use crate::chat::Message;
use crate::developer;
use crate::http;
use crate::json::{self, Value};
use crate::stats;
use crate::{Job, Out, Shared, Work};

/// The most `stop` strings, as OpenAI takes.
const MAX_STOPS: usize = 4;

/// An error in OpenAI's shape, which its client libraries know how to show.
pub fn error_json(message: &str, kind: &str, code: &str) -> String {
    format!(
        "{{\"error\":{{\"message\":{},\"type\":{},\"param\":null,\"code\":{}}}}}",
        json::string(message),
        json::string(kind),
        json::string(code)
    )
}

fn refuse(stream: &mut TcpStream, status: u16, code: &str, message: &str) {
    let kind = match status {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        500 | 501 => "server_error",
        _ => "invalid_request_error",
    };
    http::respond(stream, status, &error_json(message, kind, code));
}

fn model_object(id: &str) -> String {
    format!(
        "{{\"id\":{},\"object\":\"model\",\"created\":{},\"owned_by\":\"syrinx\"}}",
        json::string(id),
        unix_now()
    )
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Serves a request under `/v1`.
pub fn serve(mut stream: TcpStream, shared: &Shared, request: &http::Request, path: &str) {
    let settings = developer::load();
    let key = match developer::authorize(&settings, request.bearer.as_deref()) {
        Ok(key) => key,
        Err((status, code, why)) => {
            stats::refuse(&crate::peer(&stream), "OpenAI-compatible API refused");
            return refuse(&mut stream, status, code, &why);
        }
    };
    if let Some(k) = &key {
        developer::touch(&k.id);
    }
    let model = shared.info.model.as_str();
    match (request.method.as_str(), path) {
        ("GET", "/v1/models") => http::respond(
            &mut stream,
            200,
            &format!("{{\"object\":\"list\",\"data\":[{}]}}", model_object(model)),
        ),
        ("GET", p) if p.starts_with("/v1/models/") => {
            let asked = &p["/v1/models/".len()..];
            if asked == model {
                http::respond(&mut stream, 200, &model_object(model));
            } else {
                refuse(
                    &mut stream,
                    404,
                    "model_not_found",
                    &format!("The model '{asked}' isn't loaded. This server runs '{model}'."),
                );
            }
        }
        ("POST", "/v1/chat/completions") => {
            let source = stats::Source {
                id: key.as_ref().map_or_else(|| "api:local".to_string(), |k| format!("api:{}", k.id)),
                name: key.as_ref().map_or_else(|| "OpenAI-compatible API".to_string(), |k| format!("API key {}", k.name)),
                kind: "api",
            };
            chat_completion(stream, shared, &request.body, source);
        }
        ("POST", "/v1/embeddings") => refuse(
            &mut stream,
            501,
            "not_supported",
            "Embeddings aren't available yet: Syrinx has no embedding model.",
        ),
        (_, p) if p.starts_with("/v1/responses") => refuse(
            &mut stream,
            501,
            "not_supported",
            "The Responses API isn't available yet. Use /v1/chat/completions.",
        ),
        _ => refuse(
            &mut stream,
            404,
            "unknown_url",
            &format!("Unknown request: {} {path}", request.method),
        ),
    }
}

/// A chat completion request, read and checked.
#[derive(Debug, PartialEq)]
pub struct Asked {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    /// The sampling settings, as the engine reads a `generation` object.
    pub generation: Value,
    pub max_tokens: Option<usize>,
    pub stops: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    /// Whether to think first; `None` for the server's default.
    pub thinking: Option<bool>,
}

/// A message's text: a string, or a list of parts whose text parts are joined. Anything
/// else (an image) is refused.
fn text_of(content: Option<&Value>) -> Result<String, String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.str("type") {
                    Some("text") | Some("input_text") => text.push_str(part.str("text").unwrap_or("")),
                    Some(other) => {
                        return Err(format!(
                            "Message parts of type '{other}' aren't supported yet: only text."
                        ));
                    }
                    None => return Err("Each message part needs a type.".into()),
                }
            }
            Ok(text)
        }
        Some(_) => Err("A message's content must be text or a list of parts.".into()),
    }
}

/// Reads a chat completion request; or why it can't be served, for a 400.
pub fn read(body: &Value) -> Result<Asked, String> {
    let list = match body.get("messages") {
        Some(Value::Array(list)) if !list.is_empty() => list,
        _ => return Err("'messages' must be a non-empty list.".into()),
    };
    let wants_tools = match body.get("tools") {
        Some(Value::Array(t)) => !t.is_empty(),
        _ => false,
    } || matches!(body.get("functions"), Some(Value::Array(f)) if !f.is_empty());
    if wants_tools && body.str("tool_choice") != Some("none") {
        return Err("Tool calling through /v1 isn't supported yet.".into());
    }
    if let Some(format) = body.get("response_format")
        && format.str("type").is_some_and(|t| t != "text")
    {
        return Err("response_format isn't supported yet: only plain text.".into());
    }
    if body.num("n").is_some_and(|n| n != 1.0) {
        return Err("Only one choice (n = 1) is supported.".into());
    }
    let mut system: Vec<String> = Vec::new();
    let mut messages = Vec::new();
    for m in list {
        let role = m.str("role").ok_or("Each message needs a role.")?;
        let text = text_of(m.get("content"))?;
        match role {
            "system" | "developer" => system.push(text),
            "user" => messages.push(Message::new("user", crate::chat::plain(&text))),
            "assistant" => {
                if !m.array("tool_calls").is_empty() || m.get("function_call").is_some() {
                    return Err("Tool calls in the chat aren't supported yet.".into());
                }
                messages.push(Message::new(
                    "assistant",
                    crate::chat::without_thinking(&crate::chat::plain(&text)),
                ));
            }
            "tool" | "function" => return Err("Tool results in the chat aren't supported yet.".into()),
            other => return Err(format!("Unknown message role '{other}'.")),
        }
    }
    if messages.last().is_none_or(|m| m.role != "user") {
        return Err("The chat must end with a user message.".into());
    }
    let mut generation = std::collections::BTreeMap::new();
    for field in ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"] {
        match body.get(field) {
            None | Some(Value::Null) => {}
            Some(Value::Number(n)) => {
                generation.insert(field.to_string(), Value::Number(*n));
            }
            Some(_) => return Err(format!("'{field}' must be a number.")),
        }
    }
    let max_tokens = match body.get("max_completion_tokens").or_else(|| body.get("max_tokens")) {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) if *n >= 1.0 => Some(*n as usize),
        Some(_) => return Err("'max_tokens' must be a whole number of at least 1.".into()),
    };
    let stops: Vec<String> = match body.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err("'stop' must be a string or a list of strings.".to_string()),
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("'stop' must be a string or a list of strings.".into()),
    };
    if stops.len() > MAX_STOPS {
        return Err(format!("At most {MAX_STOPS} stop strings."));
    }
    let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
    let thinking = match (body.get("thinking"), body.str("reasoning_effort")) {
        (Some(Value::Bool(b)), _) => Some(*b),
        (_, Some("low" | "medium" | "high")) => Some(true),
        (_, Some("none" | "minimal")) => Some(false),
        _ => match body.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")) {
            Some(Value::Bool(b)) => Some(*b),
            _ => None,
        },
    };
    let system = (!system.is_empty()).then(|| system.join("\n\n"));
    Ok(Asked {
        system,
        messages,
        generation: Value::Object(generation),
        max_tokens,
        stops,
        stream: matches!(body.get("stream"), Some(Value::Bool(true))),
        include_usage: matches!(
            body.get("stream_options").and_then(|o| o.get("include_usage")),
            Some(Value::Bool(true))
        ),
        thinking,
    })
}

/// The answer's text, ended at the first stop string. Text that could be the start of
/// one is held back until it's clear it isn't.
#[derive(Debug, Default)]
pub struct Stopper {
    stops: Vec<String>,
    held: String,
    pub stopped: bool,
}

impl Stopper {
    pub fn new(stops: Vec<String>) -> Self {
        Stopper { stops, ..Default::default() }
    }

    /// What of `text` may be sent now.
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.held.push_str(text);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            self.stopped = true;
            let out = self.held[..at].to_string();
            self.held.clear();
            return out;
        }
        // Keep back the longest end of what's held that some stop string starts with.
        let keep = self
            .stops
            .iter()
            .flat_map(|s| {
                (1..s.len())
                    .filter(|&n| s.is_char_boundary(n))
                    .filter(|&n| self.held.ends_with(&s[..n]))
                    .max()
            })
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        out
    }

    /// What's held back, at the end.
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }
}

/// How a reply ended, as OpenAI says it.
fn finish_reason(stopped: bool, tokens: usize, limit: usize) -> &'static str {
    if !stopped && tokens >= limit { "length" } else { "stop" }
}

fn usage_json(usage: Option<&stats::Usage>, completion: usize) -> String {
    let (prompt, cached) = usage.map_or((0, 0), |u| (u.prompt_tokens, u.cached_tokens));
    format!(
        "{{\"prompt_tokens\":{prompt},\"completion_tokens\":{completion},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{cached}}}}}",
        prompt + completion
    )
}

fn chat_completion(mut stream: TcpStream, shared: &Shared, body: &[u8], source: stats::Source) {
    let body = match json::parse(&String::from_utf8_lossy(body)) {
        Ok(v) => v,
        Err(_) => return refuse(&mut stream, 400, "invalid_json", "The request body must be JSON."),
    };
    let asked = match read(&body) {
        Ok(a) => a,
        Err(why) => return refuse(&mut stream, 400, "invalid_request", &why),
    };
    // The engine writes at most half the context; with no limit, as much as that.
    let limit = asked
        .max_tokens
        .unwrap_or(usize::MAX)
        .min(shared.info.context / 2);
    let stop = Arc::new(AtomicBool::new(false));
    let (out, events) = channel();
    let job = Job {
        system: asked.system.clone(),
        messages: asked.messages.clone(),
        generation: Some(asked.generation.clone()),
        max_tokens: limit,
        stop: stop.clone(),
        out,
        client_tools: Vec::new(),
        files: None,
        access: String::new(),
        folders: Vec::new(),
        chat_id: None,
        thinking: asked.thinking.unwrap_or(crate::options().thinking),
        no_tools: true,
        api: true,
        tool_results: shared.tool_results.clone(),
        approvals: shared.approvals.clone(),
    };
    let counted = stats::begin(source);
    let started = Instant::now();
    if shared.jobs.lock().unwrap().send(Work::Reply(job)).is_err() {
        stats::finish(counted, stats::Outcome { status: "error", ..Default::default() });
        return refuse(&mut stream, 500, "engine_stopped", "The engine has stopped.");
    }
    let id = format!("chatcmpl-{:016x}{:08x}", crate::random_u64(), crate::random_u64() as u32);
    let model = shared.info.model.clone();
    let reply = Reply {
        id,
        model,
        created: unix_now(),
        limit,
        stopper: Stopper::new(asked.stops.clone()),
    };
    let outcome = if asked.stream {
        reply.stream(&mut stream, &events, &stop, asked.include_usage, started)
    } else {
        reply.whole(&mut stream, &events, &stop, started)
    };
    stats::finish(counted, outcome);
}

struct Reply {
    id: String,
    model: String,
    created: u64,
    limit: usize,
    stopper: Stopper,
}

/// What an engine event adds: thought text, or nothing to send (the answer comes as
/// `Out::Text`).
enum Piece {
    Thought(String),
    Nothing,
}

fn piece(event: &str) -> Piece {
    let Ok(v) = json::parse(event) else { return Piece::Nothing };
    match (v.str("type"), v.str("content")) {
        (Some("thinking"), Some(text)) => Piece::Thought(text.to_string()),
        _ => Piece::Nothing,
    }
}

impl Reply {
    fn chunk(&self, delta: &str, finish: Option<&str>) -> String {
        format!(
            "{{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"system_fingerprint\":null,\"choices\":[{{\"index\":0,\"delta\":{delta},\"logprobs\":null,\"finish_reason\":{}}}]}}",
            json::string(&self.id),
            self.created,
            json::string(&self.model),
            finish.map_or("null".to_string(), json::string)
        )
    }

    /// The reply as server-sent events, as OpenAI streams one.
    fn stream(
        mut self,
        stream: &mut TcpStream,
        events: &Receiver<Out>,
        stop: &AtomicBool,
        include_usage: bool,
        started: Instant,
    ) -> stats::Outcome {
        let mut outcome = stats::Outcome::default();
        if http::start_events(stream).is_err() {
            stop.store(true, Ordering::Relaxed);
            outcome.status = "cancelled";
            return outcome;
        }
        // The client has gone: no one to write for.
        let gone = std::cell::Cell::new(false);
        let mut send = |data: &str, stop: &AtomicBool| {
            if !gone.get() && http::event(stream, data).is_err() {
                gone.set(true);
                stop.store(true, Ordering::Relaxed);
            }
        };
        send(&self.chunk("{\"role\":\"assistant\",\"content\":\"\"}", None), stop);
        let mut usage = None;
        let mut tokens = 0;
        let mut failed = None;
        for event in events {
            if outcome.first_token_seconds.is_none() && matches!(event, Out::Text(_) | Out::Event(_)) {
                outcome.first_token_seconds = Some(started.elapsed().as_secs_f64());
            }
            match event {
                Out::Text(text) => {
                    let sendable = self.stopper.push(&text);
                    if !sendable.is_empty() {
                        send(&self.chunk(&format!("{{\"content\":{}}}", json::string(&sendable)), None), stop);
                    }
                    if self.stopper.stopped {
                        stop.store(true, Ordering::Relaxed);
                    }
                }
                Out::Event(e) => {
                    if let Piece::Thought(text) = piece(&e)
                        && !self.stopper.stopped
                    {
                        send(
                            &self.chunk(&format!("{{\"reasoning_content\":{}}}", json::string(&text)), None),
                            stop,
                        );
                    }
                }
                Out::Usage(u) => usage = Some(u),
                Out::Done { tokens: n, seconds } => {
                    tokens = n;
                    outcome.completion_tokens = n;
                    outcome.generation_seconds = Some(seconds);
                }
                Out::Error(e) => failed = Some(e),
            }
        }
        let rest = self.stopper.finish();
        if !rest.is_empty() {
            send(&self.chunk(&format!("{{\"content\":{}}}", json::string(&rest)), None), stop);
        }
        if let Some(e) = &failed {
            send(&error_json(e, "server_error", "engine_error"), stop);
            outcome.status = "error";
        } else {
            let reason = finish_reason(self.stopper.stopped, tokens, self.limit);
            send(&self.chunk("{}", Some(reason)), stop);
            if include_usage {
                send(
                    &format!(
                        "{{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"choices\":[],\"usage\":{}}}",
                        json::string(&self.id),
                        self.created,
                        json::string(&self.model),
                        usage_json(usage.as_ref(), tokens)
                    ),
                    stop,
                );
            }
            outcome.status = if gone.get() { "cancelled" } else { "ok" };
        }
        send("[DONE]", stop);
        outcome.usage = usage;
        outcome
    }

    /// The whole reply as one `chat.completion`.
    fn whole(mut self, stream: &mut TcpStream, events: &Receiver<Out>, stop: &AtomicBool, started: Instant) -> stats::Outcome {
        let mut outcome = stats::Outcome::default();
        let mut answer = String::new();
        let mut thought = String::new();
        let mut usage = None;
        let mut tokens = 0;
        for event in events {
            if outcome.first_token_seconds.is_none() && matches!(event, Out::Text(_) | Out::Event(_)) {
                outcome.first_token_seconds = Some(started.elapsed().as_secs_f64());
            }
            match event {
                Out::Text(text) => {
                    answer.push_str(&self.stopper.push(&text));
                    if self.stopper.stopped {
                        stop.store(true, Ordering::Relaxed);
                    }
                }
                Out::Event(e) => {
                    if let Piece::Thought(text) = piece(&e) {
                        thought.push_str(&text);
                    }
                }
                Out::Usage(u) => usage = Some(u),
                Out::Done { tokens: n, seconds } => {
                    tokens = n;
                    outcome.completion_tokens = n;
                    outcome.generation_seconds = Some(seconds);
                }
                Out::Error(e) => {
                    outcome.status = "error";
                    refuse(stream, 500, "engine_error", &e);
                    return outcome;
                }
            }
        }
        answer.push_str(&self.stopper.finish());
        let reasoning = if thought.is_empty() {
            String::new()
        } else {
            format!(",\"reasoning_content\":{}", json::string(thought.trim()))
        };
        let reason = finish_reason(self.stopper.stopped, tokens, self.limit);
        http::respond(
            stream,
            200,
            &format!(
                "{{\"id\":{},\"object\":\"chat.completion\",\"created\":{},\"model\":{},\"system_fingerprint\":null,\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}{reasoning}}},\"logprobs\":null,\"finish_reason\":{}}}],\"usage\":{}}}",
                json::string(&self.id),
                self.created,
                json::string(&self.model),
                json::string(&answer),
                json::string(reason),
                usage_json(usage.as_ref(), tokens)
            ),
        );
        outcome.status = "ok";
        outcome.usage = usage;
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asked(text: &str) -> Result<Asked, String> {
        read(&json::parse(text).unwrap())
    }

    #[test]
    fn a_request_reads_as_a_plain_chat() {
        let a = asked(
            r#"{"model":"x","messages":[
                {"role":"system","content":"Be brief."},
                {"role":"developer","content":"Use British spelling."},
                {"role":"user","content":[{"type":"text","text":"Hi"},{"type":"text","text":" there"}]},
                {"role":"assistant","content":"Hello."},
                {"role":"user","content":"Colour of the sky?"}],
              "temperature":0.2,"top_p":0.9,"max_tokens":50,"stop":"\n\n","stream":true,
              "stream_options":{"include_usage":true},"reasoning_effort":"high","seed":7}"#,
        )
        .unwrap();
        assert_eq!(a.system.as_deref(), Some("Be brief.\n\nUse British spelling."));
        assert_eq!(a.messages.len(), 3);
        assert_eq!(a.messages[0].content, "Hi there");
        assert_eq!(a.messages[2].role, "user");
        assert_eq!(a.generation.num("temperature"), Some(0.2));
        assert_eq!(a.generation.num("top_p"), Some(0.9));
        assert_eq!(a.max_tokens, Some(50));
        assert_eq!(a.stops, vec!["\n\n".to_string()]);
        assert!(a.stream && a.include_usage);
        assert_eq!(a.thinking, Some(true));
        let b = asked(r#"{"messages":[{"role":"user","content":"x"}],"max_completion_tokens":9,"chat_template_kwargs":{"enable_thinking":false}}"#).unwrap();
        assert_eq!((b.max_tokens, b.thinking, b.stream), (Some(9), Some(false), false));
        assert!(b.stops.is_empty() && b.system.is_none());
        // tool_choice "none" with tools listed is a plain chat.
        assert!(asked(r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}],"tool_choice":"none"}"#).is_ok());
    }

    #[test]
    fn what_isnt_supported_yet_is_refused_plainly() {
        for (body, says) in [
            (r#"{"messages":[]}"#, "non-empty"),
            (r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]}"#, "Tool calling"),
            (r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#, "image_url"),
            (r#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}"#, "response_format"),
            (r#"{"messages":[{"role":"user","content":"x"}],"n":2}"#, "one choice"),
            (r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}"#, "end with a user"),
            (r#"{"messages":[{"role":"tool","content":"x"}]}"#, "Tool results"),
            (r#"{"messages":[{"role":"user","content":"x"}],"stop":["a","b","c","d","e"]}"#, "At most 4"),
            (r#"{"messages":[{"role":"user","content":"x"}],"max_tokens":0}"#, "at least 1"),
            (r#"{"messages":[{"role":"user","content":"x"}],"temperature":"hot"}"#, "number"),
        ] {
            let why = asked(body).unwrap_err();
            assert!(why.contains(says), "{body}: {why}");
        }
    }

    #[test]
    fn a_stop_string_ends_the_answer_even_split_across_pieces() {
        let mut s = Stopper::new(vec!["END".into(), "\n\n".into()]);
        let mut out = String::new();
        for piece in ["Hello wor", "ld E", "N", "D and more"] {
            out.push_str(&s.push(piece));
        }
        assert!(s.stopped);
        assert_eq!(out, "Hello world ");
        assert_eq!(s.finish(), "");
        // Something that only looked like a stop's start is sent at the end.
        let mut s = Stopper::new(vec!["END".into()]);
        assert_eq!(s.push("THE EN"), "THE ");
        assert_eq!(s.push("TRY"), "ENTRY");
        assert_eq!(s.push(" E"), " ");
        assert_eq!(s.finish(), "E");
        // No stops: everything goes at once.
        let mut s = Stopper::new(Vec::new());
        assert_eq!(s.push("abc"), "abc");
        // A multi-byte character in a stop string.
        let mut s = Stopper::new(vec!["é!".into()]);
        assert_eq!(s.push("café"), "caf");
        assert_eq!(s.push("!"), "");
        assert!(s.stopped);
    }

    #[test]
    fn errors_and_reasons_are_in_openais_shape() {
        let e = json::parse(&error_json("No.", "invalid_request_error", "bad")).unwrap();
        let inner = e.get("error").unwrap();
        assert_eq!(inner.str("message"), Some("No."));
        assert_eq!(inner.str("code"), Some("bad"));
        assert_eq!(finish_reason(false, 10, 10), "length");
        assert_eq!(finish_reason(true, 10, 10), "stop");
        assert_eq!(finish_reason(false, 3, 10), "stop");
        let u = json::parse(&usage_json(None, 5)).unwrap();
        assert_eq!(u.num("total_tokens"), Some(5.0));
    }
}
