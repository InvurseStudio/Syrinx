//! `vireo-cli serve`: keeps one model loaded and answers chats over HTTP on this PC, for
//! the Windows app. It speaks the part of the Vireo server's API the app's chat uses
//! (`/api/chat/completions` streamed as server-sent events, `/api/metrics`, stop), so the
//! app talks to it as it talks to a Mac. No agent or tools yet: that's the real server
//! (PLAN.md, phase 3). Only 127.0.0.1 is served, and every request must carry the token
//! the app started it with.
//!
//! One thread runs the model; each connection gets a thread that reads the request and
//! hands chats to the model thread, one at a time, in order.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vireo_engine::{GgufFile, Tokenizer};

use crate::Engine;
use crate::json::{self, Value};

pub struct ServeOptions {
    pub port: u16,
    pub token: Option<String>,
    pub gpu: bool,
    pub threads: Option<usize>,
    pub context: usize,
    pub sampling: Sampling,
    pub thinking: bool,
    /// The most tokens one reply may spend thinking before it's closed; None for no cap.
    pub thinking_budget: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    /// Taken off every token the reply has already used, as vLLM's presence_penalty:
    /// it keeps small models from going round in circles.
    pub presence_penalty: f32,
}

impl ServeOptions {
    pub fn parse(args: &[&str]) -> Option<Self> {
        let mut o = Self {
            port: 39_281,
            token: None,
            gpu: false,
            threads: None,
            context: 16_384,
            sampling: Sampling {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 0,
                min_p: 0.0,
                presence_penalty: 0.0,
            },
            thinking: true,
            thinking_budget: None,
        };
        let mut args = args.iter();
        while let Some(&arg) = args.next() {
            let mut value = || args.next().copied();
            match arg {
                "--port" => o.port = value()?.parse().ok()?,
                "--token" => o.token = Some(value()?.to_string()),
                "--gpu" => o.gpu = true,
                "--threads" => o.threads = Some(value()?.parse().ok()?).filter(|&n| n > 0),
                "--context" => o.context = value()?.parse().ok()?,
                "--temperature" => o.sampling.temperature = value()?.parse().ok()?,
                "--top-p" => o.sampling.top_p = value()?.parse().ok()?,
                "--top-k" => o.sampling.top_k = value()?.parse().ok()?,
                "--min-p" => o.sampling.min_p = value()?.parse().ok()?,
                "--presence-penalty" => o.sampling.presence_penalty = value()?.parse().ok()?,
                "--no-thinking" => o.thinking = false,
                "--thinking-budget" => {
                    o.thinking_budget = Some(value()?.parse().ok()?).filter(|&n: &usize| n > 0)
                }
                _ => return None,
            }
        }
        Some(o)
    }
}

/// What `/api/engine/status` reports, for the app's Start/Stop controls.
#[derive(Default)]
struct Status {
    state: &'static str,
    model: String,
    device: String,
    error: Option<String>,
    context: usize,
    last_tokens_per_second: Option<f64>,
    busy: bool,
}

/// One chat for the model thread. Its events go back over `events`.
struct Job {
    run_id: String,
    messages: Vec<(String, String)>,
    system: Option<String>,
    sampling: Sampling,
    max_tokens: Option<usize>,
    events: Sender<String>,
    stop: Arc<AtomicBool>,
}

struct Shared {
    token: Option<String>,
    status: Mutex<Status>,
    jobs: Mutex<Sender<Job>>,
    /// The reply being written, and how to stop it.
    current: Mutex<Option<(String, Arc<AtomicBool>)>>,
    next_run: AtomicU64,
    defaults: Sampling,
}

pub fn serve(path: &str, options: ServeOptions) -> std::process::ExitCode {
    let listener = match TcpListener::bind(("127.0.0.1", options.port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Couldn't listen on 127.0.0.1:{}: {e}", options.port);
            return std::process::ExitCode::FAILURE;
        }
    };
    let model_name = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let (jobs, queue) = mpsc::channel::<Job>();
    let shared = Arc::new(Shared {
        token: options.token.clone(),
        status: Mutex::new(Status {
            state: "loading",
            model: model_name,
            context: options.context,
            ..Status::default()
        }),
        jobs: Mutex::new(jobs),
        current: Mutex::new(None),
        next_run: AtomicU64::new(1),
        defaults: options.sampling,
    });
    // The app waits for this line before it asks anything.
    println!("Listening on http://127.0.0.1:{}", options.port);
    let _ = std::io::stdout().flush();

    let accepting = Arc::clone(&shared);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let shared = Arc::clone(&accepting);
            std::thread::spawn(move || {
                if let Err(e) = handle(stream, &shared) {
                    eprintln!("Request failed: {e}");
                }
            });
        }
    });

    run_model(path, &options, &shared, queue)
}

/// Loads the model, then answers chats until the process is told to stop.
fn run_model(
    path: &str,
    options: &ServeOptions,
    shared: &Shared,
    queue: Receiver<Job>,
) -> std::process::ExitCode {
    let fail = |message: String| {
        eprintln!("{message}");
        let mut status = shared.status.lock().unwrap();
        status.state = "failed";
        status.error = Some(message);
        drop(status);
        // Stay up a little so the app can read why, then quit.
        std::thread::sleep(Duration::from_secs(30));
        std::process::ExitCode::FAILURE
    };
    let started = Instant::now();
    let file = match GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => return fail(format!("Couldn't open {path}: {e}")),
    };
    let tokenizer = match Tokenizer::from_gguf(&file) {
        Ok(t) => t,
        Err(e) => return fail(format!("Couldn't load the tokenizer: {e}")),
    };
    let model = match Engine::load(&file, options.gpu, options.threads) {
        Ok(m) => m,
        Err(e) => return fail(format!("Couldn't load the model: {e}")),
    };
    let template = Template::for_architecture(file.architecture().unwrap_or(""));
    let trained = file
        .arch_u64("context_length")
        .map_or(usize::MAX, |n| n as usize);
    let context = options.context.clamp(256, trained);
    {
        let mut status = shared.status.lock().unwrap();
        status.state = "ready";
        status.device = model.describe();
        status.context = context;
        if let Some(name) = file.get_str("general.name").filter(|n| n.len() >= 4) {
            status.model = name.to_string();
        }
    }
    eprintln!("Loaded in {:.1?}: {}.", started.elapsed(), model.describe());

    let mut rng = Rng::seeded();
    for job in queue {
        shared.status.lock().unwrap().busy = true;
        let result = reply(
            &model, &tokenizer, template, context, options, &job, &mut rng,
        );
        let mut status = shared.status.lock().unwrap();
        status.busy = false;
        match result {
            Ok(rate) => status.last_tokens_per_second = rate.or(status.last_tokens_per_second),
            Err(message) => {
                let _ = job
                    .events
                    .send(format!("{{\"error\":{}}}", json::quote(&message)));
            }
        }
        drop(status);
        let _ = job.events.send("[DONE]".into());
        *shared.current.lock().unwrap() = None;
    }
    std::process::ExitCode::SUCCESS
}

// Chat templates -----------------------------------------------------------------------

/// How a family's chat template lays out a conversation, written out from each GGUF's
/// `tokenizer.chat_template` for plain chats (no tools).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Template {
    Llama3,
    Qwen,
    Gemma4,
}

impl Template {
    fn for_architecture(architecture: &str) -> Self {
        match architecture {
            a if a.starts_with("qwen") => Template::Qwen,
            a if a.starts_with("gemma") => Template::Gemma4,
            _ => Template::Llama3,
        }
    }

    /// The prompt text (without BOS, which the tokenizer adds) and whether the reply
    /// starts inside the thinking block.
    fn render(
        self,
        system: Option<&str>,
        messages: &[(String, String)],
        thinking: bool,
    ) -> (String, bool) {
        let mut out = String::new();
        match self {
            Template::Llama3 => {
                if let Some(system) = system {
                    out.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    out.push_str("Cutting Knowledge Date: December 2023\n\n");
                    out.push_str(system.trim());
                    out.push_str("<|eot_id|>");
                }
                for (role, content) in messages {
                    out.push_str(&format!(
                        "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
                        content.trim()
                    ));
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                (out, false)
            }
            Template::Qwen => {
                if let Some(system) = system {
                    out.push_str(&format!(
                        "<|im_start|>system\n{}<|im_end|>\n",
                        system.trim()
                    ));
                }
                for (role, content) in messages {
                    out.push_str(&format!(
                        "<|im_start|>{role}\n{}<|im_end|>\n",
                        content.trim()
                    ));
                }
                out.push_str("<|im_start|>assistant\n");
                if thinking {
                    out.push_str("<think>\n");
                } else {
                    out.push_str("<think>\n\n</think>\n\n");
                }
                (out, thinking)
            }
            Template::Gemma4 => {
                if thinking || system.is_some() {
                    out.push_str("<|turn>system\n");
                    if thinking {
                        out.push_str("<|think|>\n");
                    }
                    if let Some(system) = system {
                        out.push_str(system.trim());
                    }
                    out.push_str("<turn|>\n");
                }
                for (role, content) in messages {
                    let role = if role == "assistant" {
                        "model"
                    } else {
                        role.as_str()
                    };
                    out.push_str(&format!("<|turn>{role}\n{}<turn|>\n", content.trim()));
                }
                out.push_str("<|turn>model\n");
                (out, false)
            }
        }
    }

    /// The model's own thinking markers, which the reply shows as <think>…</think>, as the
    /// Vireo server sends them.
    fn thinking_markers(self) -> Option<(&'static str, &'static str)> {
        match self {
            Template::Qwen => Some(("<think>", "</think>")),
            Template::Gemma4 => Some(("<|channel>thought\n", "<channel|>")),
            Template::Llama3 => None,
        }
    }

    /// What closes thinking when the budget runs out, as the model would write it.
    fn close_thinking(self) -> &'static str {
        match self {
            Template::Qwen => "\n</think>\n\n",
            Template::Gemma4 => "<channel|>",
            Template::Llama3 => "",
        }
    }
}

/// A reply's earlier turns go back without their thinking, as the templates do.
fn without_thinking(text: &str) -> String {
    match text.rfind("</think>") {
        Some(end) => text[end + "</think>".len()..].trim_start().to_string(),
        None => text.to_string(),
    }
}

// Generation ---------------------------------------------------------------------------

/// Generates one reply, sending its text as it comes. Returns the generation speed.
fn reply(
    model: &Engine,
    tokenizer: &Tokenizer,
    template: Template,
    context: usize,
    options: &ServeOptions,
    job: &Job,
    rng: &mut Rng,
) -> Result<Option<f64>, String> {
    let started = Instant::now();
    let send = |event: String| job.events.send(event).is_ok();
    send(format!(
        "{{\"type\":\"run\",\"runId\":{}}}",
        json::quote(&job.run_id)
    ));

    // The newest turns that fit, leaving room for the reply.
    let room = context.saturating_sub(job.max_tokens.unwrap_or(1024).min(context / 2));
    let mut messages = job.messages.clone();
    let (mut ids, mut thinking) = encode(tokenizer, template, job, &messages, options.thinking);
    while ids.len() > room && messages.len() > 1 {
        messages.remove(0);
        while messages.len() > 1 && messages[0].0 != "user" {
            messages.remove(0);
        }
        (ids, thinking) = encode(tokenizer, template, job, &messages, options.thinking);
    }
    if ids.len() >= context {
        return Err(format!(
            "The message is {} tokens, longer than the {context}-token context. Raise the context length in Settings › Hardware, or shorten it.",
            ids.len()
        ));
    }

    let mut session = model.session(context);
    for &id in &ids[..ids.len() - 1] {
        if job.stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        model.forward(&mut session, id, false);
    }
    let mut logits = model
        .forward(&mut session, ids[ids.len() - 1], true)
        .expect("asked for logits")
        .to_vec();
    let prompt_time = started.elapsed();

    let markers = template.thinking_markers();
    let mut filter = MarkerFilter::new(markers);
    if thinking {
        send(format!("{{\"content\":{}}}", json::quote("<think>")));
        filter.inside = true;
    }
    let generating = Instant::now();
    let max_tokens = job.max_tokens.unwrap_or(usize::MAX);
    let mut generated = 0usize;
    let mut thought = 0usize;
    let mut pending = Vec::new();
    let mut forced: Vec<u32> = Vec::new();
    let mut used = vec![false; logits.len()];
    while generated < max_tokens && session.len() < session.capacity() {
        if job.stop.load(Ordering::Relaxed) {
            break;
        }
        let next = if let Some(id) = forced.pop() {
            id
        } else {
            if job.sampling.presence_penalty != 0.0 {
                for (logit, _) in logits.iter_mut().zip(&used).filter(|(_, u)| **u) {
                    *logit -= job.sampling.presence_penalty;
                }
            }
            sample(&mut logits, &job.sampling, rng)
        };
        if let Some(u) = used.get_mut(next as usize) {
            *u = true;
        }
        if forced.is_empty() && tokenizer.is_end_of_generation(next) {
            break;
        }
        generated += 1;
        tokenizer.token_bytes(next, true, &mut pending);
        let whole = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => pending.len(),
        };
        let text = String::from_utf8_lossy(&pending[..whole]).into_owned();
        pending.drain(..whole);
        let shown = filter.push(&text);
        if !shown.is_empty() && !send(format!("{{\"content\":{}}}", json::quote(&shown))) {
            break; // the app went away
        }
        if filter.inside {
            thought += 1;
            if let Some(budget) = options.thinking_budget
                && thought >= budget
                && forced.is_empty()
            {
                // Out of thinking budget: write the model's own closing, so it answers next.
                let close = template.close_thinking();
                forced = tokenizer.encode(close, false, true);
                forced.reverse();
            }
        }
        logits.copy_from_slice(
            model
                .forward(&mut session, next, true)
                .expect("asked for logits"),
        );
    }
    let rest = filter.finish(&String::from_utf8_lossy(&pending));
    if !rest.is_empty() {
        send(format!("{{\"content\":{}}}", json::quote(&rest)));
    }
    let seconds = generating.elapsed().as_secs_f64();
    let rate = (generated > 0 && seconds > 0.0).then(|| generated as f64 / seconds);
    send(format!(
        "{{\"done\":true,\"tokens\":{generated},\"duration\":{},\"tokensPerSecond\":{},\"promptTokens\":{},\"promptSeconds\":{:.2}}}",
        json::quote(&format!("{:.1}s", started.elapsed().as_secs_f64())),
        json::quote(&rate.map_or("0".into(), |r| format!("{r:.1}"))),
        ids.len(),
        prompt_time.as_secs_f64()
    ));
    Ok(rate)
}

fn encode(
    tokenizer: &Tokenizer,
    template: Template,
    job: &Job,
    messages: &[(String, String)],
    thinking: bool,
) -> (Vec<u32>, bool) {
    let (text, starts_thinking) = template.render(job.system.as_deref(), messages, thinking);
    (tokenizer.encode(&text, true, true), starts_thinking)
}

/// Turns the model's thinking markers into <think>…</think>, holding back text that
/// might be the start of a marker until it's clear.
struct MarkerFilter {
    markers: Option<(&'static str, &'static str)>,
    held: String,
    inside: bool,
}

impl MarkerFilter {
    fn new(markers: Option<(&'static str, &'static str)>) -> Self {
        Self {
            markers,
            held: String::new(),
            inside: false,
        }
    }

    fn push(&mut self, text: &str) -> String {
        let Some((open, close)) = self.markers else {
            return text.to_string();
        };
        self.held.push_str(text);
        let mut out = String::new();
        loop {
            let marker = if self.inside { close } else { open };
            if let Some(at) = self.held.find(marker) {
                out.push_str(&self.held[..at]);
                out.push_str(if self.inside { "</think>" } else { "<think>" });
                self.held.drain(..at + marker.len());
                self.inside = !self.inside;
                continue;
            }
            // Keep any tail that could still grow into the marker.
            let keep = (1..marker.len().min(self.held.len() + 1))
                .rev()
                .find(|&n| {
                    self.held.is_char_boundary(self.held.len() - n)
                        && marker.starts_with(&self.held[self.held.len() - n..])
                })
                .unwrap_or(0);
            let split = self.held.len() - keep;
            out.push_str(&self.held[..split]);
            self.held.drain(..split);
            return out;
        }
    }

    fn finish(&mut self, text: &str) -> String {
        let mut out = self.push(text);
        out.push_str(&std::mem::take(&mut self.held));
        if self.inside {
            out.push_str("</think>");
            self.inside = false;
        }
        out
    }
}

fn sample(logits: &mut [f32], sampling: &Sampling, rng: &mut Rng) -> u32 {
    if sampling.temperature <= 0.0 {
        return crate::argmax(logits);
    }
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    let keep = if sampling.top_k > 0 {
        sampling.top_k.min(candidates.len())
    } else {
        candidates.len().min(1024)
    };
    candidates.select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
    candidates.truncate(keep);
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let top = candidates[0].1;
    let mut total = 0.0f32;
    for c in &mut candidates {
        c.1 = ((c.1 - top) / sampling.temperature).exp();
        total += c.1;
    }
    for c in &mut candidates {
        c.1 /= total;
    }
    if sampling.min_p > 0.0 {
        let floor = candidates[0].1 * sampling.min_p;
        candidates.retain(|c| c.1 >= floor);
    }
    if sampling.top_p < 1.0 {
        let mut sum = 0.0;
        let mut cut = candidates.len();
        for (i, c) in candidates.iter().enumerate() {
            sum += c.1;
            if sum >= sampling.top_p {
                cut = i + 1;
                break;
            }
        }
        candidates.truncate(cut.max(1));
    }
    let total: f32 = candidates.iter().map(|c| c.1).sum();
    let mut pick = rng.next_f32() * total;
    for c in &candidates {
        pick -= c.1;
        if pick <= 0.0 {
            return c.0;
        }
    }
    candidates[candidates.len() - 1].0
}

/// xorshift64*: enough randomness for sampling, without a dependency.
struct Rng(u64);

impl Rng {
    fn seeded() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64);
        Rng(nanos | 1)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

// HTTP ---------------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> std::io::Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let mut length = 0usize;
    let mut authorization = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = value.trim().parse().unwrap_or(0),
                "authorization" => authorization = Some(value.trim().to_string()),
                _ => {}
            }
        }
    }
    if length > 64 * 1024 * 1024 {
        return Err(std::io::Error::other("request too large"));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Request {
        method,
        path,
        authorization,
        body,
    })
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

fn detail(message: &str) -> String {
    format!("{{\"detail\":{}}}", json::quote(message))
}

fn handle(mut stream: TcpStream, shared: &Shared) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let request = read_request(&stream)?;
    if let Some(token) = &shared.token
        && request.authorization.as_deref() != Some(&format!("Bearer {token}"))
    {
        return respond(
            &mut stream,
            "401 Unauthorized",
            &detail("This engine was started by another app."),
        );
    }
    let path = request.path.split('?').next().unwrap_or("");
    match (request.method.as_str(), path) {
        ("GET", "/api/engine/status" | "/api/health") => {
            let status = shared.status.lock().unwrap();
            let body = format!(
                "{{\"state\":{},\"model\":{},\"device\":{},\"error\":{},\"contextLimit\":{},\"busy\":{},\"tokensPerSecond\":{},\"pid\":{}}}",
                json::quote(status.state),
                json::quote(&status.model),
                json::quote(&status.device),
                status.error.as_deref().map_or("null".into(), json::quote),
                status.context,
                status.busy,
                status
                    .last_tokens_per_second
                    .map_or("null".into(), |r| format!("{r:.1}")),
                std::process::id()
            );
            respond(&mut stream, "200 OK", &body)
        }
        // What the app asks a Mac, answered for this PC.
        ("GET", "/api/pairing/status") => respond(
            &mut stream,
            "200 OK",
            "{\"paired\":true,\"device\":{\"name\":\"This PC\"}}",
        ),
        ("GET", "/api/metrics") => {
            let status = shared.status.lock().unwrap();
            let model = if status.state == "ready" {
                json::quote(&status.model)
            } else {
                "null".into()
            };
            let body = format!(
                "{{\"model\":{model},\"runtime\":{},\"contextLimit\":{}}}",
                json::quote(&format!(
                    "Vireo engine, {}",
                    if status.device.contains("GPU") {
                        "GPU"
                    } else {
                        "CPU"
                    }
                )),
                status.context
            );
            respond(&mut stream, "200 OK", &body)
        }
        ("GET", "/api/capabilities") => respond(
            &mut stream,
            "200 OK",
            "{\"protocolVersion\":1,\"device\":\"windows\",\"serverPlatform\":\"windows\",\"tools\":[]}",
        ),
        ("GET", "/api/skill-library/commands") => respond(&mut stream, "200 OK", "{\"skills\":[]}"),
        ("GET", "/api/assistant/approvals") => respond(&mut stream, "200 OK", "{\"approvals\":[]}"),
        ("GET", "/api/assistant/activity") => {
            respond(&mut stream, "200 OK", "{\"latest\":0,\"entries\":[]}")
        }
        ("POST", "/api/chat/completions") => chat(stream, shared, &request.body),
        ("POST", p) if p.starts_with("/api/chat/runs/") && p.ends_with("/stop") => {
            let id = &p["/api/chat/runs/".len()..p.len() - "/stop".len()];
            if let Some((current, stop)) = shared.current.lock().unwrap().as_ref()
                && current == id
            {
                stop.store(true, Ordering::Relaxed);
            }
            respond(&mut stream, "200 OK", "{\"stopped\":true}")
        }
        ("POST", "/api/engine/shutdown") => {
            respond(&mut stream, "200 OK", "{\"stopping\":true}")?;
            eprintln!("Stopping, as asked.");
            std::process::exit(0);
        }
        _ => respond(
            &mut stream,
            "404 Not Found",
            &detail(
                "The engine on this PC doesn't do that yet: it has chat only, no tools or agent.",
            ),
        ),
    }
}

fn chat(mut stream: TcpStream, shared: &Shared, body: &[u8]) -> std::io::Result<()> {
    let state = shared.status.lock().unwrap().state;
    if state != "ready" {
        let why = if state == "loading" {
            "The model is still loading."
        } else {
            "The model couldn't be loaded."
        };
        return respond(&mut stream, "503 Service Unavailable", &detail(why));
    }
    let request = match std::str::from_utf8(body)
        .map_err(|e| e.to_string())
        .and_then(json::parse)
    {
        Ok(r) => r,
        Err(e) => {
            return respond(
                &mut stream,
                "400 Bad Request",
                &detail(&format!("Couldn't read the request: {e}")),
            );
        }
    };
    let mut system = request
        .get("systemPrompt")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut messages = Vec::new();
    for message in request
        .get("messages")
        .and_then(Value::as_array)
        .unwrap_or(&[])
    {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message.get("content").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" => system = Some(content.to_string()),
            "assistant" => messages.push(("assistant".to_string(), without_thinking(content))),
            _ => messages.push(("user".to_string(), content.to_string())),
        }
    }
    if messages.last().is_none_or(|(role, _)| role != "user") {
        return respond(
            &mut stream,
            "400 Bad Request",
            &detail("The conversation should end with a message from you."),
        );
    }
    let mut sampling = shared.defaults;
    let mut max_tokens = None;
    if let Some(generation) = request.get("generation") {
        let number = |key: &str| generation.get(key).and_then(Value::as_f64);
        if let Some(t) = number("temperature") {
            sampling.temperature = t as f32;
        }
        if let Some(p) = number("top_p") {
            sampling.top_p = p as f32;
        }
        if let Some(k) = number("top_k") {
            sampling.top_k = k as usize;
        }
        if let Some(m) = number("min_p") {
            sampling.min_p = m as f32;
        }
        if let Some(p) = number("presence_penalty") {
            sampling.presence_penalty = p as f32;
        }
        max_tokens = number("max_tokens").map(|n| n as usize).filter(|&n| n > 0);
    }

    let run_id = format!("local-{}", shared.next_run.fetch_add(1, Ordering::Relaxed));
    let stop = Arc::new(AtomicBool::new(false));
    let (events, received) = mpsc::channel();
    // Waits behind any reply already being written.
    let job = Job {
        run_id: run_id.clone(),
        messages,
        system,
        sampling,
        max_tokens,
        events,
        stop: Arc::clone(&stop),
    };
    *shared.current.lock().unwrap() = Some((run_id, Arc::clone(&stop)));
    if shared.jobs.lock().unwrap().send(job).is_err() {
        return respond(
            &mut stream,
            "503 Service Unavailable",
            &detail("The engine is stopping."),
        );
    }

    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    for event in received {
        if stream
            .write_all(format!("data: {event}\n\n").as_bytes())
            .and_then(|()| stream.flush())
            .is_err()
        {
            stop.store(true, Ordering::Relaxed); // the app stopped reading
            break;
        }
        if event == "[DONE]" {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns() -> Vec<(String, String)> {
        vec![
            ("user".into(), "Hi".into()),
            ("assistant".into(), "Hello!".into()),
            ("user".into(), " What's 2+2? ".into()),
        ]
    }

    #[test]
    fn templates_follow_the_models_own() {
        let (qwen, thinking) = Template::Qwen.render(Some("Be brief."), &turns(), true);
        assert!(thinking);
        assert_eq!(
            qwen,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n\
             <|im_start|>user\nWhat's 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        let (qwen_off, thinking) = Template::Qwen.render(None, &turns()[..1], false);
        assert!(!thinking);
        assert!(qwen_off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));

        let (gemma, _) = Template::Gemma4.render(None, &turns(), true);
        assert_eq!(
            gemma,
            "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n\
             <|turn>user\nWhat's 2+2?<turn|>\n<|turn>model\n"
        );
        let (gemma_plain, _) = Template::Gemma4.render(None, &turns()[..1], false);
        assert_eq!(gemma_plain, "<|turn>user\nHi<turn|>\n<|turn>model\n");

        let (llama, _) = Template::Llama3.render(None, &turns()[..1], true);
        assert_eq!(
            llama,
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn thinking_markers_become_think_tags_across_pieces() {
        let mut filter = MarkerFilter::new(Template::Gemma4.thinking_markers());
        let mut out = String::new();
        for piece in ["<|chan", "nel>thou", "ght\nLet me", " add.<chan", "nel|>4"] {
            out.push_str(&filter.push(piece));
        }
        out.push_str(&filter.finish(""));
        assert_eq!(out, "<think>Let me add.</think>4");

        let mut unfinished = MarkerFilter::new(Template::Qwen.thinking_markers());
        unfinished.inside = true;
        let text = unfinished.push("hmm <") + &unfinished.finish("");
        assert_eq!(text, "hmm <</think>");
    }

    #[test]
    fn earlier_replies_lose_their_thinking() {
        assert_eq!(
            without_thinking("<think>\nplan\n</think>\n\nAnswer"),
            "Answer"
        );
        assert_eq!(without_thinking("Plain"), "Plain");
    }

    #[test]
    fn sampling_respects_its_limits() {
        let mut rng = Rng(42);
        let mut logits = vec![0.0, 5.0, 4.9, -3.0];
        let greedy = Sampling {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
        };
        assert_eq!(sample(&mut logits.clone(), &greedy, &mut rng), 1);
        let top1 = Sampling {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            presence_penalty: 0.0,
        };
        for _ in 0..20 {
            assert_eq!(sample(&mut logits.clone(), &top1, &mut rng), 1);
        }
        let two = Sampling {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.5,
            presence_penalty: 0.0,
        };
        let mut seen = [false; 4];
        for _ in 0..400 {
            seen[sample(&mut logits, &two, &mut rng) as usize] = true;
        }
        assert_eq!(seen, [false, true, true, false]);
    }

    #[test]
    fn reads_its_options() {
        let o = ServeOptions::parse(&[
            "--port",
            "4000",
            "--gpu",
            "--context",
            "8192",
            "--temperature",
            "0.6",
            "--top-k",
            "20",
            "--no-thinking",
            "--thinking-budget",
            "0",
            "--threads",
            "0",
            "--presence-penalty",
            "1.5",
        ])
        .unwrap();
        assert_eq!(o.sampling.presence_penalty, 1.5);
        assert_eq!(
            (
                o.port,
                o.gpu,
                o.context,
                o.sampling.top_k,
                o.thinking,
                o.thinking_budget,
                o.threads
            ),
            (4000, true, 8192, 20, false, None, None)
        );
        assert!(ServeOptions::parse(&["--bogus"]).is_none());
    }
}
