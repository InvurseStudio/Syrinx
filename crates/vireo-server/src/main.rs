//! `vireo-server`: serves one GGUF model on this PC over the part of the Vireo app's API
//! that plain chats use, with Vireo's own engine on the GPU. The prototype app pairs with
//! it as it would with the Mac (address `localhost`, the code this prints) and chats
//! with it. No tools, files or skills yet: those come with the rest of phase 3.
//!
//! The routes, as `server/api_server.py` answers them:
//!   POST   /api/pairing/claim          {code, name} → {token}
//!   GET    /api/pairing/status         {paired, device: {name}}
//!   DELETE /api/pairing/device         unpairs the caller
//!   GET    /api/metrics                {model, runtime, contextLimit}
//!   GET    /api/capabilities           no tools
//!   GET    /api/skill-library/commands no skills
//!   GET    /api/assistant/approvals    and /activity: empty
//!   POST   /api/chat/completions       the reply as server-sent events
//!   POST   /api/chat/runs/{id}/stop    stops a reply
//! Anything else answers 404, which the app takes as "not on this server".

mod chat;
mod http;
mod json;
mod sample;
mod store;

use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chat::{Family, Message};
use json::Value;
use sample::Sampler;
use vireo_engine::cuda::Gpu;
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::models::qwen35_cuda::{Qwen35Checkpoint, Qwen35Cuda};
use vireo_engine::models::{Logits, Precision, SavedState};
use vireo_engine::{GgufFile, Tokenizer};

const USAGE: &str = "\
Usage: vireo-server <model.gguf> [--port <n>] [--context <tokens>] [--exact]
                    [--token <token>] [--with-app] [--chat-cache <folder>|off]

Serves the model on http://127.0.0.1:<port> (3000 by default; 0 for any free port) for
the Vireo Prototype app: in its Settings, connect to \"localhost\" with the code this
prints. --context is the most tokens a chat keeps (8192 by default); --exact runs f32
arithmetic. Chats' caches are saved in %LOCALAPPDATA%\\VireoPrototype\\chat-cache (or
--chat-cache's folder; off for none), so going back to a chat reads only what's new.

The app starts it itself with --port 0, --token (a token it accepts without pairing)
and --with-app (stop when the app does: when its end of standard input closes), and
reads the address from the \"Listening on\" line.";

/// The longest reply when the app doesn't say.
const DEFAULT_MAX_TOKENS: usize = 2048;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path = None;
    let mut port = 3000u16;
    let mut context = 8192usize;
    let mut precision = Precision::Int8;
    let mut app_token = None;
    let mut with_app = false;
    let mut chat_cache = Some(None);
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1);
        match args[i].as_str() {
            "--port" if value.and_then(|v| v.parse::<u16>().ok()).is_some() => {
                port = value.unwrap().parse().unwrap();
                i += 1;
            }
            "--context" if value.and_then(|v| v.parse::<usize>().ok()).is_some() => {
                context = value.unwrap().parse().unwrap();
                i += 1;
            }
            "--exact" => precision = Precision::Exact,
            "--token" if value.is_some() => {
                app_token = value.cloned();
                i += 1;
            }
            "--with-app" => with_app = true,
            "--chat-cache" if value.is_some() => {
                let v = value.unwrap();
                chat_cache = if v == "off" {
                    None
                } else {
                    Some(Some(PathBuf::from(v)))
                };
                i += 1;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            a if path.is_none() && !a.starts_with("--") => path = Some(a.to_string()),
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };

    // The engine thread owns the GPU; requests reach it over a channel.
    let (jobs, job_rx) = channel::<Work>();
    if with_app {
        // The app holds the other end of standard input: when it closes, the app has
        // gone (however it ended), and so does the server, once the chat in use is
        // saved (if that takes long, it goes anyway).
        let jobs = jobs.clone();
        std::thread::spawn(move || {
            let mut sink = Vec::new();
            let _ = std::io::Read::read_to_end(&mut std::io::stdin(), &mut sink);
            let (done, saved) = channel();
            if jobs.send(Work::Shutdown(done)).is_ok() {
                let _ = saved.recv_timeout(std::time::Duration::from_secs(10));
            }
            std::process::exit(0);
        });
    }
    let (ready_tx, ready_rx) = channel::<Result<ModelInfo, String>>();
    std::thread::spawn(move || engine(&path, context, precision, chat_cache, ready_tx, job_rx));
    let info = match ready_rx.recv() {
        Ok(Ok(info)) => info,
        Ok(Err(e)) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        Err(_) => return ExitCode::FAILURE,
    };

    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Couldn't listen on port {port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let port = listener.local_addr().map_or(port, |a| a.port());
    let mut pairing = Pairing::load();
    pairing.app_token = app_token;
    println!();
    println!("Vireo server: {} on {}", info.model, info.runtime);
    println!("Listening on http://127.0.0.1:{port}");
    println!(
        "To pair the Vireo Prototype app: Settings, connect to \"localhost\" with the code {}",
        pairing.code
    );
    println!();
    let shared = Arc::new(Shared {
        info,
        pairing: Mutex::new(pairing),
        jobs: Mutex::new(jobs),
        runs: Mutex::new(HashMap::new()),
        next_run: AtomicU64::new(1),
    });
    for stream in listener.incoming().flatten() {
        let shared = shared.clone();
        std::thread::spawn(move || serve(stream, &shared));
    }
    ExitCode::SUCCESS
}

struct ModelInfo {
    model: String,
    runtime: String,
    context: usize,
}

struct Shared {
    info: ModelInfo,
    pairing: Mutex<Pairing>,
    jobs: Mutex<Sender<Work>>,
    /// Replies under way, by run id: set to stop one.
    runs: Mutex<HashMap<String, Arc<AtomicBool>>>,
    next_run: AtomicU64,
}

// ---- Pairing ---------------------------------------------------------------------------

/// The code shown when the server starts, and the tokens it has given out, kept in
/// %LOCALAPPDATA%\VireoPrototype\server-pairings.txt (a token and a device name a line)
/// so the app stays paired across restarts.
struct Pairing {
    code: String,
    devices: Vec<(String, String)>,
    /// The token of the app that started this server, if one did: paired already.
    app_token: Option<String>,
}

fn random_u64() -> u64 {
    use std::hash::{BuildHasher, RandomState};
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    RandomState::new().hash_one(nanos)
}

impl Pairing {
    fn file() -> PathBuf {
        let base =
            std::env::var_os("LOCALAPPDATA").map_or_else(|| PathBuf::from("."), PathBuf::from);
        base.join("VireoPrototype").join("server-pairings.txt")
    }

    fn load() -> Self {
        let devices = std::fs::read_to_string(Self::file())
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.split_once('\t'))
            .map(|(t, n)| (t.to_string(), n.to_string()))
            .collect();
        Self {
            code: format!("{:06}", random_u64() % 1_000_000),
            devices,
            app_token: None,
        }
    }

    fn save(&self) {
        let path = Self::file();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let text: String = self
            .devices
            .iter()
            .map(|(t, n)| format!("{t}\t{n}\n"))
            .collect();
        let _ = std::fs::write(path, text);
    }

    fn device(&self, token: Option<&str>) -> Option<&str> {
        let token = token?;
        if self.app_token.as_deref() == Some(token) {
            return Some("This PC");
        }
        self.devices
            .iter()
            .find(|(t, _)| t == token)
            .map(|(_, n)| n.as_str())
    }
}

// ---- Requests --------------------------------------------------------------------------

fn error(detail: &str) -> String {
    format!("{{\"detail\":{}}}", json::string(detail))
}

fn serve(mut stream: TcpStream, shared: &Shared) {
    let request = match http::read_request(&stream) {
        Ok(r) => r,
        Err(e) => return http::respond(&mut stream, 400, &error(&e)),
    };
    let method = request.method.as_str();
    let path = request.path.trim_end_matches('/');

    if (method, path) == ("POST", "/api/pairing/claim") {
        let body = json::parse(&String::from_utf8_lossy(&request.body)).unwrap_or(Value::Null);
        let mut pairing = shared.pairing.lock().unwrap();
        if body.str("code") != Some(pairing.code.as_str()) {
            return http::respond(
                &mut stream,
                400,
                &error("That code doesn't match. Use the one the Vireo server printed."),
            );
        }
        let token = format!("{:016x}{:016x}", random_u64(), random_u64());
        let name = body.str("name").unwrap_or("Windows PC").to_string();
        pairing.devices.push((token.clone(), name));
        pairing.save();
        return http::respond(
            &mut stream,
            200,
            &format!("{{\"token\":{}}}", json::string(&token)),
        );
    }

    let device = shared
        .pairing
        .lock()
        .unwrap()
        .device(request.bearer.as_deref())
        .map(str::to_string);
    if (method, path) == ("GET", "/api/pairing/status") {
        let body = match &device {
            Some(name) => format!(
                "{{\"paired\":true,\"device\":{{\"name\":{}}}}}",
                json::string(name)
            ),
            None => "{\"paired\":false}".to_string(),
        };
        return http::respond(&mut stream, 200, &body);
    }
    if device.is_none() {
        return http::respond(
            &mut stream,
            401,
            "{\"code\":\"pairing_required\",\"detail\":\"This PC isn't paired with the Vireo server. Pair again with the code it printed.\"}",
        );
    }

    match (method, path) {
        ("DELETE", "/api/pairing/device") => {
            let mut pairing = shared.pairing.lock().unwrap();
            let token = request.bearer.clone().unwrap_or_default();
            pairing.devices.retain(|(t, _)| *t != token);
            pairing.save();
            http::respond(&mut stream, 200, "{\"ok\":true}");
        }
        ("GET", "/api/metrics") => {
            let i = &shared.info;
            http::respond(
                &mut stream,
                200,
                &format!(
                    "{{\"model\":{},\"runtime\":{},\"contextLimit\":{}}}",
                    json::string(&i.model),
                    json::string(&i.runtime),
                    i.context
                ),
            );
        }
        ("GET", "/api/capabilities") => {
            let device = request
                .query
                .split('&')
                .find_map(|kv| kv.strip_prefix("device="))
                .unwrap_or("windows")
                .to_string();
            http::respond(
                &mut stream,
                200,
                &format!(
                    "{{\"protocolVersion\":1,\"device\":{},\"serverPlatform\":\"windows\",\"tools\":[]}}",
                    json::string(&device)
                ),
            );
        }
        ("GET", "/api/skill-library/commands") => {
            http::respond(&mut stream, 200, "{\"skills\":[]}")
        }
        // The assistant's inbox and activity log, which the app polls: empty here.
        ("GET", "/api/assistant/approvals") => {
            http::respond(&mut stream, 200, "{\"approvals\":[]}")
        }
        ("GET", "/api/assistant/activity") => {
            http::respond(&mut stream, 200, "{\"latest\":0,\"entries\":[]}")
        }
        ("POST", "/api/chat/completions") => completion(stream, shared, &request.body),
        ("POST", p) if p.starts_with("/api/chat/runs/") && p.ends_with("/stop") => {
            let id = &p["/api/chat/runs/".len()..p.len() - "/stop".len()];
            if let Some(stop) = shared.runs.lock().unwrap().get(id) {
                stop.store(true, Ordering::Relaxed);
            }
            http::respond(&mut stream, 200, "{\"ok\":true}");
        }
        _ => http::respond(&mut stream, 404, &error("Not on this server yet.")),
    }
}

/// One reply, streamed as the app reads them: the run, the text as it comes, then done.
fn completion(mut stream: TcpStream, shared: &Shared, body: &[u8]) {
    let body = match json::parse(&String::from_utf8_lossy(body)) {
        Ok(v) => v,
        Err(e) => return http::respond(&mut stream, 400, &error(&format!("Bad request: {e}"))),
    };
    let mut messages = Vec::new();
    let mut system = body.str("systemPrompt").map(str::to_string);
    for m in body.array("messages") {
        let (Some(role), Some(content)) = (m.str("role"), m.str("content")) else {
            continue;
        };
        match role {
            "system" => system = Some(content.to_string()),
            "user" => messages.push(Message {
                role: "user".into(),
                content: content.to_string(),
            }),
            "assistant" => messages.push(Message {
                role: "assistant".into(),
                content: chat::without_thinking(content),
            }),
            _ => {}
        }
    }
    if messages.last().is_none_or(|m| m.role != "user") {
        return http::respond(
            &mut stream,
            400,
            &error("The chat should end with a message to reply to."),
        );
    }
    let generation = body.get("generation");
    let max_tokens = generation
        .and_then(|g| g.num("max_tokens"))
        .map_or(DEFAULT_MAX_TOKENS, |v| v.max(1.0) as usize);

    let run_id = format!("run-{}", shared.next_run.fetch_add(1, Ordering::Relaxed));
    let stop = Arc::new(AtomicBool::new(false));
    shared
        .runs
        .lock()
        .unwrap()
        .insert(run_id.clone(), stop.clone());
    let (out, events) = channel();
    let job = Job {
        system,
        messages,
        generation: generation.cloned(),
        max_tokens,
        stop: stop.clone(),
        out,
    };
    if shared.jobs.lock().unwrap().send(Work::Reply(job)).is_err() {
        shared.runs.lock().unwrap().remove(&run_id);
        return http::respond(&mut stream, 500, &error("The engine has stopped."));
    }
    stream_reply(&mut stream, &run_id, &events, &stop);
    shared.runs.lock().unwrap().remove(&run_id);
}

fn stream_reply(stream: &mut TcpStream, run_id: &str, events: &Receiver<Out>, stop: &AtomicBool) {
    if http::start_events(stream).is_err() {
        stop.store(true, Ordering::Relaxed);
        return;
    }
    let mut send = |data: &str| {
        if http::event(stream, data).is_err() {
            stop.store(true, Ordering::Relaxed); // the app has gone: no one to write for
        }
    };
    send(&format!(
        "{{\"type\":\"run\",\"runId\":{}}}",
        json::string(run_id)
    ));
    for event in events {
        match event {
            Out::Text(text) => send(&format!("{{\"content\":{}}}", json::string(&text))),
            Out::Done { tokens, seconds } => {
                let tps = if seconds > 0.0 {
                    tokens as f64 / seconds
                } else {
                    0.0
                };
                send(&format!(
                    "{{\"done\":true,\"tokens\":{tokens},\"duration\":\"{seconds:.2}\",\"tokensPerSecond\":\"{tps:.2}\"}}"
                ));
            }
            Out::Error(e) => send(&format!("{{\"error\":{}}}", json::string(&e))),
        }
    }
    send("[DONE]");
}

// ---- The engine ------------------------------------------------------------------------

enum Work {
    Reply(Job),
    /// Save the chat in use, then say so.
    Shutdown(Sender<()>),
}

struct Job {
    system: Option<String>,
    messages: Vec<Message>,
    generation: Option<Value>,
    max_tokens: usize,
    stop: Arc<AtomicBool>,
    out: Sender<Out>,
}

enum Out {
    Text(String),
    Done { tokens: usize, seconds: f64 },
    Error(String),
}

#[allow(clippy::large_enum_variant)] // there is one, for the life of the server
enum Model {
    Llama(LlamaCuda<'static>),
    Qwen35(Qwen35Cuda<'static>),
    Gemma4(Gemma4Cuda<'static>),
}

enum Session {
    Llama(vireo_engine::models::llama_cuda::CudaSession),
    Qwen35(vireo_engine::models::qwen35_cuda::Qwen35CudaSession),
    Gemma4(vireo_engine::models::gemma4_cuda::Gemma4CudaSession),
}

/// The one session the server keeps between replies, so a chat's next turn reads only
/// what's new: its keys and values (and Qwen's DeltaNet state) stay on the GPU.
struct Kept {
    session: Session,
    /// The tokens in the session, in order: the last prompt, then its reply.
    tokens: Vec<u32>,
    /// Qwen: its DeltaNet layers saved at the end of the last prompt's history.
    checkpoint: Option<Qwen35Checkpoint>,
}

/// How many tokens `ids` starts with that `tokens` does too, leaving at least the last of
/// `ids` to read again (for its logits).
fn shared_start(tokens: &[u32], ids: &[u32]) -> usize {
    tokens
        .iter()
        .zip(ids)
        .take_while(|(a, b)| a == b)
        .count()
        .min(ids.len().saturating_sub(1))
}

impl Model {
    fn session(&self, capacity: usize) -> Result<Session, String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        Ok(match self {
            Model::Llama(m) => Session::Llama(m.session(capacity).map_err(e)?),
            Model::Qwen35(m) => Session::Qwen35(m.session(capacity).map_err(e)?),
            Model::Gemma4(m) => Session::Gemma4(m.session(capacity).map_err(e)?),
        })
    }

    /// The prompt, in batches; the last token's logits.
    fn read(&self, s: &mut Session, ids: &[u32]) -> Result<Vec<f32>, String> {
        let r = match (self, s) {
            (Model::Llama(m), Session::Llama(s)) => m
                .forward_batch(s, ids, Logits::Last)
                .map(|l| l.map(<[f32]>::to_vec)),
            (Model::Qwen35(m), Session::Qwen35(s)) => m
                .forward_batch(s, ids, Logits::Last)
                .map(|l| l.map(<[f32]>::to_vec)),
            (Model::Gemma4(m), Session::Gemma4(s)) => m
                .forward_batch(s, ids, Logits::Last)
                .map(|l| l.map(<[f32]>::to_vec)),
            _ => unreachable!("a session belongs to its model"),
        };
        r.map_err(|e| e.to_string())?
            .ok_or_else(|| "no logits".into())
    }

    /// Reads `ids` (the history, then from `mark` on the reply's opened turn) into the
    /// kept session, starting from the longest start it shares with the tokens already
    /// there; the last token's logits, and how many tokens were reused. On an error the
    /// session's contents are unknown: drop it.
    fn continue_with(
        &self,
        kept: &mut Kept,
        ids: &[u32],
        mark: usize,
    ) -> Result<(Vec<f32>, usize), String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        let shared = shared_start(&kept.tokens, ids);
        let from = match (self, &mut kept.session) {
            (Model::Llama(_), Session::Llama(s)) => {
                s.truncate(shared);
                shared
            }
            (Model::Gemma4(_), Session::Gemma4(s)) => {
                s.truncate(shared);
                shared
            }
            // A DeltaNet state can't be cut back: go on from the end, if every token
            // there is wanted, else from the last history's end, else from empty.
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                if shared == s.len() {
                    shared
                } else if let Some(cp) = kept.checkpoint.as_ref().filter(|c| c.len() <= shared) {
                    m.restore(s, cp).map_err(e)?;
                    cp.len()
                } else {
                    m.reset(s).map_err(e)?;
                    kept.checkpoint = None;
                    0
                }
            }
            _ => unreachable!("a session belongs to its model"),
        };
        kept.tokens.truncate(from);
        // Qwen: the history read on its own, and its DeltaNet saved where the next
        // turn's prompt will part from this one.
        let mut start = from;
        if let (Model::Qwen35(m), Session::Qwen35(s)) = (self, &mut kept.session)
            && mark > from
        {
            m.forward_batch(s, &ids[from..mark], Logits::None)
                .map_err(e)?;
            m.save(s, &mut kept.checkpoint).map_err(e)?;
            start = mark;
        }
        let logits = self.read(&mut kept.session, &ids[start..])?;
        kept.tokens.extend_from_slice(&ids[from..]);
        Ok((logits, from))
    }

    /// How many of `ids` `continue_with` would keep from the session in use.
    fn reusable(&self, kept: &Kept, ids: &[u32]) -> usize {
        let shared = shared_start(&kept.tokens, ids);
        match &kept.session {
            Session::Qwen35(s) if shared != s.len() => kept
                .checkpoint
                .as_ref()
                .map(|c| c.len())
                .filter(|&n| n <= shared)
                .unwrap_or(0),
            _ => shared,
        }
    }

    /// How many of the session's tokens `export` would save.
    fn saveable(&self, kept: &Kept) -> usize {
        match &kept.session {
            Session::Qwen35(_) => kept.checkpoint.as_ref().map_or(0, |c| c.len()),
            _ => kept.tokens.len(),
        }
    }

    /// How many of `ids` a saved chat of `tokens` could give: as many as they share, but
    /// Qwen's only whole (its DeltaNet state is for the end).
    fn usable(&self, tokens: &[u32], ids: &[u32], context: usize) -> usize {
        let shared = shared_start(tokens, ids);
        let n = match self {
            Model::Qwen35(_) if shared != tokens.len() => 0,
            _ => shared,
        };
        if n > context { 0 } else { n }
    }

    /// The chat in `kept`, to save: every token (Qwen: up to its checkpoint), and the
    /// state after them. `None` when there's nothing to save.
    fn export(&self, kept: &Kept) -> Result<Option<(Vec<u32>, SavedState)>, String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        let state = match &kept.session {
            Session::Llama(s) => s.export().map_err(e)?,
            Session::Gemma4(s) => s.export().map_err(e)?,
            Session::Qwen35(s) => match &kept.checkpoint {
                Some(cp) => s.export(cp).map_err(e)?,
                None => return Ok(None),
            },
        };
        if state.is_empty() {
            return Ok(None);
        }
        Ok(Some((kept.tokens[..state.len()].to_vec(), state)))
    }

    /// Loads a saved chat's state into `kept` for its first `tokens.len()` positions.
    fn import(&self, kept: &mut Kept, state: &SavedState, tokens: Vec<u32>) -> Result<(), String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        match (self, &mut kept.session) {
            (Model::Llama(_), Session::Llama(s)) => s.import(state, tokens.len()).map_err(e)?,
            (Model::Gemma4(_), Session::Gemma4(s)) => s.import(state, tokens.len()).map_err(e)?,
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                s.import(state).map_err(e)?;
                m.save(s, &mut kept.checkpoint).map_err(e)?;
            }
            _ => unreachable!("a session belongs to its model"),
        }
        kept.tokens = tokens;
        Ok(())
    }

    /// Empties `kept`.
    fn clear(&self, kept: &mut Kept) -> Result<(), String> {
        match (self, &mut kept.session) {
            (Model::Llama(_), Session::Llama(s)) => s.truncate(0),
            (Model::Gemma4(_), Session::Gemma4(s)) => s.truncate(0),
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                m.reset(s).map_err(|e| e.to_string())?;
                kept.checkpoint = None;
            }
            _ => unreachable!("a session belongs to its model"),
        }
        kept.tokens.clear();
        Ok(())
    }

    fn next(&self, s: &mut Session, id: u32, logits: &mut [f32]) -> Result<(), String> {
        let r = match (self, s) {
            (Model::Llama(m), Session::Llama(s)) => m
                .forward(s, id, true)
                .map(|l| l.map(|l| logits.copy_from_slice(l))),
            (Model::Qwen35(m), Session::Qwen35(s)) => m
                .forward(s, id, true)
                .map(|l| l.map(|l| logits.copy_from_slice(l))),
            (Model::Gemma4(m), Session::Gemma4(s)) => m
                .forward(s, id, true)
                .map(|l| l.map(|l| logits.copy_from_slice(l))),
            _ => unreachable!("a session belongs to its model"),
        };
        r.map_err(|e| e.to_string())?
            .ok_or_else(|| "no logits".into())
    }
}

fn engine(
    path: &str,
    context: usize,
    precision: Precision,
    chat_cache: Option<Option<PathBuf>>,
    ready: Sender<Result<ModelInfo, String>>,
    jobs: Receiver<Work>,
) {
    let started = Instant::now();
    let file: &'static GgufFile = match GgufFile::open(path) {
        Ok(f) => Box::leak(Box::new(f)),
        Err(e) => {
            let _ = ready.send(Err(format!("Couldn't open {path}: {e}")));
            return;
        }
    };
    let loaded = (|| -> Result<(Model, Tokenizer, Family, String), String> {
        let family = Family::of(file.architecture())
            .ok_or_else(|| format!("{path} isn't a Llama, Qwen 3.5 or Gemma 4 model"))?;
        let tokenizer = Tokenizer::from_gguf(file).map_err(|e| e.to_string())?;
        let gpu = Gpu::open().map_err(|e| format!("No GPU for the engine: {e}"))?;
        let gpu_name = gpu.info.name.clone();
        let e = |e: vireo_engine::models::Error| e.to_string();
        let model = match family {
            Family::Llama => Model::Llama(LlamaCuda::load(file, gpu).map_err(e).map(|mut m| {
                m.set_precision(precision);
                m
            })?),
            Family::Qwen35 => {
                Model::Qwen35(Qwen35Cuda::load(file, gpu).map_err(e).map(|mut m| {
                    m.set_precision(precision);
                    m
                })?)
            }
            Family::Gemma4 => {
                Model::Gemma4(Gemma4Cuda::load(file, gpu).map_err(e).map(|mut m| {
                    m.set_precision(precision);
                    m
                })?)
            }
        };
        Ok((model, tokenizer, family, gpu_name))
    })();
    let (model, tokenizer, family, gpu_name) = match loaded {
        Ok(l) => l,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let name = std::path::Path::new(path)
        .file_stem()
        .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    let mode = if precision == Precision::Exact {
        "exact"
    } else {
        "8-bit"
    };
    eprintln!("Loaded {name} in {:.1?}.", started.elapsed());
    let _ = ready.send(Ok(ModelInfo {
        model: name,
        runtime: format!("Vireo engine ({gpu_name}, {mode})"),
        context,
    }));

    let mut store = chat_cache.map(|dir| {
        let key = store::model_key(std::path::Path::new(path), mode);
        let dir = dir.unwrap_or_else(|| store::default_dir(&key));
        let s = store::Store::open(dir, store::DISK_BUDGET);
        if !s.entries().is_empty() {
            eprintln!("{} saved chats.", s.entries().len());
        }
        s
    });
    let mut seed = random_u64();
    let mut kept: Option<Kept> = None;
    for work in jobs {
        let job = match work {
            Work::Reply(job) => job,
            Work::Shutdown(done) => {
                if let (Some(k), Some(store)) = (kept.as_ref(), store.as_mut()) {
                    stash(&model, k, store);
                    store.flush();
                }
                let _ = done.send(());
                return;
            }
        };
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if kept.is_none() {
            match model.session(context) {
                Ok(session) => {
                    kept = Some(Kept {
                        session,
                        tokens: Vec::new(),
                        checkpoint: None,
                    })
                }
                Err(e) => {
                    let _ = job.out.send(Out::Error(e));
                    continue;
                }
            }
        }
        if let Err(e) = reply(
            &model,
            kept.as_mut().unwrap(),
            store.as_mut(),
            &tokenizer,
            family,
            context,
            seed,
            &job,
        ) {
            // Whatever the session held is unknown now: the next reply starts afresh.
            kept = None;
            let _ = job.out.send(Out::Error(e));
        }
    }
}

/// Saves the chat in `kept` to `store`, if there's one worth saving.
fn stash(model: &Model, kept: &Kept, store: &mut store::Store) {
    match model.export(kept) {
        Ok(Some((tokens, state))) => store.put(tokens, state),
        Ok(None) => {}
        Err(e) => eprintln!("Couldn't save the chat's cache: {e}"),
    }
}

/// How many more tokens a saved chat must spare reading, over the session in use, to be
/// loaded instead: loading one costs about as much as reading a few dozen tokens.
const SWITCH_GAIN: usize = 128;

/// Before a reply for `ids`: when it would throw away much of the chat in the session
/// (another chat, or an early message edited), saves that chat first; and when a saved
/// chat shares more of `ids` than the session does, loads it in the session's place.
fn switch_chat(
    model: &Model,
    kept: &mut Kept,
    store: &mut store::Store,
    ids: &[u32],
    context: usize,
) -> Result<(), String> {
    let live = model.reusable(kept, ids);
    let thrown_away = model.saveable(kept).saturating_sub(live);
    let best = store
        .entries()
        .iter()
        .enumerate()
        .map(|(i, e)| (i, model.usable(&e.tokens, ids, context)))
        .max_by_key(|&(_, n)| n)
        .filter(|&(_, n)| n >= live + SWITCH_GAIN);
    let Some((i, usable)) = best else {
        if thrown_away >= SWITCH_GAIN {
            stash(model, kept, store);
        }
        return Ok(());
    };
    let started = Instant::now();
    let tokens = store.entries()[i].tokens[..usable].to_vec();
    // The state first: saving the session's chat may reorder the saved ones.
    let Some(state) = store.get(i) else {
        return Ok(());
    };
    if thrown_away >= SWITCH_GAIN {
        stash(model, kept, store);
    }
    if let Err(e) = model.import(kept, &state, tokens) {
        // Not this session's kind of state after all: forget it, and start afresh.
        eprintln!("Couldn't load a saved chat ({e}); reading it again.");
        store.forget(&state);
        model.clear(kept)?;
        return Ok(());
    }
    eprintln!(
        "Loaded {usable} tokens of a saved chat in {:.0} ms.",
        started.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)] // the engine loop's state, which it owns apart
fn reply(
    model: &Model,
    kept: &mut Kept,
    store: Option<&mut store::Store>,
    tokenizer: &Tokenizer,
    family: Family,
    context: usize,
    seed: u64,
    job: &Job,
) -> Result<(), String> {
    let today = chat::today();
    let max_tokens = job.max_tokens.min(context / 2);
    // The chat's latest messages that fit with room for the reply.
    let mut first = 0;
    let opening = tokenizer.encode(chat::opening(family), false, true);
    let (ids, mark) = loop {
        let text = chat::history(
            family,
            job.system.as_deref(),
            &job.messages[first..],
            &today,
        );
        let mut ids = tokenizer.encode(&text, false, true);
        let mark = ids.len();
        ids.extend_from_slice(&opening);
        if ids.len() + max_tokens <= context || first + 1 >= job.messages.len() {
            break (ids, mark);
        }
        first += 1;
        while first < job.messages.len() && job.messages[first].role != "user" {
            first += 1;
        }
        first = first.min(job.messages.len() - 1);
    };
    if ids.len() + max_tokens > context {
        return Err(format!(
            "The message is {} tokens, too long for this server's {context}-token chats.",
            ids.len()
        ));
    }
    let started = Instant::now();
    if let Some(store) = store {
        switch_chat(model, kept, store, &ids, context)?;
    }
    let (mut logits, reused) = model.continue_with(kept, &ids, mark)?;
    let prompt_seconds = started.elapsed().as_secs_f64();

    let settings = family.sampling().with(job.generation.as_ref());
    let mut sampler = Sampler::new(settings, seed);
    // Gemma 4's thinking channel, should it open one with thinking off: left out.
    let channel = (
        tokenizer.token_id("<|channel>"),
        tokenizer.token_id("<channel|>"),
    );
    let mut in_channel = false;
    let mut generated: Vec<u32> = Vec::new();
    let mut pending = Vec::new();
    let started = Instant::now();
    while generated.len() < max_tokens && !job.stop.load(Ordering::Relaxed) {
        let next = sampler.pick(&logits, &generated);
        if tokenizer.is_end_of_generation(next) {
            break;
        }
        generated.push(next);
        if Some(next) == channel.0 {
            in_channel = true;
        } else if Some(next) == channel.1 {
            in_channel = false;
        } else if !in_channel {
            tokenizer.token_bytes(next, false, &mut pending);
            // Whole characters only: a character's bytes can span tokens.
            let whole = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => pending.len(),
            };
            if whole > 0 {
                let text = String::from_utf8_lossy(&pending[..whole]).into_owned();
                pending.drain(..whole);
                if job.out.send(Out::Text(text)).is_err() {
                    break;
                }
            }
        }
        model.next(&mut kept.session, next, &mut logits)?;
        kept.tokens.push(next);
    }
    if !pending.is_empty() {
        let _ = job
            .out
            .send(Out::Text(String::from_utf8_lossy(&pending).into_owned()));
    }
    let seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "Reply: {} prompt tokens ({reused} kept, {} read in {prompt_seconds:.2}s, {:.0}/s), {} tokens in {seconds:.2}s ({:.1}/s){}.",
        ids.len(),
        ids.len() - reused,
        (ids.len() - reused) as f64 / prompt_seconds.max(1e-9),
        generated.len(),
        generated.len() as f64 / seconds.max(1e-9),
        if job.stop.load(Ordering::Relaxed) {
            ", stopped"
        } else {
            ""
        }
    );
    let _ = job.out.send(Out::Done {
        tokens: generated.len(),
        seconds,
    });
    Ok(())
}
