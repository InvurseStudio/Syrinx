//! `vireo-server`: serves one GGUF model on this PC over the part of the Vireo app's API
//! that chats use, with Vireo's own engine on the GPU. The prototype app pairs with it as
//! it would with the Mac (address `localhost`, the code this prints) and chats with it.
//! Replies can call tools (reading, finding and changing files and Git's status and
//! diff, which the app runs on the PC, and the date and time), each call written under its
//! grammar (`tools.rs`). The ones that change files wait for the user's approval in the
//! app first. Skills (SKILL.md, as the Mac's) are picked by scoring and read by the model
//! as the result of a `use_skill` call the server writes in (`skills.rs`).
//!
//! The routes, as `server/api_server.py` answers them:
//!   POST   /api/pairing/claim          {code, name} → {token}
//!   GET    /api/pairing/status         {paired, device: {name}}
//!   DELETE /api/pairing/device         unpairs the caller
//!   GET    /api/metrics                {model, runtime, contextLimit}
//!   GET    /api/capabilities           the tools
//!   POST   /api/agent/client-tool-result a PC tool's result, from the app
//!   POST   /api/agent/confirm          {id, approved, remember}: the user's answer to an approval
//!   GET    /api/skill-library/commands the skills this PC can follow, for slash commands
//!   GET    /api/assistant/approvals    and /activity: empty
//!   POST   /api/upload                 an attachment (multipart) → {path, docId, extractedText}
//!   GET    /uploads/{id}-{name}        a kept attachment's file
//!   POST   /api/chat/completions       the reply as server-sent events
//!   POST   /api/chat/runs/{id}/stop    stops a reply
//! Anything else answers 404, which the app takes as "not on this server".

mod attach;
mod calc;
mod chat;
mod draft;
mod files;
mod http;
mod json;
mod sample;
mod skills;
mod store;
mod tools;

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
use vireo_engine::grammar::{Grammar, TokenTrie};
use vireo_engine::models::gemma4_cuda::Gemma4Cuda;
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::models::qwen35_cuda::{Qwen35Checkpoint, Qwen35Cuda};
use vireo_engine::models::{Logits, Precision, SavedState};
use vireo_engine::{GgufFile, Tokenizer};

const USAGE: &str = "\
Usage: vireo-server <model.gguf> [--port <n>] [--context <tokens>|auto] [--exact]
                    [--token <token>] [--with-app] [--chat-cache <folder>|off]
                    [--score-choices on|off] [--tools-shown all|scored] [--heal on|off]
                    [--changes declared|scored|offered] [--skills on|off]
                    [--draft <small-model.gguf>] [--draft-tokens <n>]

Serves the model on http://127.0.0.1:<port> (3000 by default; 0 for any free port) for
the Vireo Prototype app: in its Settings, connect to \"localhost\" with the code this
prints. --context is the most tokens a chat keeps: auto (the default) is as many as the
GPU's free memory holds once the model is loaded, up to the model's own limit; --exact runs f32
arithmetic. Chats' caches are saved in the prototype's folder (on Windows
%LOCALAPPDATA%\\VireoPrototype, on the Mac ~/Library/Application Support/VireoPrototype),
in chat-cache (or
--chat-cache's folder; off for none), so going back to a chat reads only what's new.

The tools that change files are offered when one scored question says the request asks
for a change (--changes), and each waits for the user's approval in the app. The kinds
the user always allows are kept in approvals.txt in that folder.

The app starts it itself with --port 0, --token (a token it accepts without pairing)
and --with-app (stop when the app does: when its end of standard input closes), and
reads the address from the \"Listening on\" line.

--draft has a small model of the same family and vocabulary guess the next tokens (Gemma
4 E2B for 31B), --draft-tokens at a time (2 by default), for the model to check in one
pass: every token is still the model's own pick. Gemma 4 only.";

/// How replies use the tools, from the command line: features the eval turns on and off.
#[derive(Debug, Default)]
struct Options {
    /// A tool's name chosen by scoring each whole name (`--score-choices on`), not a
    /// token at a time.
    score_choices: bool,
    /// Token healing for forced text (`--heal off` to measure without it).
    heal: bool,
    /// Only the tool groups a request needs, picked by scoring (`--tools-shown scored`),
    /// in place of every tool (`all`).
    scored_groups: bool,
    /// When the tools that change files are there (`--changes`): `declared` (the
    /// default), declared and callable only when scoring says the request asks for a
    /// change; `scored`, declared always but callable only then (the prompt stays the
    /// same, but the longer tool list cost small models on other requests, and a model
    /// kept from a change it wanted called another tool); `offered`, always. Measured
    /// 2026-11-11 (PLAN.md).
    changes: Changes,
    /// Skills picked by scoring and written in as a `use_skill` call (`--skills on`, the
    /// default), or none (`off`).
    skills: bool,
    /// A draft model to guess tokens with (`--draft`), and how many a round.
    draft: Option<String>,
    draft_tokens: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
enum Changes {
    Offered,
    Scored,
    #[default]
    Declared,
}

static OPTIONS: std::sync::OnceLock<Options> = std::sync::OnceLock::new();

fn options() -> &'static Options {
    OPTIONS.get_or_init(Options::default)
}

/// The longest reply when the app doesn't say.
const DEFAULT_MAX_TOKENS: usize = 2048;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path = None;
    let mut port = 3000u16;
    let mut context: Option<usize> = None;
    let mut precision = Precision::Int8;
    let mut app_token = None;
    let mut with_app = false;
    let mut chat_cache = Some(None);
    let mut score_choices = false;
    let mut scored_groups = false;
    let mut heal = true;
    let mut changes = Changes::Declared;
    let mut use_skills = true;
    let mut draft = None;
    let mut draft_tokens = draft::DEFAULT_GUESSES;
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1);
        match args[i].as_str() {
            "--port" if value.and_then(|v| v.parse::<u16>().ok()).is_some() => {
                port = value.unwrap().parse().unwrap();
                i += 1;
            }
            "--context" if value.is_some_and(|v| v == "auto") => {
                context = None;
                i += 1;
            }
            "--context"
                if value
                    .and_then(|v| v.parse::<usize>().ok())
                    .is_some_and(|n| n >= 256) =>
            {
                context = value.unwrap().parse().ok();
                i += 1;
            }
            "--exact" => precision = Precision::Exact,
            "--token" if value.is_some() => {
                app_token = value.cloned();
                i += 1;
            }
            "--with-app" => with_app = true,
            "--heal" if matches!(value.map(String::as_str), Some("on" | "off")) => {
                heal = value.unwrap() == "on";
                i += 1;
            }
            "--tools-shown" if matches!(value.map(String::as_str), Some("all" | "scored")) => {
                scored_groups = value.unwrap() == "scored";
                i += 1;
            }
            "--changes"
                if matches!(
                    value.map(String::as_str),
                    Some("offered" | "scored" | "declared")
                ) =>
            {
                changes = match value.unwrap().as_str() {
                    "offered" => Changes::Offered,
                    "scored" => Changes::Scored,
                    _ => Changes::Declared,
                };
                i += 1;
            }
            "--skills" if matches!(value.map(String::as_str), Some("on" | "off")) => {
                use_skills = value.unwrap() == "on";
                i += 1;
            }
            "--score-choices" if matches!(value.map(String::as_str), Some("on" | "off")) => {
                score_choices = value.unwrap() == "on";
                i += 1;
            }
            "--chat-cache" if value.is_some() => {
                let v = value.unwrap();
                chat_cache = if v == "off" {
                    None
                } else {
                    Some(Some(PathBuf::from(v)))
                };
                i += 1;
            }
            "--draft" if value.is_some() => {
                draft = value.cloned();
                i += 1;
            }
            "--draft-tokens"
                if value
                    .and_then(|v| v.parse::<usize>().ok())
                    .is_some_and(|n| (1..=16).contains(&n)) =>
            {
                draft_tokens = value.unwrap().parse().unwrap();
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

    let _ = OPTIONS.set(Options {
        score_choices,
        heal,
        scored_groups,
        changes,
        skills: use_skills,
        draft,
        draft_tokens,
    });
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
        tool_results: Arc::default(),
        approvals: Arc::new(Approvals {
            waiting: Mutex::new(HashMap::new()),
            always: Mutex::new(Approvals::load()),
        }),
        uploads: attach::Uploads {
            dir: store::data_dir().join("uploads"),
        },
    });
    {
        let shared = shared.clone();
        std::thread::spawn(move || prune_uploads(&shared));
    }
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
    tool_results: ToolWaits,
    approvals: Arc<Approvals>,
    /// Attachments, and the text read from them.
    uploads: attach::Uploads,
}

/// The platform reported to the app, as the Python server names it.
const PLATFORM: &str = if cfg!(target_os = "macos") {
    "macOS"
} else {
    "windows"
};

/// Calls waiting for the app to run a tool, by request id: where its result goes.
type ToolWaits = Arc<Mutex<HashMap<String, Sender<Result<String, String>>>>>;

// ---- Approvals -------------------------------------------------------------------------

/// Changes waiting for the user's answer, and the kinds they always allow.
struct Approvals {
    /// By approval id: where the answer (approved, remember) goes.
    waiting: Mutex<HashMap<String, Sender<(bool, bool)>>>,
    /// Tools the user said not to ask about again, kept in approvals.txt in the
    /// prototype's folder (`store::data_dir`), a tool a line, as the Mac's server keeps
    /// approvals.json.
    always: Mutex<Vec<String>>,
}

/// How long an approval waits for the user, as on the Mac.
const APPROVAL_SECONDS: u64 = 120;

impl Approvals {
    /// `VIREO_APPROVALS_FILE` puts it elsewhere (vireo-eval starts each run with none).
    fn file() -> PathBuf {
        if let Some(file) = std::env::var_os("VIREO_APPROVALS_FILE") {
            return PathBuf::from(file);
        }
        store::data_dir().join("approvals.txt")
    }

    fn load() -> Vec<String> {
        std::fs::read_to_string(Self::file())
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect()
    }

    fn always_allow(&self, tool: &str) {
        let mut always = self.always.lock().unwrap();
        if !always.iter().any(|t| t == tool) {
            always.push(tool.to_string());
            let file = Self::file();
            if let Some(dir) = file.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&file, always.join("\n") + "\n");
            eprintln!("Always allowed from now on: {tool}");
        }
    }
}

// ---- Pairing ---------------------------------------------------------------------------

/// The code shown when the server starts, and the tokens it has given out, kept in
/// server-pairings.txt in the prototype's folder (a token and a device name a line)
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
        store::data_dir().join("server-pairings.txt")
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
        Err((status, e)) => return http::respond(&mut stream, status, &error(&e)),
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
                .unwrap_or(PLATFORM)
                .to_string();
            http::respond(
                &mut stream,
                200,
                &format!(
                    "{{\"protocolVersion\":1,\"device\":{},\"serverPlatform\":{},\"tools\":{}}}",
                    json::string(&device),
                    json::string(PLATFORM),
                    tools::capabilities_json(PLATFORM)
                ),
            );
        }
        ("GET", "/api/skill-library/commands") => {
            let names: Vec<String> = pc_skills(&tools::all_names())
                .iter()
                .map(|s| format!("{{\"name\":{}}}", json::string(&s.name)))
                .collect();
            http::respond(
                &mut stream,
                200,
                &format!("{{\"skills\":[{}]}}", names.join(",")),
            )
        }
        // The assistant's inbox and activity log, which the app polls: empty here.
        ("GET", "/api/assistant/approvals") => {
            http::respond(&mut stream, 200, "{\"approvals\":[]}")
        }
        ("GET", "/api/assistant/activity") => {
            http::respond(&mut stream, 200, "{\"latest\":0,\"entries\":[]}")
        }
        ("POST", "/api/upload") => {
            let id = format!("{:016x}", random_u64());
            let (status, body) = attach::upload(
                request.content_type.as_deref(),
                &request.body,
                &shared.uploads,
                &id,
            );
            http::respond(&mut stream, status, &body);
            if status == 200 {
                prune_uploads(shared);
            }
        }
        ("GET", p) if p.starts_with("/uploads/") => match shared.uploads.file(p) {
            Some((bytes, media_type)) => http::respond_with(&mut stream, 200, media_type, &bytes),
            None => http::respond(&mut stream, 404, &error("That attachment isn't kept here.")),
        },
        ("POST", "/api/chat/completions") => completion(stream, shared, &request.body),
        ("POST", "/api/agent/client-tool-result") => {
            let body = json::parse(&String::from_utf8_lossy(&request.body)).unwrap_or(Value::Null);
            let Some(id) = body.str("id") else {
                return http::respond(&mut stream, 400, &error("No id."));
            };
            let result = if matches!(body.get("ok"), Some(Value::Bool(true))) {
                Ok(body.str("output").unwrap_or("").to_string())
            } else {
                Err(body.str("error").unwrap_or("The tool failed.").to_string())
            };
            match shared.tool_results.lock().unwrap().remove(id) {
                Some(waiting) => {
                    let _ = waiting.send(result);
                    http::respond(&mut stream, 200, "{\"ok\":true}");
                }
                None => http::respond(
                    &mut stream,
                    404,
                    &error("No call is waiting for that result."),
                ),
            }
        }
        ("POST", "/api/agent/confirm") => {
            let body = json::parse(&String::from_utf8_lossy(&request.body)).unwrap_or(Value::Null);
            let Some(id) = body.str("id") else {
                return http::respond(&mut stream, 400, &error("No id."));
            };
            let approved = matches!(body.get("approved"), Some(Value::Bool(true)));
            let remember = matches!(body.get("remember"), Some(Value::Bool(true)));
            match shared.approvals.waiting.lock().unwrap().remove(id) {
                Some(waiting) => {
                    let _ = waiting.send((approved, remember));
                    http::respond(&mut stream, 200, "{\"success\":true}");
                }
                None => http::respond(
                    &mut stream,
                    404,
                    &error("Confirmation request not found or already resolved."),
                ),
            }
        }
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
    // The documents attached to user messages: the message's index, the file's name and text.
    let mut documents: Vec<(usize, String, String)> = Vec::new();
    // Attachments no longer kept: the message's index and the path.
    let mut gone = Vec::new();
    for m in body.array("messages") {
        let (Some(role), Some(content)) = (m.str("role"), m.str("content")) else {
            continue;
        };
        match role {
            "system" => system = Some(content.to_string()),
            "user" => {
                // The app sends each message's attachment path every turn.
                if let Some(path) = m.str("image").filter(|p| p.starts_with("/uploads/")) {
                    match shared.uploads.by_path(path) {
                        Some((name, text)) => documents.push((messages.len(), name, text)),
                        None => gone.push((messages.len(), path.to_string())),
                    }
                }
                messages.push(Message {
                    role: "user".into(),
                    content: content.to_string(),
                });
            }
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
    add_documents(shared, &body, &mut messages, documents, gone, max_tokens);

    let run_id = format!("run-{}", shared.next_run.fetch_add(1, Ordering::Relaxed));
    let stop = Arc::new(AtomicBool::new(false));
    shared
        .runs
        .lock()
        .unwrap()
        .insert(run_id.clone(), stop.clone());
    let (out, events) = channel();
    let client_tools = body
        .array("clientTools")
        .iter()
        .filter_map(|t| match t {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    // The files the chat may reach on this Mac, for the PC tools the app doesn't run.
    let folders = body
        .array("folders")
        .iter()
        .filter_map(|f| match f {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let files =
        files::Access::from_request(body.str("access"), folders).map(files::Files::for_user);
    let job = Job {
        system,
        messages,
        generation: generation.cloned(),
        max_tokens,
        stop: stop.clone(),
        out,
        client_tools,
        files,
        tool_results: shared.tool_results.clone(),
        approvals: shared.approvals.clone(),
    };
    if shared.jobs.lock().unwrap().send(Work::Reply(job)).is_err() {
        shared.runs.lock().unwrap().remove(&run_id);
        return http::respond(&mut stream, 500, &error("The engine has stopped."));
    }
    stream_reply(&mut stream, &run_id, &events, &stop);
    shared.runs.lock().unwrap().remove(&run_id);
}

/// Writes the attached documents into their messages. The last message's comes from
/// what was kept of its upload, else the request's `extractedText` or `docId`. They share
/// what the context has room for after the reply, the system prompt, tools and messages:
/// the newest document first, and earlier ones are cut, or left out with a note.
fn add_documents(
    shared: &Shared,
    body: &Value,
    messages: &mut [Message],
    mut documents: Vec<(usize, String, String)>,
    gone: Vec<(usize, String)>,
    max_tokens: usize,
) {
    let last = messages.len() - 1;
    if documents.last().is_none_or(|d| d.0 != last) {
        let name = body
            .str("image")
            .and_then(|p| p.rsplit('/').next())
            .map(|f| f.split_once('-').map_or(f, |(_, n)| n).to_string())
            .unwrap_or_default();
        if let Some(text) = body.str("extractedText").filter(|t| !t.trim().is_empty()) {
            documents.push((last, name, text.to_string()));
        } else if let Some((name, text)) = body.str("docId").and_then(|id| shared.uploads.by_id(id))
        {
            documents.push((last, name, text));
        }
    }
    for (i, path) in gone {
        if !documents.iter().any(|d| d.0 == i) {
            messages[i].content = attach::gone(&messages[i].content, &path);
        }
    }
    if documents.is_empty() {
        return;
    }
    // About three bytes of text a token, kept on the safe side.
    let context = shared.info.context;
    let room = context.saturating_sub(max_tokens.min(context / 2) + 2048) * 3;
    let others: usize = messages.iter().map(|m| m.content.len()).sum();
    let mut budget = room.saturating_sub(others);
    for (i, name, text) in documents.iter().rev() {
        let message = &mut messages[*i];
        message.content = attach::with_document(&message.content, name, text, budget);
        budget = budget.saturating_sub(text.len().min(budget));
    }
}

/// Removes attachments no chat has used for a while (see [`attach::KEEP_FOR`]).
fn prune_uploads(shared: &Shared) {
    let removed = shared.uploads.prune(
        std::time::SystemTime::now(),
        attach::KEEP_FOR,
        attach::KEEP_BYTES,
    );
    if removed > 0 {
        eprintln!("Removed {removed} attachments no chat has used for a while.");
    }
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
            Out::Event(event) => send(&event),
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
    /// The tools the app runs on the PC (its `clientTools`).
    client_tools: Vec<String>,
    /// What the chat may reach on this Mac (its `access`, and `folders`), for the PC tools
    /// the server runs itself; `None` for none.
    files: Option<files::Files>,
    tool_results: ToolWaits,
    approvals: Arc<Approvals>,
}

enum Out {
    Text(String),
    /// An event as it's sent: a tool call, a request for the app to run one, its result.
    Event(String),
    Done {
        tokens: usize,
        seconds: f64,
    },
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
    /// Qwen: its DeltaNet layers where scoring started, to go back to after each try.
    scratch: Option<Qwen35Checkpoint>,
    /// A small session apart, for questions the server asks the model itself (which tools a
    /// request needs), so the chat's own cache is left as it is.
    picker: Option<Session>,
    /// Each skill's share of yes for a request that fits no skill, by its question: this
    /// model's bias, scored once (`pick_skill`).
    skill_controls: HashMap<String, f64>,
    /// Where the last prompt's history ends (its tokens are still the session's first):
    /// Gemma's chat is saved up to there, which the next turn and a retry both start with,
    /// since its saved state can only be loaded whole.
    history_end: usize,
}

/// The context when the model's file doesn't say what it was trained for.
const DEFAULT_CONTEXT: usize = 8192;
/// The most `--context auto` picks, however much memory is free: further than the engine
/// has been checked against llama.cpp.
const AUTO_CONTEXT_MAX: usize = 32768;
/// GPU memory `--context auto` leaves free: for the display and other programs on a PC
/// whose GPU also drives the screen (and a tenth of what's free on top).
const GPU_HEADROOM: usize = 1 << 30;

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
    /// As many tokens as the GPU's free memory holds in a session's caches, leaving
    /// headroom, up to `trained` (and `AUTO_CONTEXT_MAX`), in whole thousands.
    fn automatic_context(&self, trained: usize) -> Result<usize, String> {
        let probe = self.session(1024)?;
        let per = self.cache_bytes_per_position(&probe).max(1);
        drop(probe);
        let free = match self {
            Model::Llama(m) => m.gpu().free_memory(),
            Model::Qwen35(m) => m.gpu().free_memory(),
            Model::Gemma4(m) => m.gpu().free_memory(),
        };
        let room = free.saturating_sub(GPU_HEADROOM + free / 10);
        let n = (room / per).min(trained).min(AUTO_CONTEXT_MAX) / 1024 * 1024;
        if n < 2048 {
            return Err(format!(
                "The GPU has {:.1} GB free once the model is loaded: not enough for chats. Close other programs using it, or choose a smaller model.",
                free as f64 / 1e9
            ));
        }
        Ok(n)
    }

    /// Empties a session (the picker's, before each question).
    fn empty(&self, s: &mut Session) -> Result<(), String> {
        match (self, s) {
            (Model::Llama(_), Session::Llama(s)) => s.truncate(0),
            (Model::Gemma4(_), Session::Gemma4(s)) => {
                s.truncate(0);
            }
            (Model::Qwen35(m), Session::Qwen35(s)) => m.reset(s).map_err(|e| e.to_string())?,
            _ => unreachable!("a session belongs to its model"),
        }
        Ok(())
    }

    fn cache_bytes_per_position(&self, s: &Session) -> usize {
        match s {
            Session::Llama(s) => s.cache_bytes_per_position(),
            Session::Qwen35(s) => s.cache_bytes_per_position(),
            Session::Gemma4(s) => s.cache_bytes_per_position(),
        }
    }

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

    /// Reads `ids`; every one's logits, a row each.
    fn read_all(&self, s: &mut Session, ids: &[u32]) -> Result<Vec<f32>, String> {
        let r = match (self, s) {
            (Model::Llama(m), Session::Llama(s)) => m
                .forward_batch(s, ids, Logits::All)
                .map(|l| l.map(<[f32]>::to_vec)),
            (Model::Qwen35(m), Session::Qwen35(s)) => m
                .forward_batch(s, ids, Logits::All)
                .map(|l| l.map(<[f32]>::to_vec)),
            (Model::Gemma4(m), Session::Gemma4(s)) => m
                .forward_batch(s, ids, Logits::All)
                .map(|l| l.map(<[f32]>::to_vec)),
            _ => unreachable!("a session belongs to its model"),
        };
        r.map_err(|e| e.to_string())?
            .ok_or_else(|| "no logits".into())
    }

    /// How likely the model finds `ids` next, given `logits` for the first of them: the
    /// sum of each token's log-probability. The session is left as it was.
    fn score(&self, kept: &mut Kept, logits: &[f32], ids: &[u32]) -> Result<f64, String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        let mut total = log_probability(logits, ids[0]);
        if ids.len() > 1 {
            let len = kept.tokens.len();
            if let (Model::Qwen35(m), Session::Qwen35(s)) = (self, &kept.session) {
                m.save(s, &mut kept.scratch).map_err(e)?;
            }
            let rows = self.read_all(&mut kept.session, &ids[..ids.len() - 1])?;
            let vocab = logits.len();
            for (i, &id) in ids[1..].iter().enumerate() {
                total += log_probability(&rows[i * vocab..(i + 1) * vocab], id);
            }
            match (self, &mut kept.session) {
                (Model::Llama(_), Session::Llama(s)) => s.truncate(len),
                // A few tokens back: well inside what the session keeps.
                (Model::Gemma4(_), Session::Gemma4(s)) => {
                    if s.truncate(len) != len {
                        return Err(
                            "The chat's cache couldn't go back to before the choices.".into()
                        );
                    }
                }
                (Model::Qwen35(m), Session::Qwen35(s)) => {
                    m.restore(s, kept.scratch.as_ref().unwrap()).map_err(e)?
                }
                _ => unreachable!("a session belongs to its model"),
            }
        }
        Ok(total)
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
            // Back to the shared start, or to the beginning when the sliding-window
            // layers no longer hold the window before it.
            (Model::Gemma4(_), Session::Gemma4(s)) => {
                let kept_len = s.truncate(shared);
                kept.history_end = kept.history_end.min(kept_len);
                kept_len
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
        kept.history_end = mark;
        Ok((logits, from))
    }

    /// Reads `ids` after what the session holds; the last one's logits.
    fn feed(&self, kept: &mut Kept, ids: &[u32]) -> Result<Vec<f32>, String> {
        let logits = self.read(&mut kept.session, ids)?;
        kept.tokens.extend_from_slice(ids);
        Ok(logits)
    }

    /// Takes the session back to `kept.tokens`, dropping what was read past them (a
    /// draft's guesses): a few tokens, well inside what the session keeps.
    fn rewind(&self, kept: &mut Kept) -> Result<(), String> {
        let len = kept.tokens.len();
        let ok = match (self, &mut kept.session) {
            (Model::Gemma4(_), Session::Gemma4(s)) => s.truncate(len) == len,
            (Model::Llama(_), Session::Llama(s)) => {
                s.truncate(len);
                true
            }
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err("The chat's cache couldn't go back to before the draft's guesses.".into())
        }
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
            Session::Gemma4(_) => kept.history_end,
            Session::Llama(_) => kept.tokens.len(),
        }
    }

    /// How many of `ids` a saved chat of `tokens` could give: as many as they share, but
    /// Qwen's only whole (its DeltaNet state is for the end), and Gemma's only whole once
    /// it's longer than the sliding window (its older positions there weren't kept).
    fn usable(&self, tokens: &[u32], ids: &[u32], context: usize) -> usize {
        let shared = shared_start(tokens, ids);
        let n = match self {
            Model::Qwen35(_) if shared != tokens.len() => 0,
            Model::Gemma4(m)
                if shared != tokens.len() && tokens.len() > m.config.sliding_window =>
            {
                0
            }
            _ => shared,
        };
        if n > context { 0 } else { n }
    }

    /// The chat in `kept`, to save: every token (Qwen: up to its checkpoint; Gemma: to its
    /// last history's end), and the state after them. `None` when there's nothing to save.
    fn export(&self, kept: &Kept) -> Result<Option<(Vec<u32>, SavedState)>, String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        let state = match &kept.session {
            Session::Llama(s) => s.export().map_err(e)?,
            Session::Gemma4(s) => s.export(kept.history_end).map_err(e)?,
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
        kept.history_end = tokens.len();
        kept.tokens = tokens;
        Ok(())
    }

    /// Empties `kept`.
    fn clear(&self, kept: &mut Kept) -> Result<(), String> {
        match (self, &mut kept.session) {
            (Model::Llama(_), Session::Llama(s)) => s.truncate(0),
            (Model::Gemma4(_), Session::Gemma4(s)) => {
                s.truncate(0);
            }
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                m.reset(s).map_err(|e| e.to_string())?;
                kept.checkpoint = None;
            }
            _ => unreachable!("a session belongs to its model"),
        }
        kept.tokens.clear();
        kept.history_end = 0;
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
    context: Option<usize>,
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
    // The default mode: 8-bit activations on CUDA, f16 on the Mac's neural accelerators.
    let mode = if precision == Precision::Exact {
        "exact"
    } else if cfg!(target_os = "macos") {
        "f16"
    } else {
        "8-bit"
    };
    eprintln!("Loaded {name} in {:.1?}.", started.elapsed());
    let trained = file
        .arch_u64("context_length")
        .map_or(DEFAULT_CONTEXT, |n| n as usize);
    let context = match context {
        Some(n) => {
            if n > trained {
                eprintln!(
                    "{name} was trained for {trained} tokens; {n} may read poorly past that."
                );
            }
            n
        }
        None => match model.automatic_context(trained) {
            Ok(n) => n,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        },
    };
    // The session for the server's life, made now so a context too large fails here.
    let mut kept: Option<Kept> = match model.session(context) {
        Ok(session) => {
            eprintln!(
                "Chats keep up to {context} tokens ({:.1} GB of the GPU's memory for their cache).",
                (model.cache_bytes_per_position(&session) * context) as f64 / 1e9
            );
            Some(Kept {
                session,
                tokens: Vec::new(),
                checkpoint: None,
                scratch: None,
                picker: None,
                skill_controls: HashMap::new(),
                history_end: 0,
            })
        }
        Err(e) => {
            let _ = ready.send(Err(format!(
                "Couldn't make room for {context}-token chats on the GPU ({e}). Try a smaller --context."
            )));
            return;
        }
    };
    let mut drafter = match &options().draft {
        Some(draft_path) if family == Family::Gemma4 => {
            let started = Instant::now();
            match draft::Drafter::load(
                draft_path,
                &tokenizer,
                precision,
                context,
                options().draft_tokens,
            ) {
                Ok(d) => {
                    eprintln!(
                        "Drafting with {}, {} tokens a round (loaded in {:.1?}).",
                        d.name,
                        d.guesses,
                        started.elapsed()
                    );
                    Some(d)
                }
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            }
        }
        Some(_) => {
            let _ = ready.send(Err("--draft works with Gemma 4 models only.".into()));
            return;
        }
        None => None,
    };
    let _ = ready.send(Ok(ModelInfo {
        model: name,
        runtime: format!("Vireo engine ({gpu_name}, {mode})"),
        context,
    }));

    // The vocabulary as a byte trie, for writing tool calls under their grammars.
    let extra: Vec<u32> = tools::call_tokens(family)
        .iter()
        .filter_map(|t| tokenizer.token_id(t))
        .collect();
    let trie = TokenTrie::with_extra(&tokenizer, &extra);
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
                        scratch: None,
                        picker: None,
                        skill_controls: HashMap::new(),
                        history_end: 0,
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
            &trie,
            family,
            context,
            seed,
            &job,
            drafter.as_mut(),
        ) {
            // Whatever the session held is unknown now: the next reply starts afresh.
            kept = None;
            let _ = job.out.send(Out::Error(e));
        }
    }
}

/// Saves the chat in `kept` to `store`, if there's one worth saving.
fn stash(model: &Model, kept: &Kept, store: &mut store::Store) {
    if model.saveable(kept) < MIN_SAVED {
        return;
    }
    match model.export(kept) {
        Ok(Some((tokens, state))) => store.put(tokens, state),
        Ok(None) => {}
        Err(e) => eprintln!("Couldn't save the chat's cache: {e}"),
    }
}

/// The fewest tokens worth saving a chat's cache for: a shorter chat reads again in a
/// few tenths of a second, about what loading it takes, and its file would still be
/// megabytes (Qwen's DeltaNet state alone is 18 MB on the 0.8B).
const MIN_SAVED: usize = 1024;

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
    trie: &TokenTrie,
    family: Family,
    context: usize,
    seed: u64,
    job: &Job,
    mut drafter: Option<&mut draft::Drafter>,
) -> Result<(), String> {
    let today = chat::today();
    let max_tokens = job.max_tokens.min(context / 2);
    let mut offered = tools::offered(&job.client_tools, job.files.is_some());
    if options().scored_groups
        && !offered.is_empty()
        && let Some(request) = job.messages.last()
    {
        let groups = pick_groups(model, kept, tokenizer, family, &request.content)?;
        offered.retain(|t| groups.contains(&t.group));
    }
    // The tools a call may name: all those offered, less the ones that change files
    // when the request doesn't ask for a change.
    let mut callable = offered.clone();
    if options().changes != Changes::Offered
        && offered.iter().any(|t| t.asks.is_some())
        && let Some(request) = job.messages.last()
        && !asks_for_change(model, kept, tokenizer, family, &request.content)?
    {
        callable.retain(|t| t.asks.is_none());
        if options().changes == Changes::Declared {
            offered.retain(|t| t.asks.is_none());
        }
    }
    // The skill the request asks for, if one fits: its instructions come in as the result
    // of a use_skill call, written after the request, before the model writes.
    let mut skill_ids = Vec::new();
    if options().skills
        && !offered.is_empty()
        && let Some(request) = job.messages.last()
        && let Some(skill) = pick_skill(model, kept, tokenizer, family, &offered, &request.content)?
    {
        let call = tools::Call {
            tool: "use_skill".into(),
            args: vec![("name".into(), tools::Arg::Text(skill.name.clone()))],
        };
        let result = skills::instructions(&skill);
        let (before, body, after) = tools::after_call(family, "use_skill", &result);
        skill_ids = tokenizer.encode(&tools::call_text(family, &call), false, true);
        skill_ids.extend(tokenizer.encode(&before, false, true));
        skill_ids.extend(tokenizer.encode(&body, false, false));
        skill_ids.extend(tokenizer.encode(&after, false, true));
        let _ = job.out.send(Out::Event(format!(
            "{{\"type\":\"tool_call\",\"tool\":\"use_skill\",\"args\":{}}}",
            call.args_json()
        )));
        let _ = job.out.send(Out::Event(format!(
            "{{\"type\":\"tool_result\",\"tool\":\"use_skill\",\"result\":{}}}",
            json::string(&preview(&result))
        )));
        eprintln!("Skill: {} ({} tokens)", skill.name, skill_ids.len());
    }
    let declared = tools::declarations(family, &offered);
    // With tools, the system prompt opens with what they're for.
    let system = match (offered.is_empty(), job.system.as_deref().map(str::trim)) {
        (true, s) => s.map(str::to_string),
        (false, Some(s)) if !s.is_empty() => Some(format!("{}\n\n{s}", tools::PREAMBLE)),
        (false, _) => Some(tools::PREAMBLE.to_string()),
    };
    // The chat's latest messages that fit with room for the reply.
    let mut first = 0;
    let opening = tokenizer.encode(chat::opening(family), false, true);
    let (ids, mark) = loop {
        let text = chat::history(
            family,
            system.as_deref(),
            &declared,
            &job.messages[first..],
            &today,
        );
        if std::env::var_os("VIREO_DEBUG_PROMPT").is_some() {
            eprintln!("[prompt] {text}{}", chat::opening(family));
        }
        let mut ids = tokenizer.encode(&text, false, true);
        let mark = ids.len();
        ids.extend_from_slice(&opening);
        ids.extend_from_slice(&skill_ids);
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
    if let Some(d) = drafter.as_deref_mut() {
        d.sync(&kept.tokens)?;
    }
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
    // Tool calls: the grammar they're written under, and how one starts.
    let grammar = Grammar::new(&tools::call_grammar(family, &callable));
    let start = tools::call_start(family);
    let start_token = tokenizer.token_id(start.token);
    let mut last_call: Option<tools::Call> = None;
    let (mut calls, mut call_tokens) = (0usize, 0usize);
    let mut turn_start = true;
    // Once calls are used up (or the model repeats one), it may only answer: the tokens
    // that start a call are taken away from it.
    let mut may_call = !offered.is_empty();
    let mut brace_tokens: Option<Vec<u32>> = None;
    // Drafted tokens the model has read past `kept.tokens`, each with its logits after.
    let mut ahead: std::collections::VecDeque<(u32, Vec<f32>)> = Default::default();
    // With a draft: the last token picked, in `kept.tokens` but not read yet. It's read
    // with the next round's guesses, in the same pass, rather than on its own.
    let mut unread: Option<u32> = None;
    if let Some(d) = drafter.as_deref_mut() {
        d.start_reply();
    }
    // After a tool's result (or the skill written in) the model must say something: its
    // end-of-turn tokens are taken away until it writes one token. Gemma 4 E4B, with a
    // longer list of tools, ended its turn on the result three times in 39 cases,
    // answering nothing.
    let mut answer_due = !skill_ids.is_empty();
    let mut end_tokens: Option<Vec<u32>> = None;
    while generated.len() + call_tokens < max_tokens && !job.stop.load(Ordering::Relaxed) {
        let room = (max_tokens - generated.len() - call_tokens)
            .min(context.saturating_sub(kept.tokens.len() + 64));
        // Out of guesses: a new round (after the unread token, if there's one), read by
        // the model in one pass with it.
        if ahead.is_empty() {
            match drafter.as_deref_mut() {
                Some(d) if room > 1 => {
                    let guesses = d.guess(&kept.tokens, d.guesses.min(room - 1))?;
                    let first = usize::from(unread.is_some());
                    let batch: Vec<u32> = unread
                        .take()
                        .into_iter()
                        .chain(guesses.iter().copied())
                        .collect();
                    let rows = model.read_all(&mut kept.session, &batch)?;
                    let vocab = rows.len() / batch.len();
                    if first == 1 {
                        logits.copy_from_slice(&rows[..vocab]);
                    }
                    ahead.extend(guesses.iter().enumerate().map(|(i, &g)| {
                        (
                            g,
                            rows[(first + i) * vocab..(first + i + 1) * vocab].to_vec(),
                        )
                    }));
                }
                _ => {
                    if let Some(u) = unread.take() {
                        model.next(&mut kept.session, u, &mut logits)?;
                    }
                }
            }
        }
        if !may_call && !offered.is_empty() {
            if let Some(t) = start_token {
                logits[t as usize] = f32::NEG_INFINITY;
            }
            if start.brace && turn_start {
                let braces = brace_tokens.get_or_insert_with(|| {
                    let mut b = Vec::new();
                    (0..tokenizer.vocab_size() as u32)
                        .filter(|&id| {
                            b.clear();
                            tokenizer.token_bytes(id, false, &mut b);
                            b.first() == Some(&b'{')
                        })
                        .collect()
                });
                for &t in braces.iter() {
                    logits[t as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if answer_due {
            let ends = end_tokens.get_or_insert_with(|| {
                (0..tokenizer.vocab_size() as u32)
                    .filter(|&id| tokenizer.is_end_of_generation(id))
                    .collect()
            });
            for &t in ends.iter() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
        }
        let next = sampler.pick(&logits, &generated);
        if tokenizer.is_end_of_generation(next) {
            break;
        }
        answer_due = false;
        if debug_calls() && turn_start {
            eprintln!(
                "[calls] turn starts with {:?}",
                tokenizer.decode(&[next], true)
            );
        }
        let by_token = Some(next) == start_token;
        let by_brace = start.brace && turn_start && {
            let mut b = Vec::new();
            tokenizer.token_bytes(next, false, &mut b);
            b.first() == Some(&b'{')
        };
        let opens_call = may_call && (by_token || by_brace);
        // The guess is taken when it's the token picked, for plain text; otherwise the
        // model goes back to before the guesses.
        let guessed = !opens_call && ahead.front().is_some_and(|(g, _)| *g == next);
        if !guessed && !ahead.is_empty() {
            ahead.clear();
            model.rewind(kept)?;
        }
        if opens_call {
            flush_text(job, &mut pending);
            if by_token {
                // The call's own start token is the model's; Llama's brace is the grammar's.
                model.next(&mut kept.session, next, &mut logits)?;
                kept.tokens.push(next);
                call_tokens += 1;
            }
            let call_started = Instant::now();
            let text = write_call(
                model,
                kept,
                tokenizer,
                trie,
                &grammar,
                &mut logits,
                &mut call_tokens,
                context,
            )?;
            let call = tools::parse_call(family, &text)
                .ok_or_else(|| format!("Couldn't read back a tool call: {text:?}"))?;
            calls += 1;
            let args = call.args_json();
            let written = call_started.elapsed();
            let _ = job.out.send(Out::Event(format!(
                "{{\"type\":\"tool_call\",\"tool\":{},\"args\":{args}}}",
                json::string(&call.tool)
            )));
            // The same call again, with the same arguments, would get the same answer: a
            // small model can loop on one. It isn't run again, and the model must answer.
            let repeated = last_call.as_ref() == Some(&call);
            let result = if repeated {
                "This is the same call as the last one, and its result is above. Answer the user from it now."
                    .to_string()
            } else {
                match run_tool(&call, &offered, job) {
                    Some(r) => r,
                    None => break, // stopped while the app ran it
                }
            };
            if repeated || calls == MAX_CALLS {
                may_call = false;
            }
            last_call = Some(call.clone());
            eprintln!(
                "Tool: {} {args} (written in {:.0} ms), {} characters back.",
                call.tool,
                written.as_secs_f64() * 1000.0,
                result.len()
            );
            let _ = job.out.send(Out::Event(format!(
                "{{\"type\":\"tool_result\",\"tool\":{},\"result\":{}}}",
                json::string(&call.tool),
                json::string(&preview(&result))
            )));
            let (before, body, after) = tools::after_call(family, &call.tool, &shaped(&result));
            let mut ids = tokenizer.encode(&before, false, true);
            ids.extend(tokenizer.encode(&body, false, false));
            ids.extend(tokenizer.encode(&after, false, true));
            if kept.tokens.len() + ids.len() + 64 > context {
                return Err("The tool's result doesn't fit in this chat's context.".into());
            }
            logits = model.feed(kept, &ids)?;
            turn_start = true;
            answer_due = true;
            continue;
        }
        turn_start = false;
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
        if guessed {
            logits = ahead.pop_front().unwrap().1;
            if let Some(d) = drafter.as_deref_mut() {
                d.taken += 1;
            }
        } else if drafter.is_some() {
            unread = Some(next);
        } else {
            model.next(&mut kept.session, next, &mut logits)?;
        }
        kept.tokens.push(next);
    }
    // Guesses read past the end of the reply (it ended, or was stopped): not the chat's.
    // Nor a last token never read: the next turn reads it with its prompt.
    if !ahead.is_empty() {
        model.rewind(kept)?;
    }
    if unread.is_some() {
        kept.tokens.pop();
    }
    flush_text(job, &mut pending);
    let seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "Reply: {} prompt tokens ({reused} kept, {} read in {prompt_seconds:.2}s, {:.0}/s), {} tokens in {seconds:.2}s ({:.1}/s), {calls} tool calls{}{}.",
        ids.len(),
        ids.len() - reused,
        (ids.len() - reused) as f64 / prompt_seconds.max(1e-9),
        generated.len() + call_tokens,
        (generated.len() + call_tokens) as f64 / seconds.max(1e-9),
        if job.stop.load(Ordering::Relaxed) {
            ", stopped"
        } else {
            ""
        },
        drafter.as_deref().map_or(String::new(), |d| format!(
            ", {} of {} guesses taken ({:.0}%, {:.2}s guessing)",
            d.taken,
            d.guessed,
            100.0 * d.taken as f64 / d.guessed.max(1) as f64,
            d.seconds
        ))
    );
    let _ = job.out.send(Out::Done {
        tokens: generated.len() + call_tokens,
        seconds,
    });
    Ok(())
}

/// The tool groups a request needs, asked of the model in its picker session: one pass
/// over a short multiple-choice question, its answer letters scored. Groups at least
/// `GROUP_SHARE` likely are offered (the likeliest always, unless it's "none").
/// The skills this PC can follow with these tools (`skills::usable`).
fn pc_skills(tools: &[&str]) -> Vec<skills::Skill> {
    let built_in = skills::built_in_folder();
    let user = skills::user_folder();
    skills::load(built_in.as_deref(), user.as_deref())
        .into_iter()
        .filter(|s| skills::usable(s, tools))
        .collect()
}

/// The skill a request asks for: the one it names (an app's slash command), or the one
/// that fits best by scoring (`skills::fit_question`, a pass each), if its share of " yes"
/// is at least `SKILL_SHARE` and at least `OVER_CONTROL` above the share it gets for a
/// request that fits no skill (`skills::CONTROL`, scored once per skill while this model
/// runs). A model whose yes means nothing (Qwen 3.5 0.8B said yes to nearly every skill
/// for every request) so picks none, as before skills, instead of a wrong one.
fn pick_skill(
    model: &Model,
    kept: &mut Kept,
    tokenizer: &Tokenizer,
    family: Family,
    offered: &[tools::Tool],
    request: &str,
) -> Result<Option<skills::Skill>, String> {
    const SKILL_SHARE: f64 = 0.5;
    const OVER_CONTROL: f64 = 0.4;
    let names: Vec<&str> = offered.iter().map(|t| t.name).collect();
    let mut skills = pc_skills(&names);
    if let Some(skill) = skills::named(request, &skills) {
        return Ok(Some(skill.clone()));
    }
    let mut best: Option<(f64, usize)> = None;
    for (i, skill) in skills.iter().enumerate() {
        let question = skills::fit_question(skill, request);
        let share = yes_share(model, kept, tokenizer, family, &question, "Fits:")?;
        let control_question = skills::fit_question(skill, skills::CONTROL);
        let control = match kept.skill_controls.get(&control_question) {
            Some(c) => *c,
            None => {
                let c = yes_share(model, kept, tokenizer, family, &control_question, "Fits:")?;
                kept.skill_controls.insert(control_question, c);
                c
            }
        };
        if debug_calls() {
            eprintln!(
                "[calls] skill {}: {share:.2} (control {control:.2})",
                skill.name
            );
        }
        if share >= SKILL_SHARE
            && share - control >= OVER_CONTROL
            && best.is_none_or(|(b, _)| share > b)
        {
            best = Some((share, i));
        }
    }
    Ok(best.map(|(_, i)| skills.swap_remove(i)))
}

/// The share of " yes" against " no" as the next word after `question`, asked as a user
/// turn and answered from `lead` ("Fits:"), in the picker's own session.
fn yes_share(
    model: &Model,
    kept: &mut Kept,
    tokenizer: &Tokenizer,
    family: Family,
    question: &str,
    lead: &str,
) -> Result<f64, String> {
    let question = Message {
        role: "user".into(),
        content: question.to_string(),
    };
    let text =
        chat::history(family, None, "", &[question], &chat::today()) + chat::opening(family) + lead;
    let ids = tokenizer.encode(&text, false, true);
    if kept.picker.is_none() {
        kept.picker = Some(model.session(1024)?);
    }
    let session = kept.picker.as_mut().unwrap();
    model.empty(session)?;
    let logits = model.read(session, &ids[..ids.len().min(1000)])?;
    let score = |word: &str| {
        let id = tokenizer.encode(&format!(" {word}"), false, false)[0];
        f64::from(logits[id as usize])
    };
    Ok(1.0 / (1.0 + (score("no") - score("yes")).exp()))
}

/// Whether a request asks to change files, scored in one pass over a question with worked
/// examples (`tools::change_question`), as `pick_groups` scores groups. "Yes" needs only
/// `CHANGE_SHARE`: hiding a change the user asked for costs more than an approval they
/// can decline.
fn asks_for_change(
    model: &Model,
    kept: &mut Kept,
    tokenizer: &Tokenizer,
    family: Family,
    request: &str,
) -> Result<bool, String> {
    const CHANGE_SHARE: f64 = 0.3;
    let share = yes_share(
        model,
        kept,
        tokenizer,
        family,
        &tools::change_question(request),
        "Changes:",
    )?;
    if debug_calls() {
        eprintln!("[calls] asks for a change: {share:.2}");
    }
    Ok(share >= CHANGE_SHARE)
}

fn pick_groups(
    model: &Model,
    kept: &mut Kept,
    tokenizer: &Tokenizer,
    family: Family,
    request: &str,
) -> Result<Vec<tools::Group>, String> {
    const GROUP_SHARE: f64 = 0.15;
    let question = Message {
        role: "user".into(),
        content: tools::group_question(request),
    };
    // The answer opens with the examples' "Needs:", so its next token is the letter.
    let text = chat::history(family, None, "", &[question], &chat::today())
        + chat::opening(family)
        + "Needs:";
    let ids = tokenizer.encode(&text, false, true);
    if kept.picker.is_none() {
        kept.picker = Some(model.session(1024)?);
    }
    let session = kept.picker.as_mut().unwrap();
    model.empty(session)?;
    let logits = model.read(session, &ids[..ids.len().min(1000)])?;
    let scores: Vec<f64> = tools::GROUP_CHOICES
        .iter()
        .map(|(letter, _)| {
            let id = tokenizer.encode(&format!(" {letter}"), false, false)[0];
            f64::from(logits[id as usize])
        })
        .collect();
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = scores.iter().map(|s| (s - max).exp()).sum();
    let shares: Vec<f64> = scores.iter().map(|s| (s - max).exp() / sum).collect();
    if debug_calls() {
        eprintln!(
            "[calls] groups: files {:.2}, git {:.2}, time {:.2}, none {:.2}",
            shares[0], shares[1], shares[2], shares[3]
        );
    }
    let top = (0..shares.len())
        .max_by(|&a, &b| shares[a].total_cmp(&shares[b]))
        .unwrap();
    let mut groups: Vec<tools::Group> = tools::GROUP_CHOICES
        .iter()
        .zip(&shares)
        .filter(|(_, share)| **share >= GROUP_SHARE)
        .filter_map(|((_, group), _)| *group)
        .collect();
    if let Some(group) = tools::GROUP_CHOICES[top].1
        && !groups.contains(&group)
    {
        groups.push(group);
    }
    // No tools only when that's clear: hiding a tool a request needs costs more than
    // showing one it doesn't.
    if tools::GROUP_CHOICES[top].1.is_none() && shares[top] >= 0.7 {
        groups.clear();
    } else if groups.is_empty() {
        let best = (0..tools::GROUP_CHOICES.len() - 1)
            .max_by(|&a, &b| shares[a].total_cmp(&shares[b]))
            .unwrap();
        groups.extend(tools::GROUP_CHOICES[best].1);
    }
    Ok(groups)
}

/// `ln p(id)` under `logits`.
fn log_probability(logits: &[f32], id: u32) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    let sum: f64 = logits.iter().map(|&x| f64::from(x - max).exp()).sum();
    f64::from(logits[id as usize] - max) - sum.ln()
}

/// Whether to log how tool calls are written (VIREO_DEBUG_CALLS=1).
fn debug_calls() -> bool {
    std::env::var_os("VIREO_DEBUG_CALLS").is_some()
}

/// The most tool calls one reply makes.
const MAX_CALLS: usize = 6;
/// The most tokens one call may take: a path or a search, not an essay.
const MAX_CALL_TOKENS: usize = 1024;
/// The most of a tool's result the model reads, in characters (about 2,000 tokens).
const MAX_RESULT: usize = 8000;

/// Sends the text held back so far (only whole characters are sent as they come).
fn flush_text(job: &Job, pending: &mut Vec<u8>) {
    if !pending.is_empty() {
        let _ = job
            .out
            .send(Out::Text(String::from_utf8_lossy(pending).into_owned()));
        pending.clear();
    }
}

/// Writes a tool call under `grammar`: what the grammar forces goes in without asking the
/// model (read in one pass), and elsewhere the model's likeliest allowed token. The
/// call's text, from just after its start.
#[allow(clippy::too_many_arguments)]
fn write_call(
    model: &Model,
    kept: &mut Kept,
    tokenizer: &Tokenizer,
    trie: &TokenTrie,
    grammar: &Grammar,
    logits: &mut Vec<f32>,
    tokens: &mut usize,
    context: usize,
) -> Result<String, String> {
    let mut m = grammar.matcher();
    let mut text = Vec::new();
    let first = *tokens;
    let room = |kept: &Kept, n: usize| {
        if kept.tokens.len() + n + 64 > context {
            Err("The tool call doesn't fit in this chat's context.".to_string())
        } else {
            Ok(())
        }
    };
    loop {
        let forced = m.forced();
        // Token healing: forced text's last token is left for the model to write, under
        // the grammar, so it can merge it with what follows as its training did (Qwen's
        // "=read", not "=" then "read"). Tokens no text of the model's would contain
        // otherwise push it off course: Qwen 3.5 9B read files with git_diff.
        let mut held: Vec<u8> = Vec::new();
        if !forced.is_empty() {
            let ids = tokenizer.encode(&String::from_utf8_lossy(&forced), false, true);
            let ends = {
                let mut t = m;
                t.advance_all(&forced) && t.is_complete()
            };
            let mut keep = if ends || !options().heal {
                ids.len()
            } else {
                ids.len() - 1
            };
            let mut fed_bytes = Vec::new();
            for &id in &ids[..keep] {
                tokenizer.token_bytes(id, true, &mut fed_bytes);
            }
            if !forced.starts_with(&fed_bytes) {
                // Tokens that don't spell the text exactly: write it all.
                keep = ids.len();
                fed_bytes = forced.clone();
            }
            held = forced[fed_bytes.len()..].to_vec();
            if keep > 0 {
                m.advance_all(&fed_bytes);
                text.extend_from_slice(&fed_bytes);
                room(kept, keep)?;
                *logits = model.feed(kept, &ids[..keep])?;
                *tokens += keep;
            }
        }
        if m.is_complete() {
            break;
        }
        // A branch (a tool's name): each option scored whole, and the likeliest written.
        // (Each is scored with the text held back by healing before it, read as one.)
        let ahead = {
            let mut t = m;
            t.advance_all(&held);
            t
        };
        if options().score_choices
            && let Some(choices) = ahead.scored_options().filter(|c| c.len() > 1)
        {
            let mut best: Option<(f64, Vec<u8>, Vec<u32>)> = None;
            for option in choices {
                let bytes = [held.as_slice(), &option].concat();
                let ids = tokenizer.encode(&String::from_utf8_lossy(&bytes), false, true);
                if ids.is_empty() {
                    continue;
                }
                let score = model.score(kept, logits, &ids)?;
                if debug_calls() {
                    eprintln!(
                        "[calls] score {score:8.3} for {:?}",
                        String::from_utf8_lossy(&bytes)
                    );
                }
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, bytes, ids));
                }
            }
            let (_, bytes, ids) = best.ok_or("No option of a choice could be written.")?;
            if !m.advance_all(&bytes) {
                return Err("A scored option didn't fit its grammar.".into());
            }
            text.extend_from_slice(&bytes);
            room(kept, ids.len())?;
            *logits = model.feed(kept, &ids)?;
            *tokens += ids.len();
            continue;
        }
        let allowed = m.allowed(trie);
        let next = (0..logits.len() as u32)
            .filter(|&id| allowed[id as usize / 64] >> (id % 64) & 1 == 1)
            .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]))
            .ok_or("The tool call's grammar left no way on.")?;
        if debug_calls() {
            let free = (0..logits.len() as u32)
                .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]))
                .unwrap_or(0);
            eprintln!(
                "[calls] after {:?}: the model's {:?}, the grammar's {:?}",
                String::from_utf8_lossy(&text),
                tokenizer.decode(&[free], true),
                tokenizer.decode(&[next], true)
            );
        }
        let mut bytes = Vec::new();
        tokenizer.token_bytes(next, false, &mut bytes);
        if !m.advance_all(&bytes) {
            return Err("A token the grammar allowed didn't fit it.".into());
        }
        text.extend_from_slice(&bytes);
        room(kept, 1)?;
        model.next(&mut kept.session, next, logits)?;
        kept.tokens.push(next);
        *tokens += 1;
        if *tokens - first > MAX_CALL_TOKENS {
            return Err("A tool call ran too long.".into());
        }
    }
    String::from_utf8(text).map_err(|_| "A tool call wasn't text.".to_string())
}

/// Runs a call: the server's own tools here (with the Mac's files when the app doesn't
/// run the PC's tools), the PC's by asking the app and waiting for its answer (`/api/agent/client-tool-result`). The result, or what went wrong as the
/// model should read it. `None` if the reply was stopped meanwhile.
fn run_tool(call: &tools::Call, offered: &[tools::Tool], job: &Job) -> Option<String> {
    let tool = offered.iter().find(|t| t.name == call.tool)?;
    if let Some(ask) = tool.asks {
        // The app checks first that the change can be made (the file is there, the name
        // is free), so nobody is asked to approve a guess. An app that can't check says
        // so, and the approval is asked as before.
        if job.client_tools.iter().any(|t| t == CHECK_CHANGE) {
            let args = format!(
                "{{\"tool\":{},\"args\":{}}}",
                json::string(&call.tool),
                call.args_json()
            );
            if let Err(why) = ask_app(CHECK_CHANGE, &args, job)? {
                return Some(format!("Error: {why}"));
            }
        }
        match approve(call, ask, job) {
            Some(Ok(())) => {}
            Some(Err(why)) => return Some(format!("Error: {why}")),
            None => return None,
        }
    }
    let result = match tool.runs {
        tools::Runs::Server => tools::run_here(call, job.files.as_ref()),
        // The app answers create_docx with the file's path alone; the server words it.
        tools::Runs::App if call.tool == "create_docx" => {
            ask_app(&call.tool, &call.args_json(), job)?
                .map(|path| format!("Word document saved: {}", path.trim()))
        }
        tools::Runs::App => ask_app(&call.tool, &call.args_json(), job)?,
    };
    Some(result.unwrap_or_else(|e| format!("Error: {e}")))
}

/// What the app runs, not a tool the model sees: whether a change could be made, without
/// making it (`PcTools.CheckChange`). The app lists it in `clientTools` when it has it.
const CHECK_CHANGE: &str = "check_change";

/// Asks the app to run one of its tools and waits for the answer. `None` if the reply was
/// stopped meanwhile.
fn ask_app(tool: &str, args_json: &str, job: &Job) -> Option<Result<String, String>> {
    let id = format!("call-{:016x}", random_u64());
    let (tx, rx) = channel();
    job.tool_results.lock().unwrap().insert(id.clone(), tx);
    let _ = job.out.send(Out::Event(format!(
        "{{\"type\":\"client_tool_request\",\"id\":{},\"tool\":{},\"args\":{}}}",
        json::string(&id),
        json::string(tool),
        args_json
    )));
    let asked = Instant::now();
    loop {
        match rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(r) => return Some(r),
            Err(_) if job.stop.load(Ordering::Relaxed) => {
                job.tool_results.lock().unwrap().remove(&id);
                return None;
            }
            Err(_) if asked.elapsed().as_secs() > 120 => {
                job.tool_results.lock().unwrap().remove(&id);
                return Some(Err("The PC didn't answer within two minutes.".into()));
            }
            Err(_) => {}
        }
    }
}

/// Asks the user to approve a change (`confirm_required`, as the Mac's server sends it)
/// and waits for the answer at `/api/agent/confirm`, unless they always allow this kind.
/// `Err` says, for the model, why it won't happen; `None` if the reply was stopped.
fn approve(call: &tools::Call, ask: tools::Ask, job: &Job) -> Option<Result<(), String>> {
    if ask.rememberable && job.approvals.always.lock().unwrap().contains(&call.tool) {
        eprintln!("Approved without asking (always allowed): {}", call.tool);
        return Some(Ok(()));
    }
    let id = format!("confirm-{:016x}", random_u64());
    let (tx, rx) = channel();
    job.approvals.waiting.lock().unwrap().insert(id.clone(), tx);
    let preview = tools::approval_preview(call, |p| on_this_pc(p).exists());
    let remember = if ask.rememberable {
        format!(",\"rememberLabel\":{}", json::string(&ask.remember_label()))
    } else {
        String::new()
    };
    let _ = job.out.send(Out::Event(format!(
        "{{\"type\":\"confirm_required\",\"id\":{},\"tool\":{},\"args\":{},\"category\":\"{}\",\"preview\":{}{remember}}}",
        json::string(&id),
        json::string(&call.tool),
        call.args_json(),
        if ask.rememberable { "automatic" } else { "restricted" },
        json::string(&preview)
    )));
    let asked = Instant::now();
    let answer = loop {
        match rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(answer) => break Some(answer),
            Err(_) if job.stop.load(Ordering::Relaxed) => {
                job.approvals.waiting.lock().unwrap().remove(&id);
                return None;
            }
            Err(_) if asked.elapsed().as_secs() > APPROVAL_SECONDS => {
                job.approvals.waiting.lock().unwrap().remove(&id);
                break None;
            }
            Err(_) => {}
        }
    };
    eprintln!(
        "Approval for {}: {}",
        call.tool,
        match answer {
            Some((true, _)) => "approved",
            Some((false, _)) => "declined",
            None => "no answer",
        }
    );
    Some(match answer {
        Some((true, remember)) => {
            if remember && ask.rememberable {
                job.approvals.always_allow(&call.tool);
            }
            Ok(())
        }
        Some((false, _)) => Err(
            "The user declined this, so it wasn't done. Don't try it another way: tell them it wasn't done."
                .into(),
        ),
        None => Err(format!(
            "Nobody answered the approval within {} minutes, so it wasn't done. Tell the user.",
            APPROVAL_SECONDS / 60
        )),
    })
}

/// A path as the app's tools take it (`PcTools.Resolve`): full, or relative to the home
/// folder. For the approval's wording only: the app checks access when it runs the call.
fn on_this_pc(path: &str) -> PathBuf {
    let p = path.trim().trim_matches('"');
    let p = p
        .strip_prefix("~/")
        .or_else(|| p.strip_prefix("~\\"))
        .unwrap_or(p);
    let path = PathBuf::from(p);
    if path.is_absolute() {
        path
    } else {
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map_or(path.clone(), |home| PathBuf::from(home).join(&path))
    }
}

/// A tool's result as the model reads it: at most `MAX_RESULT` characters, saying how much
/// was left out.
fn shaped(result: &str) -> String {
    let chars = result.chars().count();
    if chars <= MAX_RESULT {
        return result.to_string();
    }
    let kept: String = result.chars().take(MAX_RESULT).collect();
    format!(
        "{kept}\n[... {} more characters not shown]",
        chars - MAX_RESULT
    )
}

/// A result's first few hundred characters, for the step the app shows.
fn preview(result: &str) -> String {
    let short: String = result.chars().take(300).collect();
    if short.len() < result.len() {
        format!("{short}…")
    } else {
        short
    }
}
