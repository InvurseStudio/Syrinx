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
mod chats;
mod draft;
mod files;
mod http;
mod json;
mod memory;
mod sample;
mod skills;
mod stats;
mod store;
mod tasks;
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
use vireo_engine::models::gemma4_cuda::{Gemma4Cuda, assistant as gemma4_assistant};
use vireo_engine::models::llama_cuda::LlamaCuda;
use vireo_engine::models::qwen35_cuda::{Qwen35Checkpoint, Qwen35Cuda};
use vireo_engine::models::{Logits, Precision, SavedState};
use vireo_engine::{GgufFile, Tokenizer};

const USAGE: &str = "\
Usage: vireo-server <model.gguf> [--port <n>] [--context <tokens>|auto] [--exact]
                    [--token <token>] [--with-app] [--chat-cache <folder>|off]
                    [--score-choices on|off] [--tools-shown all|scored] [--heal on|off]
                    [--changes declared|scored|offered] [--skills on|off]
                    [--skill-shortlist on|off]
                    [--draft <small-model.gguf>] [--draft-tokens <n>] [--lookup <n>]
                    [--mtp on|off] [--thinking on|off]

Serves the model on http://127.0.0.1:<port> (3000 by default; 0 for any free port) for
the Vireo Prototype app: in its Settings, connect to \"localhost\" with the code this
prints. --context is the most tokens a chat keeps: auto (the default) is as many as the
GPU's free memory holds once the model is loaded, up to the model's own limit; --exact runs f32
arithmetic. Chats' caches are saved in the prototype's folder (on Windows
%LOCALAPPDATA%\\VireoPrototype, on the Mac ~/Library/Application Support/VireoPrototype),
in chat-cache (or
--chat-cache's folder; off for none), so going back to a chat reads only what's new.

The tools that change files are offered when one scored question says the request asks
for a change (--changes), and stay declared in that chat from then on (its cache is read
again once, not at every such turn). Each waits for the user's approval in the app. The kinds
the user always allows are kept in approvals.txt in that folder.

The app starts it itself with --port 0, --token (a token it accepts without pairing)
and --with-app (stop when the app does: when its end of standard input closes), and
reads the address from the \"Listening on\" line.

--draft has a small model of the same family and vocabulary guess the next tokens (Gemma
4 E2B for 31B), --draft-tokens at a time (2 by default), for the model to check in one
pass: every token is still the model's own pick. Gemma 4 only. Without one, --lookup
guesses up to <n> tokens a round from the chat itself (what followed the last few
tokens where they appeared before; 3 by default, 0 for none), checked the same way.
Gemma 4, and Qwen 3.5 on the Mac. A Qwen model with its own multi-token-prediction
layer (Qwen 3.8 27B) guesses with that instead (--mtp, on by default on the Mac), and a
Gemma 4 with Google's draft model (gemma-4-31B-it-assistant, or the E2B's or E4B's;
safetensors) beside it or in the Mac app's DraftModels folder with that, or with --draft's
folder. On CUDA it guesses only where it's sure, and the chat's long runs are still taken.

--thinking is what a request gets when it doesn't say (\"thinking\": true or false): off by
default. On, Gemma 4 and Qwen think before they answer and after each tool's result (at
most 2,000 tokens a thought), sent to the app as \"thinking\" events apart from the
answer, and Muse reasons at high strength.";

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
    /// change, and from then on declared for the rest of that chat (callable only when
    /// asked for, as `scored`), so the chat's cached start changes once at most; `scored`, declared always but callable only then (the prompt stays the
    /// same, but the longer tool list cost small models on other requests, and a model
    /// kept from a change it wanted called another tool); `offered`, always. Measured
    /// 2026-11-11 (PLAN.md).
    changes: Changes,
    /// Skills picked by scoring and written in as a `use_skill` call (`--skills on`, the
    /// default), or none (`off`).
    skills: bool,
    /// With more than two skills, one question about them all first (each skill's name
    /// scored as its answer) and only the two likeliest asked about one by one (`--skill-shortlist on`, the default); `off` asks
    /// about every skill, as before 2026-11-12. Measured then (PLAN.md).
    skill_shortlist: bool,
    /// A draft model to guess tokens with (`--draft`), and how many a round.
    draft: Option<String>,
    draft_tokens: usize,
    /// Guesses a round taken from the chat itself (`--lookup <n>`, 0 for none), for a
    /// model that can go back a few tokens (Gemma 4, Llama) and has no draft model.
    lookup: usize,
    /// Qwen's multi-token-prediction layer as the draft, where the model has one
    /// (`--mtp on`, the default; `off` for none).
    mtp: bool,
    /// Thinking for a request that doesn't say (`--thinking on`; off by default).
    thinking: bool,
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
    let mut skill_shortlist = true;
    let mut draft = None;
    let mut draft_tokens = draft::DEFAULT_GUESSES;
    let mut lookup = draft::DEFAULT_LOOKUP;
    let mut mtp = true;
    let mut thinking = false;
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
            "--skill-shortlist" if matches!(value.map(String::as_str), Some("on" | "off")) => {
                skill_shortlist = value.unwrap() == "on";
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
            "--lookup"
                if value
                    .and_then(|v| v.parse::<usize>().ok())
                    .is_some_and(|n| n <= 8) =>
            {
                lookup = value.unwrap().parse().unwrap();
                i += 1;
            }
            "--mtp" if matches!(value.map(String::as_str), Some("on" | "off")) => {
                mtp = value.map(String::as_str) == Some("on");
                i += 1;
            }
            "--thinking" if matches!(value.map(String::as_str), Some("on" | "off")) => {
                thinking = value.map(String::as_str) == Some("on");
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
        skill_shortlist,
        draft,
        draft_tokens,
        lookup,
        mtp,
        thinking,
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
    tasks::set_port(port);
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
    stats::start();
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
            stats::refuse(&peer(&stream), "Wrong pairing code");
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

    // A link task's link, opened by another app on this computer: its token is its key.
    if let Some(token) = path.strip_prefix("/api/tasks/link/")
        && matches!(method, "GET" | "POST")
    {
        let sent = String::from_utf8_lossy(&request.body);
        return match tasks::open_link(token, &sent) {
            Some(name) => http::respond(
                &mut stream,
                200,
                &format!("{{\"ok\":true,\"task\":{}}}", json::string(&name)),
            ),
            None => http::respond(&mut stream, 404, &error("No task has that link.")),
        };
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
        stats::refuse(
            &peer(&stream),
            if request.bearer.is_some() {
                "Unknown token"
            } else {
                "Not paired"
            },
        );
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
        ("GET", "/api/stats") => {
            let i = &shared.info;
            http::respond(
                &mut stream,
                200,
                &stats::json(&i.model, &i.runtime, i.context),
            );
        }
        ("POST", "/api/chat/completions") => {
            let app = request.bearer.is_some()
                && shared.pairing.lock().unwrap().app_token == request.bearer;
            let source = stats::Source::of(
                app,
                device.as_deref().unwrap_or(""),
                request.user_agent.as_deref(),
            );
            completion(stream, shared, &request.body, source)
        }
        // Tasks (`tasks.rs`): the app lists them, and asks which are due to run.
        ("GET", "/api/tasks") => http::respond(&mut stream, 200, &tasks::list_json()),
        ("GET", "/api/tasks/due") => http::respond(&mut stream, 200, &tasks::due_json()),
        ("POST", "/api/tasks") => {
            let body = json::parse(&String::from_utf8_lossy(&request.body)).unwrap_or(Value::Null);
            match tasks::add_from_json(&body) {
                Ok(task) => http::respond(&mut stream, 200, &task),
                Err(why) => http::respond(&mut stream, 400, &error(&why)),
            }
        }
        ("POST", p) if p.starts_with("/api/tasks/") && p.ends_with("/run") => {
            let id = &p["/api/tasks/".len()..p.len() - "/run".len()];
            if tasks::run_now(id) {
                http::respond(&mut stream, 200, "{\"ok\":true}");
            } else {
                http::respond(&mut stream, 404, &error("No such task."));
            }
        }
        ("POST", p) if p.starts_with("/api/tasks/") && p.ends_with("/enabled") => {
            let id = &p["/api/tasks/".len()..p.len() - "/enabled".len()];
            let body = json::parse(&String::from_utf8_lossy(&request.body)).unwrap_or(Value::Null);
            let enabled = matches!(body.get("enabled"), Some(Value::Bool(true)));
            if tasks::set_enabled(id, enabled) {
                http::respond(&mut stream, 200, "{\"ok\":true}");
            } else {
                http::respond(&mut stream, 404, &error("No such task."));
            }
        }
        ("DELETE", p) if p.starts_with("/api/tasks/") => {
            if tasks::delete(&p["/api/tasks/".len()..]) {
                http::respond(&mut stream, 200, "{\"ok\":true}");
            } else {
                http::respond(&mut stream, 404, &error("No such task."));
            }
        }
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
fn completion(mut stream: TcpStream, shared: &Shared, body: &[u8], source: stats::Source) {
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
                messages.push(Message::new("user", chat::plain(content)));
            }
            "assistant" => messages.push(Message {
                role: "assistant".into(),
                content: chat::without_thinking(&chat::plain(content)),
                // What the reply's tools returned, as the app kept it (`turn_record`).
                record: m.str("record").and_then(chat::Record::parse),
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
    let folders: Vec<String> = body
        .array("folders")
        .iter()
        .filter_map(|f| match f {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let files = files::Access::from_request(body.str("access"), folders.clone())
        .map(files::Files::for_user);
    let thinking = match body.get("thinking") {
        Some(Value::Bool(on)) => *on,
        _ => options().thinking,
    };
    // `"tools": false`: a plain chat, no tools offered at all (the server's own too).
    let no_tools = matches!(body.get("tools"), Some(Value::Bool(false)));
    let job = Job {
        system,
        messages,
        generation: generation.cloned(),
        max_tokens,
        stop: stop.clone(),
        out,
        client_tools,
        files,
        access: body.str("access").unwrap_or("").to_string(),
        folders,
        chat_id: body.str("chatId").map(str::to_string),
        thinking,
        no_tools,
        tool_results: shared.tool_results.clone(),
        approvals: shared.approvals.clone(),
    };
    let counted = stats::begin(source);
    let asked = Instant::now();
    if shared.jobs.lock().unwrap().send(Work::Reply(job)).is_err() {
        shared.runs.lock().unwrap().remove(&run_id);
        stats::finish(
            counted,
            stats::Outcome {
                status: "error",
                ..Default::default()
            },
        );
        return http::respond(&mut stream, 500, &error("The engine has stopped."));
    }
    let outcome = stream_reply(&mut stream, &run_id, &events, &stop, asked);
    shared.runs.lock().unwrap().remove(&run_id);
    stats::finish(counted, outcome);
}

/// The address a request came from, for a refused one.
fn peer(stream: &TcpStream) -> String {
    stream
        .peer_addr()
        .map_or_else(|_| "unknown".into(), |a| a.ip().to_string())
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
    let others: usize = messages.iter().map(Message::bytes).sum();
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

/// Streams the reply, and says how it went (for `stats`): `asked` is when the request came.
fn stream_reply(
    stream: &mut TcpStream,
    run_id: &str,
    events: &Receiver<Out>,
    stop: &AtomicBool,
    asked: Instant,
) -> stats::Outcome {
    let mut outcome = stats::Outcome::default();
    if http::start_events(stream).is_err() {
        stop.store(true, Ordering::Relaxed);
        outcome.status = "cancelled";
        return outcome;
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
        if matches!(event, Out::Text(_) | Out::Event(_)) && outcome.first_token_seconds.is_none() {
            outcome.first_token_seconds = Some(asked.elapsed().as_secs_f64());
        }
        match event {
            Out::Text(text) => send(&format!("{{\"content\":{}}}", json::string(&text))),
            Out::Event(event) => send(&event),
            Out::Usage(usage) => outcome.usage = Some(usage),
            Out::Done { tokens, seconds } => {
                outcome.status = "ok";
                outcome.completion_tokens = tokens;
                outcome.generation_seconds = Some(seconds);
                let tps = if seconds > 0.0 {
                    tokens as f64 / seconds
                } else {
                    0.0
                };
                send(&format!(
                    "{{\"done\":true,\"tokens\":{tokens},\"duration\":\"{seconds:.2}\",\"tokensPerSecond\":\"{tps:.2}\"}}"
                ));
            }
            Out::Error(e) => {
                outcome.status = "error";
                send(&format!("{{\"error\":{}}}", json::string(&e)))
            }
        }
    }
    send("[DONE]");
    if outcome.status.is_empty() || (outcome.status == "ok" && stop.load(Ordering::Relaxed)) {
        outcome.status = if stop.load(Ordering::Relaxed) {
            "cancelled"
        } else {
            "error"
        };
    }
    outcome
}

// ---- The engine ------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)] // one at a time, taken off the channel at once
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
    /// The request's `access` and `folders` as sent: what a task set up in this chat may
    /// reach when it runs.
    access: String,
    folders: Vec<String>,
    /// The chat's id in the app, which a search of earlier chats leaves out.
    chat_id: Option<String>,
    /// Whether the model thinks before it answers (the request's `thinking`, else
    /// `--thinking`).
    thinking: bool,
    /// No tools at all (the request's `"tools": false`).
    no_tools: bool,
    tool_results: ToolWaits,
    approvals: Arc<Approvals>,
}

enum Out {
    Text(String),
    /// An event as it's sent: a tool call, a request for the app to run one, its result.
    Event(String),
    /// What the reply read and how (for `stats`), just before `Done`; not sent on.
    Usage(stats::Usage),
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

#[allow(clippy::large_enum_variant)] // a few, for the life of the server
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
    /// Qwen: its DeltaNet layers where a prompt parted from the one before (for a new
    /// chat, the end of the system prompt and tools it shares with the last), so the
    /// next new chat doesn't read them again.
    base: Option<Qwen35Checkpoint>,
    /// Qwen: its DeltaNet layers at the end of the last system turn read (its
    /// instructions and tools), so the next chat with the same ones reads none of it.
    turn: Option<Qwen35Checkpoint>,
    /// Qwen: its DeltaNet layers where scoring started, to go back to after each try.
    scratch: Option<Qwen35Checkpoint>,
    /// Small sessions apart, for questions the server asks the model itself (which tools a
    /// request needs, whether a skill fits), so the chat's own cache is left as it is.
    /// One a kind of question (up to [`PICKERS`]), each going on from the start its last
    /// question shares with the next ([`Model::ask`]).
    pickers: Vec<Picker>,
    /// Each skill's share of yes for a request that fits no skill, by its question: this
    /// model's bias, scored once (`pick_skill`).
    skill_controls: HashMap<String, f64>,
    /// Where the last prompt's history ends (its tokens are still the session's first):
    /// Gemma's chat is saved up to there, which the next turn and a retry both start with,
    /// since its saved state can only be loaded whole.
    history_end: usize,
}

/// A session for the server's own questions, and what's in it.
struct Picker {
    session: Session,
    /// The tokens in the session: the last question read.
    tokens: Vec<u32>,
    /// Qwen: its DeltaNet layers where the last two questions part (their shared start).
    checkpoint: Option<Qwen35Checkpoint>,
    /// When it was last used (a count of questions), to reuse the oldest.
    used: u64,
}

/// Whether `--draft` names Google's draft model for Gemma 4 (a folder, or a .safetensors
/// file) rather than a small model to draft with (a GGUF file).
fn is_assistant(path: &str) -> bool {
    let p = std::path::Path::new(path);
    p.is_dir() || p.extension().is_some_and(|e| e == "safetensors")
}

/// Guesses a round from Qwen's multi-token-prediction layer, or Gemma 4's draft model
/// (Google's for 31B: the first right 74% of the time, 1.72 of 3 taken a round,
/// `vireo-engine`'s `tests/gemma_assistant.rs`). On Qwen 3.8 27B the first is
/// right 86% of the time, the first two 64% and all three 43% (`tests/mtp.rs`). While
/// checking 4 tokens cost 1.8 times one, 2 a round beat 3; with the 8-token tile it costs
/// 1.24 times one (46.7 ms one token, 57.7 four: `tests/check_speed.rs`), and through the
/// server 2 / 3 / 4 guesses write a copied document at 44 / 53 / 59 tokens/s and a new
/// story at 30.1 / 29.6 / 26.5: 3.
const MTP_GUESSES: usize = 3;

/// How sure Gemma 4's draft model must be of a guess for the model to check it (its
/// probability for the token; a round's guesses stop before the first it gives less). On
/// CUDA a guess checked costs the small models about a third of a token (every layer and
/// the logits again for each: Gemma 4 E2B 10.6 ms one token, 3.4 more a guess; E4B 19.8
/// and 5.3), and the draft knows when it's guessing: of first guesses it gave 0.9 or more,
/// 90% were the model's pick, of those under 0.7, one in five (`tests/gemma_assistant.rs`).
/// Writing greedy, 3 guesses a round, any / 0.5 / 0.7 / 0.9: E2B 1.01 / 1.14 / 1.25 / 1.21
/// times one token at a time, E4B 1.28 / 1.42 / 1.41 / 1.36. Not measured on the Mac, where
/// the 31B's draft model guesses as before.
const MTP_FLOOR: f32 = if cfg!(target_os = "macos") { 0.0 } else { 0.7 };

/// With a draft model, the chat's own guesses are still taken where they're a long run
/// (this many or more: what followed 4 or more of the chat's last tokens before), since a
/// copied line is right far more often than a draft and costs nothing to guess. Shorter
/// ones are left to the draft model. On CUDA; the Mac drafts from one or the other.
const CHAT_RUN_OVER_DRAFT: usize = 3;

/// The most picker sessions kept: the change question, the skill questions and the group
/// question each start their own way.
const PICKERS: usize = 4;
/// The most tokens a picker session reads at once. Its questions are a few hundred tokens,
/// mostly shared with the last one, and a session's buffers grow with its batch: at 1,024
/// Qwen 3.5 9B's are 0.4 GB a session, and three pickers on top of the chat's left an 8 GB
/// GPU with none free (Windows then keeps GPU memory in the PC's: a token took 27 ms, not
/// 19.5, over nine eval cases). 256 on CUDA: 0.1 GB each. The Mac keeps 1,024, as measured
/// there.
const PICKER_BATCH: usize = if cfg!(target_os = "macos") { 1024 } else { 256 };
/// A question goes to a new or the oldest picker unless one shares at least this many
/// tokens with it (more than the chat template's opening).
const PICKER_MIN_SHARED: usize = 48;

/// The context when the model's file doesn't say what it was trained for.
const DEFAULT_CONTEXT: usize = 8192;
/// The most `--context auto` picks, however much memory is free: further than the engine
/// has been checked against llama.cpp.
const AUTO_CONTEXT_MAX: usize = 32768;
/// GPU memory `--context auto` leaves free: for the display and other programs on a PC
/// whose GPU also drives the screen (and a tenth of what's free on top).
const GPU_HEADROOM: usize = 1 << 30;
/// GPU memory left free after a Qwen session's DeltaNet steps (kept so the chat's guesses
/// can be undone): without it Windows moves GPU memory into the PC's, and everything slows.
const STEPS_HEADROOM: usize = 384 << 20;

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

/// With VIREO_TIME_LOG=<file>: each pass through the model (`read`, `read_all`, `next`),
/// its tokens and milliseconds, a line each, appended. How a reply's time splits between
/// writing, checking guesses and reading batches of each size (vireo-eval passes the
/// variable on to the server it starts).
struct TimeLog(&'static str, usize, std::time::Instant);
impl TimeLog {
    /// When logging, the GPU work already queued (a prompt read without logits, a
    /// checkpoint's copies) finishes first, so its time isn't put down to this pass.
    fn start(model: &Model, kind: &'static str, n: usize) -> Self {
        if std::env::var_os("VIREO_TIME_LOG").is_some() {
            let _ = match model {
                Model::Llama(m) => m.gpu().synchronize(),
                Model::Qwen35(m) => m.gpu().synchronize(),
                Model::Gemma4(m) => m.gpu().synchronize(),
            };
        }
        TimeLog(kind, n, std::time::Instant::now())
    }
}
impl Drop for TimeLog {
    fn drop(&mut self) {
        if let Some(path) = std::env::var_os("VIREO_TIME_LOG") {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(
                    f,
                    "{} {} {:.2}",
                    self.0,
                    self.1,
                    self.2.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }
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

    /// Reads a question of the server's own (`ids`, from its start) in a picker session,
    /// reading only what it doesn't share with the last question there; the last token's
    /// logits. Gemma and Llama go back to the shared start; Qwen, whose DeltaNet state
    /// can't be cut back, to a checkpoint saved where the last two questions parted (the
    /// skill questions share their worked examples, ~230 tokens, and differ after).
    fn ask(&self, kept: &mut Kept, ids: &[u32]) -> Result<Vec<f32>, String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        let count = kept.pickers.iter().map(|p| p.used).max().unwrap_or(0) + 1;
        let best = (0..kept.pickers.len())
            .max_by_key(|&i| shared_start(&kept.pickers[i].tokens, ids))
            .filter(|&i| shared_start(&kept.pickers[i].tokens, ids) >= PICKER_MIN_SHARED);
        let at = match best {
            Some(i) => i,
            None if kept.pickers.len() < PICKERS => {
                kept.pickers.push(Picker {
                    session: self.picker_session()?,
                    tokens: Vec::new(),
                    checkpoint: None,
                    used: 0,
                });
                kept.pickers.len() - 1
            }
            None => (0..kept.pickers.len())
                .min_by_key(|&i| kept.pickers[i].used)
                .unwrap(),
        };
        let p = &mut kept.pickers[at];
        p.used = count;
        let shared = shared_start(&p.tokens, ids);
        let mut from = match (self, &mut p.session) {
            (Model::Llama(_), Session::Llama(s)) => {
                s.truncate(shared);
                shared
            }
            (Model::Gemma4(_), Session::Gemma4(s)) => s.truncate(shared),
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                if shared == s.len() {
                    shared
                } else if let Some(cp) = p.checkpoint.as_ref().filter(|c| c.len() <= shared) {
                    m.restore(s, cp).map_err(e)?;
                    cp.len()
                } else {
                    m.reset(s).map_err(e)?;
                    p.checkpoint = None;
                    0
                }
            }
            _ => unreachable!("a session belongs to its model"),
        };
        // Qwen: a checkpoint where this question parts from the last, a few tokens early
        // (the next may part a token sooner), if it's well past where reading starts.
        if let (Model::Qwen35(m), Session::Qwen35(s)) = (self, &mut p.session) {
            let mark = shared.saturating_sub(8);
            if mark >= from + 32 {
                m.forward_batch(s, &ids[from..mark], Logits::None)
                    .map_err(e)?;
                m.save(s, &mut p.checkpoint).map_err(e)?;
                from = mark;
            }
        }
        p.tokens.clear();
        let logits = self.read(&mut p.session, &ids[from..]);
        if logits.is_ok() {
            p.tokens.extend_from_slice(ids);
        } else {
            // Unknown contents: start over next time.
            p.checkpoint = None;
            self.empty(&mut p.session)?;
        }
        logits
    }

    /// Empties a session.
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

    /// A session for the server's own questions: 1,024 positions, read on CUDA in batches
    /// of [`PICKER_BATCH`], since a session's buffers are sized by its batch.
    fn picker_session(&self) -> Result<Session, String> {
        let e = |e: vireo_engine::models::Error| e.to_string();
        let (capacity, batch) = (1024, PICKER_BATCH);
        Ok(match self {
            Model::Llama(m) => Session::Llama(m.session_with_batch(capacity, batch).map_err(e)?),
            Model::Qwen35(m) => {
                Session::Qwen35(m.session_with_batch(capacity, batch).map_err(e)?)
            }
            Model::Gemma4(m) => {
                Session::Gemma4(m.session_with_batch(capacity, batch).map_err(e)?)
            }
        })
    }

    /// The session chats run in: on a Qwen model with a multi-token-prediction layer
    /// (and `--mtp on`), one that keeps the layer's cache up, to draft with; on a Gemma 4
    /// with its draft model loaded, one with room for it.
    fn chat_session(&self, capacity: usize) -> Result<Session, String> {
        let mut session = self.session(capacity)?;
        match (self, &mut session) {
            (Model::Qwen35(m), Session::Qwen35(s)) if options().mtp && m.has_mtp() => {
                m.use_mtp(s).map_err(|e| e.to_string())?;
            }
            (Model::Gemma4(m), Session::Gemma4(s)) if m.has_mtp() => {
                m.use_mtp(s).map_err(|e| e.to_string())?;
            }
            _ => {}
        }
        Ok(session)
    }

    /// Up to `k` guesses from Qwen's multi-token-prediction layer, or Gemma 4's draft
    /// model, at the tokens after `first` (picked, not read yet); none for a session that
    /// doesn't keep it.
    fn mtp_draft(&self, kept: &mut Kept, first: u32, k: usize) -> Result<Vec<u32>, String> {
        match (self, &mut kept.session) {
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                m.mtp_draft(s, first, k).map_err(|e| e.to_string())
            }
            (Model::Gemma4(m), Session::Gemma4(s)) => m
                .mtp_draft_sure(s, first, k, MTP_FLOOR)
                .map(|g| g.into_iter().map(|(token, _)| token).collect())
                .map_err(|e| e.to_string()),
            _ => Ok(Vec::new()),
        }
    }

    /// The prompt, in batches; the last token's logits.
    fn read(&self, s: &mut Session, ids: &[u32]) -> Result<Vec<f32>, String> {
        let _t = TimeLog::start(self, "read", ids.len());
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
        let _t = TimeLog::start(self, "read_all", ids.len());
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
    /// there; the last token's logits, and how many tokens were reused. `head` is where
    /// the system turn ends in `ids` (0 when not known): what a new chat would share. On
    /// an error the session's contents are unknown: drop it.
    fn continue_with(
        &self,
        kept: &mut Kept,
        ids: &[u32],
        mark: usize,
        head: usize,
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
                } else if let Some(cp) = kept.turn.as_ref().filter(|c| c.len() <= shared) {
                    m.restore(s, cp).map_err(e)?;
                    cp.len()
                } else if let Some(cp) = kept.base.as_ref().filter(|c| c.len() <= shared) {
                    // Another system turn from here on: the one kept is no longer the
                    // session's.
                    kept.turn = None;
                    m.restore(s, cp).map_err(e)?;
                    cp.len()
                } else {
                    m.reset(s).map_err(e)?;
                    kept.checkpoint = None;
                    kept.base = None;
                    kept.turn = None;
                    0
                }
            }
            _ => unreachable!("a session belongs to its model"),
        };
        kept.tokens.truncate(from);
        let reused = from;
        // Qwen, reading a system turn (or a good part of one) again: two checkpoints on
        // the way (each a few tokens early, as the next chat may part a token sooner).
        // One where this prompt parts from the last, when none is kept: what chats with
        // different tools still share. One at the system turn's end, for the next chat
        // with the same instructions and tools: each of the benchmark's tasks read
        // Vireo's ~1,100 tokens of tools again without it (the one checkpoint was where
        // the first two chats parted, before the tools).
        let mut from = from;
        if let (Model::Qwen35(m), Session::Qwen35(s)) = (self, &mut kept.session) {
            let head_at = head.saturating_sub(8);
            let parted_at = shared.saturating_sub(8);
            if kept.base.is_none() && parted_at >= from + 64 && parted_at < mark {
                m.forward_batch(s, &ids[from..parted_at], Logits::None)
                    .map_err(e)?;
                m.save(s, &mut kept.base).map_err(e)?;
                kept.tokens.extend_from_slice(&ids[from..parted_at]);
                from = parted_at;
            }
            if head_at >= from + 64 && head_at < mark {
                m.forward_batch(s, &ids[from..head_at], Logits::None)
                    .map_err(e)?;
                m.save(s, &mut kept.turn).map_err(e)?;
                kept.tokens.extend_from_slice(&ids[from..head_at]);
                from = head_at;
            }
        }
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
        Ok((logits, reused))
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
            // Inside the last round read with its DeltaNet steps kept (`keep_steps`).
            (Model::Qwen35(m), Session::Qwen35(s)) => {
                m.rewind(s, len).map_err(|e| e.to_string())?
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
            Session::Qwen35(s) if shared != s.len() => [&kept.checkpoint, &kept.turn, &kept.base]
                .into_iter()
                .flatten()
                .map(|c| c.len())
                .filter(|&n| n <= shared)
                .max()
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
                kept.base = None;
                kept.turn = None;
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
                kept.base = None;
                kept.turn = None;
            }
            _ => unreachable!("a session belongs to its model"),
        }
        kept.tokens.clear();
        kept.history_end = 0;
        Ok(())
    }

    fn next(&self, s: &mut Session, id: u32, logits: &mut [f32]) -> Result<(), String> {
        let _t = TimeLog::start(self, "next", 1);
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
        let family = Family::of(file.architecture()).ok_or_else(|| {
            format!("{path} isn't a Llama, Qwen 3.5, Gemma 4 or Muse Glimmer model")
        })?;
        let tokenizer = Tokenizer::from_gguf(file).map_err(|e| e.to_string())?;
        chat::follow_template(file.get_str("tokenizer.chat_template"));
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
            // Muse Glimmer runs on Gemma 4's model code (`Config::muse`).
            Family::Gemma4 | Family::Muse => {
                let mut m = Gemma4Cuda::load(file, gpu).map_err(e)?;
                m.set_precision(precision);
                // Google's draft model for it: --draft's folder or .safetensors, else one
                // Vireo keeps beside the model or in the app's DraftModels (--mtp on).
                let given = options().draft.as_deref().filter(|d| is_assistant(d));
                let found = match (&options().draft, family) {
                    (None, Family::Gemma4) if options().mtp => {
                        gemma4_assistant::find(std::path::Path::new(path))
                    }
                    _ => None,
                };
                if let Some(draft) = given.map(std::path::PathBuf::from).or(found) {
                    let started = Instant::now();
                    match m.load_assistant(&draft) {
                        Ok(()) => eprintln!(
                            "Drafting with {}, Google's draft model, {MTP_GUESSES} tokens a round (loaded in {:.1?}).",
                            draft.display(),
                            started.elapsed()
                        ),
                        // One given must work; one found is only a speed-up.
                        Err(err) if given.is_some() => {
                            return Err(format!("Couldn't load the draft model {}: {err}", draft.display()));
                        }
                        Err(err) => eprintln!(
                            "Not drafting with {} ({err}).",
                            draft.display()
                        ),
                    }
                }
                Model::Gemma4(m)
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
    // The weights are in GPU memory now: let go of the file's pages read on the way.
    file.release_pages();
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
    let mut kept: Option<Kept> = match model.chat_session(context) {
        Ok(session) => {
            eprintln!(
                "Chats keep up to {context} tokens ({:.1} GB of the GPU's memory for their cache).",
                (model.cache_bytes_per_position(&session) * context) as f64 / 1e9
            );
            Some(Kept {
                session,
                tokens: Vec::new(),
                checkpoint: None,
                base: None,
                turn: None,
                scratch: None,
                pickers: Vec::new(),
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
        // Google's draft model, loaded with the model.
        Some(draft_path) if is_assistant(draft_path) => None,
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
        let full_chat = kept
            .as_ref()
            .map_or(0, |k| model.cache_bytes_per_position(&k.session) * context);
        let budget = store::budget_for(full_chat as u64);
        let s = store::Store::open(dir, budget);
        eprintln!(
            "{} saved chats; they may take up to {:.0} GB on disk.",
            s.entries().len(),
            budget as f64 / (1u64 << 30) as f64
        );
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
            match model.chat_session(context) {
                Ok(session) => {
                    kept = Some(Kept {
                        session,
                        tokens: Vec::new(),
                        checkpoint: None,
                        base: None,
                    turn: None,
                        scratch: None,
                        pickers: Vec::new(),
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

/// A reply's prompt, written one of the ways `reply` may write it.
struct Prompt {
    ids: Vec<u32>,
    /// Where the history ends in `ids`: what the next turn's prompt starts with.
    mark: usize,
    /// The first of the chat's messages in it (later than 0 for a chat too long to fit).
    first: usize,
    /// Its system prompt and tool declarations.
    system: Option<String>,
    declared: String,
}

/// A chat too long for the context starts a third of the context further on each time
/// (`chat::cut_points`, in bytes: about this many a token), so it reads everything again
/// once in that many tokens, not at every turn. A quarter at a time was tried on a
/// 64-turn chat (Gemma 4 31B, `evals/long-run`): 25 s more reading, and the same early
/// notes forgotten.
const CUT_BYTES_PER_TOKEN: usize = 4;
const CUT_STEP: usize = 3;

/// The share of the context (one part in this many) left free for the request and a
/// skill's instructions when a long chat's start is chosen from its history alone.
const REQUEST_ROOM: usize = 8;

/// How many facts back a chat's instructions may stop at for its cache to be kept: the
/// facts remembered while it's cached, before it's read again with all of them.
const OLDER_FACTS: usize = 8;

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
    // Thinking, where the request asks for it and the model has the tokens that open and
    // close a thought: Gemma 4's thought channel, Qwen's <think>.
    let think = (job.thinking && family.thinks_aloud())
        .then_some(match family {
            Family::Qwen35 => ("<think>", "</think>"),
            _ => ("<|channel>", "<channel|>"),
        })
        .and_then(|(open, close)| Some((tokenizer.token_id(open)?, tokenizer.token_id(close)?)));
    let thinking_on = think.is_some();
    // What the prompt says of it (Muse's is its reasoning strength).
    let prompt_thinking = thinking_on || (job.thinking && family == Family::Muse);
    let mut offered = if job.no_tools {
        Vec::new()
    } else {
        tools::offered(&job.client_tools, job.files.is_some())
    };
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
    let every_tool = offered.clone();
    // `declared`, and this request doesn't ask for a change: the tools offered with the
    // ones that change files still among them, for a chat whose cache declares them.
    let mut with_changes: Option<Vec<tools::Tool>> = None;
    if options().changes != Changes::Offered
        && offered.iter().any(|t| t.changes_files())
        && let Some(request) = job.messages.last()
        && !asks_for_change(model, kept, tokenizer, family, &request.content)?
    {
        callable.retain(|t| !t.changes_files());
        if options().changes == Changes::Declared {
            with_changes = Some(offered.clone());
            offered.retain(|t| !t.changes_files());
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
        // A skill that names a tool that changes files asks for the change itself, whatever
        // the request's own words ("use the weekly-report skill for this" doesn't say the
        // report is saved: Gemma 4 31B, the tool kept from it, read and summed in circles
        // in the Nov 13 benchmark).
        if callable.len() < every_tool.len()
            && every_tool
                .iter()
                .any(|t| t.changes_files() && result.contains(t.name))
        {
            callable = every_tool.clone();
            offered = every_tool.clone();
            with_changes = None;
        }
        let (before, body, after) = tools::after_call(family, "use_skill", &result, thinking_on);
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
    // The prompt. A chat's cache holds its start as it was first written, and two things
    // would change that start from one turn to the next: the tools that change files
    // (declared only when asked for) and a fact remembered since. So a chat whose cache
    // was written another way keeps that way: those tools stay declared (though not
    // callable now), and its instructions keep the facts they began with, the newer
    // ones told at the end of the request.
    let facts = memory::all();
    // A skill written in comes before any thought: the turn opens without one, and the
    // thought follows the skill's instructions, as it follows any tool's result.
    let opens_thinking = thinking_on && skill_ids.is_empty();
    let opening = tokenizer.encode(chat::opening(family, opens_thinking), false, true);
    let cuts = chat::cut_points(&job.messages, context / CUT_STEP * CUT_BYTES_PER_TOKEN);
    // Earlier replies' tool results (`chat::Record`) are kept whole while the chat fits;
    // then only the start of each; then not at all, before the chat's start moves on.
    let views: Vec<Vec<Message>> = if job.messages.iter().any(|m| m.record.is_some()) {
        [chat::Results::Whole, chat::Results::Short, chat::Results::Left]
            .iter()
            .map(|&keep| job.messages.iter().map(|m| m.with_results(keep)).collect())
            .collect()
    } else {
        vec![job.messages.clone()]
    };
    // With tools, a request ends with today's date (Llama's template gives it): a memo
    // was dated from the model's training otherwise. After the history, so the chat's
    // cache is the same from one day to the next.
    let dated = if !offered.is_empty() && family != Family::Llama {
        format!("\n\n(Today is {}.)", tools::today_line())
    } else {
        String::new()
    };
    let debug_prompt = std::env::var_os("VIREO_DEBUG_PROMPT").is_some();
    let system_turn = |tools: &[tools::Tool], known: usize| -> (Option<String>, String) {
        // With tools, the system prompt opens with what they're for.
        let preamble = || {
            let mut text = match tools::family_notes(family) {
                Some(notes) => format!("{}\n\n{notes}", tools::PREAMBLE),
                None => tools::PREAMBLE.to_string(),
            };
            // Gemma with thinking off: the small ones (E2B, E4B) otherwise write their
            // plan as the answer under these instructions, and run out of reply before the
            // answer itself (0 and 1 of the Nov 13 benchmark's 6 tasks right).
            if family == Family::Gemma4 && !prompt_thinking {
                text.push_str(tools::GEMMA_NO_THINKING);
            }
            text
        };
        let system = match (tools.is_empty(), job.system.as_deref().map(str::trim)) {
            (true, s) => s.filter(|s| !s.is_empty()).map(str::to_string),
            (false, Some(s)) if !s.is_empty() => Some(format!("{}\n\n{s}", preamble())),
            (false, _) => Some(preamble()),
        };
        // What the user asked to be remembered, after the rest (the Mac's order).
        let system = match (system, memory::instructions_for(&facts[..known])) {
            (Some(s), Some(m)) => Some(format!("{s}\n\n{m}")),
            (s, m) => s.or(m),
        };
        (system, tools::declarations(family, tools))
    };
    let build = |tools: &[tools::Tool], known: usize| -> Prompt {
        let (system, declared) = system_turn(tools, known);
        let note = format!("{}{dated}", memory::since(&facts[known..]));
        // The chat's latest messages that fit with room for the reply: all of them, else
        // from the first of `cuts` that fits, else from the first message that does. At
        // each start, with as much of earlier tool results as fits (`views`).
        let mut starts = std::iter::once(0)
            .chain(cuts.iter().copied())
            .flat_map(|start| (0..views.len()).map(move |view| (start, view)));
        let mut first = 0;
        let mut view;
        loop {
            match starts.next() {
                Some((start, results)) => (first, view) = (start, results),
                None => {
                    view = views.len() - 1;
                    first += 1;
                    while first < job.messages.len() && job.messages[first].role != "user" {
                        first += 1;
                    }
                    first = first.min(job.messages.len() - 1);
                }
            }
            let messages = &views[view];
            let text = chat::history(
                family,
                system.as_deref(),
                &declared,
                &messages[first..],
                &today,
                prompt_thinking,
            );
            // The note goes at the end of the request, read apart from the history
            // before it: the next turn's prompt is the same up to there (Qwen's
            // DeltaNet is saved at `mark`, and must be wanted whole).
            let end = text.len() - chat::closing(family).len();
            let (history, rest) = if note.is_empty() {
                (text.as_str(), String::new())
            } else {
                (&text[..end], format!("{note}{}", &text[end..]))
            };
            if debug_prompt {
                eprintln!(
                    "[prompt] {history}{rest}{}",
                    chat::opening(family, opens_thinking)
                );
            }
            let mut ids = chat::encode(tokenizer, history);
            let mark = ids.len();
            ids.extend(tokenizer.encode(&rest, false, true));
            ids.extend_from_slice(&opening);
            ids.extend_from_slice(&skill_ids);
            // A start is settled on what came before the request, with room left for a
            // request and a skill's instructions (`REQUEST_ROOM`): that only grows, so
            // once a chat has moved to a later start it never goes back to an earlier one
            // on a turn that happens to be shorter (a short question after a long one, or
            // no skill after a skill), which would swap its cache out and in again.
            let last = job.messages.len() - 1;
            let settled = first >= last || {
                let before = chat::history(
                    family,
                    system.as_deref(),
                    &declared,
                    &messages[first..last],
                    &today,
                    prompt_thinking,
                );
                chat::encode(tokenizer, &before).len() + max_tokens + context / REQUEST_ROOM
                    <= context
            };
            if (settled && ids.len() + max_tokens <= context) || first >= last {
                break Prompt {
                    ids,
                    mark,
                    first,
                    system,
                    declared,
                };
            }
        }
    };
    let mut prompt = build(&offered, facts.len());
    let mut keep_changes = false;
    if job.messages.len() - prompt.first > 1 && (with_changes.is_some() || !facts.is_empty()) {
        let caches: Vec<&[u32]> = std::iter::once(kept.tokens.as_slice())
            .chain(
                store
                    .as_deref()
                    .into_iter()
                    .flat_map(|s| s.entries().iter().map(|e| e.tokens.as_slice())),
            )
            .collect();
        // Whether a cached chat starts as `p` does through its first message at least
        // (another chat with the same system turn doesn't count).
        let cached = |p: &Prompt| {
            let head = chat::history(
                family,
                p.system.as_deref(),
                &p.declared,
                &job.messages[p.first..p.first + 1],
                &today,
                prompt_thinking,
            );
            let head = chat::encode(tokenizer, &head).len();
            caches.iter().any(|c| shared_start(c, &p.ids) >= head)
        };
        if !cached(&prompt) {
            let oldest = facts.len().saturating_sub(OLDER_FACTS);
            'ways: for known in (oldest..=facts.len()).rev() {
                let plain = (known < facts.len()).then_some(offered.as_slice());
                for (tools, changes) in [(plain, false), (with_changes.as_deref(), true)] {
                    let Some(tools) = tools else { continue };
                    // Cheaply first: is any cache's system turn written this way?
                    let (system, declared) = system_turn(tools, known);
                    let turn = chat::history(
                        family,
                        system.as_deref(),
                        &declared,
                        &[],
                        &today,
                        prompt_thinking,
                    );
                    let turn = chat::encode(tokenizer, &turn);
                    if !caches.iter().any(|c| c.starts_with(&turn)) {
                        continue;
                    }
                    let other = build(tools, known);
                    if job.messages.len() - other.first > 1 && cached(&other) {
                        eprintln!(
                            "Prompt as the chat's cache has it: {}{} of {} facts in its instructions.",
                            if changes {
                                "the tools that change files declared, "
                            } else {
                                ""
                            },
                            known,
                            facts.len()
                        );
                        prompt = other;
                        keep_changes = changes;
                        break 'ways;
                    }
                }
            }
        }
    }
    if keep_changes && let Some(tools) = with_changes {
        offered = tools;
    }
    let Prompt {
        ids,
        mark,
        system,
        declared,
        ..
    } = prompt;
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
    // Qwen: where the system turn ends, for a saved start for new chats.
    let head = if matches!(family, Family::Qwen35) {
        let turn = chat::history(
            family,
            system.as_deref(),
            &declared,
            &[],
            &today,
            prompt_thinking,
        );
        shared_start(&chat::encode(tokenizer, &turn), &ids)
    } else {
        0
    };
    let (mut logits, reused) = model.continue_with(kept, &ids, mark, head)?;
    if let Some(d) = drafter.as_deref_mut() {
        d.sync(&kept.tokens)?;
    }
    let prompt_seconds = started.elapsed().as_secs_f64();

    let settings = family.sampling().with(job.generation.as_ref());
    let mut sampler = Sampler::new(settings, seed);
    // Gemma 4's thinking channel, for a thought it opens with thinking off (with it on,
    // `think` has these tokens).
    let channel = if thinking_on {
        (None, None)
    } else {
        (
            tokenizer.token_id("<|channel>"),
            tokenizer.token_id("<channel|>"),
        )
    };
    let mut in_channel = false;
    let mut generated: Vec<u32> = Vec::new();
    let mut shown = Shown::default();
    // Thinking on: whether the model is in a thought now (Qwen's turn opens in one; so
    // does the turn after a skill's instructions), the tokens this thought has taken, and
    // whether Gemma's own label ("thought") is still to come at its start, to leave out.
    let mut in_thought = thinking_on && (family == Family::Qwen35 || !skill_ids.is_empty());
    let mut thought_tokens = 0usize;
    // The most this thought may take: `THOUGHT_TOKENS`, or `SHORT_THOUGHT_TOKENS` after a
    // change was made or once the reply's thoughts have had their share.
    let mut thought_cap = THOUGHT_TOKENS;
    let mut label = false;
    if in_thought {
        shown.tag(job, THINK_OPEN, true);
    }
    // The reply's tool calls and their results, for the app to keep (`chat::Record`).
    let mut steps: Vec<chat::Step> = Vec::new();
    let started = Instant::now();
    // Tool calls: the grammar they're written under, and how one starts.
    let grammar = Grammar::new(&tools::call_grammar(family, &callable));
    // Muse's recipients without its reasoning (once it has used its budget), and without
    // tools (once calls are used up, or none are offered).
    let muse_grammars = (family == Family::Muse).then(|| {
        (
            Grammar::new(&tools::muse_call_grammar(&callable, false)),
            Grammar::new(&tools::muse_call_grammar(&[], true)),
        )
    });
    // Muse writing to itself, and the tokens it has so far this reply.
    let mut thinking = false;
    let mut thought = 0usize;
    let muse_thought_tokens = if job.thinking {
        MUSE_THOUGHT_TOKENS * 2
    } else {
        MUSE_THOUGHT_TOKENS
    };
    let start = tools::call_start(family);
    let start_token = tokenizer.token_id(start.token);
    // The calls made since the reply last changed anything: the same one again would
    // get the same answer.
    let mut seen: Vec<tools::Call> = Vec::new();
    let (mut calls, mut call_tokens, mut repeats) = (0usize, 0usize, 0usize);
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
    // Without a draft model: guesses from the chat itself, for a model that can go back.
    // Qwen goes back through the DeltaNet steps its chat session keeps for short batches:
    // to draft with its multi-token-prediction layer, or from the chat. On the Mac only on
    // a large model: on a small one a check of a few tokens costs too much against one
    // token for the chat's guesses to pay (Qwen 3.5 4B, eight eval cases: writing 5.5 →
    // 5.9 s; 27B 58.4 → 52.9 s). On CUDA a check of 4 costs Qwen 3.5 4B 1.4 times one
    // token, and it pays on every size (4B rewriting a document as a list 86 → 199
    // tokens/s, 9B returning an edited code file 52 → 157; a new story 1-3% slower),
    // where the GPU has the memory for the steps besides what's kept free.
    // Inside a tool call the guesses are nearly all taken (a path or a number the chat
    // already holds), so there the chat's guesses pay on the small ones too: `text_drafts`
    // says whether the reply's own text is guessed at as well.
    let mut text_drafts = true;
    let can_go_back = match (model, &mut kept.session) {
        (Model::Gemma4(_) | Model::Llama(_), _) => true,
        (Model::Qwen35(m), Session::Qwen35(s)) if m.can_rewind() => {
            text_drafts = s.keeps_mtp() || m.config.dim >= 5120 || !cfg!(target_os = "macos");
            let room = cfg!(target_os = "macos")
                || s.keeps_steps()
                || m.gpu().free_memory() >= m.steps_bytes() + STEPS_HEADROOM;
            (text_drafts || !offered.is_empty()) && room && m.keep_steps(s).is_ok()
        }
        _ => false,
    };
    // Qwen with its multi-token-prediction layer kept up: guesses from that, not the chat.
    let mtp = drafter.is_none()
        && can_go_back
        && match &kept.session {
            Session::Qwen35(s) => s.keeps_mtp(),
            Session::Gemma4(s) => s.keeps_mtp(),
            _ => false,
        };
    let mut mtp_stats = (0usize, 0usize); // guessed, taken
    // Gemma 4 with its draft model on CUDA guesses from the chat too, where the chat has
    // a long run to offer (`CHAT_RUN_OVER_DRAFT`).
    let both = mtp && matches!(kept.session, Session::Gemma4(_)) && !cfg!(target_os = "macos");
    let mut lookup = (drafter.is_none() && (!mtp || both) && options().lookup > 0 && can_go_back)
        .then(|| draft::Lookup::new(options().lookup));
    // Whether the guesses read ahead are the draft model's (or the layer's), not the chat's.
    let mut ahead_drafted = false;
    let drafting = drafter.is_some() || (lookup.is_some() && text_drafts) || mtp;
    // After a tool's result (or the skill written in) the model must say something: its
    // end-of-turn tokens are taken away until it writes one token. Gemma 4 E4B, with a
    // longer list of tools, ended its turn on the result three times in 39 cases,
    // answering nothing.
    let mut answer_due = !skill_ids.is_empty();
    let mut end_tokens: Option<Vec<u32>> = None;
    // `max_tokens` is the model's own writing; calls have their own limits (`MAX_CALLS`,
    // `MAX_CALL_TOKENS`), so a task of many calls still has room to answer.
    while generated.len() < max_tokens && !job.stop.load(Ordering::Relaxed) {
        // Thoughts and tools' results aren't counted in `max_tokens`: the context's end
        // is the limit for them.
        if kept.tokens.len() + 8 >= context {
            break;
        }
        // Muse names each message's recipient first. With no call to make, the user: the
        // model writes to them from here. Otherwise the call's grammar has it choose (the
        // user is one of its options), below.
        let muse_turn = family == Family::Muse && turn_start;
        let may_think = thought < muse_thought_tokens;
        if muse_turn && !may_call && !may_think {
            let ids = tokenizer.encode(tools::MUSE_TO_USER, false, true);
            if kept.tokens.len() + ids.len() + 64 > context {
                break;
            }
            logits = model.feed(kept, &ids)?;
            turn_start = false;
            continue;
        }
        let room =
            (max_tokens - generated.len()).min(context.saturating_sub(kept.tokens.len() + 64));
        // Out of guesses: a new round (after the unread token, if there's one), read by
        // the model in one pass with it.
        if ahead.is_empty() && !muse_turn {
            // The chat's guesses, where it guesses from the chat (in a reply's text only
            // where `text_drafts`): all of them without a draft model, only a long run with
            // one.
            let from_chat = match lookup.as_mut() {
                Some(l) if room > 1 && text_drafts => {
                    let g = l.guess(&kept.tokens, l.max.min(room - 1));
                    if mtp && g.len() < CHAT_RUN_OVER_DRAFT {
                        l.guessed -= g.len();
                        Vec::new()
                    } else {
                        g
                    }
                }
                _ => Vec::new(),
            };
            ahead_drafted = false;
            let guesses = match (drafter.as_deref_mut(), unread) {
                (Some(d), _) if room > 1 => d.guess(&kept.tokens, d.guesses.min(room - 1))?,
                (None, _) if !from_chat.is_empty() => from_chat,
                // The layer guesses what follows the token just picked.
                (None, Some(first)) if mtp && room > 1 => {
                    let g = model.mtp_draft(kept, first, MTP_GUESSES.min(room - 1))?;
                    mtp_stats.0 += g.len();
                    ahead_drafted = true;
                    g
                }
                _ => Vec::new(),
            };
            match guesses {
                guesses if !guesses.is_empty() => {
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
        if !may_call && !offered.is_empty() && !thinking {
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
        let mut next = sampler.pick(&logits, &generated);
        // A thought ends with its closing token, here even when the model would end its
        // turn inside it, or has thought as long as one thought may: its answer is still
        // to come.
        if in_thought
            && let Some((_, close)) = think
            && (tokenizer.is_end_of_generation(next) || thought_tokens >= thought_cap)
        {
            next = close;
        }
        // Muse's reasoning ends its message with <|eom|>, here even when it would end its
        // turn, or runs out of room: its answer is still to come.
        if thinking
            && (tokenizer.is_end_of_generation(next) || thought >= muse_thought_tokens)
            && let Some(eom) = start_token
        {
            next = eom;
        }
        if tokenizer.is_end_of_generation(next) && !muse_turn {
            break;
        }
        // A thought isn't the answer that's due, shown (thinking on) or left out (Gemma's
        // channel with thinking off): a reply that thought and then ended said nothing.
        let of_a_thought = in_thought
            || in_channel
            || think.is_some_and(|(open, _)| next == open)
            || (channel.0.is_some() && Some(next) == channel.0);
        if !muse_turn && !thinking && !of_a_thought {
            answer_due = false;
        }
        if debug_calls() && turn_start {
            eprintln!(
                "[calls] turn starts with {:?}",
                tokenizer.decode(&[next], true)
            );
        }
        let by_token = !muse_turn && Some(next) == start_token;
        // Muse's message start is the grammar's, as Llama's brace is (its pick is left).
        let by_brace = muse_turn
            || start.brace && turn_start && {
                let mut b = Vec::new();
                tokenizer.token_bytes(next, false, &mut b);
                b.first() == Some(&b'{')
            };
        // Muse's message start is always a choice, and its <|eom|> (taken away unless it
        // may call or is thinking) always opens another message.
        let opens_call = if family == Family::Muse {
            muse_turn || by_token
        } else {
            may_call && (by_token || by_brace)
        };
        // The guess is taken when it's the token picked, for plain text; otherwise the
        // model goes back to before the guesses.
        let guessed = !opens_call && ahead.front().is_some_and(|(g, _)| *g == next);
        if !guessed && !ahead.is_empty() {
            ahead.clear();
            model.rewind(kept)?;
        }
        if opens_call {
            // Where the call starts, to take it back if it runs too long.
            let call_begin = kept.tokens.len();
            if in_thought {
                in_thought = false;
                shown.tag(job, THINK_CLOSE, false);
            } else {
                shown.flush(job, true);
            }
            if by_token {
                // The call's own start token is the model's; Llama's brace is the grammar's.
                model.next(&mut kept.session, next, &mut logits)?;
                kept.tokens.push(next);
                call_tokens += 1;
                if family == Family::Muse {
                    // Muse's <|eom|>: another message follows, its recipient chosen at its start.
                    let ids = tokenizer.encode(chat::opening(family, false), false, true);
                    logits = model.feed(kept, &ids)?;
                    call_tokens += ids.len();
                    turn_start = true;
                    thinking = false;
                    continue;
                }
            }
            let call_started = Instant::now();
            let grammar = match &muse_grammars {
                Some((_, answers)) if !may_call => answers,
                Some((no_thinking, _)) if !may_think => no_thinking,
                _ => &grammar,
            };
            let text = match write_call(
                model,
                kept,
                tokenizer,
                trie,
                grammar,
                &mut logits,
                &mut call_tokens,
                context,
                call_draft(mtp, &mut mtp_stats, &mut lookup),
            ) {
                // Too long to be a call a tool would take: taken back, and the model
                // answers from what it has (failing the reply left the user with nothing
                // after minutes of writing).
                Err(e) if e == CALL_TOO_LONG && family != Family::Muse => {
                    eprintln!("A tool call ran past its limit: taken back, no more calls this reply.");
                    logits = take_back(model, kept, call_begin)?;
                    if let Some(d) = drafter.as_deref_mut() {
                        d.sync(&kept.tokens)?;
                    }
                    ahead.clear();
                    may_call = false;
                    continue;
                }
                other => other?,
            };
            if family == Family::Muse
                && (text == tools::MUSE_TO_USER || text == tools::MUSE_TO_SELF)
            {
                // To the user, or to itself (not shown): plain text from here.
                turn_start = false;
                thinking = text == tools::MUSE_TO_SELF;
                continue;
            }
            let call = tools::parse_call(family, &text, &callable)
                .ok_or_else(|| format!("Couldn't read back a tool call: {text:?}"))?;
            // The call as it was written, from its start token (Muse's from its recipient).
            let written_call = if by_token && family != Family::Muse {
                format!("{}{text}", start.token)
            } else {
                text.clone()
            };
            let mut group = vec![(call, written_call, call_started.elapsed())];
            // Gemma 4 may write several calls in one step, one after another, before any
            // result (its template's form for them): after each call's end, its next
            // token says whether another follows. Otherwise it would write the results'
            // start, which is the server's to write.
            let close = (family == Family::Gemma4)
                .then(|| tokenizer.encode(tools::GEMMA_CALL_CLOSE, false, true));
            if let Some(close) = &close {
                logits = model.feed(kept, close)?;
                call_tokens += close.len();
                while may_call && calls + group.len() < MAX_CALLS && group.len() < MAX_TOGETHER {
                    let peek = sampler.pick(&logits, &generated);
                    if Some(peek) != start_token {
                        break;
                    }
                    let started = Instant::now();
                    let call_begin = kept.tokens.len();
                    model.next(&mut kept.session, peek, &mut logits)?;
                    kept.tokens.push(peek);
                    call_tokens += 1;
                    let text = match write_call(
                        model,
                        kept,
                        tokenizer,
                        trie,
                        grammar,
                        &mut logits,
                        &mut call_tokens,
                        context,
                        call_draft(mtp, &mut mtp_stats, &mut lookup),
                    ) {
                        // As above: this call is taken back, the ones before it stand.
                        Err(e) if e == CALL_TOO_LONG => {
                            eprintln!("A tool call ran past its limit: taken back, no more calls this reply.");
                            // (The results read next give the logits to go on from.)
                            take_back(model, kept, call_begin)?;
                            if let Some(d) = drafter.as_deref_mut() {
                                d.sync(&kept.tokens)?;
                            }
                            may_call = false;
                            break;
                        }
                        other => other?,
                    };
                    let call = tools::parse_call(family, &text, &callable)
                        .ok_or_else(|| format!("Couldn't read back a tool call: {text:?}"))?;
                    group.push((call, format!("{}{text}", start.token), started.elapsed()));
                    logits = model.feed(kept, close)?;
                    call_tokens += close.len();
                }
            }
            // A thought after the results, as the template opens one: a short one after
            // changes only (a file saved, an event made: 0.9's routine step) or once the
            // reply's thoughts have had their share, so a long task still thinks between
            // its steps.
            let thinks_next = thinking_on;
            let routine = group
                .iter()
                .all(|(c, _, _)| offered.iter().any(|t| t.name == c.tool && t.asks.is_some()));
            let together = group.len();
            let mut ids = Vec::new();
            let mut stopped = false;
            let mut round_repeated = false;
            for (i, (call, written_call, written)) in group.into_iter().enumerate() {
                let last = i + 1 == together;
                calls += 1;
                let args = call.args_json();
                let _ = job.out.send(Out::Event(format!(
                    "{{\"type\":\"tool_call\",\"tool\":{},\"args\":{args}}}",
                    json::string(&call.tool)
                )));
                // The same call again, with the same arguments, would get the same answer: a
                // model can loop on one, or on a round of several (Gemma 4 31B read, summed
                // and read again four times over in the Nov 13 benchmark's skill task, never
                // reaching the file it was to write). It isn't run again, whether it was the
                // last call or an earlier one, as long as nothing was changed in between. The
                // first round that repeats, the model may still call another tool (Qwen 3.5
                // 4B repeated a sum in the test gauntlet, then had no way left to fill the
                // template it was asked to); the second, it must answer.
                let repeated = seen.contains(&call);
                if repeated && !round_repeated {
                    repeats += 1;
                    round_repeated = true;
                }
                let run_started = Instant::now();
                let result = if repeated && repeats >= 2 {
                    "This is the same call as one before, and its result is above. Answer the user from it now."
                        .to_string()
                } else if repeated {
                    "This is the same call as one before, and its result is above. Go on from it: call the next tool the task needs, or answer."
                        .to_string()
                } else {
                    match run_tool(&call, &offered, job) {
                        Some(r) => r,
                        None => {
                            stopped = true; // stopped while the app ran it
                            break;
                        }
                    }
                };
                if repeats >= 2 || calls == MAX_CALLS {
                    may_call = false;
                }
                if offered.iter().any(|t| t.name == call.tool && t.asks.is_some()) {
                    // A change: what the earlier calls read may read differently now.
                    seen.clear();
                } else if !repeated {
                    seen.push(call.clone());
                }
                eprintln!(
                    "Tool: {} {args} (written in {:.0} ms, run in {:.0} ms{}), {} characters back.",
                    call.tool,
                    written.as_secs_f64() * 1000.0,
                    run_started.elapsed().as_secs_f64() * 1000.0,
                    if together > 1 {
                        format!(", {} of {together} together", i + 1)
                    } else {
                        String::new()
                    },
                    result.len()
                );
                let _ = job.out.send(Out::Event(format!(
                    "{{\"type\":\"tool_result\",\"tool\":{},\"result\":{}}}",
                    json::string(&call.tool),
                    json::string(&preview(&result))
                )));
                let mut read = shaped(
                    &chat::plain(&result),
                    tools::result_limit(&call, MAX_RESULT),
                );
                if calls == MAX_CALLS {
                    // Told so, the model answers from what it has; left to find its calls
                    // taken away, Gemma 4 31B ended a reply with nothing said (2026-11-12).
                    read.push_str(LAST_CALL_NOTE);
                }
                let (before, body, after) =
                    tools::after_call(family, &call.tool, &read, thinks_next && last);
                // The call's end was read already (Gemma's, to see whether another follows).
                let before = match close {
                    Some(_) => before
                        .strip_prefix(tools::GEMMA_CALL_CLOSE)
                        .unwrap_or(&before)
                        .to_string(),
                    None => before,
                };
                steps.push(chat::Step {
                    at: shown.said.trim_start().len(),
                    tool: call.tool.clone(),
                    call: written_call,
                    result: read,
                    then: !last,
                });
                ids.extend(tokenizer.encode(&before, false, true));
                ids.extend(tokenizer.encode(&body, false, false));
                ids.extend(tokenizer.encode(&after, false, true));
            }
            if stopped {
                break;
            }
            if kept.tokens.len() + ids.len() + 64 > context {
                return Err("The tool's result doesn't fit in this chat's context.".into());
            }
            logits = model.feed(kept, &ids)?;
            turn_start = true;
            answer_due = true;
            if thinks_next {
                in_thought = true;
                thought_tokens = 0;
                thought_cap = if routine || thought >= REPLY_THOUGHT_TOKENS {
                    SHORT_THOUGHT_TOKENS
                } else {
                    THOUGHT_TOKENS
                };
                label = false;
                shown.tag(job, THINK_OPEN, true);
            }
            continue;
        }
        turn_start = false;
        let opens_thought = !in_thought && think.is_some_and(|(open, _)| next == open);
        let closes_thought = in_thought && think.is_some_and(|(_, close)| next == close);
        if thinking {
            // Muse's reasoning: the model's, not the reply's.
            thought += 1;
        } else if opens_thought {
            thought += 1;
            in_thought = true;
            thought_tokens = 0;
            thought_cap = if thought >= REPLY_THOUGHT_TOKENS {
                SHORT_THOUGHT_TOKENS
            } else {
                THOUGHT_TOKENS
            };
            label = family == Family::Gemma4;
            shown.tag(job, THINK_OPEN, true);
        } else if closes_thought {
            thought += 1;
            in_thought = false;
            shown.tag(job, THINK_CLOSE, false);
        } else if in_thought {
            // Sent as it comes, as the model's reasoning.
            thought += 1;
            thought_tokens += 1;
            tokenizer.token_bytes(next, false, &mut shown.pending);
            if label {
                label = shown.drop_label(GEMMA_THOUGHT_LABEL);
            }
            if !label && !shown.send(job, false) {
                break;
            }
        } else {
            generated.push(next);
            // Gemma 4's thinking channel, should it open one with thinking off: left out.
            if channel.0.is_some() && Some(next) == channel.0 {
                in_channel = true;
            } else if channel.1.is_some() && Some(next) == channel.1 {
                in_channel = false;
            } else if !in_channel {
                tokenizer.token_bytes(next, false, &mut shown.pending);
                if !shown.send(job, true) {
                    break;
                }
            }
        }
        if guessed {
            logits = ahead.pop_front().unwrap().1;
            if let Some(d) = drafter.as_deref_mut() {
                d.taken += 1;
            }
            if ahead_drafted {
                mtp_stats.1 += 1;
            } else if let Some(l) = lookup.as_mut() {
                l.taken += 1;
            }
        } else if drafting {
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
    if in_thought {
        shown.tag(job, THINK_CLOSE, false);
    } else {
        shown.flush(job, true);
    }
    if !steps.is_empty() {
        let record = chat::Record {
            family: family.name().into(),
            steps,
        };
        let _ = job.out.send(Out::Event(format!(
            "{{\"type\":\"turn_record\",\"record\":{}}}",
            json::string(&record.to_json())
        )));
    }
    let seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "Reply: {} prompt tokens ({reused} kept, {} read in {prompt_seconds:.2}s, {:.0}/s), {} tokens in {seconds:.2}s ({:.1}/s), {calls} tool calls{}{}{}.",
        ids.len(),
        ids.len() - reused,
        (ids.len() - reused) as f64 / prompt_seconds.max(1e-9),
        generated.len() + call_tokens + thought,
        (generated.len() + call_tokens + thought) as f64 / seconds.max(1e-9),
        if thought > 0 {
            format!(", {thought} of them reasoning")
        } else {
            String::new()
        },
        if job.stop.load(Ordering::Relaxed) {
            ", stopped"
        } else {
            ""
        },
        drafter
            .as_deref()
            .map(|d| format!(
                ", {} of {} guesses taken ({:.0}%, {:.2}s guessing)",
                d.taken,
                d.guessed,
                100.0 * d.taken as f64 / d.guessed.max(1) as f64,
                d.seconds
            ))
            .unwrap_or_else(|| {
                let chat = lookup.as_ref().map(|l| {
                    format!(
                        ", {} of {} guesses from the chat taken ({:.0}%)",
                        l.taken,
                        l.guessed,
                        100.0 * l.taken as f64 / l.guessed.max(1) as f64
                    )
                });
                let drafted = mtp.then(|| {
                    format!(
                        ", {} of {} guesses from the {} taken ({:.0}%)",
                        mtp_stats.1,
                        mtp_stats.0,
                        if matches!(kept.session, Session::Gemma4(_)) {
                            "draft model"
                        } else {
                            "multi-token-prediction layer"
                        },
                        100.0 * mtp_stats.1 as f64 / mtp_stats.0.max(1) as f64
                    )
                });
                chat.unwrap_or_default() + &drafted.unwrap_or_default()
            })
    );
    let (guessed, taken) = match (drafter.as_deref(), lookup.as_ref()) {
        (Some(d), _) => (d.guessed, d.taken),
        (None, l) => (
            l.map_or(0, |l| l.guessed) + mtp_stats.0,
            l.map_or(0, |l| l.taken) + mtp_stats.1,
        ),
    };
    let _ = job.out.send(Out::Usage(stats::Usage {
        prompt_tokens: ids.len(),
        cached_tokens: reused,
        prefill_seconds: prompt_seconds,
        context_tokens: kept.tokens.len(),
        tool_calls: calls,
        guessed,
        taken,
    }));
    let _ = job.out.send(Out::Done {
        tokens: generated.len() + call_tokens + thought,
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
/// is at least `SKILL_SHARE` and at least `OVER_CONTROL` above the highest share it gets
/// for requests that fit no skill (`skills::CONTROLS`, scored once per skill while this
/// model runs). A model whose yes means nothing (Qwen 3.5 0.8B said yes to nearly every skill
/// for every request) so picks none, as before skills, instead of a wrong one.
/// How many skills are asked about one by one, after the shortlist question.
const SKILLS_ASKED: usize = 2;

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
    // With more than a couple of skills, one question about them all first: only the
    // likeliest are then asked about one by one, so a turn's cost doesn't grow with the
    // skills. A skill is still taken only on its own question's yes, as before.
    // (A list too long for a picker's session is left out, and every skill asked about.)
    let mut shortlist_ids = None;
    if options().skill_shortlist && skills.len() > SKILLS_ASKED {
        let question = Message::new("user", skills::shortlist_question(&skills, request));
        let text = chat::history(family, None, "", &[question], &chat::today(), false)
            + chat::answer_opening(family)
            + skills::SHORTLIST_LEAD;
        let ids = tokenizer.encode(&text, false, true);
        shortlist_ids = Some(ids);
    }
    let asked: Vec<usize> = if let Some(ids) = shortlist_ids.filter(|ids| ids.len() <= 1000) {
        let logits = model.ask(kept, &ids)?;
        let mut ranked: Vec<(f32, usize)> = (0..skills.len().min(skills::SHORTLISTED))
            .map(|i| {
                // The first token of the skill's name, as the next word.
                let id = tokenizer.encode(&format!(" {}", skills[i].name), false, false)[0];
                (logits[id as usize], i)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        if debug_calls() {
            let names: Vec<&str> = ranked
                .iter()
                .take(SKILLS_ASKED)
                .map(|&(_, i)| skills[i].name.as_str())
                .collect();
            eprintln!("[calls] skills shortlisted: {}", names.join(", "));
        }
        ranked.iter().take(SKILLS_ASKED).map(|&(_, i)| i).collect()
    } else {
        (0..skills.len()).collect()
    };
    let mut best: Option<(f64, usize)> = None;
    for i in asked {
        let skill = &skills[i];
        let question = skills::fit_question(skill, request);
        let share = yes_share(model, kept, tokenizer, family, &question, "Fits:")?;
        let mut control: f64 = 0.0;
        for request in skills::CONTROLS {
            let control_question = skills::fit_question(skill, request);
            let c = match kept.skill_controls.get(&control_question) {
                Some(c) => *c,
                None => {
                    let c = yes_share(model, kept, tokenizer, family, &control_question, "Fits:")?;
                    kept.skill_controls.insert(control_question, c);
                    c
                }
            };
            control = control.max(c);
        }
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
    let question = Message::new("user", question);
    let text = chat::history(family, None, "", &[question], &chat::today(), false)
        + chat::answer_opening(family)
        + lead;
    let ids = tokenizer.encode(&text, false, true);
    let logits = model.ask(kept, &ids[..ids.len().min(1000)])?;
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
    let question = Message::new("user", tools::group_question(request));
    // The answer opens with the examples' "Needs:", so its next token is the letter.
    let text = chat::history(family, None, "", &[question], &chat::today(), false)
        + chat::answer_opening(family)
        + "Needs:";
    let ids = tokenizer.encode(&text, false, true);
    let logits = model.ask(kept, &ids[..ids.len().min(1000)])?;
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

/// The most tool calls one reply makes: enough for a task over several files (read a
/// guide, six invoices and a register, work out the sums, write the rows).
const MAX_CALLS: usize = 30;
/// The most calls Gemma 4 may write together in one step.
const MAX_TOGETHER: usize = 8;
/// What the last call's result ends with.
const LAST_CALL_NOTE: &str = "\n\n[That was the last tool call this reply can make. Answer now from what you have, and say plainly what is still to do.]";
/// The most tokens one thought may take (thinking on; the Mac server's budget), and the
/// share of a reply's thoughts past which each later one is short. Past its cap a thought
/// is closed for the model. Before 2026-11-13 thoughts stopped altogether past 8,000, and
/// a long task went on without thinking between its steps.
const THOUGHT_TOKENS: usize = 2000;
const REPLY_THOUGHT_TOKENS: usize = 8000;
/// A short thought: after a change (0.9's 400-token thought after a routine step), and
/// once a reply's thoughts have had their share.
const SHORT_THOUGHT_TOKENS: usize = 400;
/// A thought's start and end as the app is told of them (`thinking` events, with the
/// thought's text between as their `content`): it shows a thought as the model's
/// reasoning, apart from the answer.
const THINK_OPEN: &str = "\"start\"";
const THINK_CLOSE: &str = "\"done\"";
/// What Gemma 4 opens a thought of its own with.
const GEMMA_THOUGHT_LABEL: &[u8] = b"thought\n";
/// The most tokens Muse may reason in, a reply (at "Reasoning strength: low"). Past it
/// its reasoning message is ended for it and it answers or calls without more.
const MUSE_THOUGHT_TOKENS: usize = 2048;
/// The most tokens one call may take: a document's text or a spreadsheet's rows at most.
const MAX_CALL_TOKENS: usize = 4096;
/// A `calculate` call's: the tool takes 2,000 characters at most, and a model that writes
/// more is copying its data into the sum (Qwen 3.8 27B wrote a log's numbers out for five
/// minutes in the Nov 13 benchmark).
const MAX_CALCULATE_TOKENS: usize = 768;
/// What `write_call` gives back for a call past its limit: the reply takes the call back
/// and goes on without tools (`take_back`).
const CALL_TOO_LONG: &str = "A tool call ran too long.";
/// The most of a tool's result the model reads, in characters (about 2,000 tokens).
const MAX_RESULT: usize = 8000;

/// A reply's text on its way to the app: what's held back (only whole characters are
/// sent as they come: a character's bytes can span tokens), and the answer sent so far,
/// which is where a tool call is said to have been made (`chat::Step::at`).
#[derive(Default)]
struct Shown {
    pending: Vec<u8>,
    /// The answer's text so far, thoughts left out.
    said: String,
}

impl Shown {
    /// The answer's text goes as the reply's content; a thought's as `thinking` events,
    /// which an app shows apart from the answer (or not at all).
    fn out(&mut self, job: &Job, text: String, answer: bool) -> bool {
        if answer {
            self.said.push_str(&text);
            job.out.send(Out::Text(text)).is_ok()
        } else {
            job.out
                .send(Out::Event(format!(
                    "{{\"type\":\"thinking\",\"content\":{}}}",
                    json::string(&text)
                )))
                .is_ok()
        }
    }

    /// Sends the whole characters held back. `false` once nobody is reading.
    fn send(&mut self, job: &Job, answer: bool) -> bool {
        let whole = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.pending.len(),
        };
        if whole == 0 {
            return true;
        }
        let text = String::from_utf8_lossy(&self.pending[..whole]).into_owned();
        self.pending.drain(..whole);
        self.out(job, text, answer)
    }

    /// Sends all that's held back.
    fn flush(&mut self, job: &Job, answer: bool) {
        if !self.pending.is_empty() {
            let text = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.out(job, text, answer);
        }
    }

    /// Says a thought has started or ended (`mark`), after what's held back (`answer`:
    /// whether that was the answer's, as before a thought starts, or the thought's).
    fn tag(&mut self, job: &Job, mark: &str, answer: bool) {
        self.flush(job, answer);
        let _ = job
            .out
            .send(Out::Event(format!("{{\"type\":\"thinking\",{mark}:true}}")));
    }

    /// Takes `label` off the start of what's held back. Whether it may still be coming
    /// (what's held is only its start so far).
    fn drop_label(&mut self, label: &[u8]) -> bool {
        if self.pending.starts_with(label) {
            self.pending.drain(..label.len());
            false
        } else {
            label.starts_with(&self.pending)
        }
    }
}

/// Where a tool call's guesses come from, as the reply's own do: the model's draft layer
/// or draft model, or the chat's own tokens. (Not a separate small model's: `Drafter`
/// would have to read the call too.)
enum CallDraft<'a> {
    None,
    Mtp(&'a mut (usize, usize)),
    Lookup(&'a mut draft::Lookup),
}

/// The most guesses a round inside a tool call.
const CALL_GUESSES: usize = 8;

fn call_draft<'a>(
    mtp: bool,
    stats: &'a mut (usize, usize),
    lookup: &'a mut Option<draft::Lookup>,
) -> CallDraft<'a> {
    match lookup {
        _ if mtp => CallDraft::Mtp(stats),
        Some(l) => CallDraft::Lookup(l),
        None => CallDraft::None,
    }
}

/// Takes the session back to its first `len` tokens (a call that ran too long, from its
/// start token on): the last of them's logits. Whatever the model's cache can't be cut
/// back to is read again from the chat's last kept point.
fn take_back(model: &Model, kept: &mut Kept, len: usize) -> Result<Vec<f32>, String> {
    let ids = kept.tokens[..len].to_vec();
    let mark = kept.history_end.min(len);
    let (logits, _) = model.continue_with(kept, &ids, mark, 0)?;
    Ok(logits)
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
    mut draft: CallDraft,
) -> Result<String, String> {
    let mut m = grammar.matcher();
    let timing = std::env::var_os("VIREO_CALL_TIME").is_some();
    let (mut t_forced, mut t_scored, mut t_free, mut t_allowed) = (0f64, 0f64, 0f64, 0f64);
    let (mut n_forced, mut n_scored, mut n_free) = (0usize, 0usize, 0usize);
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
                let t0 = Instant::now();
                *logits = model.feed(kept, &ids[..keep])?;
                t_forced += t0.elapsed().as_secs_f64();
                n_forced += keep;
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
            let t0 = Instant::now();
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
            t_scored += t0.elapsed().as_secs_f64();
            n_scored += ids.len();
            *tokens += ids.len();
            continue;
        }
        let t0 = Instant::now();
        let allowed = m.allowed(trie);
        t_allowed += t0.elapsed().as_secs_f64();
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
        room(kept, 1 + CALL_GUESSES)?;
        let t0 = Instant::now();
        // The token is read with the draft layer's guesses at what follows it, where the
        // model is free to write (a path, a sum, a file's text): each guess the model
        // would have picked itself, under the grammar, is taken without a pass of its
        // own, as in the reply's own text. Where the grammar writes the next text or
        // scores a choice, the guesses are dropped and it goes on as before. (Tool calls
        // were 70% of the writing in Gemma 4 31B's tool tasks, a token at a time.)
        let free_next = |m: &vireo_engine::grammar::Matcher| {
            !m.is_complete() && m.forced().is_empty() && m.scored_options().is_none()
        };
        kept.tokens.push(next);
        *tokens += 1;
        n_free += 1;
        let guesses = match &mut draft {
            CallDraft::Mtp(stats) if free_next(&m) => {
                let g = model.mtp_draft(kept, next, MTP_GUESSES)?;
                stats.0 += g.len();
                g
            }
            CallDraft::Lookup(l) if free_next(&m) => {
                l.guess(&kept.tokens, l.max.min(CALL_GUESSES))
            }
            _ => Vec::new(),
        };
        if guesses.is_empty() {
            model.next(&mut kept.session, next, logits)?;
        } else {
            let batch: Vec<u32> = std::iter::once(next).chain(guesses.iter().copied()).collect();
            let rows = model.read_all(&mut kept.session, &batch)?;
            let vocab = rows.len() / batch.len();
            let mut taken = 0;
            for &guess in &guesses {
                if !free_next(&m) {
                    break;
                }
                let row = &rows[taken * vocab..(taken + 1) * vocab];
                let allowed = m.allowed(trie);
                let pick = (0..vocab as u32)
                    .filter(|&id| allowed[id as usize / 64] >> (id % 64) & 1 == 1)
                    .max_by(|&a, &b| row[a as usize].total_cmp(&row[b as usize]));
                if pick != Some(guess) {
                    break;
                }
                bytes.clear();
                tokenizer.token_bytes(guess, false, &mut bytes);
                if !m.advance_all(&bytes) {
                    return Err("A token the grammar allowed didn't fit it.".into());
                }
                text.extend_from_slice(&bytes);
                kept.tokens.push(guess);
                *tokens += 1;
                n_free += 1;
                taken += 1;
            }
            logits.copy_from_slice(&rows[taken * vocab..(taken + 1) * vocab]);
            if taken < guesses.len() {
                model.rewind(kept)?;
            }
            match &mut draft {
                CallDraft::Mtp(stats) => stats.1 += taken,
                CallDraft::Lookup(l) => l.taken += taken,
                CallDraft::None => {}
            }
        }
        t_free += t0.elapsed().as_secs_f64();
        let limit = if text.len() >= 48 && text[..48].windows(9).any(|w| w == b"calculate") {
            MAX_CALCULATE_TOKENS
        } else {
            MAX_CALL_TOKENS
        };
        if *tokens - first > limit {
            return Err(CALL_TOO_LONG.into());
        }
    }
    if timing {
        eprintln!(
            "[call time] forced {n_forced} tokens {:.0} ms, scored {n_scored} tokens {:.0} ms, free {n_free} tokens {:.0} ms, allowed sets {:.0} ms",
            t_forced * 1e3,
            t_scored * 1e3,
            t_free * 1e3,
            t_allowed * 1e3
        );
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
        if tool.runs == tools::Runs::App && job.client_tools.iter().any(|t| t == CHECK_CHANGE) {
            let args = format!(
                "{{\"tool\":{},\"args\":{}}}",
                json::string(&call.tool),
                call.args_json()
            );
            if let Err(why) = ask_app(CHECK_CHANGE, &args, job)? {
                return Some(format!("Error: {why}"));
            }
        }
        // A task's schedule is read before anyone is asked to approve it.
        if call.tool == "schedule_task"
            && let Err(why) = tasks::When::parse(call.text("when").unwrap_or(""))
        {
            return Some(format!("Error: {why}"));
        }
        // The same check for a change the server makes itself.
        if tool.runs == tools::Runs::Server
            && let Err(why) = tools::check_here(call, job.files.as_ref())
        {
            return Some(format!("Error: {why}"));
        }
        match approve(call, ask, tool.runs == tools::Runs::Server, job) {
            Some(Ok(())) => {}
            Some(Err(why)) => return Some(format!("Error: {why}")),
            None => return None,
        }
    }
    let result = match tool.runs {
        tools::Runs::Server if call.tool == "search_chats" => {
            chats::search(call.text("query").unwrap_or(""), job.chat_id.as_deref())
        }
        tools::Runs::Server if call.tool == "schedule_task" => schedule(call, job),
        tools::Runs::Server => tools::run_here(call, job.files.as_ref()),
        // Several files in one call: the app is asked for each.
        tools::Runs::App if tools::several_paths(call).is_some() => {
            let paths = tools::several_paths(call).unwrap_or_default();
            let stopped = std::cell::Cell::new(false);
            let text = tools::read_several(&paths, |path| {
                if stopped.get() {
                    return Err("stopped".into());
                }
                match ask_app("read_file", &tools::read_one(path).args_json(), job) {
                    Some(result) => result,
                    None => {
                        stopped.set(true);
                        Err("stopped".into())
                    }
                }
            });
            if stopped.get() {
                return None;
            }
            Ok(text)
        }
        // The app answers create_docx with the file's path alone; the server words it.
        tools::Runs::App if call.tool == "create_docx" => {
            ask_app(&call.tool, &call.args_json(), job)?
                .map(|path| format!("Word document saved: {}", path.trim()))
        }
        tools::Runs::App => ask_app(&call.tool, &call.args_json(), job)?,
    };
    Some(result.unwrap_or_else(|e| format!("Error: {e}")))
}

/// Sets up the task a `schedule_task` call asks for (`tasks.rs`), with what this chat may
/// reach: a folder to watch must be one of them.
fn schedule(call: &tools::Call, job: &Job) -> Result<String, String> {
    let mut when = tasks::When::parse(call.text("when").unwrap_or(""))?;
    if let tasks::When::Folder { path } = &mut when {
        let files = job
            .files
            .as_ref()
            .ok_or("This chat can't reach the user's files, so it can't watch a folder.")?;
        *path = files.resolve(path)?;
        if std::path::Path::new(path.as_str()).is_file() {
            return Err(format!("{path} is a file, not a folder."));
        }
    }
    tasks::add(
        call.text("name").unwrap_or(""),
        call.text("prompt").unwrap_or(""),
        when,
        &job.access,
        &job.folders,
    )
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
fn approve(
    call: &tools::Call,
    ask: tools::Ask,
    here: bool,
    job: &Job,
) -> Option<Result<(), String>> {
    if ask.rememberable && job.approvals.always.lock().unwrap().contains(&call.tool) {
        eprintln!("Approved without asking (always allowed): {}", call.tool);
        return Some(Ok(()));
    }
    let id = format!("confirm-{:016x}", random_u64());
    let (tx, rx) = channel();
    job.approvals.waiting.lock().unwrap().insert(id.clone(), tx);
    let preview = tools::approval_preview(call, |p| on_this_pc(p).exists());
    // A file the server deletes itself goes to this Mac's Trash, not an app's Recycle Bin.
    let preview = if here && cfg!(target_os = "macos") {
        preview.replace("to the Recycle Bin", "to the Trash")
    } else {
        preview
    };
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

/// A tool's result as the model reads it: at most `limit` characters (`MAX_RESULT`, or
/// more for a call that read several files), saying how much was left out.
fn shaped(result: &str, limit: usize) -> String {
    let chars = result.chars().count();
    if chars <= limit {
        return result.to_string();
    }
    let kept: String = result.chars().take(limit).collect();
    format!("{kept}\n[... {} more characters not shown]", chars - limit)
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
