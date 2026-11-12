//! `vireo-eval`: how well each model uses Vireo's tools. It starts vireo-server with a
//! model, stands in for the app (a PC of a few files in memory, `pc.rs`, running the
//! app's tools the way the app does), sends each case (`cases.rs`) as a new chat, and
//! scores the reply: the tools it called, their arguments, and the answer. Replies are
//! greedy (temperature 0), so a run gives the same scores every time.
//!
//! Twelve easy cases (`cases.rs`), twelve harder ones (`hard_cases.rs`) and the ones
//! that change files (`change_cases.rs`), counted apart in the report. The stand-in app
//! answers approvals as the case's user does (`Pc::declining`), and any case not about
//! changes fails if the reply changes a file.
//!
//!   vireo-eval [--models a.gguf,b.gguf] [--cases read,git] [--out results.json]
//!              [--server-arg <arg>]...
//!
//! Models are paths, or names in VireoPrototype/models; every test model by default.
//! `--server-arg` passes an argument on to vireo-server, to compare a feature on and off.

mod cases;
mod change_cases;
mod hard_cases;
#[path = "../../vireo-server/src/json.rs"]
#[allow(dead_code)]
mod json;
mod pc;
mod skill_cases;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitCode, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cases::{Call, Case, Run};
use json::Value;

/// Small to large in each family: where a feature stops mattering shows along each.
/// Models people choose today. Llama 3.2 1B (2024) isn't one any more, and Qwen 3.5 0.8B
/// was left out on 2026-11-11 (too small to use the tools well: it said yes to every skill
/// and saved a poem to a file); both stay the engine's references in its own tests, and run
/// here on request (`--models`).
const DEFAULT_MODELS: [&str; 4] = [
    "gemma-4-E2B_q4_0-it.gguf",
    "gemma-4-E4B_q4_0-it.gguf",
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf",
];
const TOKEN: &str = "vireo-eval";
/// The PC tools the stand-in app runs (its `clientTools`).
const CLIENT_TOOLS: [&str; 11] = [
    "read_file",
    "list_directory",
    "search_files",
    "file_exists",
    "git_status",
    "git_diff",
    "write_file",
    "rename_file",
    "move_file",
    "delete_file",
    // Not a tool: the server checks a change with it before asking for approval.
    "check_change",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut models, mut only, mut out, mut server_args) =
        (Vec::new(), Vec::new(), None, Vec::new());
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).cloned();
        match (args[i].as_str(), value) {
            ("--models", Some(v)) => models = v.split(',').map(str::to_string).collect(),
            ("--cases", Some(v)) => only = v.split(',').map(str::to_string).collect(),
            ("--out", Some(v)) => out = Some(PathBuf::from(v)),
            ("--server-arg", Some(v)) => server_args.push(v),
            _ => {
                eprintln!(
                    "Usage: vireo-eval [--models a,b] [--cases a,b] [--out file.json] [--server-arg arg]..."
                );
                return ExitCode::FAILURE;
            }
        }
        i += 2;
    }
    if models.is_empty() {
        models = DEFAULT_MODELS.iter().map(|s| s.to_string()).collect();
    }
    let here = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let Some(here) = here else {
        return ExitCode::FAILURE;
    };
    let server = here.join("vireo-server.exe");
    let server = if server.exists() {
        server
    } else {
        here.join("vireo-server")
    };
    let models_dir = here.join("../../models");

    let cases: Vec<Case> = cases::all()
        .into_iter()
        .filter(|c| only.is_empty() || only.iter().any(|o| o == c.id))
        .collect();
    let mut report = Vec::new();
    for model in &models {
        let path = if Path::new(model).exists() {
            PathBuf::from(model)
        } else {
            models_dir.join(model)
        };
        let name = path
            .file_stem()
            .map_or(model.clone(), |s| s.to_string_lossy().into_owned());
        let mut running = match Server::start(&server, &path, &server_args) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{name}: {e}");
                continue;
            }
        };
        println!("\n{name}");
        let mut runs = Vec::new();
        for case in &cases {
            let run = ask(running.port, case);
            let verdict = match (&run.error, (case.check)(&run)) {
                (Some(e), _) => Err(format!("error: {e}")),
                (None, _) if !case.changes && !case.skills && !run.changes_made().is_empty() => {
                    Err(format!(
                        "called {} when nobody asked for a change",
                        run.changes_made()[0].tool
                    ))
                }
                (None, v) => v,
            };
            let tools: Vec<String> = run.calls.iter().map(|c| c.tool.clone()).collect();
            println!(
                "  {} {:<16} {:<40} {:>5.1} s  {}{}",
                if verdict.is_ok() { "pass" } else { "FAIL" },
                case.id,
                case.title,
                run.seconds,
                if tools.is_empty() {
                    "no tools".into()
                } else {
                    tools.join(" > ")
                },
                verdict
                    .as_ref()
                    .err()
                    .map_or(String::new(), |why| format!("  ({why})"))
            );
            runs.push((case.id, verdict, run));
        }
        let passed = runs.iter().filter(|r| r.1.is_ok()).count();
        let seconds: f64 = runs.iter().map(|r| r.2.seconds).sum();
        let calls: usize = runs.iter().map(|r| r.2.calls.len()).sum();
        let part = |hard: bool, changes: bool, skills: bool| {
            let of: Vec<_> = runs
                .iter()
                .filter(|r| r.2.hard == hard && r.2.changes == changes && r.2.skill_case == skills)
                .collect();
            (of.iter().filter(|r| r.1.is_ok()).count(), of.len())
        };
        let ((easy_passed, easy), (hard_passed, hard)) =
            (part(false, false, false), part(true, false, false));
        let (changes_passed, changes) = part(false, true, false);
        let (skills_passed, skill_cases) = part(false, false, true);
        // Skills picked where a case didn't ask for one: worth watching, not a failure.
        let stray = runs
            .iter()
            .filter(|r| !r.2.skill_case && !r.2.skills.is_empty())
            .count();
        println!(
            "  {passed}/{} passed ({easy_passed}/{easy} easy, {hard_passed}/{hard} hard, {changes_passed}/{changes} changes, {skills_passed}/{skill_cases} skills; a skill in {stray} others), {:.1} s a case on average, {calls} tool calls",
            runs.len(),
            seconds / runs.len().max(1) as f64
        );
        running.stop();
        report.push((name, runs));
    }

    if let Some(out) = out {
        if let Err(e) = std::fs::write(&out, results_json(&report, &server_args)) {
            eprintln!("Couldn't write {}: {e}", out.display());
            return ExitCode::FAILURE;
        }
        println!("\nResults in {}", out.display());
    }
    ExitCode::SUCCESS
}

type Report<'a> = Vec<(String, Vec<(&'a str, Result<(), String>, Run)>)>;

fn results_json(report: &Report, server_args: &[String]) -> String {
    let models: Vec<String> = report
        .iter()
        .map(|(name, runs)| {
            let cases: Vec<String> = runs
                .iter()
                .map(|(id, verdict, run)| {
                    let tools: Vec<String> = run
                        .calls
                        .iter()
                        .map(|c| format!("{{\"tool\":{},\"args\":{}}}", json::string(&c.tool), to_json(&c.args)))
                        .collect();
                    format!(
                        "{{\"id\":{},\"hard\":{},\"changes\":{},\"skillCase\":{},\"skills\":[{}],\"asked\":[{}],\"passed\":{},\"problem\":{},\"calls\":[{}],\"seconds\":{:.2},\"tokens\":{},\"answer\":{}}}",
                        json::string(id),
                        run.hard,
                        run.changes,
                        run.skill_case,
                        run.skills.iter().map(|t| json::string(t)).collect::<Vec<_>>().join(","),
                        run.asked.iter().map(|t| json::string(t)).collect::<Vec<_>>().join(","),
                        verdict.is_ok(),
                        verdict.as_ref().err().map_or("null".into(), |p| json::string(p)),
                        tools.join(","),
                        run.seconds,
                        run.tokens,
                        json::string(&run.answer)
                    )
                })
                .collect();
            format!(
                "{}:{{\"passed\":{},\"total\":{},\"cases\":[\n    {}\n  ]}}",
                json::string(name),
                runs.iter().filter(|r| r.1.is_ok()).count(),
                runs.len(),
                cases.join(",\n    ")
            )
        })
        .collect();
    let args: Vec<String> = server_args.iter().map(|a| json::string(a)).collect();
    format!(
        "{{\"serverArgs\":[{}],\"models\":{{\n  {}\n}}}}\n",
        args.join(","),
        models.join(",\n  ")
    )
}

fn to_json(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => json::string(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(to_json).collect::<Vec<_>>().join(",")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}:{}", json::string(k), to_json(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

// ---- The server ---------------------------------------------------------------------------

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    port: u16,
}

impl Server {
    fn start(exe: &Path, model: &Path, extra: &[String]) -> Result<Server, String> {
        let mut child = Command::new(exe)
            .arg(model)
            .args([
                "--port",
                "0",
                "--token",
                TOKEN,
                "--with-app",
                "--chat-cache",
                "off",
            ])
            .args(extra)
            // Its own "always allow" list, which stays empty: every change asks.
            .env(
                "VIREO_APPROVALS_FILE",
                std::env::temp_dir()
                    .join(format!("vireo-eval-approvals-{}.txt", std::process::id())),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("couldn't start {}: {e}", exe.display()))?;
        // Its log is drained as it comes, so it never blocks on a full pipe.
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let stderr = child.stderr.take().unwrap();
        let sink = log.clone();
        // With VIREO_DEBUG_CALLS set (the server reads it too), its "[calls]" lines, the
        // scores behind each choice, are shown as they come.
        let show = std::env::var_os("VIREO_DEBUG_CALLS").is_some();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if show && line.starts_with("[calls]") {
                    eprintln!("    {line}");
                }
                sink.lock().unwrap().push(line);
            }
        });
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let mut port = None;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(p) = line.strip_prefix("Listening on http://127.0.0.1:") {
                port = p.trim().parse().ok();
                break;
            }
        }
        match port {
            Some(port) => Ok(Server { child, stdin, port }),
            None => {
                let _ = child.wait();
                Err(log.lock().unwrap().join("\n"))
            }
        }
    }

    fn stop(&mut self) {
        drop(self.stdin.take()); // it ends when its standard input closes
        let _ = self.child.wait();
    }
}

// ---- The stand-in app ---------------------------------------------------------------------

fn request(port: u16, path: &str, body: &str) -> std::io::Result<TcpStream> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(stream)
}

/// One case as a new chat: its events read as the app reads them, its tool requests
/// answered from the case's PC.
fn ask(port: u16, case: &Case) -> Run {
    let started = Instant::now();
    let mut run = Run::default();
    let tools: Vec<String> = CLIENT_TOOLS.iter().map(|t| json::string(t)).collect();
    run.hard = case.hard;
    run.changes = case.changes;
    run.skill_case = case.skills;
    let mut pc = case.pc.clone();
    // The earlier turns, then the prompt.
    let messages: Vec<String> = case
        .history
        .iter()
        .map(|(role, text)| (*role, text.as_str()))
        .chain([("user", case.prompt.as_str())])
        .map(|(role, text)| {
            format!(
                "{{\"role\":{},\"content\":{}}}",
                json::string(role),
                json::string(text)
            )
        })
        .collect();
    let body = format!(
        "{{\"messages\":[{}],\"clientTools\":[{}],\"generation\":{{\"max_tokens\":400,\"temperature\":0}}}}",
        messages.join(","),
        tools.join(",")
    );
    let stream = match request(port, "/api/chat/completions", &body) {
        Ok(s) => s,
        Err(e) => {
            run.error = Some(e.to_string());
            return run;
        }
    };
    for line in BufReader::new(stream).lines().map_while(Result::ok) {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(event) = json::parse(data) else {
            continue;
        };
        match event.str("type") {
            // The skill the server picked comes as a use_skill call; it's kept apart from
            // the model's own calls.
            Some("tool_call") if event.str("tool") == Some("use_skill") => {
                let args = event.get("args").cloned().unwrap_or(Value::Null);
                run.skills.push(args.str("name").unwrap_or("").to_string());
            }
            Some("tool_result") if event.str("tool") == Some("use_skill") => {}
            Some("tool_call") => run.calls.push(Call {
                tool: event.str("tool").unwrap_or("").to_string(),
                args: event.get("args").cloned().unwrap_or(Value::Null),
            }),
            Some("tool_result") => run
                .results
                .push(event.str("result").unwrap_or("").to_string()),
            Some("client_tool_request") => {
                let id = event.str("id").unwrap_or("");
                let args = event.get("args").cloned().unwrap_or(Value::Null);
                let answer = match pc.run(event.str("tool").unwrap_or(""), &args) {
                    Ok(output) => format!(
                        "{{\"id\":{},\"ok\":true,\"output\":{}}}",
                        json::string(id),
                        json::string(&output)
                    ),
                    Err(error) => format!(
                        "{{\"id\":{},\"ok\":false,\"error\":{}}}",
                        json::string(id),
                        json::string(&error)
                    ),
                };
                if let Ok(mut s) = request(port, "/api/agent/client-tool-result", &answer) {
                    let mut sink = Vec::new();
                    let _ = s.read_to_end(&mut sink);
                }
            }
            Some("confirm_required") => {
                let tool = event.str("tool").unwrap_or("").to_string();
                let answer = format!(
                    "{{\"id\":{},\"approved\":{},\"remember\":false}}",
                    json::string(event.str("id").unwrap_or("")),
                    pc.approves(&tool)
                );
                run.asked.push(tool);
                if let Ok(mut s) = request(port, "/api/agent/confirm", &answer) {
                    let mut sink = Vec::new();
                    let _ = s.read_to_end(&mut sink);
                }
            }
            Some(_) => {}
            None => {
                if let Some(text) = event.str("content") {
                    run.answer.push_str(text);
                } else if let Some(e) = event.str("error") {
                    run.error = Some(e.to_string());
                } else if let Some(n) = event.num("tokens") {
                    run.tokens = n as usize;
                }
            }
        }
    }
    run.seconds = started.elapsed().as_secs_f64();
    run.pc = pc;
    run
}
