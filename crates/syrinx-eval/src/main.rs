//! `syrinx-eval`: how well each model uses Vireo's tools. It starts syrinx-server with a
//! model, stands in for the app (a PC of a few files in memory, `pc.rs`, running the
//! app's tools the way the app does), sends each case (`cases.rs`) as a new chat, and
//! scores the reply: the tools it called, their arguments, and the answer. Replies are
//! greedy (temperature 0), so a run gives the same scores every time.
//!
//! Twelve easy cases (`cases.rs`), twelve harder ones (`hard_cases.rs`), the ones that
//! change files (`change_cases.rs`), skills (`skill_cases.rs`) and the later tools
//! (`more_cases.rs`), counted apart in the report. The stand-in app answers approvals as
//! the case's user does (`Pc::declining`), and any case not about changes fails if the
//! reply changes a file.
//!
//!   syrinx-eval [--full | --models a.gguf,b.gguf] [--cases read,git] [--out results.json]
//!              [--server-arg <arg>]...
//!
//! By default a short run on two models, one from each family (`QUICK_MODELS`, about 6
//! minutes for every case on the RTX 5060): the check while developing. `--full` runs all
//! four eval models (`FULL_MODELS`, about 18 minutes), for in-depth testing and the record
//! kept in `evals/` before a pull request. `--cases` takes case ids or categories (easy,
//! hard, changes, skills, more): a new tool or skill is checked on its own cases, and the
//! whole set saved for in-depth runs. `--models` names others: paths, or names in
//! Syrinx/models. `--server-arg` passes an argument on to syrinx-server, to compare
//! a feature on and off.

mod cases;
mod change_cases;
mod hard_cases;
#[path = "../../syrinx-server/src/json.rs"]
#[allow(dead_code)]
mod json;
mod more_cases;
mod office_cases;
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

/// The short run's models: the smallest Gemma, where tools break first, and the Qwen that
/// does as well as the 9B in a third of the time (2026-11-11: 39 and 38 of 39).
const QUICK_MODELS: [&str; 2] = ["gemma-4-E2B_q4_0-it.gguf", "Qwen3.5-4B-Q4_K_M.gguf"];

/// `--full`: small to large in each family, where a feature stops mattering shows along
/// each. Models people choose today. Llama 3.2 1B (2024) isn't one any more, and Qwen 3.5
/// 0.8B was left out on 2026-11-11 (too small to use the tools well: it said yes to every
/// skill and saved a poem to a file); both stay the engine's references in its own tests,
/// and run here on request (`--models`).
const FULL_MODELS: [&str; 4] = [
    "gemma-4-E2B_q4_0-it.gguf",
    "gemma-4-E4B_q4_0-it.gguf",
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Qwen3.5-9B-Q4_K_M.gguf",
];
const TOKEN: &str = "syrinx-eval";

/// A case's category, which `--cases` takes as well as ids: `--cases more` runs only the
/// later tools' cases, `--cases skills` the skills', so a new tool or skill is checked on
/// its own cases in a minute or two.
fn category(case: &Case) -> &'static str {
    if case.more {
        "more"
    } else if case.skills {
        "skills"
    } else if case.changes {
        "changes"
    } else if case.hard {
        "hard"
    } else {
        "easy"
    }
}
/// The PC tools the stand-in app runs (its `clientTools`).
const CLIENT_TOOLS: [&str; 18] = [
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
    "read_spreadsheet",
    "write_spreadsheet",
    "fill_docx_template",
    "create_docx",
    "run_tests",
    "run_build",
    "run_linter",
    // Not a tool: the server checks a change with it before asking for approval.
    "check_change",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut models, mut only, mut out, mut server_args) =
        (Vec::new(), Vec::new(), None, Vec::new());
    let mut full = false;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--full" {
            full = true;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).cloned();
        match (args[i].as_str(), value) {
            ("--models", Some(v)) => models = v.split(',').map(str::to_string).collect(),
            ("--cases", Some(v)) => only = v.split(',').map(str::to_string).collect(),
            ("--out", Some(v)) => out = Some(PathBuf::from(v)),
            ("--server-arg", Some(v)) => server_args.push(v),
            _ => {
                eprintln!(
                    "Usage: syrinx-eval [--full | --models a,b] [--cases a,b] [--out file.json] [--server-arg arg]..."
                );
                return ExitCode::FAILURE;
            }
        }
        i += 2;
    }
    if models.is_empty() {
        let set: &[&str] = if full { &FULL_MODELS } else { &QUICK_MODELS };
        models = set.iter().map(|s| s.to_string()).collect();
    }
    let here = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let Some(here) = here else {
        return ExitCode::FAILURE;
    };
    let server = here.join("syrinx-server.exe");
    let server = if server.exists() {
        server
    } else {
        here.join("syrinx-server")
    };
    let models_dir = here.join("../../models");

    let cases: Vec<Case> = cases::all()
        .into_iter()
        .filter(|c| only.is_empty() || only.iter().any(|o| o == c.id || o == category(c)))
        .collect();
    if cases.is_empty() {
        eprintln!(
            "No cases match --cases {}: give case ids or a category (easy, hard, changes, skills, more).",
            only.join(",")
        );
        return ExitCode::FAILURE;
    }
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
            let logged = running.log.lock().unwrap().len();
            let run = ask(running.port, case);
            let speeds = running.speeds_since(logged);
            let verdict = match (&run.error, (case.check)(&run)) {
                (Some(e), _) => Err(format!("error: {e}")),
                (None, _)
                    if !case.changes
                        && !case.skills
                        && !case.more
                        && !run.changes_made().is_empty() =>
                {
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
            runs.push((case.id, verdict, run, speeds));
        }
        let passed = runs.iter().filter(|r| r.1.is_ok()).count();
        let seconds: f64 = runs.iter().map(|r| r.2.seconds).sum();
        let calls: usize = runs.iter().map(|r| r.2.calls.len()).sum();
        let part = |hard: bool, changes: bool, skills: bool, more: bool| {
            let of: Vec<_> = runs
                .iter()
                .filter(|r| {
                    r.2.hard == hard
                        && r.2.changes == changes
                        && r.2.skill_case == skills
                        && r.2.more_case == more
                })
                .collect();
            (of.iter().filter(|r| r.1.is_ok()).count(), of.len())
        };
        let ((easy_passed, easy), (hard_passed, hard)) = (
            part(false, false, false, false),
            part(true, false, false, false),
        );
        let (changes_passed, changes) = part(false, true, false, false);
        let (skills_passed, skill_cases) = part(false, false, true, false);
        let (more_passed, more_cases) = part(false, false, false, true);
        // Skills picked where a case didn't ask for one: worth watching, not a failure.
        let stray = runs
            .iter()
            .filter(|r| !r.2.skill_case && !r.2.skills.is_empty())
            .count();
        println!(
            "  {passed}/{} passed ({easy_passed}/{easy} easy, {hard_passed}/{hard} hard, {changes_passed}/{changes} changes, {skills_passed}/{skill_cases} skills, {more_passed}/{more_cases} more tools; a skill in {stray} others), {:.1} s a case on average, {calls} tool calls",
            runs.len(),
            seconds / runs.len().max(1) as f64
        );
        let mut all = Speeds::default();
        for r in &runs {
            all.add(&r.3);
        }
        let (call_ms, run_ms) = all.tool_ms();
        println!(
            "  read {} tokens at {:.0}/s ({} more kept), wrote {} at {:.1}/s, {} of {} guesses taken; a tool call written in {:.0} ms and run in {:.0} ms on average",
            all.read_tokens,
            Speeds::per_second(all.read_tokens, all.read_seconds),
            all.kept_tokens,
            all.written_tokens,
            Speeds::per_second(all.written_tokens, all.write_seconds),
            all.taken,
            all.guessed,
            call_ms / all.tools.len().max(1) as f64,
            run_ms / all.tools.len().max(1) as f64
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

type Report<'a> = Vec<(String, Vec<(&'a str, Result<(), String>, Run, Speeds)>)>;

fn results_json(report: &Report, server_args: &[String]) -> String {
    let models: Vec<String> = report
        .iter()
        .map(|(name, runs)| {
            let cases: Vec<String> = runs
                .iter()
                .map(|(id, verdict, run, speeds)| {
                    let tools: Vec<String> = run
                        .calls
                        .iter()
                        .map(|c| format!("{{\"tool\":{},\"args\":{}}}", json::string(&c.tool), to_json(&c.args)))
                        .collect();
                    format!(
                        "{{\"id\":{},\"hard\":{},\"changes\":{},\"skillCase\":{},\"moreCase\":{},\"skills\":[{}],\"asked\":[{}],\"passed\":{},\"problem\":{},\"calls\":[{}],\"seconds\":{:.2},\"tokens\":{},{},\"answer\":{}}}",
                        json::string(id),
                        run.hard,
                        run.changes,
                        run.skill_case,
                        run.more_case,
                        run.skills.iter().map(|t| json::string(t)).collect::<Vec<_>>().join(","),
                        run.asked.iter().map(|t| json::string(t)).collect::<Vec<_>>().join(","),
                        verdict.is_ok(),
                        verdict.as_ref().err().map_or("null".into(), |p| json::string(p)),
                        tools.join(","),
                        run.seconds,
                        run.tokens,
                        speeds.json(true),
                        json::string(&run.answer)
                    )
                })
                .collect();
            let mut all = Speeds::default();
            for r in runs {
                all.add(&r.3);
            }
            format!(
                "{}:{{\"passed\":{},\"total\":{},\"seconds\":{:.2},\"speeds\":{{{}}},\"cases\":[\n    {}\n  ]}}",
                json::string(name),
                runs.iter().filter(|r| r.1.is_ok()).count(),
                runs.len(),
                runs.iter().map(|r| r.2.seconds).sum::<f64>(),
                all.json(false),
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
    /// Its log so far, a line each.
    log: Arc<Mutex<Vec<String>>>,
}

/// What the server's log says of a case's replies: how much it read and wrote and how
/// fast ("Reply:" lines), and each tool call's time ("Tool:" lines). Saved with every
/// case on every machine, so runs can be compared for speed as well as score.
#[derive(Clone, Debug, Default, PartialEq)]
struct Speeds {
    /// The prompt's tokens, those kept from the chat's cache, and those read.
    prompt_tokens: usize,
    kept_tokens: usize,
    read_tokens: usize,
    read_seconds: f64,
    /// Tokens written (the answer, calls and thoughts) and the time from the prompt's
    /// end to the reply's, tools' runs included.
    written_tokens: usize,
    write_seconds: f64,
    /// Each call: the tool, the time writing the call and the time the tool ran, ms.
    tools: Vec<(String, f64, f64)>,
    /// Guesses checked (a draft model's, the chat's own) and taken.
    guessed: usize,
    taken: usize,
}

impl Speeds {
    /// From the log lines of one case.
    fn read(lines: &[String]) -> Self {
        // The number before `after` in `line` (after the text before it, if given).
        fn before(line: &str, after: &str) -> Option<f64> {
            let head = &line[..line.find(after)?];
            let start = head
                .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
                .map_or(0, |i| i + 1);
            head[start..].parse().ok()
        }
        let mut out = Speeds::default();
        for line in lines {
            if let Some(rest) = line.strip_prefix("Reply: ") {
                let n = |after: &str| before(rest, after).unwrap_or(0.0);
                out.prompt_tokens += n(" prompt tokens") as usize;
                out.kept_tokens += n(" kept,") as usize;
                out.read_tokens += n(" read in ") as usize;
                out.read_seconds += n("s, ");
                // "), 123 tokens in 4.56s (": after the prompt's part.
                if let Some(at) = rest.find("/s), ") {
                    let written = &rest[at + 5..];
                    out.written_tokens += before(written, " tokens in ").unwrap_or(0.0) as usize;
                    out.write_seconds += before(written, "s (").unwrap_or(0.0);
                }
                let mut guesses = rest;
                while let Some(at) = guesses.find(" guesses") {
                    // "12 of 34 guesses".
                    let head = &guesses[..at];
                    out.guessed += before(guesses, " guesses").unwrap_or(0.0) as usize;
                    out.taken += before(head, " of ").unwrap_or(0.0) as usize;
                    guesses = &guesses[at + 8..];
                }
            } else if let Some(rest) = line.strip_prefix("Tool: ") {
                let name = rest.split(' ').next().unwrap_or("").to_string();
                // The times follow the arguments: the last "(written in".
                let times = rest.rfind("(written in ").map_or("", |at| &rest[at..]);
                out.tools.push((
                    name,
                    before(times, " ms").unwrap_or(0.0),
                    times
                        .find("run in ")
                        .and_then(|at| before(&times[at..], " ms"))
                        .unwrap_or(0.0),
                ));
            }
        }
        out
    }

    fn per_second(tokens: usize, seconds: f64) -> f64 {
        if seconds > 0.0 { tokens as f64 / seconds } else { 0.0 }
    }

    fn add(&mut self, other: &Speeds) {
        self.prompt_tokens += other.prompt_tokens;
        self.kept_tokens += other.kept_tokens;
        self.read_tokens += other.read_tokens;
        self.read_seconds += other.read_seconds;
        self.written_tokens += other.written_tokens;
        self.write_seconds += other.write_seconds;
        self.tools.extend(other.tools.iter().cloned());
        self.guessed += other.guessed;
        self.taken += other.taken;
    }

    /// Time writing calls and running tools, ms.
    fn tool_ms(&self) -> (f64, f64) {
        self.tools
            .iter()
            .fold((0.0, 0.0), |sum, t| (sum.0 + t.1, sum.1 + t.2))
    }

    /// The fields of a case's (or a model's) JSON object, without braces.
    fn json(&self, with_calls: bool) -> String {
        let (call_ms, run_ms) = self.tool_ms();
        let mut out = format!(
            "\"promptTokens\":{},\"keptTokens\":{},\"readTokens\":{},\"readSeconds\":{:.3},\"readTokensPerSecond\":{:.0},\"writtenTokens\":{},\"writeSeconds\":{:.3},\"writtenTokensPerSecond\":{:.1},\"toolCalls\":{},\"toolCallWriteMs\":{:.0},\"toolRunMs\":{:.0},\"guessed\":{},\"guessesTaken\":{}",
            self.prompt_tokens,
            self.kept_tokens,
            self.read_tokens,
            self.read_seconds,
            Self::per_second(self.read_tokens, self.read_seconds),
            self.written_tokens,
            self.write_seconds,
            Self::per_second(self.written_tokens, self.write_seconds),
            self.tools.len(),
            call_ms,
            run_ms,
            self.guessed,
            self.taken
        );
        if with_calls {
            let calls: Vec<String> = self
                .tools
                .iter()
                .map(|(tool, write, run)| {
                    format!(
                        "{{\"tool\":{},\"writeMs\":{write:.0},\"runMs\":{run:.0}}}",
                        json::string(tool)
                    )
                })
                .collect();
            out += &format!(",\"toolTimes\":[{}]", calls.join(","));
        }
        out
    }
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
                    .join(format!("syrinx-eval-approvals-{}.txt", std::process::id())),
            )
            // Its own memory, which stays empty (the stand-in user declines remembering).
            .env(
                "VIREO_MEMORY_FILE",
                std::env::temp_dir()
                    .join(format!("syrinx-eval-memory-{}.txt", std::process::id())),
            )
            // Only the built-in skills, not the ones on this machine: the cases must score
            // the same wherever they run. (`INVURSE_SKILLS_DIR` set by hand is kept.)
            .env(
                "INVURSE_SKILLS_DIR",
                std::env::var_os("INVURSE_SKILLS_DIR").unwrap_or_else(|| {
                    let empty = std::env::temp_dir()
                        .join(format!("syrinx-eval-skills-{}", std::process::id()));
                    let _ = std::fs::create_dir_all(&empty);
                    empty.into_os_string()
                }),
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
        // scores behind each choice, are shown as they come, with each reply's speeds
        // ("Reply:") and its calls ("Tool:").
        let show = std::env::var_os("VIREO_DEBUG_CALLS").is_some();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if show && ["[calls]", "Reply:", "Tool:"].iter().any(|p| line.starts_with(p)) {
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
            Some(port) => Ok(Server {
                child,
                stdin,
                port,
                log,
            }),
            None => {
                let _ = child.wait();
                Err(log.lock().unwrap().join("\n"))
            }
        }
    }

    /// The speeds in the log's lines from line `from` on: a case's, once its reply is
    /// done. The log comes through a pipe, a moment after the reply's last event.
    fn speeds_since(&self, from: usize) -> Speeds {
        for _ in 0..50 {
            let log = self.log.lock().unwrap();
            if log[from.min(log.len())..]
                .iter()
                .any(|l| l.starts_with("Reply:"))
            {
                break;
            }
            drop(log);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // A line more may follow the first "Reply:" (a second request's): a moment for it.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let log = self.log.lock().unwrap();
        Speeds::read(&log[from.min(log.len())..])
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
    run.more_case = case.more;
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
    // Office tasks get the app's own budget (the server's default): their answers are
    // tables over several files, which 400 tokens cut off before the last step.
    let max_tokens = if case.id.starts_with("office-") { 2048 } else { 400 };
    let body = format!(
        "{{\"messages\":[{}],\"clientTools\":[{}],\"generation\":{{\"max_tokens\":{max_tokens},\"temperature\":0}}}}",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_cases_speeds_from_the_servers_log() {
        let lines: Vec<String> = [
            "[calls] something else",
            "Tool: read_file {\"path\":\"notes (written in haste).txt\"} (written in 212 ms, run in 3 ms), 1840 characters back.",
            "Tool: calculate {\"expression\":\"2+2\"} (written in 95 ms, run in 0 ms, 2 of 2 together), 1 characters back.",
            "Reply: 1563 prompt tokens (1477 kept, 86 read in 0.04s, 2150/s), 120 tokens in 1.50s (80.0/s), 2 tool calls, 30 of 40 guesses from the chat taken (75%), 5 of 10 guesses from the draft model taken (50%).",
        ]
        .iter()
        .map(|l| l.to_string())
        .collect();
        let s = Speeds::read(&lines);
        assert_eq!(
            (s.prompt_tokens, s.kept_tokens, s.read_tokens, s.written_tokens),
            (1563, 1477, 86, 120)
        );
        assert_eq!((s.read_seconds, s.write_seconds), (0.04, 1.5));
        assert_eq!((s.guessed, s.taken), (50, 35));
        assert_eq!(
            s.tools,
            vec![
                ("read_file".to_string(), 212.0, 3.0),
                ("calculate".to_string(), 95.0, 0.0)
            ]
        );
        assert_eq!(s.tool_ms(), (307.0, 3.0));
        // An older server's line, without the run time.
        let old = Speeds::read(&["Tool: list_dir {} (written in 50 ms), 9 characters back.".to_string()]);
        assert_eq!(old.tools, vec![("list_dir".to_string(), 50.0, 0.0)]);
        assert!(s.json(true).contains("\"readTokensPerSecond\":2150,"));
        assert!(s.json(true).contains("\"toolTimes\":[{\"tool\":\"read_file\",\"writeMs\":212,\"runMs\":3}"));
    }
}
