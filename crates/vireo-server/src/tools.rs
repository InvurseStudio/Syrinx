//! The tools a reply may call, and each model family's way of writing them: how they're
//! declared in the system prompt, the grammar a call is written under, how a call is read
//! back, and how its result is handed to the model. Each family's form is the one its
//! chat template (`tokenizer.chat_template` in its GGUF file) uses, so the model writes
//! what it was trained on.
//!
//! The app runs the PC's tools (`PcTools` in the Windows app) when the server asks
//! (`client_tool_request`); the server runs its own (the date and time). Only tools that
//! change nothing so far: the ones that do come with approvals.

use vireo_engine::grammar::{Property, RecordStyle, Schema, json_string};

use crate::chat::Family;
use crate::json;

/// What an argument holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Text,
    #[allow(dead_code)] // for the integer arguments to come (lines, depth)
    Integer,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: &'static str,
    pub kind: Kind,
    pub description: &'static str,
    pub required: bool,
}

/// Who runs a tool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Runs {
    /// The app, on the PC (it says which it runs in `clientTools`).
    App,
    /// This server.
    Server,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub params: Vec<Param>,
    pub runs: Runs,
}

const fn text(name: &'static str, description: &'static str) -> Param {
    Param {
        name,
        kind: Kind::Text,
        description,
        required: true,
    }
}

/// Every tool this server knows. Optional arguments are left out for now: small models
/// fill them in where they shouldn't (Llama 3.2 1B asked for one line of a file), so
/// they'll come back where the eval says a model uses them well.
fn catalog() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file",
            description: "Read a text file, PDF, or Word, Excel or PowerPoint document on this PC.",
            params: vec![text(
                "path",
                "The file's path: a full path, or one relative to the home folder (Documents/report.pdf). Don't make one up.",
            )],
            runs: Runs::App,
        },
        Tool {
            name: "list_directory",
            description: "List the files and folders in a folder on this PC.",
            params: vec![text(
                "path",
                "The folder's path: a full path, or one relative to the home folder (Documents).",
            )],
            runs: Runs::App,
        },
        Tool {
            name: "search_files",
            description: "Find files and folders by name, or files containing some text, in a folder on this PC and its subfolders.",
            params: vec![
                text("pattern", "A name or part of one, or the text to look for."),
                text(
                    "directory",
                    "The folder to search: a full path, or one relative to the home folder.",
                ),
            ],
            runs: Runs::App,
        },
        Tool {
            name: "file_exists",
            description: "Check whether a file or folder exists on this PC, and which it is.",
            params: vec![text("path", "The path to check.")],
            runs: Runs::App,
        },
        Tool {
            name: "git_status",
            description: "Show a Git project's status (changed and new files) without changing anything.",
            params: vec![text("path", "The project's folder.")],
            runs: Runs::App,
        },
        Tool {
            name: "git_diff",
            description: "Show a Git project's or file's changes without changing anything.",
            params: vec![text("path", "The project's folder, or a file in it.")],
            runs: Runs::App,
        },
        Tool {
            name: "get_current_datetime",
            description: "Get the current date, time and day of the week.",
            params: vec![],
            runs: Runs::Server,
        },
    ]
}

/// The tools a reply offers: the server's own, and those of the app's that it runs.
pub fn offered(client_tools: &[String]) -> Vec<Tool> {
    catalog()
        .into_iter()
        .filter(|t| t.runs == Runs::Server || client_tools.iter().any(|c| c == t.name))
        .collect()
}

/// The capability list `/api/capabilities` answers with (the app's Settings count them).
pub fn capabilities_json() -> String {
    let items: Vec<String> = catalog()
        .iter()
        .map(|t| {
            format!(
                "{{\"name\":{},\"label\":{},\"description\":{},\"category\":\"automatic\",\"platforms\":[\"windows\"],\"available\":true}}",
                json::string(t.name),
                json::string(&t.name.replace('_', " ")),
                json::string(t.description)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

// ---- Declaring them ---------------------------------------------------------------------

/// A JSON value, written the way the templates' `tojson` writes it.
enum J {
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn compact(&self, out: &mut String) {
        match self {
            J::Str(s) => out.push_str(&json_string(s)),
            J::Arr(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.compact(out);
                }
                out.push(']');
            }
            J::Obj(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&json_string(k));
                    out.push_str(": ");
                    v.compact(out);
                }
                out.push('}');
            }
        }
    }

    /// Python's `json.dumps(indent=4)`, as Llama's template writes tools.
    fn indented(&self, depth: usize, out: &mut String) {
        let pad = |d: usize| "    ".repeat(d);
        match self {
            J::Str(s) => out.push_str(&json_string(s)),
            J::Arr(items) if items.is_empty() => out.push_str("[]"),
            J::Obj(fields) if fields.is_empty() => out.push_str("{}"),
            J::Arr(items) => {
                out.push_str("[\n");
                for (i, v) in items.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    v.indented(depth + 1, out);
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push(']');
            }
            J::Obj(fields) => {
                out.push_str("{\n");
                for (i, (k, v)) in fields.iter().enumerate() {
                    out.push_str(&pad(depth + 1));
                    out.push_str(&json_string(k));
                    out.push_str(": ");
                    v.indented(depth + 1, out);
                    out.push_str(if i + 1 < fields.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(depth));
                out.push('}');
            }
        }
    }
}

/// A tool as the OpenAI-style function declaration Qwen's and Llama's templates write.
fn declaration(tool: &Tool) -> J {
    let properties = tool
        .params
        .iter()
        .map(|p| {
            let kind = match p.kind {
                Kind::Text => "string",
                Kind::Integer => "integer",
            };
            (
                p.name.to_string(),
                J::Obj(vec![
                    ("type".into(), J::Str(kind.into())),
                    ("description".into(), J::Str(p.description.into())),
                ]),
            )
        })
        .collect();
    let required = tool
        .params
        .iter()
        .filter(|p| p.required)
        .map(|p| J::Str(p.name.into()))
        .collect();
    J::Obj(vec![
        ("type".into(), J::Str("function".into())),
        (
            "function".into(),
            J::Obj(vec![
                ("name".into(), J::Str(tool.name.into())),
                ("description".into(), J::Str(tool.description.into())),
                (
                    "parameters".into(),
                    J::Obj(vec![
                        ("type".into(), J::Str("object".into())),
                        ("properties".into(), J::Obj(properties)),
                        ("required".into(), J::Arr(required)),
                    ]),
                ),
            ]),
        ),
    ])
}

const QUOTE: &str = "<|\"|>";

/// Gemma 4's declaration: `declaration:name{description:...,parameters:{...}}`, its own
/// syntax, keys in alphabetical order.
fn gemma_declaration(tool: &Tool) -> String {
    let q = |s: &str| format!("{QUOTE}{s}{QUOTE}");
    let mut params: Vec<&Param> = tool.params.iter().collect();
    params.sort_by_key(|p| p.name);
    let mut out = format!(
        "declaration:{}{{description:{}",
        tool.name,
        q(tool.description)
    );
    out.push_str(",parameters:{");
    if !params.is_empty() {
        out.push_str("properties:{");
        let props: Vec<String> = params
            .iter()
            .map(|p| {
                let kind = match p.kind {
                    Kind::Text => "STRING",
                    Kind::Integer => "INTEGER",
                };
                format!(
                    "{}:{{description:{},type:{}}}",
                    p.name,
                    q(p.description),
                    q(kind)
                )
            })
            .collect();
        out.push_str(&props.join(","));
        out.push_str("},");
        let required: Vec<String> = tool
            .params
            .iter()
            .filter(|p| p.required)
            .map(|p| q(p.name))
            .collect();
        if !required.is_empty() {
            out.push_str(&format!("required:[{}],", required.join(",")));
        }
    }
    out.push_str(&format!("type:{}}}}}", q("OBJECT")));
    out
}

/// What the system prompt says about the tools, in the family's own words. Where it goes
/// is the family's too (`chat::history` puts it there).
pub fn declarations(family: Family, tools: &[Tool]) -> String {
    match family {
        Family::Qwen35 => {
            let mut out =
                String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for t in tools {
                out.push('\n');
                declaration(t).compact(&mut out);
            }
            out.push_str("\n</tools>");
            out.push_str(
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
                 <tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n\
                 </parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n\
                 that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
                 - Function calls MUST follow the specified format: an inner <function=...></function> block must \
                 be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n\
                 - You may provide optional reasoning for your function call in natural language BEFORE the \
                 function call, but NOT after\n- If there is no function call available, answer the question \
                 like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
            );
            out
        }
        Family::Llama => {
            let mut out = String::from(
                "You have access to the following functions. To call a function, please respond with JSON for a \
                 function call.Respond in the format {\"name\": function name, \"parameters\": dictionary of \
                 argument name and its value}.Do not use variables.\n\n",
            );
            for t in tools {
                declaration(t).indented(0, &mut out);
                out.push_str("\n\n");
            }
            out
        }
        Family::Gemma4 => tools
            .iter()
            .map(|t| format!("<|tool>{}<tool|>", gemma_declaration(t)))
            .collect(),
    }
}

// ---- Calling them -----------------------------------------------------------------------

/// The longest text argument, in bytes: a path or a search, not an essay.
const MAX_ARGUMENT: usize = 2048;

/// How a call starts: with this special token (the model's), or, for Llama, with a turn
/// that opens with `{` (the grammar's own first byte).
pub struct Start {
    pub token: &'static str,
    pub brace: bool,
}

pub fn call_start(family: Family) -> Start {
    match family {
        Family::Qwen35 => Start {
            token: "<tool_call>",
            brace: false,
        },
        Family::Gemma4 => Start {
            token: "<|tool_call>",
            brace: false,
        },
        // With "Environment: ipython" in its system prompt, Llama 3.2 starts a call with
        // <|python_tag|> as often as with a bare JSON object.
        Family::Llama => Start {
            token: "<|python_tag|>",
            brace: true,
        },
    }
}

/// What the system prompt says first when there are tools: what they're for, in a line,
/// so a small model doesn't answer that it can't reach the user's files.
pub const PREAMBLE: &str = "You are Vireo, running on the user's PC. With the tools below you can read and find      their files, check Git projects and get the date and time. When a question needs one of them, call it,      then answer from what it returns.";

/// Special tokens a call is written with, inside it.
pub fn call_tokens(family: Family) -> &'static [&'static str] {
    match family {
        Family::Gemma4 => &[QUOTE],
        _ => &[],
    }
}

/// The grammar a call to one of `tools` is written under, from just after its start (for
/// Llama, from the turn's start).
pub fn call_grammar(family: Family, tools: &[Tool]) -> Schema {
    match family {
        Family::Llama => {
            let value = |p: &Param| match p.kind {
                Kind::Text => Schema::String {
                    max_bytes: Some(MAX_ARGUMENT),
                },
                Kind::Integer => Schema::Integer,
            };
            Schema::Seq(vec![
                Schema::Literal("{\"name\": ".into()),
                Schema::Branch(
                    tools
                        .iter()
                        .map(|t| {
                            (
                                format!("{}, \"parameters\": ", json_string(t.name)),
                                Schema::Object(fields(&t.params, value)),
                            )
                        })
                        .collect(),
                ),
                Schema::Literal("}".into()),
            ])
        }
        Family::Qwen35 => {
            const END: &str = "\n</parameter>\n";
            let value = |p: &Param| match p.kind {
                Kind::Text => Schema::Text {
                    until: END.into(),
                    max_bytes: Some(MAX_ARGUMENT),
                },
                Kind::Integer => Schema::Seq(vec![Schema::Integer, Schema::Literal(END.into())]),
            };
            let style = RecordStyle {
                open: String::new(),
                key_open: "<parameter=".into(),
                key_close: ">\n".into(),
                separator: String::new(),
                close: "</function>\n".into(),
                json_keys: false,
            };
            Schema::Seq(vec![
                Schema::Literal("\n<function=".into()),
                Schema::Branch(
                    tools
                        .iter()
                        .map(|t| {
                            (
                                format!("{}>\n", t.name),
                                Schema::Record {
                                    fields: fields(&t.params, value),
                                    style: style.clone(),
                                },
                            )
                        })
                        .collect(),
                ),
            ])
        }
        Family::Gemma4 => {
            let value = |p: &Param| match p.kind {
                Kind::Text => Schema::Seq(vec![
                    Schema::Literal(QUOTE.into()),
                    Schema::Text {
                        until: QUOTE.into(),
                        max_bytes: Some(MAX_ARGUMENT),
                    },
                ]),
                Kind::Integer => Schema::Integer,
            };
            let style = RecordStyle {
                open: "{".into(),
                key_open: String::new(),
                key_close: ":".into(),
                separator: ",".into(),
                close: "}".into(),
                json_keys: false,
            };
            Schema::Seq(vec![
                Schema::Literal("call:".into()),
                Schema::Branch(
                    tools
                        .iter()
                        .map(|t| {
                            let mut params = t.params.clone();
                            params.sort_by_key(|p| p.name);
                            (
                                t.name.to_string(),
                                Schema::Record {
                                    fields: fields(&params, value),
                                    style: style.clone(),
                                },
                            )
                        })
                        .collect(),
                ),
            ])
        }
    }
}

fn fields(params: &[Param], value: impl Fn(&Param) -> Schema) -> Vec<Property> {
    params
        .iter()
        .map(|p| Property {
            name: p.name.into(),
            schema: value(p),
            required: p.required,
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Text(String),
    Integer(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub tool: String,
    pub args: Vec<(String, Arg)>,
}

impl Call {
    /// The arguments as a JSON object.
    pub fn args_json(&self) -> String {
        let parts: Vec<String> = self
            .args
            .iter()
            .map(|(k, v)| {
                let v = match v {
                    Arg::Text(s) => json::string(s),
                    Arg::Integer(n) => n.to_string(),
                };
                format!("{}:{v}", json::string(k))
            })
            .collect();
        format!("{{{}}}", parts.join(","))
    }
}

/// A call written under `call_grammar`, read back. The grammar guarantees its shape.
pub fn parse_call(family: Family, text: &str) -> Option<Call> {
    match family {
        Family::Llama => {
            let v = json::parse(text).ok()?;
            let tool = v.str("name")?.to_string();
            let mut args = Vec::new();
            if let Some(json::Value::Object(params)) = v.get("parameters") {
                for (k, v) in params {
                    args.push((
                        k.clone(),
                        match v {
                            json::Value::String(s) => Arg::Text(s.clone()),
                            json::Value::Number(n) => Arg::Integer(*n as i64),
                            _ => return None,
                        },
                    ));
                }
            }
            Some(Call { tool, args })
        }
        Family::Qwen35 => {
            let rest = text.strip_prefix("\n<function=")?;
            let (tool, mut rest) = rest.split_once(">\n")?;
            let mut args = Vec::new();
            while let Some(after) = rest.strip_prefix("<parameter=") {
                let (key, after) = after.split_once(">\n")?;
                let (value, after) = after.split_once("\n</parameter>\n")?;
                args.push((
                    key.to_string(),
                    value
                        .parse::<i64>()
                        .map_or_else(|_| Arg::Text(value.to_string()), Arg::Integer),
                ));
                rest = after;
            }
            Some(Call {
                tool: tool.to_string(),
                args,
            })
        }
        Family::Gemma4 => {
            let rest = text.strip_prefix("call:")?;
            let (tool, mut rest) = rest.split_once('{')?;
            let mut args = Vec::new();
            while !rest.starts_with('}') {
                let (key, after) = rest.split_once(':')?;
                let (value, after) = if let Some(quoted) = after.strip_prefix(QUOTE) {
                    let (value, after) = quoted.split_once(QUOTE)?;
                    (Arg::Text(value.to_string()), after)
                } else {
                    let end = after.find([',', '}'])?;
                    (Arg::Integer(after[..end].parse().ok()?), &after[end..])
                };
                args.push((key.to_string(), value));
                rest = after.strip_prefix(',').unwrap_or(after);
            }
            Some(Call {
                tool: tool.to_string(),
                args,
            })
        }
    }
}

/// After a call: the text that closes it, then its result in the family's form, up to
/// where the model goes on. Three parts: special-token text before the result, the
/// result itself (read as plain text, so nothing in it can pass for a special token),
/// and special-token text after it.
pub fn after_call(family: Family, tool: &str, result: &str) -> (String, String, String) {
    match family {
        Family::Qwen35 => (
            "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n".into(),
            result.into(),
            "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into(),
        ),
        Family::Llama => (
            "<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n".into(),
            result.into(),
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        ),
        Family::Gemma4 => (
            format!("<tool_call|><|tool_response>response:{tool}{{value:{QUOTE}"),
            result.into(),
            format!("{QUOTE}}}<tool_response|>"),
        ),
    }
}

// ---- The server's own tools -------------------------------------------------------------

/// Runs one of the server's own tools.
pub fn run_here(call: &Call) -> Result<String, String> {
    match call.tool.as_str() {
        "get_current_datetime" => Ok(now()),
        other => Err(format!("{other} isn't one of this server's tools")),
    }
}

/// "Wednesday, November 11, 2026, 1:05 PM (local time)", as the Mac's server says it.
fn now() -> String {
    let (y, mo, d, wd, h, mi) = local_time();
    const DAYS: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let (h12, ampm) = match h {
        0 => (12, "AM"),
        1..=11 => (h, "AM"),
        12 => (12, "PM"),
        _ => (h - 12, "PM"),
    };
    format!(
        "{}, {} {d}, {y}, {h12}:{mi:02} {ampm} (local time)",
        DAYS[wd as usize % 7],
        MONTHS[(mo as usize).clamp(1, 12) - 1]
    )
}

/// (year, month, day, weekday from Sunday, hour, minute), local.
#[cfg(windows)]
fn local_time() -> (u16, u16, u16, u16, u16, u16) {
    #[repr(C)]
    #[derive(Default)]
    struct SystemTime {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        milliseconds: u16,
    }
    unsafe extern "system" {
        fn GetLocalTime(time: *mut SystemTime);
    }
    let mut t = SystemTime::default();
    unsafe { GetLocalTime(&mut t) };
    (t.year, t.month, t.day, t.day_of_week, t.hour, t.minute)
}

#[cfg(not(windows))]
fn local_time() -> (u16, u16, u16, u16, u16, u16) {
    // UTC, from the system clock (the Mac's server is the one that runs elsewhere).
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let days = secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let wd = (days + 4).rem_euclid(7);
    let s = secs.rem_euclid(86_400);
    (
        year as u16,
        month as u16,
        day as u16,
        wd as u16,
        (s / 3600) as u16,
        (s / 60 % 60) as u16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vireo_engine::grammar::Grammar;

    fn written(family: Family, text: &str) -> Option<Call> {
        let tools = offered(&["read_file".into(), "search_files".into()]);
        let g = Grammar::new(&call_grammar(family, &tools));
        let mut m = g.matcher();
        assert!(
            m.advance_all(text.as_bytes()) && m.is_complete(),
            "{family:?}: {text:?}"
        );
        parse_call(family, text)
    }

    #[test]
    fn each_family_writes_and_reads_back_a_call() {
        let want = Call {
            tool: "search_files".into(),
            args: vec![
                ("pattern".into(), Arg::Text("tax \"2026\"".into())),
                ("directory".into(), Arg::Text("C:\\Users\\me".into())),
            ],
        };
        let mut sorted = want.clone();
        sorted.args.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            written(
                Family::Llama,
                r#"{"name": "search_files", "parameters": {"pattern": "tax \"2026\"", "directory": "C:\\Users\\me"}}"#
            ),
            Some(sorted.clone()),
            "Llama's JSON object comes back in key order"
        );
        assert_eq!(
            written(
                Family::Qwen35,
                "\n<function=search_files>\n<parameter=pattern>\ntax \"2026\"\n</parameter>\n<parameter=directory>\nC:\\Users\\me\n</parameter>\n</function>\n"
            ),
            Some(want.clone())
        );
        assert_eq!(
            written(
                Family::Gemma4,
                "call:search_files{directory:<|\"|>C:\\Users\\me<|\"|>,pattern:<|\"|>tax \"2026\"<|\"|>}"
            ),
            Some(sorted)
        );
        assert_eq!(
            written(Family::Gemma4, "call:get_current_datetime{}"),
            Some(Call {
                tool: "get_current_datetime".into(),
                args: vec![]
            })
        );
    }

    #[test]
    fn declarations_follow_the_templates() {
        let tools = offered(&["read_file".into()]);
        let qwen = declarations(Family::Qwen35, &tools);
        assert!(qwen.contains(
            "\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a text file"
        ));
        assert!(qwen.ends_with("</IMPORTANT>"));
        let llama = declarations(Family::Llama, &tools);
        assert!(llama.contains(
            "{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"read_file\""
        ));
        let gemma = declarations(Family::Gemma4, &tools);
        assert!(
            gemma.starts_with("<|tool>declaration:read_file{description:<|\"|>Read a text file")
        );
        assert!(gemma.contains(",parameters:{properties:{path:{description:<|\"|>The file's path"));
        assert!(gemma.contains("required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"));
        assert!(gemma.contains(
            "<|tool>declaration:get_current_datetime{description:<|\"|>Get the current date, time and day of the week.<|\"|>,parameters:{type:<|\"|>OBJECT<|\"|>}}<tool|>"
        ));
        assert_eq!(
            tools.len(),
            2,
            "read_file, and the server's own date and time"
        );
    }

    #[test]
    fn the_date_reads_like_the_macs() {
        let n = now();
        assert!(n.ends_with("(local time)") && n.contains(", 20"), "{n}");
    }
}
