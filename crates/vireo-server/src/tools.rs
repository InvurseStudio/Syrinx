//! The tools a reply may call, and each model family's way of writing them: how they're
//! declared in the system prompt, the grammar a call is written under, how a call is read
//! back, and how its result is handed to the model. Each family's form is the one its
//! chat template (`tokenizer.chat_template` in its GGUF file) uses, so the model writes
//! what it was trained on.
//!
//! The PC's tools (files and Git) run on the server when the chat gives it access
//! (`access`) and the app doesn't run them itself: the Mac app and this server share the
//! Mac, so the server reads the files itself (`files.rs`). An app that says it runs them
//! (`clientTools`, as the Windows app's `PcTools` and the eval's stand-in PC do) runs them
//! instead, when the server asks (`client_tool_request`). The date and time are always the
//! server's. A tool that changes something (`asks`) waits for the user's approval first, as
//! the Mac's server asks for it (`confirm_required`, answered at `/api/agent/confirm`);
//! those run only in an app that runs them for now, since `files.rs` doesn't make changes
//! yet.

use vireo_engine::grammar::{Property, RecordStyle, Schema, json_string};

use crate::chat::Family;
use crate::files::Files;
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
    /// The app, on the PC (it says which it runs in `clientTools`). In the catalog: a PC
    /// tool, which this server runs itself when the app doesn't and the chat has access.
    App,
    /// This server.
    Server,
}

/// What a tool is for: a reply can be offered only the groups its request needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Group {
    Files,
    /// A Git or code project: its status and diff, its tests, build and linter.
    Git,
    Time,
    /// Arithmetic (`calculate`).
    Numbers,
}

/// The question that picks a request's groups (`--tools-shown scored`), each answer's
/// letter with the group it stands for; `None` is "no tools".
pub const GROUP_CHOICES: [(&str, Option<Group>); 5] = [
    ("A", Some(Group::Files)),
    ("B", Some(Group::Git)),
    ("C", Some(Group::Time)),
    ("D", Some(Group::Numbers)),
    ("E", None),
];

/// The question, with worked examples (small models sort far better from examples than
/// from a description), ending where the answer's letter goes. The examples are worded
/// apart from the eval's cases.
pub fn group_question(request: &str) -> String {
    let request = request.trim().replace('\n', " ");
    format!(
        "Sort each request to an assistant on a PC by what answering it needs.\n\
         A = files or folders on the PC (reading, listing, finding, checking or changing one)\n\
         B = a Git or code project (its status, changes, tests or build)\n\
         C = the date or the time now\n\
         D = arithmetic: totals, percentages, sums\n\
         E = nothing: it can be answered directly\n\n\
         Request: What's in D:\\Work\\report-q3.docx?\nNeeds: A\n\n\
         Request: What's 15% of 80?\nNeeds: D\n\n\
         Request: Did I commit everything in my website repo?\nNeeds: B\n\n\
         Request: What's the date next Tuesday?\nNeeds: C\n\n\
         Request: Write a haiku about autumn.\nNeeds: E\n\n\
         Request: Find my tax return somewhere in Documents.\nNeeds: A\n\n\
         Request: Do I have a file named plan.pdf on the Desktop?\nNeeds: A\n\n\
         Request: What time is it?\nNeeds: C\n\n\
         Request: {request}\nNeeds:"
    )
}

/// The question that decides whether a request's reply may call the tools that change
/// files, with worked examples (worded apart from the eval's cases), ending where the
/// answer goes: " yes" or " no".
pub fn change_question(request: &str) -> String {
    let request = request.trim().replace('\n', " ");
    format!(
        "Does each request to an assistant on a PC ask it to create, change, rename, move or \
         delete a file or folder? Reading, finding, explaining, suggesting or writing an answer \
         in the chat is no.\n\n\
         Request: Put these meeting notes in a file on my Desktop.\nChanges: yes\n\n\
         Request: What does budget.xlsx in Documents say about travel?\nChanges: no\n\n\
         Request: Rename the scans in Pictures so they start with the date.\nChanges: yes\n\n\
         Request: Write me a short story about a fox.\nChanges: no\n\n\
         Request: Get rid of the old backup zip in Downloads.\nChanges: yes\n\n\
         Request: Which of my folders holds the tax forms?\nChanges: no\n\n\
         Request: Put 'renew passport' on the list in Documents/errands.md.\nChanges: yes\n\n\
         Request: Suggest a clearer title for my essay, but leave the file alone.\nChanges: no\n\n\
         Request: Move the screenshots off my Desktop into a Screenshots folder.\nChanges: yes\n\n\
         Request: {request}\nChanges:"
    )
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub params: Vec<Param>,
    pub runs: Runs,
    pub group: Group,
    /// Set for a tool that changes something: it runs only once the user approves.
    pub asks: Option<Ask>,
}

/// How a tool that changes something is approved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ask {
    /// What it does, as the app's Settings and approvals name it ("Rename files").
    pub label: &'static str,
    /// Whether the user may always allow it ("Don't ask again to rename files"). Deleting
    /// always asks, as on the Mac.
    pub rememberable: bool,
}

impl Ask {
    /// The approval's "always allow" choice, as the Mac's server words it.
    pub fn remember_label(&self) -> String {
        let mut chars = self.label.chars();
        let first = chars.next().map(|c| c.to_lowercase().to_string());
        format!(
            "Don't ask again to {}{}",
            first.unwrap_or_default(),
            chars.as_str()
        )
    }
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
            group: Group::Files,
            asks: None,
        },
        Tool {
            name: "list_directory",
            description: "List the files and folders in a folder on this PC.",
            params: vec![text(
                "path",
                "The folder's path: a full path, or one relative to the home folder (Documents).",
            )],
            runs: Runs::App,
            group: Group::Files,
            asks: None,
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
            group: Group::Files,
            asks: None,
        },
        Tool {
            name: "file_exists",
            description: "Check whether a file or folder exists on this PC, and which it is.",
            params: vec![text("path", "The path to check.")],
            runs: Runs::App,
            group: Group::Files,
            asks: None,
        },
        Tool {
            name: "git_status",
            description: "Show a Git project's status (changed and new files) without changing anything.",
            params: vec![text("path", "The project's folder.")],
            runs: Runs::App,
            group: Group::Git,
            asks: None,
        },
        Tool {
            name: "git_diff",
            description: "Show a Git project's or file's changes without changing anything.",
            params: vec![text("path", "The project's folder, or a file in it.")],
            runs: Runs::App,
            group: Group::Git,
            asks: None,
        },
        Tool {
            name: "get_current_datetime",
            description: "Get the current date, time and day of the week.",
            params: vec![],
            runs: Runs::Server,
            group: Group::Time,
            asks: None,
        },
        Tool {
            name: "calculate",
            description: "Work out arithmetic exactly. Use it for every total, sum, difference, percentage or average instead of doing the maths yourself, e.g. \"412.50 + 129.30 + 400\" or \"sum([12.40, 8.99, 31.00]) * 1.13\". Dollar signs, thousands commas and % are fine.",
            params: vec![text(
                "expression",
                "Numbers with + - * / % ** and brackets, or round, abs, min, max, sum. For several results, one expression a line.",
            )],
            runs: Runs::Server,
            group: Group::Numbers,
            asks: None,
        },
        // The Windows app's own tools for documents and projects (PcTools), offered when
        // the app says it runs them. `files.rs` doesn't have them, so on the Mac they wait.
        Tool {
            name: "read_spreadsheet",
            description: "Read an Excel workbook (.xlsx) on this PC: its sheet names, then the cells of its first sheet, row by row.",
            params: vec![text(
                "path",
                "The workbook's path: a full path, or one relative to the home folder (Documents/budget.xlsx).",
            )],
            runs: Runs::App,
            group: Group::Files,
            asks: None,
        },
        Tool {
            name: "create_docx",
            description: "Create a Word document (.docx) on this PC from a title and Markdown content (headings, lists, bold). Asks the user for approval.",
            params: vec![
                text(
                    "path",
                    "Where to save it, ending in .docx: a full path, or one relative to the home folder.",
                ),
                text("title", "The document's title."),
                text("content", "The document's text in Markdown."),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Export a DOCX document",
                rememberable: true,
            }),
        },
        Tool {
            name: "run_tests",
            description: "Run a project's tests, with the test command its project files set up. Use it directly when asked to run tests. Changes nothing, and runs no other command.",
            params: vec![text("path", "The project's folder.")],
            runs: Runs::App,
            group: Group::Git,
            asks: None,
        },
        Tool {
            name: "run_build",
            description: "Build a project with the build command its project files set up. Runs no other command.",
            params: vec![text("path", "The project's folder.")],
            runs: Runs::App,
            group: Group::Git,
            asks: None,
        },
        Tool {
            name: "run_linter",
            description: "Run a project's linter or format check, as its project files set it up. Changes nothing, and runs no other command.",
            params: vec![text("path", "The project's folder.")],
            runs: Runs::App,
            group: Group::Git,
            asks: None,
        },
        // The tools that change files, as the app's PcTools runs them (its undo journal
        // keeps what each reply changed). Their descriptions say they ask, as the Mac's do,
        // so a model calls them when asked instead of asking again in the chat.
        Tool {
            name: "write_file",
            description: "Create a text file on this PC, or replace one's whole text. Asks the user for approval.",
            params: vec![
                text(
                    "path",
                    "The file's path: a full path, or one relative to the home folder.",
                ),
                text("content", "The file's whole new text."),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Create or edit files",
                rememberable: true,
            }),
        },
        Tool {
            name: "rename_file",
            description: "Rename a file or folder on this PC, keeping it where it is. Asks the user for approval.",
            params: vec![
                text("path", "The file's or folder's path."),
                text(
                    "new_name",
                    "The new name only, not a path (notes-2026.txt).",
                ),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Rename files",
                rememberable: true,
            }),
        },
        Tool {
            name: "move_file",
            description: "Move a file or folder on this PC to another folder. Asks the user for approval.",
            params: vec![
                text("source", "The file's or folder's path now."),
                text(
                    "destination",
                    "Its full new path, name included (Documents/Invoices/acme.pdf).",
                ),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Move files",
                rememberable: true,
            }),
        },
        Tool {
            name: "delete_file",
            description: "Delete a file on this PC (it goes to the Recycle Bin). Asks the user for approval.",
            params: vec![text("path", "The file's path.")],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Delete files",
                rememberable: false,
            }),
        },
    ]
}

/// What the approval says the call will do, one line for the app's sheet (which shows the
/// arguments, a written file's text among them, below it). `exists` tells whether a path
/// is there now.
pub fn approval_preview(call: &Call, exists: impl Fn(&str) -> bool) -> String {
    let arg = |name: &str| call.text(name).unwrap_or("").trim().to_string();
    match call.tool.as_str() {
        "write_file" => {
            let path = arg("path");
            let verb = if exists(&path) { "Replace" } else { "Create" };
            format!("{verb} {path}")
        }
        "rename_file" => format!("Rename {} to {}", arg("path"), arg("new_name")),
        "move_file" => format!("Move {} to {}", arg("source"), arg("destination")),
        "delete_file" => format!("Move {} to the Recycle Bin", arg("path")),
        "create_docx" => {
            let path = arg("path");
            let verb = if exists(&path) { "Replace" } else { "Create" };
            format!("{verb} the Word document {path}")
        }
        other => format!("Run {other}"),
    }
}

/// The app's tools that `files.rs` can run on the server instead, for an app that doesn't
/// run them (the Mac app).
const SERVER_FILE_TOOLS: [&str; 6] = [
    "read_file",
    "list_directory",
    "search_files",
    "file_exists",
    "git_status",
    "git_diff",
];

/// Every tool's name, offered or not.
pub fn all_names() -> Vec<&'static str> {
    catalog().iter().map(|t| t.name).collect()
}

/// The tools a reply offers: the server's own; the PC's that the app runs (its
/// `clientTools`); and, when the chat reaches files here (`files_here`), the rest of the
/// PC's, which this server then runs, but for the ones that change files: `files.rs`
/// doesn't make changes yet, so those come only from an app that runs them.
pub fn offered(client_tools: &[String], files_here: bool) -> Vec<Tool> {
    catalog()
        .into_iter()
        .filter_map(|mut t| {
            if t.runs == Runs::App && !client_tools.iter().any(|c| c == t.name) {
                if !files_here || t.asks.is_some() || !SERVER_FILE_TOOLS.contains(&t.name) {
                    return None;
                }
                t.runs = Runs::Server;
            }
            Some(t)
        })
        .collect()
}

/// The capability list `/api/capabilities` answers with (the app's Settings count them).
pub fn capabilities_json(platform: &str) -> String {
    let items: Vec<String> = catalog()
        .iter()
        .map(|t| {
            let (label, category) = match t.asks {
                // The Mac's categories: changes that ask are "automatic" there too (the
                // approval is per call); deleting is "restricted".
                Some(ask) if !ask.rememberable => (ask.label.to_string(), "restricted"),
                Some(ask) => (ask.label.to_string(), "automatic"),
                None => (t.name.replace('_', " "), "automatic"),
            };
            format!(
                "{{\"name\":{},\"label\":{},\"description\":{},\"category\":\"{category}\",\"platforms\":[{}],\"available\":true}}",
                json::string(t.name),
                json::string(&label),
                json::string(t.description),
                json::string(platform)
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
    /// A text argument, if the call has it.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.args.iter().find_map(|(k, v)| match v {
            Arg::Text(s) if k == name => Some(s.as_str()),
            _ => None,
        })
    }

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
/// A call as the model would write it, from the token that starts it, for a call the
/// server writes into a reply itself (the skill it picked, as `use_skill`). Text arguments
/// only; `parse_call` reads it back.
pub fn call_text(family: Family, call: &Call) -> String {
    let start = call_start(family).token;
    let texts: Vec<(&str, &str)> = call
        .args
        .iter()
        .filter_map(|(k, v)| match v {
            Arg::Text(s) => Some((k.as_str(), s.as_str())),
            Arg::Integer(_) => None,
        })
        .collect();
    match family {
        Family::Qwen35 => {
            let params: String = texts
                .iter()
                .map(|(k, v)| format!("<parameter={k}>\n{v}\n</parameter>\n"))
                .collect();
            format!("{start}\n<function={}>\n{params}</function>\n", call.tool)
        }
        Family::Gemma4 => {
            let params: Vec<String> = texts
                .iter()
                .map(|(k, v)| format!("{k}:{QUOTE}{v}{QUOTE}"))
                .collect();
            format!("{start}call:{}{{{}}}", call.tool, params.join(","))
        }
        Family::Llama => {
            let params: Vec<String> = texts
                .iter()
                .map(|(k, v)| format!("{}: {}", json::string(k), json::string(v)))
                .collect();
            format!(
                "{start}{{\"name\": {}, \"parameters\": {{{}}}}}",
                json::string(&call.tool),
                params.join(", ")
            )
        }
    }
}

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

/// Runs one of the server's own tools, with the files the chat may reach, if any.
pub fn run_here(call: &Call, files: Option<&Files>) -> Result<String, String> {
    match (call.tool.as_str(), files) {
        ("get_current_datetime", _) => Ok(now()),
        ("calculate", _) => crate::calc::calculate_many(call.text("expression").unwrap_or("")),
        (_, Some(files)) => files.run(call),
        (other, None) => Err(format!("{other} isn't one of this server's tools")),
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

/// The Mac's local time, from `localtime_r` (the user's time zone, as the app shows it).
#[cfg(target_os = "macos")]
fn local_time() -> (u16, u16, u16, u16, u16, u16) {
    #[repr(C)]
    struct Tm {
        sec: i32,
        min: i32,
        hour: i32,
        mday: i32,
        mon: i32,
        year: i32,
        wday: i32,
        yday: i32,
        isdst: i32,
        gmtoff: i64,
        zone: *const u8,
    }
    unsafe extern "C" {
        fn time(t: *mut i64) -> i64;
        fn localtime_r(t: *const i64, out: *mut Tm) -> *mut Tm;
    }
    let mut tm = Tm {
        sec: 0,
        min: 0,
        hour: 0,
        mday: 1,
        mon: 0,
        year: 70,
        wday: 4,
        yday: 0,
        isdst: 0,
        gmtoff: 0,
        zone: std::ptr::null(),
    };
    let now = unsafe { time(std::ptr::null_mut()) };
    unsafe { localtime_r(&now, &mut tm) };
    (
        (tm.year + 1900) as u16,
        (tm.mon + 1) as u16,
        tm.mday as u16,
        tm.wday as u16,
        tm.hour as u16,
        tm.min as u16,
    )
}

#[cfg(not(any(windows, target_os = "macos")))]
fn local_time() -> (u16, u16, u16, u16, u16, u16) {
    // UTC, from the system clock.
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
        let tools = offered(&["read_file".into(), "search_files".into()], false);
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
        let tools = offered(&["read_file".into()], false);
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
            3,
            "read_file, and the server's own date and time and calculate"
        );
    }

    #[test]
    fn the_date_reads_like_the_macs() {
        let n = now();
        assert!(n.ends_with("(local time)") && n.contains(", 20"), "{n}");
    }

    #[test]
    fn a_call_the_server_writes_reads_back() {
        let call = Call {
            tool: "use_skill".into(),
            args: vec![("name".into(), Arg::Text("code-review".into()))],
        };
        for family in [Family::Qwen35, Family::Llama, Family::Gemma4] {
            let text = call_text(family, &call);
            let start = call_start(family).token;
            assert!(text.starts_with(start), "{text}");
            assert_eq!(
                parse_call(family, &text[start.len()..]),
                Some(call.clone()),
                "{text}"
            );
        }
    }

    #[test]
    fn changes_ask_as_the_macs_do() {
        let all: Vec<String> = [
            "read_file",
            "write_file",
            "rename_file",
            "move_file",
            "delete_file",
        ]
        .map(String::from)
        .to_vec();
        let tools = offered(&all, false);
        let ask = |name: &str| tools.iter().find(|t| t.name == name).unwrap().asks;
        assert_eq!(ask("read_file"), None);
        assert_eq!(
            ask("rename_file").unwrap().remember_label(),
            "Don't ask again to rename files"
        );
        assert!(
            !ask("delete_file").unwrap().rememberable,
            "deleting always asks"
        );
        // Only what the app runs is offered.
        assert!(
            offered(&["read_file".into()], false)
                .iter()
                .all(|t| t.asks.is_none())
        );

        let call = |tool: &str, args: &[(&str, &str)]| Call {
            tool: tool.into(),
            args: args
                .iter()
                .map(|(k, v)| (k.to_string(), Arg::Text(v.to_string())))
                .collect(),
        };
        let write = call(
            "write_file",
            &[("path", "Documents/a.txt"), ("content", "x")],
        );
        assert_eq!(
            approval_preview(&write, |_| false),
            "Create Documents/a.txt"
        );
        assert_eq!(
            approval_preview(&write, |_| true),
            "Replace Documents/a.txt"
        );
        assert_eq!(
            approval_preview(
                &call(
                    "move_file",
                    &[("source", "a.pdf"), ("destination", "Invoices/a.pdf")]
                ),
                |_| true
            ),
            "Move a.pdf to Invoices/a.pdf"
        );
        // Each family declares them and writes a call to one, content and all.
        for family in [Family::Qwen35, Family::Llama, Family::Gemma4] {
            let declared = declarations(family, &tools);
            assert!(
                declared.contains("new_name") && declared.contains("Asks the user for approval")
            );
        }
    }

    #[test]
    fn the_pc_tools_run_where_the_app_says() {
        let runs =
            |tools: &[Tool], name: &str| tools.iter().find(|t| t.name == name).map(|t| t.runs);
        // An app that runs them (the Windows app, the eval's stand-in PC) is asked to.
        let app = offered(&["read_file".into()], false);
        assert_eq!(runs(&app, "read_file"), Some(Runs::App));
        assert_eq!(runs(&app, "list_directory"), None);
        // With access and no client tools (the Mac app), the server runs what files.rs
        // has: not the changes, nor the app's document and project tools.
        let here = offered(&[], true);
        assert_eq!(
            here.iter().map(|t| t.name).collect::<Vec<_>>(),
            [
                "read_file",
                "list_directory",
                "search_files",
                "file_exists",
                "git_status",
                "git_diff",
                "get_current_datetime",
                "calculate"
            ]
        );
        assert!(
            here.iter()
                .all(|t| t.runs == Runs::Server && t.asks.is_none())
        );
        // An app that runs a change tool is still asked to, with access or without.
        let both = offered(&["write_file".into()], true);
        assert_eq!(runs(&both, "write_file"), Some(Runs::App));
        // Without access, only the server's own.
        let none = offered(&[], false);
        assert_eq!(
            none.iter().map(|t| t.name).collect::<Vec<_>>(),
            ["get_current_datetime", "calculate"]
        );
    }
}
