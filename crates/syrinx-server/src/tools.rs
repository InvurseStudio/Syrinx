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
//!
//! Some tools lean on the app besides: the calendar's are the app's alone (the Mac app,
//! which holds the permission); `schedule_task` is offered to an app that runs the tasks
//! this server keeps (`tasks.rs`); and `search_chats` runs here for an app whose chats
//! are on this machine (`chats.rs`). The app says which in `clientTools`.

use syrinx_engine::grammar::{Property, RecordStyle, Schema, json_string};

use crate::chat::Family;
use crate::files::Files;
use crate::json;

/// What an argument holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Text,
    #[allow(dead_code)] // for the integer arguments to come (lines, depth)
    Integer,
    /// One plain value of any kind: text, a number, true or false, or null (to skip): a
    /// spreadsheet's cell.
    Cell,
    /// A list of these.
    List(&'static Kind),
    /// An object with these fields.
    Object(&'static [Param]),
    /// An object whose keys the model names, each with text: a template's placeholders.
    Map,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// What Vireo remembers about the user, and their earlier chats.
    Memory,
    /// A Git or code project: its status and diff, its tests, build and linter.
    Git,
    Time,
    /// Arithmetic (`calculate`).
    Numbers,
    /// The user's calendar and reminders, which the app reaches (the Mac's EventKit).
    Calendar,
    /// Tasks Vireo runs later on its own (`schedule_task`).
    Tasks,
    /// The web: searching it and reading a page (`web.rs`).
    Web,
    /// The Mac itself: its Shortcuts, its screen, Messages.
    Mac,
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
/// The most of a request a scoring question holds, in characters: its start and its end.
const SCORED_START: usize = 300;
const SCORED_END: usize = 300;

/// A request as the server's own questions about it hold it (which tools it needs,
/// whether it asks for a change, whether a skill fits): on one line, and a long one as
/// its start and its end with the middle left out. What a request asks for is said at
/// one end or the other, and what's pasted in between would be read once for every
/// question (and, past the questions' limit, would push the question's own end out).
pub fn for_scoring(request: &str) -> String {
    let request = request.trim().replace('\n', " ");
    let count = request.chars().count();
    if count <= SCORED_START + SCORED_END {
        return request;
    }
    let start: String = request.chars().take(SCORED_START).collect();
    let end: String = request.chars().skip(count - SCORED_END).collect();
    format!("{} ... {}", start.trim_end(), end.trim_start())
}

pub fn group_question(request: &str) -> String {
    let request = for_scoring(request);
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
    let request = for_scoring(request);
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

/// The middle of a long request, which `for_scoring` leaves out of the server's questions;
/// `None` for a request they hold whole.
pub fn unscored(request: &str) -> Option<String> {
    let request = request.trim().replace('\n', " ");
    let count = request.chars().count();
    (count > SCORED_START + SCORED_END).then(|| {
        request
            .chars()
            .skip(SCORED_START)
            .take(count - SCORED_START - SCORED_END)
            .collect()
    })
}

/// Whether text says in so many words to change a file: a tool that does, or the usual
/// ways of asking. Wide on purpose: it only ever shows the tools that change files, each
/// of which still asks the user before it runs.
pub fn names_a_change(text: &str) -> bool {
    const WORDS: [&str; 30] = [
        "append", "add a row", "add row", "add it to", "add them to", "add each", "add one row",
        "write to", "write it to", "write the", "save ", "update the", "update it", "log it",
        "log the", "log each", "log them", "record it", "record the", "record each", "rename",
        "move the", "move it", "delete", "fill in", "fill out", "create a file", "create the file",
        "write_", "create_",
    ];
    let text = text.to_lowercase();
    WORDS.iter().any(|w| text.contains(w)) || text.contains("fill_docx_template")
}

/// The files and folders a message names, as the user writes them ("Invoices/November",
/// "Policies/AP-Guide.md", "budget.xlsx", or quoted paths with spaces like `"Q3 Budget.xlsx"`),
/// in order, without repeats: for reading ahead. Not web addresses, nor words joined by a
/// slash ("and/or", "24/7").
pub fn named_paths(message: &str) -> Vec<String> {
    const TYPES: [&str; 9] = ["md", "txt", "csv", "pdf", "docx", "xlsx", "pptx", "json", "eml"];
    const MOST: usize = 8;
    const AROUND: &str = "\"'`()[]{}<>*,;:!?";
    let is_path = |raw: &str| -> Option<String> {
        let word = raw
            .trim()
            .trim_start_matches(|c: char| AROUND.contains(c))
            .trim_end_matches(|c: char| AROUND.contains(c) || c == '.' || c == '/')
            .trim_start_matches("./");
        if word.is_empty()
            || word.len() > 200
            || word.contains('\n')
            || word.contains("://")
            || word.contains('@')
            || word.starts_with('~')
        {
            return None;
        }
        let parts: Vec<&str> = word.split('/').filter(|p| !p.is_empty()).collect();
        if parts.is_empty() || parts.iter().any(|p| *p == "." || *p == "..") {
            return None;
        }
        let typed = parts
            .last()
            .and_then(|name| name.rsplit_once('.'))
            .is_some_and(|(stem, ext)| !stem.is_empty() && TYPES.contains(&ext.to_lowercase().as_str()));
        // Words joined by a slash, fractions and dates aren't paths.
        let joined = !typed
            && parts
                .iter()
                .all(|p| p.chars().count() <= 3 || p.chars().all(|c| c.is_ascii_digit()));
        if !(typed || word.contains('/')) || joined {
            return None;
        }
        Some(word.to_string())
    };

    let mut found: Vec<String> = Vec::new();
    // First scan quoted spans ("...", '...', `...`) so paths with spaces are preserved
    // whole instead of being split on whitespace.
    let mut i = 0;
    let bytes = message.as_bytes();
    while i < bytes.len() && found.len() < MOST {
        let b = bytes[i];
        if matches!(b, b'"' | b'\'' | b'`') {
            if let Some(rel_end) = message[i + 1..].find(b as char) {
                let inner = &message[i + 1..i + 1 + rel_end];
                if inner.contains(' ')
                    && !inner.contains('\n')
                    && let Some(p) = is_path(inner)
                    && !found.iter().any(|f| f == &p)
                {
                    found.push(p);
                }
                i += rel_end + 2;
                continue;
            }
        }
        i += 1;
    }
    for word in message.split_whitespace() {
        if found.len() == MOST {
            break;
        }
        if let Some(p) = is_path(word) {
            // Skip fragments that are already part of a quoted path with spaces.
            if !found.iter().any(|f| f == &p || (f.contains(' ') && f.contains(&p))) {
                found.push(p);
            }
        }
    }
    found
}

/// `use_skill` as a tool the model calls itself (a model that thinks: see `reply`). The
/// skills are listed in its instructions (`skills::listing`), so this stays the same
/// whatever skills there are.
pub fn use_skill_tool() -> Tool {
    Tool {
        name: "use_skill",
        description: "Load a skill's saved instructions for one kind of task, then follow them. The skills are listed in your instructions.",
        params: vec![text("name", "The skill's name, as the Skills list gives it.")],
        runs: Runs::Server,
        group: Group::Memory,
        asks: None,
    }
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
    param(name, Kind::Text, description)
}

const fn param(name: &'static str, kind: Kind, description: &'static str) -> Param {
    Param {
        name,
        kind,
        description,
        required: true,
    }
}

/// A spreadsheet's rows: lists of cells.
const ROWS: Kind = Kind::List(&Kind::List(&Kind::Cell));
/// A slide of a deck, as `create_pptx` takes it.
const SLIDE: [Param; 2] = [
    text("title", "The slide's title."),
    param(
        "bullets",
        Kind::List(&Kind::Text),
        "The slide's bullet points, each a short line.",
    ),
];
const SLIDES: Kind = Kind::List(&Kind::Object(&SLIDE));

impl Tool {
    /// Whether it changes the user's files: offered only when a request asks for a change.
    /// (Remembering asks for approval too, but a request that tells something about the
    /// user is no request to change a file.)
    pub fn changes_files(&self) -> bool {
        self.asks.is_some() && self.group == Group::Files
    }
}

/// Every tool this server knows. Optional arguments are left out for now: small models
/// fill them in where they shouldn't (Llama 3.2 1B asked for one line of a file), so
/// they'll come back where the eval says a model uses them well.
fn catalog() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file",
            description: "Read a text file, PDF, or Word, Excel or PowerPoint document on this PC. To read several files, give all their paths in one call, one a line, rather than a call for each.",
            params: vec![text(
                "path",
                "The file's path: a full path, or one relative to the home folder (Documents/report.pdf). Several files: one path a line. Don't make one up.",
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
            description: "Work out arithmetic exactly. Use it for every total, sum, difference, percentage or average instead of doing the maths yourself, e.g. \"412.50 + 129.30 + 400\" or \"sum([12.40, 8.99, 31.00]) * 1.13\". Dollar signs, thousands commas and % are fine. Work out everything you need in one call, one expression a line, rather than a call for each.",
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
            name: "write_spreadsheet",
            description: "Write rows of values into an Excel workbook (.xlsx) on this PC, creating the workbook if needed. To add rows to a sheet that already has some (a register, a tracker, a log), set start_cell to \"append\": they go after the last row, in the order of the header's columns. A text value starting with = is a formula, such as =SUM(B2:B9). Other cells are left as they are. Asks the user for approval.",
            params: vec![
                text(
                    "path",
                    "The workbook's path: a full path, or one relative to the home folder.",
                ),
                text(
                    "start_cell",
                    "The top-left cell to write from, such as A1, or \"append\" to add the rows after the last one.",
                ),
                param(
                    "rows",
                    ROWS,
                    "The rows to write, each a list of cell values (text, numbers, true/false, or null to skip a cell).",
                ),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Write to spreadsheets",
                rememberable: true,
            }),
        },
        Tool {
            name: "fill_docx_template",
            description: "Fill a Word template on this PC: replace each {{placeholder}} in a .docx with a value, keeping its formatting, and save the result as a new document. Read the template first to see its placeholders. Asks the user for approval.",
            params: vec![
                text("template", "The template's path (.docx)."),
                text(
                    "path",
                    "Where to save the filled document (.docx). A file name on its own saves it in the template's folder.",
                ),
                param(
                    "values",
                    Kind::Map,
                    "Each placeholder's name, without the braces, and its text, e.g. {\"client\": \"Acme\", \"date\": \"3 November 2026\"}.",
                ),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Fill Word templates",
                rememberable: true,
            }),
        },
        Tool {
            name: "create_pptx",
            description: "Create a PowerPoint deck (.pptx) on this PC from slide titles and bullet points. Asks the user for approval.",
            params: vec![
                text(
                    "path",
                    "Where to save it, ending in .pptx: a full path, or one relative to the home folder.",
                ),
                text("title", "The deck's title, shown on its first slide."),
                param("slides", SLIDES, "The slides after the title slide."),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Create slide decks",
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
        // Memory: facts about the user kept by this server for later chats, and the app's
        // own chats, which it searches.
        Tool {
            name: "remember",
            description: "Save something about the user to remember in later chats: a preference, a fact about them, people or projects that matter to them, where their files are. Use it when they ask you to remember something, or tell you something they will clearly want remembered. One short fact per call. Asks the user for approval.",
            params: vec![text(
                "fact",
                "The fact in one sentence, e.g. \"Prefers metric units.\"",
            )],
            runs: Runs::Server,
            group: Group::Memory,
            asks: Some(Ask {
                label: "Remember",
                rememberable: true,
            }),
        },
        Tool {
            name: "forget",
            description: "Delete a remembered fact the user wants forgotten. Asks the user for approval.",
            params: vec![text(
                "fact",
                "The fact to forget, in the words it was saved or close to them.",
            )],
            runs: Runs::Server,
            group: Group::Memory,
            asks: Some(Ask {
                label: "Forget",
                rememberable: false,
            }),
        },
        Tool {
            name: "search_chats",
            description: "Search the user's earlier chats with you. Use it first when they refer to a past conversation or to something worked out in one (\"what did we decide about\", \"last time\", \"which did we\"), before searching files. Returns matching chats with dates and excerpts. Never guess what was said earlier.",
            params: vec![text(
                "query",
                "Words that would appear in the earlier chat, e.g. \"Harbor Lane memo budget\".",
            )],
            runs: Runs::App,
            group: Group::Memory,
            asks: None,
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
        // The user's calendar and reminders: the app's, where it has them (the Mac app,
        // through EventKit, which holds the permission). Every argument is asked for, with
        // "" for none, as the rest of the catalog has no optional ones yet.
        Tool {
            name: "list_events",
            description: "List the events in the user's calendars between two days or times. Check it before adding an event that may already be there.",
            params: vec![
                text(
                    "start",
                    "The first day, e.g. 2026-11-20, or a time, 2026-11-20T09:00.",
                ),
                text("end", "The last day (it's included), or a time."),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: None,
        },
        Tool {
            name: "create_event",
            description: "Add an event to the user's calendar. Times are this computer's local time: for something in another time zone, work out the local time here and say the time there in the notes. One event a call: for a trip or a programme, an event for each part its sources give (every flight and drive, each session), not the main event alone, and none for a part marked cancelled. Asks the user for approval.",
            params: vec![
                text("title", "The event's title."),
                text(
                    "start",
                    "When it starts, e.g. 2026-11-26T18:00, or 2026-11-26 for an all-day event. Always with the year.",
                ),
                text(
                    "end",
                    "When it ends, in the same form. For a one-day all-day event, the same day as start.",
                ),
                text(
                    "notes",
                    "What's worth keeping with it (the place, an amount, who it's with), or \"\" for nothing.",
                ),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: Some(Ask {
                label: "Add calendar events",
                rememberable: true,
            }),
        },
        Tool {
            name: "create_reminder",
            description: "Add a reminder to the user's Reminders. One reminder a call. Asks the user for approval.",
            params: vec![
                text("title", "What to be reminded of."),
                text(
                    "due",
                    "When it's due, e.g. 2026-11-25 or 2026-11-25T09:00 (local time, with the year), or \"\" for no date.",
                ),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: Some(Ask {
                label: "Add reminders",
                rememberable: true,
            }),
        },
        // A task Vireo runs later on its own, in a chat of its own: this server keeps it
        // and says when it's due, and the app runs it (it lists `run_tasks` when it does).
        Tool {
            name: "schedule_task",
            description: "Set up a task you run on your own later, in a new chat each time: at a time of day (\"every weekday at 8:00, list what's due\"), once at a date and time, whenever a new file lands in a folder (\"when an invoice arrives in Downloads/Invoices, file it\"), or whenever the task's own link is opened by another app or a website. Use it when the user asks for something to happen later, regularly, or whenever something arrives; it does not do the work now. Asks the user for approval.",
            params: vec![
                text("name", "A short name, e.g. \"Morning summary\"."),
                text(
                    "prompt",
                    "What to do each time, written as a full instruction to yourself, with every path, rule and file it needs: the run won't see this chat.",
                ),
                text(
                    "when",
                    "One of: \"daily 08:00\"; days and a time, \"weekdays 08:00\" or \"mon,thu 16:30\"; \"once 2026-11-20T14:00\"; \"folder \" and the folder's path, to run when new files land in it; or \"link\", to run when the task's link is opened (the result gives the link).",
                ),
            ],
            runs: Runs::Server,
            group: Group::Tasks,
            asks: Some(Ask {
                label: "Set up tasks",
                rememberable: false,
            }),
        },
        Tool {
            name: "list_scheduled_tasks",
            description: "List the tasks you run on your own later (set up with schedule_task): each one's name, when it runs, whether it's on, and what it does.",
            params: vec![],
            runs: Runs::Server,
            group: Group::Tasks,
            asks: None,
        },
        // Changing part of a file, and a PDF: as write_file and create_docx are run.
        Tool {
            name: "edit_file",
            description: "Change part of a text file on this PC by replacing exact text, leaving the rest as it is. Use it rather than write_file for any change to part of a file, and read the file first. Asks the user for approval.",
            params: vec![
                text(
                    "path",
                    "The file's path: a full path, or one relative to the home folder.",
                ),
                text(
                    "old_text",
                    "The text to replace, copied exactly from the file, with enough around it to appear only once.",
                ),
                text(
                    "new_text",
                    "What replaces old_text, and only that: not the rest of the file (\"\" to remove it).",
                ),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Edit files",
                rememberable: true,
            }),
        },
        Tool {
            name: "create_pdf",
            description: "Create a PDF document on this PC from a title and Markdown content (headings, lists, bold, tables). Asks the user for approval.",
            params: vec![
                text(
                    "path",
                    "Where to save it, ending in .pdf: a full path, or one relative to the home folder.",
                ),
                text("title", "The document's title."),
                text("content", "The document's text in Markdown."),
            ],
            runs: Runs::App,
            group: Group::Files,
            asks: Some(Ask {
                label: "Export a PDF document",
                rememberable: true,
            }),
        },
        Tool {
            name: "run_shell_command",
            description: "Run a shell command on this PC when no other tool can do the job. It starts in the home folder and is stopped after two minutes. The user sees the command and approves each run.",
            params: vec![text("command", "The command, as typed at a terminal.")],
            runs: Runs::App,
            group: Group::Git,
            asks: Some(Ask {
                label: "Run shell commands",
                rememberable: false,
            }),
        },
        // The web, for an app that turns it on (`WEB`).
        Tool {
            name: "search_web",
            description: "Search the web for current information, or anything you can't answer from what you know. Returns results with their addresses; read a page with fetch_url.",
            params: vec![
                text(
                    "query",
                    "A focused query with the specific names, numbers or phrase to find.",
                ),
                text(
                    "domain",
                    "One site to search within, such as wikipedia.org (no https://), or \"\" for the whole web.",
                ),
            ],
            runs: Runs::Server,
            group: Group::Web,
            asks: None,
        },
        Tool {
            name: "fetch_url",
            description: "Read a web page: its text, without the page's menus and scripts.",
            params: vec![text(
                "url",
                "The page's full address, starting with https:// or http://.",
            )],
            runs: Runs::Server,
            group: Group::Web,
            asks: None,
        },
        // Skills the user keeps, written by the model when asked (`skills.rs`).
        Tool {
            name: "save_skill",
            description: "Save steps as a new skill the user can reuse in later chats, when they ask to keep how to do something (\"save this as a skill\"). Write the steps so a chat that never saw this one can follow them. Asks the user for approval.",
            params: vec![
                text(
                    "name",
                    "A short lowercase name with hyphens, e.g. \"weekly-report\".",
                ),
                text(
                    "description",
                    "What it does and when to use it, in one or two sentences.",
                ),
                text("instructions", "The steps and rules, in Markdown."),
            ],
            runs: Runs::Server,
            group: Group::Memory,
            asks: Some(Ask {
                label: "Save skills",
                rememberable: true,
            }),
        },
        Tool {
            name: "update_skill",
            description: "Rewrite one of the user's skills, when they ask to change it or one you followed turned out wrong or missing a step. Read it with use_skill first, then send the complete new instructions, keeping what still holds. The old version is kept. Asks the user for approval.",
            params: vec![
                text("name", "The skill to change, e.g. \"weekly-report\"."),
                text("instructions", "The complete new steps and rules, in Markdown."),
                text(
                    "description",
                    "A new one-sentence description, or \"\" to keep the one it has.",
                ),
            ],
            runs: Runs::Server,
            group: Group::Memory,
            asks: Some(Ask {
                label: "Change skills",
                rememberable: true,
            }),
        },
        // The Mac's Shortcuts, run by this server there (`shortcuts.rs`).
        Tool {
            name: "run_shortcut",
            description: "Run one of the user's Shortcuts on the Mac, which can reach other apps and devices. Call it with name \"\" to list them first. The user approves each run.",
            params: vec![
                text(
                    "name",
                    "The shortcut's name, as listed, or \"\" to list the shortcuts.",
                ),
                text("input", "Text to give the shortcut as its input, or \"\" for none."),
            ],
            runs: Runs::Server,
            group: Group::Mac,
            asks: Some(Ask {
                label: "Run shortcuts",
                rememberable: false,
            }),
        },
        // More of the calendar and reminders, and the Mac's Messages and screen: the Mac
        // app's, which holds the permissions.
        Tool {
            name: "list_reminders",
            description: "List the user's reminders that aren't done yet, with their due dates and ids.",
            params: vec![],
            runs: Runs::App,
            group: Group::Calendar,
            asks: None,
        },
        Tool {
            name: "complete_reminder",
            description: "Mark a reminder done. Use its id from list_reminders. Asks the user for approval.",
            params: vec![
                text("id", "The reminder's id, from list_reminders."),
                text("title", "The reminder's title, as list_reminders shows it."),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: Some(Ask {
                label: "Complete reminders",
                rememberable: true,
            }),
        },
        Tool {
            name: "move_event",
            description: "Move a calendar event to a new time. Use its id from list_events. Asks the user for approval.",
            params: vec![
                text("id", "The event's id, from list_events."),
                text("title", "The event's title, as list_events shows it."),
                text(
                    "start",
                    "Its new start, e.g. 2026-11-26T18:00, or 2026-11-26 for all day. Always with the year.",
                ),
                text(
                    "end",
                    "Its new end in the same form, or \"\" to keep its length.",
                ),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: Some(Ask {
                label: "Move calendar events",
                rememberable: true,
            }),
        },
        Tool {
            name: "delete_event",
            description: "Delete a calendar event (only this occurrence of a repeating one). Use its id from list_events. Asks the user for approval.",
            params: vec![
                text("id", "The event's id, from list_events."),
                text("title", "The event's title, as list_events shows it."),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: Some(Ask {
                label: "Delete calendar events",
                rememberable: false,
            }),
        },
        Tool {
            name: "find_free_time",
            description: "Find open slots in the user's calendars between two days, within working hours (9:00 to 17:00, weekdays).",
            params: vec![
                text("start", "The first day to look on, e.g. 2026-11-20."),
                text("end", "The last day to look on (it's included)."),
                text(
                    "minutes",
                    "How long the slot must be, in minutes, e.g. \"30\".",
                ),
            ],
            runs: Runs::App,
            group: Group::Calendar,
            asks: None,
        },
        Tool {
            name: "send_message",
            description: "Send a text message through the Mac's Messages app (iMessage, or SMS through the user's iPhone). Use it only when the user asks you to message someone, with the wording they want. Asks the user for approval.",
            params: vec![
                text(
                    "to",
                    "A contact's name, a phone number or an email address, or \"me\" for the user themselves.",
                ),
                text("text", "The message."),
            ],
            runs: Runs::App,
            group: Group::Mac,
            asks: Some(Ask {
                label: "Send messages",
                rememberable: true,
            }),
        },
        Tool {
            name: "read_screen",
            description: "Read the text on the user's Mac screen: the front window of the app they used just before Vireo, or with target \"screen\" the whole display. Use it when they refer to \"this\", \"my screen\", or something open in another app. Asks the user for approval.",
            params: vec![text(
                "target",
                "\"window\" for the front window of the app they were using, or \"screen\" for the whole display.",
            )],
            runs: Runs::App,
            group: Group::Mac,
            asks: Some(Ask {
                label: "Read your screen",
                rememberable: true,
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
        "write_spreadsheet" => {
            let rows = call
                .json("rows")
                .and_then(|r| json::parse(r).ok())
                .map_or(0, |v| match v {
                    json::Value::Array(rows) => rows.len(),
                    _ => 0,
                });
            let s = if rows == 1 { "" } else { "s" };
            if arg("start_cell").eq_ignore_ascii_case("append") {
                format!("Add {rows} row{s} to {}", arg("path"))
            } else {
                format!("Write {rows} row{s} to {}", arg("path"))
            }
        }
        "fill_docx_template" => format!("Fill {} and save it as {}", arg("template"), arg("path")),
        "create_pptx" => {
            let path = arg("path");
            let verb = if exists(&path) { "Replace" } else { "Create" };
            format!("{verb} the slide deck {path}")
        }
        "create_event" => format!("Add \u{201c}{}\u{201d} to the calendar, {}", arg("title"), {
            let (start, end) = (arg("start"), arg("end"));
            if end.is_empty() || end == start {
                start
            } else {
                format!("{start} to {end}")
            }
        }),
        "create_reminder" => {
            let due = arg("due");
            if due.is_empty() {
                format!("Add the reminder \u{201c}{}\u{201d}", arg("title"))
            } else {
                format!("Add the reminder \u{201c}{}\u{201d}, due {due}", arg("title"))
            }
        }
        "schedule_task" => format!(
            "Set up the task \u{201c}{}\u{201d}: {}",
            arg("name"),
            crate::tasks::When::parse(&arg("when"))
                .map_or_else(|_| arg("when"), |w| w.described())
        ),
        "remember" => format!("Remember: {}", arg("fact")),
        "forget" => format!("Forget: {}", arg("fact")),
        "edit_file" => format!("Edit {}", arg("path")),
        "create_pdf" => {
            let path = arg("path");
            let verb = if exists(&path) { "Replace" } else { "Create" };
            format!("{verb} the PDF {path}")
        }
        "run_shell_command" => format!("Run this command: {}", arg("command")),
        "save_skill" => format!("Save the skill \u{201c}{}\u{201d}", arg("name")),
        "update_skill" => format!("Rewrite the skill \u{201c}{}\u{201d} (the old version is kept)", arg("name")),
        "run_shortcut" => format!("Run the shortcut \u{201c}{}\u{201d}", arg("name")),
        "complete_reminder" => format!("Mark the reminder \u{201c}{}\u{201d} done", arg("title")),
        "move_event" => format!("Move \u{201c}{}\u{201d} to {}", arg("title"), arg("start")),
        "delete_event" => format!("Delete the calendar event \u{201c}{}\u{201d}", arg("title")),
        "send_message" => format!("Send a message to {}", arg("to")),
        "read_screen" => {
            if arg("target") == "screen" {
                "Read the whole screen".to_string()
            } else {
                "Read the window you were using before Vireo".to_string()
            }
        }
        other => format!("Run {other}"),
    }
}

/// The app's tools that `files.rs` can run on the server instead, for an app that doesn't
/// run them (the Mac app).
const SERVER_FILE_TOOLS: [&str; 7] = [
    "read_file",
    "read_spreadsheet",
    "list_directory",
    "search_files",
    "file_exists",
    "git_status",
    "git_diff",
];

/// The changes `files.rs` makes itself on the Mac, where the app runs no tools (on Windows
/// the app makes them, with its undo journal).
const SERVER_CHANGE_TOOLS: [&str; 11] = [
    "write_file",
    "rename_file",
    "move_file",
    "delete_file",
    "write_spreadsheet",
    "create_docx",
    "fill_docx_template",
    "create_pptx",
    "edit_file",
    "create_pdf",
    "run_shell_command",
];

/// The tools offered only to an app that turns them on in `clientTools`, with the word
/// it lists: they reach past this machine's files (the web), run anything (a shell
/// command, a shortcut), and so stay off for an app or a test runner that doesn't ask.
const TURNED_ON: [(&str, &str); 4] = [
    ("search_web", WEB),
    ("fetch_url", WEB),
    ("run_shell_command", SHELL),
    ("run_shortcut", SHORTCUTS),
];
/// The app turns on the web tools.
pub const WEB: &str = "web";
/// The app turns on shell commands (they run on this server's machine).
pub const SHELL: &str = "shell";
/// The app turns on the Mac's Shortcuts.
pub const SHORTCUTS: &str = "shortcuts";

/// What an app lists in `clientTools` besides the tools it runs, for what it does that
/// this server's own tools lean on. Not tools the model sees.
/// The app runs the tasks this server says are due (it asks at `/api/tasks/due`).
pub const RUNS_TASKS: &str = "run_tasks";
/// The app keeps its chats in the prototype's folder on this machine, where this server
/// can search them (`chats.rs`): the Mac app.
pub const CHATS_HERE: &str = "chats_here";

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
            let lists = |name: &str| client_tools.iter().any(|c| c == name);
            // A task needs an app that runs it when it's due.
            if (t.name == "schedule_task" || t.name == "list_scheduled_tasks") && !lists(RUNS_TASKS) {
                return None;
            }
            if let Some((_, word)) = TURNED_ON.iter().find(|(name, _)| *name == t.name)
                && !lists(word)
            {
                return None;
            }
            // Shortcuts are the Mac's, run here.
            if t.name == "run_shortcut" && !cfg!(target_os = "macos") {
                return None;
            }
            if t.name == "search_chats" && !lists(t.name) && lists(CHATS_HERE) {
                t.runs = Runs::Server;
                return Some(t);
            }
            if t.runs == Runs::App && !client_tools.iter().any(|c| c == t.name) {
                let here = SERVER_FILE_TOOLS.contains(&t.name)
                    || (cfg!(target_os = "macos") && SERVER_CHANGE_TOOLS.contains(&t.name));
                if !files_here || !here {
                    return None;
                }
                t.runs = Runs::Server;
                if t.name == "delete_file" {
                    t.description = "Delete a file on this Mac (it goes to the Trash). Asks the user for approval.";
                }
            }
            Some(t)
        })
        .collect()
}

/// Whether a change the server would make itself could be made now (`Files::check`).
pub fn check_here(call: &Call, files: Option<&Files>) -> Result<(), String> {
    files.map_or(Ok(()), |f| f.check(call))
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

/// An argument's JSON schema, as the Mac's server declares it: its type, description,
/// then its items or properties. A cell (any plain value) is `{}`.
fn kind_json(kind: Kind, description: Option<&str>) -> J {
    let kind_name = match kind {
        Kind::Text => "string",
        Kind::Integer => "integer",
        Kind::Cell => return J::Obj(vec![]),
        Kind::List(_) => "array",
        Kind::Object(_) | Kind::Map => "object",
    };
    let mut fields = vec![("type".to_string(), J::Str(kind_name.into()))];
    if let Some(d) = description {
        fields.push(("description".into(), J::Str(d.into())));
    }
    match kind {
        Kind::List(item) => fields.push(("items".into(), kind_json(*item, None))),
        Kind::Object(params) => {
            fields.push(("properties".into(), properties_json(params)));
            fields.push(("required".into(), required_json(params)));
        }
        _ => {}
    }
    J::Obj(fields)
}

fn properties_json(params: &[Param]) -> J {
    J::Obj(
        params
            .iter()
            .map(|p| (p.name.to_string(), kind_json(p.kind, Some(p.description))))
            .collect(),
    )
}

fn required_json(params: &[Param]) -> J {
    J::Arr(
        params
            .iter()
            .filter(|p| p.required)
            .map(|p| J::Str(p.name.into()))
            .collect(),
    )
}

/// A tool as the OpenAI-style function declaration Qwen's and Llama's templates write.
fn declaration(tool: &Tool) -> J {
    let properties = properties_json(&tool.params);
    let required = required_json(&tool.params);
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
                        ("properties".into(), properties),
                        ("required".into(), required),
                    ]),
                ),
            ]),
        ),
    ])
}

const QUOTE: &str = "<|\"|>";

fn q(s: &str) -> String {
    format!("{QUOTE}{s}{QUOTE}")
}

/// A kind's type as Gemma 4's declarations name it.
fn gemma_type(kind: Kind) -> &'static str {
    match kind {
        Kind::Text | Kind::Cell => "STRING",
        Kind::Integer => "INTEGER",
        Kind::List(_) => "ARRAY",
        Kind::Object(_) | Kind::Map => "OBJECT",
    }
}

/// Parameters as Gemma 4's template's `format_parameters` writes them: keys in alphabetical
/// order, each `name:{description:...,[items or properties],type:...}`.
fn gemma_properties(params: &[Param]) -> String {
    let mut params = params.to_vec();
    params.sort_by_key(|p| p.name);
    params
        .iter()
        .map(|p| gemma_property(p.name, p.kind, p.description))
        .collect::<Vec<_>>()
        .join(",")
}

fn gemma_required(params: &[Param]) -> Option<String> {
    let required: Vec<String> = params
        .iter()
        .filter(|p| p.required)
        .map(|p| q(p.name))
        .collect();
    (!required.is_empty()).then(|| format!("required:[{}]", required.join(",")))
}

fn gemma_property(name: &str, kind: Kind, description: &str) -> String {
    let mut parts = Vec::new();
    if !description.is_empty() {
        parts.push(format!("description:{}", q(description)));
    }
    match kind {
        Kind::List(item) => {
            if let Some(items) = gemma_items(*item) {
                parts.push(format!("items:{{{items}}}"));
            }
        }
        Kind::Object(params) => {
            parts.push(format!("properties:{{{}}}", gemma_properties(params)));
            parts.extend(gemma_required(params));
        }
        // An object without properties: the template lists none.
        Kind::Map => parts.push("properties:{}".into()),
        _ => {}
    }
    parts.push(format!("type:{}", q(gemma_type(kind))));
    format!("{name}:{{{}}}", parts.join(","))
}

/// A list's `items` schema, as the template writes it (its keys in alphabetical order:
/// items, properties, required, type). None for a cell's empty schema, which it leaves out.
fn gemma_items(item: Kind) -> Option<String> {
    match item {
        Kind::Cell => None,
        Kind::List(inner) => {
            // A nested `items` is written as an argument: keys quoted.
            let mut inner_json = String::new();
            gemma_argument(&kind_json(*inner, None), &mut inner_json);
            Some(format!("items:{inner_json},type:{}", q("ARRAY")))
        }
        Kind::Object(params) => {
            let mut parts = vec![format!("properties:{{{}}}", gemma_properties(params))];
            parts.extend(gemma_required(params));
            parts.push(format!("type:{}", q("OBJECT")));
            Some(parts.join(","))
        }
        other => Some(format!("type:{}", q(gemma_type(other)))),
    }
}

/// A value as the template's `format_argument` writes it (keys quoted, in order).
fn gemma_argument(value: &J, out: &mut String) {
    match value {
        J::Str(s) => out.push_str(&q(s)),
        J::Arr(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                gemma_argument(v, out);
            }
            out.push(']');
        }
        J::Obj(fields) => {
            let mut fields: Vec<&(String, J)> = fields.iter().collect();
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            out.push('{');
            for (i, (k, v)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&q(k));
                out.push(':');
                gemma_argument(v, out);
            }
            out.push('}');
        }
    }
}

/// Gemma 4's declaration: `declaration:name{description:...,parameters:{...}}`, its own
/// syntax, keys in alphabetical order.
fn gemma_declaration(tool: &Tool) -> String {
    let params = &tool.params;
    let mut out = format!(
        "declaration:{}{{description:{}",
        tool.name,
        q(tool.description)
    );
    out.push_str(",parameters:{");
    if !params.is_empty() {
        out.push_str("properties:{");
        out.push_str(&gemma_properties(params));
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
        Family::Muse => muse_declarations(tools),
    }
}

/// Muse's template's `render_tool_defs`, then its `render_system_meta` (the recipients
/// the tools add), as its system turn writes them. A tool's namespace is its name up to a
/// dot; Vireo's names have none, so each tool is its own.
fn muse_declarations(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "In this environment you have access to a set of tools you can use to answer the user's question.\n\n\
         You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n\
         <atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n\
         <atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n\
         </atem:invoke>\n</atem:function_calls>\n\n\
         String and scalar parameters should be specified as is, while lists and objects should use JSON format. \
         Note that spaces for string values are not stripped. The output is not expected to be valid XML and is \
         parsed with regular expressions.\n\
         Here are the functions available in JSONSchema format:\n// Tool metadata\n",
    );
    let mut namespaces: Vec<&str> = Vec::new();
    for t in tools {
        let ns = t.name.split('.').next().unwrap_or(t.name);
        if !namespaces.contains(&ns) {
            namespaces.push(ns);
        }
    }
    for ns in &namespaces {
        out.push_str(&format!(
            "{{\"name\": {}, \"description\": \"\"}}\n",
            json_string(ns)
        ));
    }
    out.push_str("// Function schemas");
    for t in tools {
        let J::Obj(fields) = declaration(t) else {
            unreachable!()
        };
        let Some((_, J::Obj(function))) = fields.into_iter().find(|(k, _)| k == "function") else {
            unreachable!()
        };
        out.push('\n');
        J::Obj(function).compact(&mut out);
    }
    out.push_str(
        "\n\nHere's an example of how to call a function in the tool set:\n\
         (If the tool namespace is not specified, invoke the function directly as `example_function_name` rather \
         than `example_tool_name.example_function_name`)\n\n\
         to=example_tool_name.example_function_name\n\n\
         <atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n\
         <atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n\
         <atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\n\
         that can span\n\"multiple\" lines\n</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
    );
    out.push_str("\n\n# Valid recipients: \"self\"");
    for ns in &namespaces {
        out.push_str(&format!(", \"{ns}.*\""));
    }
    out.push_str(", \"user\".");
    out
}

// ---- Calling them -----------------------------------------------------------------------

/// The longest text argument, in bytes: room for a file's or a document's text.
const MAX_ARGUMENT: usize = 12_000;
/// The longest cell of a spreadsheet, a placeholder's name, and the most items in a list.
const MAX_CELL: usize = 1000;
const MAX_KEY: usize = 100;
const MAX_ITEMS: usize = 200;

/// A value in JSON, as Llama writes it and Qwen's template writes a structured argument
/// (`tojson`: ", " and ": ", an object's keys in alphabetical order).
fn json_value(kind: Kind) -> Schema {
    match kind {
        Kind::Text => Schema::String {
            max_bytes: Some(MAX_ARGUMENT),
        },
        Kind::Integer => Schema::Integer,
        Kind::Cell => Schema::AnyOf(vec![
            Schema::String {
                max_bytes: Some(MAX_CELL),
            },
            Schema::Number,
            Schema::Boolean,
            Schema::Null,
        ]),
        Kind::List(item) => Schema::Array {
            items: Box::new(json_value(*item)),
            max_items: Some(MAX_ITEMS),
        },
        Kind::Object(params) => {
            let mut params = params.to_vec();
            params.sort_by_key(|p| p.name);
            Schema::Object(fields(&params, |p| json_value(p.kind)))
        }
        Kind::Map => Schema::List {
            items: Box::new(Schema::Seq(vec![
                Schema::String {
                    max_bytes: Some(MAX_KEY),
                },
                Schema::Literal(": ".into()),
                Schema::String {
                    max_bytes: Some(MAX_ARGUMENT),
                },
            ])),
            max_items: Some(MAX_ITEMS),
            open: b'{',
            separator: ", ".into(),
            close: b'}',
        },
    }
}

/// Text between Gemma 4's quote marks.
fn gemma_string(max: usize) -> Schema {
    Schema::Seq(vec![
        Schema::Literal(QUOTE.into()),
        Schema::Text {
            until: QUOTE.into(),
            max_bytes: Some(max),
        },
    ])
}

fn gemma_style() -> RecordStyle {
    RecordStyle {
        open: "{".into(),
        key_open: String::new(),
        key_close: ":".into(),
        separator: ",".into(),
        close: "}".into(),
        json_keys: false,
    }
}

/// A value as Gemma 4's template writes a call's arguments (`format_argument`, keys bare):
/// `<|"|>text<|"|>`, `[a,b]`, `{key:value,...}` in alphabetical order.
fn gemma_value(kind: Kind) -> Schema {
    match kind {
        Kind::Text => gemma_string(MAX_ARGUMENT),
        Kind::Integer => Schema::Integer,
        Kind::Cell => Schema::AnyOf(vec![
            gemma_string(MAX_CELL),
            Schema::Number,
            Schema::Boolean,
            Schema::Null,
        ]),
        Kind::List(item) => Schema::List {
            items: Box::new(gemma_value(*item)),
            max_items: Some(MAX_ITEMS),
            open: b'[',
            separator: ",".into(),
            close: b']',
        },
        Kind::Object(params) => {
            let mut params = params.to_vec();
            params.sort_by_key(|p| p.name);
            Schema::Record {
                fields: fields(&params, |p| gemma_value(p.kind)),
                style: gemma_style(),
            }
        }
        Kind::Map => Schema::List {
            items: Box::new(Schema::Seq(vec![
                Schema::Text {
                    until: ":".into(),
                    max_bytes: Some(MAX_KEY),
                },
                gemma_string(MAX_ARGUMENT),
            ])),
            max_items: Some(MAX_ITEMS),
            open: b'{',
            separator: ",".into(),
            close: b'}',
        },
    }
}

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
        // Muse chooses each message's recipient at its start (`call_grammar`, from
        // `chat::opening`); a message that ends with <|eom|> has another after it, so
        // that's where a call may start mid-reply.
        Family::Muse => Start {
            token: "<|eom|>",
            brace: false,
        },
    }
}

/// Muse's recipient, as `call_grammar` writes it, when the message is to the user: plain
/// text follows.
pub const MUSE_TO_USER: &str = " to=user<|message|>";
/// The same to itself: its reasoning, which the user isn't shown. Muse can't do without
/// it (the Mac's server can only lower its strength); held to the user, every eval case
/// went unanswered by a tool.
pub const MUSE_TO_SELF: &str = " to=self<|message|>";

/// What the system prompt says first when there are tools: what they're for, so a small
/// model doesn't answer that it can't reach the user's files, and how to work, from the
/// Mac's 0.9 server (`SYSTEM_PROMPT` in `server/api_server.py`), whose rules each answer a
/// miss in the test gauntlet: sums never added up, a file in the folder left unread, a
/// policy's steps left undone, a request recorded as approved. The same all day, for the
/// prompt cache; the family's own lines follow ([`family_notes`]).
pub const PREAMBLE: &str = "You are Vireo, an assistant that does real work for the user on this computer: their files, spreadsheets and documents, and whatever else the tools below reach. You run locally and privately on this device.

How to work:
- Do the task with your tools rather than describing how it could be done, and read a file rather than guess what it says.
- The user's own files come first. For rates, prices, limits, rules and approvals, use the files in their folders (policies, guides, rate tables) before what you know, and say which file a figure or rule came from.
- Cover everything the task covers: every file in the folder asked about and every row, not a sample.
- When a task is about money, travel, cards, gifts, donations or approvals and their folders have a policy or guide for it, read that policy before you answer, even when the request does not name it, and check the task against its limits and caps.
- Finish the job. When a policy or guide you read gives steps for this kind of task (log it in a register or tracker, name the approver, attach a form), do the ones you can and say which are still open. When you list amounts for someone to approve or pay, give the total.
- Use calculate for every sum, conversion and total, and before you give a count, check it against the items you listed: seven names are seven, not eight.
- When you are done, give the result first. Then list what the user needs to know: anything that breaks a rule or goes over a limit, duplicates, missing or conflicting information, deadlines, and anything you could not do. Do not leave these out to be brief; do leave out raw tool output.
- Record what is true now. Do not write that something is approved, paid, sent or done unless a file or the user says it has happened: if it still needs someone's approval, record it as pending approval and name who must approve it.
- Before you approve, pay, send, record or change something because someone said it's fine, check it against the user's own rules where you know them or can read them (their policies, limits, who signs off what). If a rule, a policy or the user's own instruction forbids it, don't do it or record it as done, and don't offer a way round: say which rule it breaks, what the rule allows and the proper route, and ask how they want to go on. Never put full card numbers, security codes or passwords into a message, draft or file; point to the secure form or process instead.
- When the user tells you about themselves (their name and role, the people they work with, where their files are), save each fact with remember.
- If something you need is missing or unclear in a way that changes the result, ask instead of assuming. If you can't do something, say so plainly.

Changing files, the calendar or reminders and scheduling tasks ask the user for approval in the app first; if the user declines, don't try the same action again. Because the app asks, when the user has asked for one of these, do it rather than asking for confirmation in the chat.

Do not put tool calls or tokenizer markers in your answer's text.";

/// After Gemma's lines when thinking is off (see `reply`'s system turn).
pub const GEMMA_NO_THINKING: &str = "\n\nThinking is off for this reply: do not write out your reasoning, a plan, or what the user asked for. Begin with the answer itself, or with a tool call when the task needs one.";

/// For a family trained to make several calls in one step: ask for all a step needs at
/// once (0.9's `PARALLEL_CALLS_RULE`). Not Muse's (one call at a time, Meta's guide) nor
/// Llama's.
pub const PARALLEL_CALLS: &str = "Every step costs the user time, so ask for everything a step needs at once: when you need several things that do not depend on each other (reading several files, checking the calendar and reminders), call all of those tools in the same step instead of one per step.";

/// The family's own lines after [`PREAMBLE`] (0.9's `model_profiles.py`: each family's
/// `tool_note`, and Qwen's `system_note`). Muse's are `chat::MUSE_NOTE`, in its system turn.
///
/// `thinking` is the reply's: Qwen's line says to call "after your thinking ends" only
/// when it thinks. With thinking off the words named something that wasn't there, and Qwen
/// 3.8 27B took the reply for the place to report a call, not make one: asked to create a
/// small file it began with "[" twice as often as with <tool_call> and wrote "[Tool call:
/// write_file]" (or "I'll create that file for you.") and stopped, 8 times in 12; without
/// them, none in 12, <tool_call> at 0.81-0.86.
pub fn family_notes(family: Family, thinking: bool) -> Option<String> {
    match family {
        Family::Gemma4 => Some(format!(
            "{PARALLEL_CALLS}\n\nTool-call format: call tools with Gemma's own tool-call format, as the tool list below describes. Write every argument value in full: text quoted, lists in [ ], objects in {{ }} with quoted keys. Make each call outside your thinking, after it ends."
        )),
        // Not the parallel line yet: the server runs Qwen's calls one at a time.
        Family::Qwen35 => Some(format!(
            "Tool-call format: call tools only in the <tool_call> format the tools section describes, one <tool_call> block per call{}. Give each parameter its full value.\n\nKeep the final reply short: the result, then the points the user must know. Do not restate rules, file contents or your steps, do not list the sources again, and do not offer a menu of next steps. Never shorten a draft, a document or a list the user asked for.",
            if thinking { ", after your thinking ends" } else { "" }
        )),
        Family::Muse | Family::Llama => None,
    }
}

/// Special tokens a call is written with, inside it.
pub fn call_tokens(family: Family) -> &'static [&'static str] {
    match family {
        Family::Gemma4 => &[QUOTE],
        _ => &[],
    }
}

/// The grammar a call to one of `tools` is written under, from just after its start (for
/// Llama, from the turn's start; for Muse, from the message's start, with the user as a
/// recipient too: `MUSE_TO_USER`).
pub fn call_grammar(family: Family, tools: &[Tool]) -> Schema {
    match family {
        Family::Llama => {
            let value = |p: &Param| json_value(p.kind);
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
                other => Schema::Seq(vec![json_value(other), Schema::Literal(END.into())]),
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
            let value = |p: &Param| gemma_value(p.kind);
            let style = gemma_style();
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
        Family::Muse => muse_call_grammar(tools, true),
    }
}

/// Muse's message start: its recipient (the user, itself if `may_think`, or one of
/// `tools`), then for a tool the call.
pub fn muse_call_grammar(tools: &[Tool], may_think: bool) -> Schema {
    const END: &str = "</atem:parameter>\n";
    let value = |p: &Param| match p.kind {
        Kind::Text => Schema::Text {
            until: END.into(),
            max_bytes: Some(MAX_ARGUMENT),
        },
        other => Schema::Seq(vec![json_value(other), Schema::Literal(END.into())]),
    };
    let style = RecordStyle {
        open: String::new(),
        key_open: "<atem:parameter name=\"".into(),
        key_close: "\">".into(),
        separator: String::new(),
        close: "</atem:invoke>\n</atem:function_calls>".into(),
        json_keys: false,
    };
    let mut options = vec![(
        MUSE_TO_USER.strip_prefix(" to=").unwrap().to_string(),
        Schema::Seq(vec![]),
    )];
    if may_think {
        options.push((
            MUSE_TO_SELF.strip_prefix(" to=").unwrap().to_string(),
            Schema::Seq(vec![]),
        ));
    }
    options.extend(tools.iter().map(|t| {
        (
            format!("{}<|message|>", t.name),
            Schema::Seq(vec![
                Schema::Literal(format!(
                    "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
                    t.name
                )),
                Schema::Record {
                    fields: fields(&t.params, value),
                    style: style.clone(),
                },
            ]),
        )
    }));
    Schema::Seq(vec![
        Schema::Literal(" to=".into()),
        Schema::Branch(options),
    ])
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
    /// A list or object (or a number that isn't whole), as compact JSON.
    Json(String),
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

    /// A structured argument, as JSON, if the call has it.
    pub fn json(&self, name: &str) -> Option<&str> {
        self.args.iter().find_map(|(k, v)| match v {
            Arg::Json(s) if k == name => Some(s.as_str()),
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
                    Arg::Json(j) => j.clone(),
                };
                format!("{}:{v}", json::string(k))
            })
            .collect();
        format!("{{{}}}", parts.join(","))
    }
}

/// A JSON value as an argument: text and whole numbers as themselves, the rest as JSON.
fn json_arg(v: &json::Value) -> Arg {
    match v {
        json::Value::String(s) => Arg::Text(s.clone()),
        json::Value::Number(n) if n.fract() == 0.0 && n.abs() < 1e15 => Arg::Integer(*n as i64),
        other => Arg::Json(other.to_json()),
    }
}

/// A value Gemma 4 wrote (`format_argument`'s syntax), as JSON, and the text after it.
fn gemma_json(s: &str) -> Option<(String, &str)> {
    if let Some(rest) = s.strip_prefix(QUOTE) {
        let (text, after) = rest.split_once(QUOTE)?;
        return Some((json::string(text), after));
    }
    if let Some(mut rest) = s.strip_prefix('[') {
        let mut items = Vec::new();
        loop {
            if let Some(after) = rest.strip_prefix(']') {
                return Some((format!("[{}]", items.join(",")), after));
            }
            let (item, after) = gemma_json(rest)?;
            items.push(item);
            rest = after.strip_prefix(',').unwrap_or(after);
        }
    }
    if let Some(mut rest) = s.strip_prefix('{') {
        let mut fields = Vec::new();
        loop {
            if let Some(after) = rest.strip_prefix('}') {
                return Some((format!("{{{}}}", fields.join(",")), after));
            }
            let (key, after) = rest.split_once(':')?;
            let (value, after) = gemma_json(after)?;
            fields.push(format!("{}:{value}", json::string(key)));
            rest = after.strip_prefix(',').unwrap_or(after);
        }
    }
    // A number, true, false or null: up to the next punctuation.
    let end = s.find([',', '}', ']']).unwrap_or(s.len());
    let word = s[..end].trim();
    json::parse(word).ok()?;
    Some((word.to_string(), &s[end..]))
}

/// A call written under `call_grammar` for `tools`, read back. The grammar guarantees its
/// shape. (Muse's message to the user, `MUSE_TO_USER`, is no call: `None`.)
pub fn parse_call(family: Family, text: &str, tools: &[Tool]) -> Option<Call> {
    let kind_of = |tool: &str, key: &str| {
        tools
            .iter()
            .find(|t| t.name == tool)
            .and_then(|t| t.params.iter().find(|p| p.name == key))
            .map(|p| p.kind)
    };
    // A value written as is (text, or a whole number) or, for a list or object, as JSON.
    let plain_or_json = |kind: Option<Kind>, value: &str| -> Option<Arg> {
        Some(match kind {
            Some(Kind::Text) => Arg::Text(value.to_string()),
            Some(Kind::List(_) | Kind::Object(_) | Kind::Map | Kind::Cell) => {
                Arg::Json(json::parse(value).ok()?.to_json())
            }
            _ => value
                .parse::<i64>()
                .map_or_else(|_| Arg::Text(value.to_string()), Arg::Integer),
        })
    };
    match family {
        Family::Muse => {
            let (tool, rest) = text.strip_prefix(" to=")?.split_once("<|message|>")?;
            if tool == "user" || tool == "self" {
                return None;
            }
            let mut rest = rest
                .strip_prefix("<atem:function_calls>\n<atem:invoke name=\"")?
                .strip_prefix(tool)?
                .strip_prefix("\">\n")?;
            let mut args = Vec::new();
            while let Some(after) = rest.strip_prefix("<atem:parameter name=\"") {
                let (key, after) = after.split_once("\">")?;
                let (value, after) = after.split_once("</atem:parameter>\n")?;
                args.push((key.to_string(), plain_or_json(kind_of(tool, key), value)?));
                rest = after;
            }
            rest.strip_prefix("</atem:invoke>\n</atem:function_calls>")?;
            Some(Call {
                tool: tool.to_string(),
                args,
            })
        }
        Family::Llama => {
            let v = json::parse(text).ok()?;
            let tool = v.str("name")?.to_string();
            let mut args = Vec::new();
            if let Some(json::Value::Object(params)) = v.get("parameters") {
                for (k, v) in params {
                    args.push((k.clone(), json_arg(v)));
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
                args.push((key.to_string(), plain_or_json(kind_of(tool, key), value)?));
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
                    let (value, after) = gemma_json(after)?;
                    (json_arg(&json::parse(&value).ok()?), after)
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
/// only; `parse_call` reads it back. (Muse's, from just after `chat::opening`: its
/// recipient, then the call.)
pub fn call_text(family: Family, call: &Call) -> String {
    let start = call_start(family).token;
    let texts: Vec<(&str, &str)> = call
        .args
        .iter()
        .filter_map(|(k, v)| match v {
            Arg::Text(s) => Some((k.as_str(), s.as_str())),
            Arg::Integer(_) | Arg::Json(_) => None,
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
        Family::Muse => {
            let params: String = texts
                .iter()
                .map(|(k, v)| format!("<atem:parameter name=\"{k}\">{v}</atem:parameter>\n"))
                .collect();
            let tool = &call.tool;
            format!(
                " to={tool}<|message|><atem:function_calls>\n<atem:invoke name=\"{tool}\">\n{params}</atem:invoke>\n</atem:function_calls>"
            )
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

/// What ends a Gemma 4 call. After it the model writes another call, or the results
/// follow (`<|tool_response>`, which it would write to stop).
pub const GEMMA_CALL_CLOSE: &str = "<tool_call|>";

pub fn after_call(
    family: Family,
    tool: &str,
    result: &str,
    thinking: bool,
) -> (String, String, String) {
    match family {
        Family::Qwen35 => (
            "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n".into(),
            result.into(),
            // The turn that follows opens as `chat::opening` does: a thought to write, or
            // the empty one.
            if thinking {
                "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n".into()
            } else {
                "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into()
            },
        ),
        Family::Llama => (
            "<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n".into(),
            result.into(),
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        ),
        Family::Gemma4 => (
            format!("{GEMMA_CALL_CLOSE}<|tool_response>response:{tool}{{value:{QUOTE}"),
            result.into(),
            // Thinking on, the template opens a thought after a tool's result.
            if thinking {
                format!("{QUOTE}}}<tool_response|><|channel>thought\n")
            } else {
                format!("{QUOTE}}}<tool_response|>")
            },
        ),
        // The template ends a call's message with <|eot|> when the tool's turn follows,
        // and the assistant's next message names its recipient again.
        Family::Muse => (
            format!("<|eot|><|start|>tool {tool}<|message|><tool_output name=\"{tool}\">\n"),
            result.into(),
            "\n</tool_output><|eot|><|start|>assistant".into(),
        ),
    }
}

// ---- The server's own tools -------------------------------------------------------------

/// Runs one of the server's own tools, with the files the chat may reach, if any.
pub fn run_here(call: &Call, files: Option<&Files>) -> Result<String, String> {
    match (call.tool.as_str(), files) {
        // Several files in one call: each read as its own.
        ("read_file", Some(files)) if several_paths(call).is_some() => {
            let paths = several_paths(call).unwrap_or_default();
            Ok(read_several(&paths, |path| files.run(&read_one(path))))
        }
        ("get_current_datetime", _) => Ok(now()),
        ("calculate", _) => crate::calc::calculate_many(call.text("expression").unwrap_or("")),
        ("remember", _) => crate::memory::remember(call.text("fact").unwrap_or("")),
        ("forget", _) => crate::memory::forget(call.text("fact").unwrap_or("")),
        ("search_chats", _) => crate::chats::search(call.text("query").unwrap_or(""), None),
        ("list_scheduled_tasks", _) => Ok(crate::tasks::listing()),
        ("search_web", _) => crate::web::search(
            call.text("query").unwrap_or(""),
            call.text("domain").unwrap_or(""),
        ),
        ("fetch_url", _) => crate::web::fetch(call.text("url").unwrap_or("")),
        ("save_skill", _) => crate::skills::save(
            call.text("name").unwrap_or(""),
            call.text("description").unwrap_or(""),
            call.text("instructions").unwrap_or(""),
        ),
        ("update_skill", _) => crate::skills::update(
            call.text("name").unwrap_or(""),
            call.text("instructions").unwrap_or(""),
            call.text("description").unwrap_or(""),
        ),
        ("run_shortcut", _) => crate::shortcuts::run(
            call.text("name").unwrap_or(""),
            call.text("input").unwrap_or(""),
        ),
        (_, Some(files)) => files.run(call),
        (other, None) => Err(format!("{other} isn't one of this server's tools")),
    }
}

/// The most files one `read_file` call reads.
pub const MAX_FILES_A_CALL: usize = 10;

/// The paths of a `read_file` call that names several, one a line; `None` for one.
pub fn several_paths(call: &Call) -> Option<Vec<String>> {
    if call.tool != "read_file" {
        return None;
    }
    let paths: Vec<String> = call
        .text("path")?
        .lines()
        .map(|l| l.trim().trim_start_matches("- ").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    (paths.len() > 1).then_some(paths)
}

/// A `read_file` call for one path.
pub fn read_one(path: &str) -> Call {
    Call {
        tool: "read_file".into(),
        args: vec![("path".into(), Arg::Text(path.into()))],
    }
}

/// The most of one tool's result the model reads on a small context, in characters (about
/// 2,000 tokens), and on any context.
pub const MIN_RESULT: usize = 8000;
const MAX_ONE_RESULT: usize = 48_000;

/// What `result_chars` gives for the chats this server keeps (`set_context`).
static ONE_RESULT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(MIN_RESULT);

/// How much of one tool's result the model reads in a chat of `context` tokens, in
/// characters: three quarters of a character a token (a fifth to a third of the chat),
/// between `MIN_RESULT` and `MAX_ONE_RESULT`. It was 8,000 on every context, so of a
/// 20-page policy or a long sheet the model read the first four pages and answered from
/// those; 0.9 gave a result a third of the context.
pub fn result_chars(context: usize) -> usize {
    (context / 4 * 3).clamp(MIN_RESULT, MAX_ONE_RESULT)
}

/// Sizes results for chats of `context` tokens.
pub fn set_context(context: usize) {
    ONE_RESULT.store(result_chars(context), std::sync::atomic::Ordering::Relaxed);
}

/// The most of one result the model reads, in characters, on this server's context.
pub fn one_result() -> usize {
    ONE_RESULT.load(std::sync::atomic::Ordering::Relaxed)
}

/// How much of a tool's result the model reads, in characters: `one` for a call that
/// reads one thing, and for several files at least three times `MIN_RESULT` between them.
pub fn result_limit(call: &Call, one: usize) -> usize {
    several_paths(call).map_or(one, |paths| several_limit(paths.len(), one))
}

/// What `n` files read in one call may take between them.
fn several_limit(n: usize, one: usize) -> usize {
    one.max(MIN_RESULT * n.min(3))
}

/// Several files read by `read` in one result, each under its path, sharing what one
/// result may hold (`result_limit`) by water-filling so short files don't starve long ones:
/// every file shorter than the current equal share gets its full text, and the unused room
/// goes to the remaining longer files.
pub fn read_several(paths: &[String], read: impl Fn(&str) -> Result<String, String>) -> String {
    let read_now = &paths[..paths.len().min(MAX_FILES_A_CALL)];
    let texts: Vec<(String, usize)> = read_now
        .iter()
        .map(|path| {
            let text = read(path)
                .unwrap_or_else(|e| format!("Error: {e}"))
                .trim_end()
                .to_string();
            let chars = text.chars().count();
            (text, chars)
        })
        .collect();
    let total_budget = several_limit(read_now.len(), one_result())
        .saturating_sub(read_now.len() * 120);
    // Water-filling: give short files their exact size and redistribute leftover budget.
    let mut order: Vec<usize> = (0..texts.len()).collect();
    order.sort_by_key(|&i| texts[i].1);
    let mut limits = vec![0usize; texts.len()];
    let mut remaining = total_budget;
    for (pos, &idx) in order.iter().enumerate() {
        let left = texts.len() - pos;
        let fair = remaining / left;
        let take = texts[idx].1.min(fair);
        limits[idx] = take;
        remaining = remaining.saturating_sub(take);
    }
    let mut out = String::new();
    for ((path, (text, chars)), share) in read_now.iter().zip(&texts).zip(limits) {
        out.push_str(&format!("=== {path} ===\n"));
        if *chars > share {
            if share >= 800 {
                let head_n = share * 3 / 4;
                let tail_n = share - head_n;
                let head: String = text.chars().take(head_n).collect();
                let tail: String = text.chars().skip(*chars - tail_n).collect();
                out.push_str(&head);
                out.push_str(&format!(
                    "\n[... {} characters omitted from middle: read this file on its own for the rest ...]\n",
                    chars - share
                ));
                out.push_str(&tail);
            } else {
                out.extend(text.chars().take(share));
                out.push_str(&format!(
                    "\n[... {} more characters not shown: read this file on its own for the rest]",
                    chars - share
                ));
            }
        } else {
            out.push_str(text);
        }
        out.push_str("\n\n");
    }
    if paths.len() > read_now.len() {
        out.push_str(&format!(
            "[Only the first {MAX_FILES_A_CALL} files were read. Call read_file again for: {}]",
            paths[read_now.len()..].join(", ")
        ));
    }
    out.trim_end().to_string()
}

/// "Friday, November 13, 2026": today, for a request's note of the date.
pub fn today_line() -> String {
    let now = now();
    now.rsplit_once(", ").map_or(now.clone(), |(date, _)| date.to_string())
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
pub fn local_time() -> (u16, u16, u16, u16, u16, u16) {
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
pub fn local_time() -> (u16, u16, u16, u16, u16, u16) {
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
pub fn local_time() -> (u16, u16, u16, u16, u16, u16) {
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

    #[test]
    fn a_message_names_its_files_and_folders() {
        assert_eq!(
            named_paths("Process the invoices in Invoices/November following `Policies/AP-and-Coding-Guide.md`, then update (budget.xlsx)."),
            ["Invoices/November", "Policies/AP-and-Coding-Guide.md", "budget.xlsx"]
        );
        assert_eq!(named_paths("Using Events/Gala-2026, who hasn't replied? See Events/Gala-2026/."), ["Events/Gala-2026"]);
        // Not paths: words joined by a slash, dates, web addresses, mail, a way out.
        assert!(named_paths("Yes and/or no, 24/7, on 11/15/2026 at https://example.com/a/b.pdf, from a@b.example, ../x.md, e.g. v2.").is_empty());
        assert_eq!(named_paths("Read /Users/sam/Work/notes.md"), ["/Users/sam/Work/notes.md"]);
    }

    #[test]
    fn a_long_requests_middle_can_ask_for_a_change() {
        assert_eq!(unscored("Short one."), None);
        let long = format!("{}5. Append a row to the tracker.{}", "a".repeat(SCORED_START), "b".repeat(SCORED_END));
        let middle = unscored(&long).unwrap();
        assert_eq!(middle, "5. Append a row to the tracker.");
        assert!(!for_scoring(&long).contains("Append"));
        assert!(names_a_change(&middle));
        assert!(names_a_change("then call write_spreadsheet with the totals"));
        assert!(!names_a_change("Organization: Fire Brigade. Amount: CLP 4,000,000. Purpose: breathing apparatus."));
    }

    #[test]
    fn a_result_is_sized_to_the_context() {
        // A small context reads what it always did; a large one more, up to a limit.
        assert_eq!(result_chars(4096), 8000);
        assert_eq!(result_chars(8192), 8000);
        assert_eq!(result_chars(32768), 24576);
        assert_eq!(result_chars(262_144), 48_000);
        // Several files share at least what they did (three results' worth from three up),
        // and never less than one result on a large context.
        assert_eq!(several_limit(2, 8000), 16_000);
        assert_eq!(several_limit(6, 8000), 24_000);
        assert_eq!(several_limit(6, 24_576), 24_576);
        assert_eq!(several_limit(2, 48_000), 48_000);
    }

    #[test]
    fn a_long_request_is_scored_by_its_start_and_end() {
        let short = "Read notes.txt\nand sum it up.";
        assert_eq!(for_scoring(short), "Read notes.txt and sum it up.");
        let exact = "x".repeat(600);
        assert_eq!(for_scoring(&exact), exact);
        let long = format!("Here are my notes. {} Save them to notes.md.", "word ".repeat(400));
        let scored = for_scoring(&long);
        assert!(scored.starts_with("Here are my notes."));
        assert!(scored.ends_with("Save them to notes.md."));
        assert!(scored.contains(" ... "));
        assert!(scored.chars().count() <= 605);
        assert!(change_question(&long).ends_with("Save them to notes.md.\nChanges:"));
        // Characters, not bytes: nothing is cut inside one.
        let wide = "é".repeat(700);
        assert_eq!(for_scoring(&wide).chars().filter(|c| *c == 'é').count(), 600);
    }
    use syrinx_engine::grammar::Grammar;

    fn written(family: Family, text: &str) -> Option<Call> {
        let tools = offered(&["read_file".into(), "search_files".into()], false);
        let g = Grammar::new(&call_grammar(family, &tools));
        let mut m = g.matcher();
        assert!(
            m.advance_all(text.as_bytes()) && m.is_complete(),
            "{family:?}: {text:?}"
        );
        parse_call(family, text, &tools)
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
            written(
                Family::Muse,
                " to=search_files<|message|><atem:function_calls>\n<atem:invoke name=\"search_files\">\n<atem:parameter name=\"pattern\">tax \"2026\"</atem:parameter>\n<atem:parameter name=\"directory\">C:\\Users\\me</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
            ),
            Some(want.clone())
        );
        assert_eq!(
            written(
                Family::Muse,
                " to=get_current_datetime<|message|><atem:function_calls>\n<atem:invoke name=\"get_current_datetime\">\n</atem:invoke>\n</atem:function_calls>"
            ),
            Some(Call {
                tool: "get_current_datetime".into(),
                args: vec![]
            })
        );
        // Muse's message to the user, or to itself, is a choice of the same grammar, and no
        // call; without thinking, it may not write to itself.
        assert_eq!(written(Family::Muse, MUSE_TO_USER), None);
        assert_eq!(written(Family::Muse, MUSE_TO_SELF), None);
        let tools = offered(&["read_file".into()], false);
        let g = Grammar::new(&muse_call_grammar(&tools, false));
        assert!(!g.matcher().advance_all(MUSE_TO_SELF.as_bytes()));
        assert!(g.matcher().advance_all(MUSE_TO_USER.as_bytes()));
        assert_eq!(
            written(Family::Gemma4, "call:get_current_datetime{}"),
            Some(Call {
                tool: "get_current_datetime".into(),
                args: vec![]
            })
        );
    }

    fn written_with(family: Family, names: &[&str], text: &str) -> Option<Call> {
        let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        let tools = offered(&names, false);
        let g = Grammar::new(&call_grammar(family, &tools));
        let mut m = g.matcher();
        assert!(
            m.advance_all(text.as_bytes()) && m.is_complete(),
            "{family:?}: {text:?}"
        );
        parse_call(family, text, &tools)
    }

    #[test]
    fn lists_and_objects_in_each_familys_form() {
        let rows = r#"[["INV-1001",1234.5,null],["x",true]]"#;
        let spreadsheet = |family, text: &str| {
            let call = written_with(family, &["write_spreadsheet"], text).unwrap();
            assert_eq!(call.tool, "write_spreadsheet");
            assert_eq!(call.text("start_cell"), Some("append"), "{family:?}");
            assert_eq!(call.json("rows"), Some(rows), "{family:?}");
            assert_eq!(
                approval_preview(&call, |_| true),
                "Add 2 rows to Invoices/R.xlsx"
            );
        };
        spreadsheet(
            Family::Qwen35,
            "\n<function=write_spreadsheet>\n<parameter=path>\nInvoices/R.xlsx\n</parameter>\n<parameter=start_cell>\nappend\n</parameter>\n<parameter=rows>\n[[\"INV-1001\", 1234.5, null], [\"x\", true]]\n</parameter>\n</function>\n",
        );
        spreadsheet(
            Family::Gemma4,
            "call:write_spreadsheet{path:<|\"|>Invoices/R.xlsx<|\"|>,rows:[[<|\"|>INV-1001<|\"|>,1234.5,null],[<|\"|>x<|\"|>,true]],start_cell:<|\"|>append<|\"|>}",
        );
        spreadsheet(
            Family::Llama,
            r#"{"name": "write_spreadsheet", "parameters": {"path": "Invoices/R.xlsx", "start_cell": "append", "rows": [["INV-1001", 1234.5, null], ["x", true]]}}"#,
        );
        spreadsheet(
            Family::Muse,
            " to=write_spreadsheet<|message|><atem:function_calls>\n<atem:invoke name=\"write_spreadsheet\">\n<atem:parameter name=\"path\">Invoices/R.xlsx</atem:parameter>\n<atem:parameter name=\"start_cell\">append</atem:parameter>\n<atem:parameter name=\"rows\">[[\"INV-1001\", 1234.5, null], [\"x\", true]]</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
        );

        // A template's values: keys the model names.
        let values = r#"{"amount":"4,200.00","client name":"Riverside"}"#;
        for (family, text) in [
            (
                Family::Gemma4,
                "call:fill_docx_template{path:<|\"|>out.docx<|\"|>,template:<|\"|>T.docx<|\"|>,values:{amount:<|\"|>4,200.00<|\"|>,client name:<|\"|>Riverside<|\"|>}}",
            ),
            (
                Family::Qwen35,
                "\n<function=fill_docx_template>\n<parameter=template>\nT.docx\n</parameter>\n<parameter=path>\nout.docx\n</parameter>\n<parameter=values>\n{\"amount\": \"4,200.00\", \"client name\": \"Riverside\"}\n</parameter>\n</function>\n",
            ),
        ] {
            let call = written_with(family, &["fill_docx_template"], text).unwrap();
            assert_eq!(call.json("values"), Some(values), "{family:?}");
            assert_eq!(call.text("template"), Some("T.docx"));
        }

        // Slides: objects in a list, their keys in alphabetical order.
        let slides = r#"[{"bullets":["a","b"],"title":"One"}]"#;
        for (family, text) in [
            (
                Family::Gemma4,
                "call:create_pptx{path:<|\"|>d.pptx<|\"|>,slides:[{bullets:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],title:<|\"|>One<|\"|>}],title:<|\"|>Deck<|\"|>}",
            ),
            (
                Family::Qwen35,
                "\n<function=create_pptx>\n<parameter=path>\nd.pptx\n</parameter>\n<parameter=title>\nDeck\n</parameter>\n<parameter=slides>\n[{\"bullets\": [\"a\", \"b\"], \"title\": \"One\"}]\n</parameter>\n</function>\n",
            ),
        ] {
            let call = written_with(family, &["create_pptx"], text).unwrap();
            assert_eq!(call.json("slides"), Some(slides), "{family:?}");
        }
        // A slide without its title isn't a call the grammar lets through.
        let tools = offered(&["create_pptx".into()], false);
        let g = Grammar::new(&call_grammar(Family::Gemma4, &tools));
        assert!(!g.matcher().advance_all(
            "call:create_pptx{path:<|\"|>d.pptx<|\"|>,slides:[{bullets:[]}".as_bytes()
        ));
    }

    #[test]
    fn structured_arguments_are_declared_as_the_templates_do() {
        let tools = offered(
            &[
                "write_spreadsheet".into(),
                "fill_docx_template".into(),
                "create_pptx".into(),
            ],
            false,
        );
        let gemma = declarations(Family::Gemma4, &tools);
        assert!(gemma.contains(
            ",items:{items:{},type:<|\"|>ARRAY<|\"|>},type:<|\"|>ARRAY<|\"|>},start_cell:"
        ));
        assert!(gemma.contains(
            "items:{properties:{bullets:{description:<|\"|>The slide's bullet points, each a short line.<|\"|>,items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>},title:{description:<|\"|>The slide's title.<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>title<|\"|>,<|\"|>bullets<|\"|>],type:<|\"|>OBJECT<|\"|>},type:<|\"|>ARRAY<|\"|>}"
        ));
        assert!(gemma.contains(",properties:{},type:<|\"|>OBJECT<|\"|>}},required:"));
        let qwen = declarations(Family::Qwen35, &tools);
        assert!(qwen.contains(
            "\"rows\": {\"type\": \"array\", \"description\": \"The rows to write, each a list of cell values (text, numbers, true/false, or null to skip a cell).\", \"items\": {\"type\": \"array\", \"items\": {}}}"
        ));
        assert!(qwen.contains("\"values\": {\"type\": \"object\", \"description\":"));
        assert!(qwen.contains("\"required\": [\"title\", \"bullets\"]}}"));
    }

    #[test]
    fn memory_is_offered_without_asking_for_a_change() {
        let tools = offered(&[], false);
        let remember = tools.iter().find(|t| t.name == "remember").unwrap();
        assert!(remember.asks.is_some() && !remember.changes_files());
        assert_eq!(remember.runs, Runs::Server);
        assert!(tools.iter().all(|t| t.name != "search_chats"), "the app runs it");
        let app = offered(&["search_chats".into()], false);
        assert!(app.iter().any(|t| t.name == "search_chats" && t.runs == Runs::App));
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
        let muse = declarations(Family::Muse, &tools);
        assert!(muse.starts_with(
            "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
        ));
        assert!(muse.contains(
            "Here are the functions available in JSONSchema format:\n// Tool metadata\n{\"name\": \"read_file\", \"description\": \"\"}\n"
        ));
        assert!(muse.contains(
            "// Function schemas\n{\"name\": \"read_file\", \"description\": \"Read a text file"
        ));
        assert!(muse.contains("\"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\""));
        assert!(muse.contains(
            "\n{\"name\": \"get_current_datetime\", \"description\": \"Get the current date, time and day of the week.\", \"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}\n"
        ));
        assert!(muse.contains(
            "multiple\" lines\n</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n\n# Valid recipients: \"self\", \"read_file.*\", \"get_current_datetime.*\""
        ));
        assert!(muse.contains("}\n\nHere's an example of how to call a function"));
        assert!(muse.ends_with(", \"user\"."), "{muse}");
        assert_eq!(declarations(Family::Muse, &[]), "");
        assert_eq!(
            tools.len(),
            7,
            "read_file, and the server's own date and time, calculate, remember, forget, save_skill and update_skill"
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
        // Muse's, after the turn's opening: its recipient, then the call.
        let text = call_text(Family::Muse, &call);
        assert_eq!(
            text,
            " to=use_skill<|message|><atem:function_calls>\n<atem:invoke name=\"use_skill\">\n<atem:parameter name=\"name\">code-review</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
        );
        assert_eq!(parse_call(Family::Muse, &text, &[]), Some(call.clone()));
        for family in [Family::Qwen35, Family::Llama, Family::Gemma4] {
            let text = call_text(family, &call);
            let start = call_start(family).token;
            assert!(text.starts_with(start), "{text}");
            assert_eq!(
                parse_call(family, &text[start.len()..], &[]),
                Some(call.clone()),
                "{text}"
            );
        }
    }

    #[test]
    fn muses_results_come_back_as_its_template_writes_them() {
        let (before, body, after) = after_call(Family::Muse, "read_file", "hello", false);
        assert_eq!(
            before,
            "<|eot|><|start|>tool read_file<|message|><tool_output name=\"read_file\">\n"
        );
        assert_eq!(body, "hello");
        // The next message's recipient is the model's to choose again.
        assert_eq!(
            after,
            format!(
                "\n</tool_output><|eot|>{}",
                crate::chat::opening(Family::Muse, false)
            )
        );
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
                .all(|t| !t.changes_files())
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
        for family in [Family::Qwen35, Family::Llama, Family::Gemma4, Family::Muse] {
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
        // has: on the Mac the four changes too, but not the app's document and project
        // tools.
        let here = offered(&[], true);
        let mut expected = vec![
            "read_file",
            "list_directory",
            "search_files",
            "file_exists",
            "git_status",
            "git_diff",
            "get_current_datetime",
            "calculate",
            "read_spreadsheet",
        ];
        if cfg!(target_os = "macos") {
            expected.extend([
                "create_docx",
                "write_spreadsheet",
                "fill_docx_template",
                "create_pptx",
                "write_file",
                "rename_file",
                "move_file",
            ]);
        }
        expected.extend(["remember", "forget"]);
        if cfg!(target_os = "macos") {
            expected.extend(["delete_file", "edit_file", "create_pdf"]);
        }
        expected.extend(["save_skill", "update_skill"]);
        assert_eq!(here.iter().map(|t| t.name).collect::<Vec<_>>(), expected);
        assert!(here.iter().all(|t| t.runs == Runs::Server));
        assert_eq!(
            here.iter().any(|t| t.changes_files()),
            cfg!(target_os = "macos")
        );
        if cfg!(target_os = "macos") {
            let delete = here.iter().find(|t| t.name == "delete_file").unwrap();
            assert!(delete.description.contains("Trash"));
        }
        // An app that runs a change tool is still asked to, with access or without.
        let both = offered(&["write_file".into()], true);
        assert_eq!(runs(&both, "write_file"), Some(Runs::App));
        // Without access, only the server's own.
        let none = offered(&[], false);
        assert_eq!(
            none.iter().map(|t| t.name).collect::<Vec<_>>(),
            ["get_current_datetime", "calculate", "remember", "forget", "save_skill", "update_skill"]
        );
        // The web, shell commands and shortcuts only when the app turns them on.
        let names = |tools: &[Tool]| tools.iter().map(|t| t.name).collect::<Vec<_>>();
        for name in ["search_web", "fetch_url", "run_shell_command", "run_shortcut", "list_scheduled_tasks"] {
            assert!(!names(&here).contains(&name), "{name}");
        }
        let on = offered(
            &[WEB.into(), SHELL.into(), SHORTCUTS.into(), RUNS_TASKS.into()],
            true,
        );
        let on = names(&on);
        for name in ["search_web", "fetch_url", "list_scheduled_tasks", "schedule_task"] {
            assert!(on.contains(&name), "{name}");
        }
        assert_eq!(on.contains(&"run_shell_command"), cfg!(target_os = "macos"));
        assert_eq!(on.contains(&"run_shortcut"), cfg!(target_os = "macos"));
        // Calendar and Mac tools come only from an app that runs them.
        assert!(!on.contains(&"send_message") && !on.contains(&"move_event"));
        let app = offered(&["send_message".into(), "move_event".into()], false);
        assert_eq!(runs(&app, "send_message"), Some(Runs::App));
        assert_eq!(runs(&app, "move_event"), Some(Runs::App));
    }

    #[test]
    fn the_preamble_says_how_to_work_and_each_family_its_own() {
        assert!(!PREAMBLE.contains("  "), "no runs of spaces");
        for rule in [
            "Use calculate for every sum",
            "every file in the folder",
            "pending approval",
            "give the total",
        ] {
            assert!(PREAMBLE.contains(rule), "{rule}");
        }
        let gemma = family_notes(Family::Gemma4, true).unwrap();
        assert!(
            gemma.starts_with(PARALLEL_CALLS) && gemma.contains("Gemma's own tool-call format")
        );
        let qwen = family_notes(Family::Qwen35, true).unwrap();
        assert!(qwen.contains("after your thinking ends"));
        // Thinking off: nothing about thinking, the rest the same.
        let plain = family_notes(Family::Qwen35, false).unwrap();
        assert!(!plain.contains("thinking"));
        assert_eq!(plain, qwen.replace(", after your thinking ends", ""));
        assert!(qwen.contains("<tool_call>") && qwen.contains("Keep the final reply short"));
        assert!(!qwen.contains(PARALLEL_CALLS));
        assert_eq!(family_notes(Family::Muse, true), None);
        assert_eq!(family_notes(Family::Llama, false), None);
    }

    #[test]
    fn named_paths_keeps_quoted_paths_with_spaces() {
        let msg = r#"Check "Invoices/Nov 2026/AP Policy.md" and 'Q3 Budget.xlsx' against notes.txt."#;
        let found = named_paths(msg);
        assert_eq!(
            found,
            vec!["Invoices/Nov 2026/AP Policy.md", "Q3 Budget.xlsx", "notes.txt"]
        );
    }

    #[test]
    fn read_several_water_fills_so_short_files_dont_starve_long_ones() {
        let paths: Vec<String> = (0..6).map(|i| format!("file-{i}.txt")).collect();
        let long_body = format!("START\n{}\nEND_TOTAL=999", "x".repeat(8_000));
        let out = read_several(&paths, |p| {
            if p == "file-5.txt" {
                Ok(long_body.clone())
            } else {
                Ok("short receipt: $10".to_string())
            }
        });
        assert!(out.contains("END_TOTAL=999"), "long file kept whole: len={}", out.len());
        assert!(!out.contains("omitted"), "no unnecessary truncation");
    }
}
