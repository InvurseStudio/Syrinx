//! The read-only tools this server runs on the Mac it's on: reading, listing and searching
//! files, and Git's status and diff. The app and the server share the Mac, so the server
//! reaches the files itself rather than asking the app (`client_tool_request`, which stays
//! for an app on another PC). The answers are worded as the Windows app's `PcTools` and
//! the eval's stand-in PC (`vireo-eval`'s `pc.rs`) word theirs, so the model reads them the
//! same way. Nothing here changes a file: tools that do will come with approvals.
//!
//! What a chat may reach is its access level, sent with each chat: the home folder without
//! `~/Library`, the folders the user chose (sent with it), or the whole Mac.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::json;
use crate::tools::{Arg, Call};

pub const MAX_READ_BYTES: u64 = 1024 * 1024;
pub const MAX_SEARCHED_FILE_BYTES: u64 = 500 * 1024;
pub const MAX_SEARCH_RESULTS: usize = 50;
pub const MAX_SEARCH_DEPTH: usize = 8;
pub const MAX_LISTING_DEPTH: usize = 3;
pub const MAX_LISTED_ENTRIES: usize = 400;
pub const SEARCH_TIME_LIMIT: Duration = Duration::from_secs(15);
pub const GIT_TIMEOUT: Duration = Duration::from_secs(30);
/// Dependency and package folders, which a search or a deep listing walks past.
const SKIPPED_FOLDERS: [&str; 4] = ["node_modules", "venv", "__pycache__", "site-packages"];
/// Read with the Mac's document readers rather than as text.
const RICH_TEXT_EXTENSIONS: [&str; 5] = ["docx", "doc", "rtf", "rtfd", "odt"];
/// Documents these tools can't read yet.
const UNREAD_DOCUMENTS: [&str; 6] = ["xlsx", "xlsm", "pptx", "pages", "numbers", "key"];

/// How much of the Mac a chat may reach (the app's `AgentAccess`).
#[derive(Clone, Debug, PartialEq)]
pub enum Access {
    /// The user's folder, without `~/Library`.
    Home,
    /// Only the folders the user chose.
    Folders(Vec<String>),
    /// Anywhere.
    Full,
}

impl Access {
    /// A chat request's `access` and `folders`; `None` when the chat reaches no files.
    pub fn from_request(access: Option<&str>, folders: Vec<String>) -> Option<Access> {
        match access? {
            "home" => Some(Access::Home),
            "folders" => Some(Access::Folders(folders)),
            "full" => Some(Access::Full),
            _ => None,
        }
    }
}

/// The tools, for one chat.
#[derive(Clone, Debug)]
pub struct Files {
    /// The user's folder, where relative paths start.
    home: String,
    access: Access,
}

impl Files {
    pub fn new(home: &str, access: Access) -> Files {
        let access = match access {
            Access::Folders(f) => Access::Folders(f.iter().map(|f| standardized(f)).collect()),
            a => a,
        };
        Files {
            home: standardized(home),
            access,
        }
    }

    /// For the user this server runs as.
    pub fn for_user(access: Access) -> Files {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
        Files::new(&home, access)
    }

    /// `~/Library` holds other apps' data and sign-ins, so home access leaves it out (as
    /// the Windows app leaves out AppData).
    fn library(&self) -> String {
        format!("{}/Library", self.home)
    }

    /// Runs one of the tools: its answer, or what went wrong, worded for the model.
    pub fn run(&self, call: &Call) -> Result<String, String> {
        match call.tool.as_str() {
            "read_file" => self.read_file(&arg(call, "path")?),
            "list_directory" => self.list_directory(&arg(call, "path")?, int_arg(call, "depth")),
            "file_exists" => self.file_exists(&arg(call, "path")?),
            "search_files" => self.search_files(
                &arg(call, "pattern")?,
                &arg(call, "directory")?,
                optional_arg(call, "file_ext"),
            ),
            "git_status" => self.git_status(&arg(call, "path")?),
            "git_diff" => self.git_diff(&arg(call, "path")?),
            other => Err(format!("{other} can't run on this Mac.")),
        }
    }

    // ---- Paths ---------------------------------------------------------------------------

    /// A path this chat may reach. Relative ones ("Desktop/notes.txt") start in the user's
    /// folder; `~` is the user's folder.
    pub fn resolve(&self, path: &str) -> Result<String, String> {
        let mut expanded = path.trim();
        if expanded.len() >= 2 && expanded.starts_with('"') && expanded.ends_with('"') {
            expanded = &expanded[1..expanded.len() - 1];
        }
        if expanded == "~" {
            expanded = "";
        } else if let Some(rest) = expanded.strip_prefix("~/") {
            expanded = rest;
        }
        let full = standardized(&if expanded.starts_with('/') {
            expanded.to_string()
        } else if expanded.is_empty() {
            self.home.clone()
        } else {
            format!("{}/{expanded}", self.home)
        });
        match &self.access {
            Access::Full => Ok(full),
            Access::Folders(folders) => {
                if folders.iter().any(|f| is_within(&full, f)) {
                    return Ok(full);
                }
                let allowed = if folders.is_empty() {
                    "none".to_string()
                } else {
                    folders.join(", ")
                };
                Err(format!(
                    "{full} is outside the folders this chat may access ({allowed}). The user can change access in the menu above the chat box, or add the folder in Settings."
                ))
            }
            Access::Home => {
                if !is_within(&full, &self.home) {
                    return Err(format!(
                        "{full} is outside the user's folder ({}), which is the only one these tools can reach. The user can change access in the menu above the chat box.",
                        self.home
                    ));
                }
                if is_within(&full, &self.library()) {
                    return Err(format!(
                        "{full} is in ~/Library, which these tools can't reach."
                    ));
                }
                Ok(full)
            }
        }
    }

    fn reachable(&self, path: &str) -> bool {
        match &self.access {
            Access::Full => true,
            Access::Folders(folders) => folders.iter().any(|f| is_within(path, f)),
            Access::Home => is_within(path, &self.home) && !is_within(path, &self.library()),
        }
    }

    // ---- Files ---------------------------------------------------------------------------

    fn read_file(&self, path: &str) -> Result<String, String> {
        let file = self.resolve(path)?;
        let Ok(meta) = fs::metadata(&file) else {
            return Err(format!("File not found: {file}"));
        };
        if meta.is_dir() {
            return Err(format!(
                "{file} is a folder. Use list_directory to see what's in it."
            ));
        }
        let name = file_name(&file);
        let ext = Path::new(&file)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if ext == "pdf" {
            return match documents::pdf_text(&file) {
                Some(text) if !text.trim().is_empty() => Ok(text),
                _ => Err(format!(
                    "Couldn't read the text of {name}. It may be scanned pages without text."
                )),
            };
        }
        if RICH_TEXT_EXTENSIONS.contains(&ext.as_str()) {
            return documents::rich_text(&file).ok_or(format!("Couldn't read {name}."));
        }
        if UNREAD_DOCUMENTS.contains(&ext.as_str()) {
            return Err(format!(
                "{name} is a document this Mac's tools can't read yet. read_file can read text files, PDFs and Word documents."
            ));
        }
        let size = meta.len();
        if size > MAX_READ_BYTES {
            return Err(format!(
                "File too large ({} KB). Maximum is 1 MB.",
                size / 1024
            ));
        }
        let data = fs::read(&file).map_err(|_| format!("Couldn't read {file}."))?;
        let text = String::from_utf8(data).map_err(|_| {
            format!(
                "{name} is not a text file. read_file can read text files, PDFs and Word documents."
            )
        })?;
        Ok(text
            .strip_prefix('\u{FEFF}')
            .map(str::to_string)
            .unwrap_or(text))
    }

    /// A folder's contents, and with `depth` 2 or 3 the folders inside it, indented.
    fn list_directory(&self, path: &str, depth: Option<i64>) -> Result<String, String> {
        let folder = self.resolve(path)?;
        if !is_dir(&folder) {
            return Err(format!("Folder not found: {folder}"));
        }
        let levels = depth.unwrap_or(1).clamp(1, MAX_LISTING_DEPTH as i64) as usize;
        let mut lines = Vec::new();
        let mut cut = false;
        self.walk(&folder, 0, levels, &mut lines, &mut cut)?;
        if lines.is_empty() {
            return Ok("(empty directory)".into());
        }
        // The full path first, so the next call can use it as it is.
        let mut text = if levels > 1 {
            format!("{folder}:\n")
        } else {
            String::new()
        };
        text.push_str(&lines.join("\n"));
        if cut {
            text.push_str(&format!(
                "\n[... the first {MAX_LISTED_ENTRIES} entries; list a subfolder for the rest ...]"
            ));
        }
        Ok(text)
    }

    fn walk(
        &self,
        current: &str,
        level: usize,
        levels: usize,
        lines: &mut Vec<String>,
        cut: &mut bool,
    ) -> Result<(), String> {
        let pad = " ".repeat(2 * level);
        let names = match names_in(current) {
            Ok(names) => names,
            Err(e) if level == 0 => return Err(format!("Couldn't list {current}: {e}")),
            Err(e) => {
                lines.push(format!("{pad}(could not be listed: {e})"));
                return Ok(());
            }
        };
        let mut entries: Vec<(String, bool)> = names
            .into_iter()
            .map(|name| {
                let folder = is_dir(&join(current, &name));
                (name, folder)
            })
            .collect();
        // Folders first, then files, each by name, as the Windows app lists them.
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (name, folder) in entries {
            // Below the folder asked for, hidden files and packages are noise.
            if level > 0 && (name.starts_with('.') || SKIPPED_FOLDERS.contains(&name.as_str())) {
                continue;
            }
            if level == 0 && name == ".DS_Store" {
                continue;
            }
            if lines.len() >= MAX_LISTED_ENTRIES {
                *cut = true;
                return Ok(());
            }
            let kind = if folder { "[dir] " } else { "[file]" };
            lines.push(format!("{pad}{kind} {name}"));
            let full = join(current, &name);
            if folder && level + 1 < levels && !is_symlink(&full) && self.reachable(&full) {
                self.walk(&full, level + 1, levels, lines, cut)?;
                if *cut {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn file_exists(&self, path: &str) -> Result<String, String> {
        let full = self.resolve(path)?;
        let (exists, kind) = match fs::metadata(&full) {
            Ok(m) if m.is_dir() => (true, json::string("directory")),
            Ok(_) => (true, json::string("file")),
            Err(_) => (false, "null".to_string()),
        };
        Ok(format!(
            "{{\"exists\":{exists},\"path\":{},\"type\":{kind}}}",
            json::string(&full)
        ))
    }

    fn search_files(
        &self,
        pattern: &str,
        directory: &str,
        file_ext: Option<String>,
    ) -> Result<String, String> {
        let root = self.resolve(directory)?;
        if !is_dir(&root) {
            return Err(format!("Folder not found: {root}"));
        }
        // Every word of the search in a file or folder name counts ("STEM camp" finds
        // Northern-Lakes-STEM-Girls-Camp.docx).
        let words = words(&pattern.to_lowercase());
        // A file's name is looked for by name only: reading every file on the way is what
        // made a search of the home folder take minutes.
        let names_only = is_file_name(pattern.trim());
        let needle = pattern.to_lowercase();
        let deadline = Instant::now() + SEARCH_TIME_LIMIT;
        let mut timed_out = false;
        let mut results: Vec<String> = Vec::new();
        let ext = file_ext.map(|e| e.to_lowercase()).filter(|e| !e.is_empty());

        // Breadth first, so files near the starting folder are found before deep ones.
        let mut queue: Vec<(String, usize)> = vec![(root.clone(), 0)];
        let mut next = 0;
        'search: while next < queue.len() && results.len() < MAX_SEARCH_RESULTS {
            if Instant::now() > deadline {
                timed_out = true;
                break;
            }
            let (folder, depth) = queue[next].clone();
            next += 1;
            let Ok(mut names) = names_in(&folder) else {
                continue;
            };
            names.sort();
            for name in names {
                if results.len() >= MAX_SEARCH_RESULTS {
                    break 'search;
                }
                if Instant::now() > deadline {
                    timed_out = true;
                    break 'search;
                }
                let full = join(&folder, &name);
                let Ok(meta) = fs::metadata(&full) else {
                    continue;
                };
                if meta.is_dir() {
                    let skipped = name.starts_with('.')
                        || SKIPPED_FOLDERS.contains(&name.as_str())
                        || is_symlink(&full)
                        || !self.reachable(&full);
                    if skipped {
                        continue;
                    }
                    if name_matches(&name, &words) {
                        results.push(format!("{full}/ (folder name matches)"));
                    }
                    if depth < MAX_SEARCH_DEPTH {
                        queue.push((full, depth + 1));
                    }
                } else {
                    if let Some(ext) = &ext
                        && !name.to_lowercase().ends_with(ext.as_str())
                    {
                        continue;
                    }
                    if name_matches(&name, &words) {
                        results.push(format!("{full} (file name matches)"));
                        continue;
                    }
                    if names_only || meta.len() > MAX_SEARCHED_FILE_BYTES {
                        continue;
                    }
                    let Ok(text) = fs::read(&full).map(String::from_utf8) else {
                        continue;
                    };
                    let Ok(text) = text else {
                        continue;
                    };
                    for (number, line) in text.split('\n').enumerate() {
                        if results.len() >= MAX_SEARCH_RESULTS {
                            break;
                        }
                        if line.to_lowercase().contains(&needle) {
                            results.push(format!("{full}:{}: {}", number + 1, line.trim()));
                        }
                    }
                }
            }
        }

        let seconds = SEARCH_TIME_LIMIT.as_secs();
        if results.is_empty() {
            let more = if timed_out {
                format!(" in the first {seconds} s. Search a smaller folder.")
            } else {
                String::new()
            };
            return Ok(format!(
                "No matches found for \"{pattern}\" in {root}{more}"
            ));
        }
        let header = if results.len() >= MAX_SEARCH_RESULTS {
            format!("First {MAX_SEARCH_RESULTS} matches")
        } else {
            format!(
                "{} match{}",
                results.len(),
                if results.len() == 1 { "" } else { "es" }
            )
        };
        let note = if timed_out {
            format!(
                "\n\nStopped after {seconds} s before searching everything under {root}. Search a smaller folder for the rest."
            )
        } else {
            String::new()
        };
        Ok(format!(
            "{header} for \"{pattern}\":\n\n{}{note}",
            results.join("\n")
        ))
    }

    // ---- Git -----------------------------------------------------------------------------

    fn git_status(&self, path: &str) -> Result<String, String> {
        let root = self.git_root(&self.project_folder(path)?)?;
        let (code, output) = git(&["-C", &root, "status", "--short"], &root)?;
        let status = output.trim();
        Ok(format!(
            "{{\"ok\":{},\"path\":{},\"status\":{}}}",
            code == 0,
            json::string(&root),
            json::string(if status.is_empty() { "clean" } else { status })
        ))
    }

    fn git_diff(&self, path: &str) -> Result<String, String> {
        let full = self.resolve(path)?;
        let folder = if is_dir(&full) {
            full.clone()
        } else {
            parent(&full)
        };
        let root = self.git_root(&folder)?;
        let relative = if full.len() > root.len() {
            full.get(root.len() + 1..).unwrap_or(".")
        } else {
            "."
        };
        let (code, output) = git(&["-C", &root, "diff", "--", relative], &root)?;
        if code != 0 {
            let error = output.trim();
            return Err(if error.is_empty() {
                "Git diff failed".into()
            } else {
                error.into()
            });
        }
        Ok(if output.is_empty() {
            format!("No tracked Git diff for {full}")
        } else {
            output
        })
    }

    fn project_folder(&self, path: &str) -> Result<String, String> {
        let full = self.resolve(path)?;
        match fs::metadata(&full) {
            Ok(m) if m.is_dir() => Ok(full),
            Ok(_) => Ok(parent(&full)),
            Err(_) => Err(format!("Project directory not found: {full}")),
        }
    }

    fn git_root(&self, folder: &str) -> Result<String, String> {
        let (code, output) = git(&["-C", folder, "rev-parse", "--show-toplevel"], folder)?;
        if code != 0 {
            return Err(format!("No Git repository found for {folder}"));
        }
        // A repository that starts above the folders this chat may reach would show files
        // outside them.
        self.resolve(output.trim())
    }
}

/// Runs Git with its output and errors together, giving up after `GIT_TIMEOUT`.
pub fn git(arguments: &[&str], folder: &str) -> Result<(i32, String), String> {
    let Some(git) = [
        "/usr/bin/git",
        "/opt/homebrew/bin/git",
        "/usr/local/bin/git",
    ]
    .into_iter()
    .find(|p| Path::new(p).is_file()) else {
        return Err(
            "Git isn't installed on this Mac. Installing Xcode's command line tools adds it."
                .into(),
        );
    };
    let (mut reader, writer) = std::io::pipe().map_err(|e| format!("Couldn't run Git: {e}"))?;
    let errors = writer
        .try_clone()
        .map_err(|e| format!("Couldn't run Git: {e}"))?;
    // No pager, no prompts for credentials, and messages in English for the model.
    let mut child = Command::new(git)
        .args(arguments)
        .current_dir(folder)
        .env("GIT_PAGER", "cat")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(writer)
        .stderr(errors)
        .spawn()
        .map_err(|e| format!("Couldn't run Git: {e}"))?;
    // The pipe's write ends now belong to Git alone, so reading ends when it does.
    let collected = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = reader.read_to_end(&mut bytes);
        bytes
    });
    let deadline = Instant::now() + GIT_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "Git didn't finish within {} s.",
                    GIT_TIMEOUT.as_secs()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => return Err(format!("Couldn't run Git: {e}")),
        }
    };
    let bytes = collected.join().unwrap_or_default();
    Ok((
        status.code().unwrap_or(-1),
        String::from_utf8_lossy(&bytes).into_owned(),
    ))
}

// ---- Helpers ----------------------------------------------------------------------------

/// A path with `.` and `..` worked out, no doubled or trailing `/`, and `/private` left off
/// where the rest names the same place (`/var`, `/tmp`), as the Mac's
/// `standardizingPath` writes it.
pub fn standardized(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|p| *p != "..") {
                    parts.pop();
                } else if !absolute {
                    parts.push("..");
                }
            }
            p => parts.push(p),
        }
    }
    let joined = parts.join("/");
    let result = if absolute {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    };
    if let Some(rest) = result.strip_prefix("/private/")
        && ["var", "tmp", "etc"]
            .iter()
            .any(|d| rest == *d || rest.starts_with(&format!("{d}/")))
        && Path::new(&format!("/{rest}")).exists()
    {
        return format!("/{rest}");
    }
    result
}

/// The Mac's disks are case-insensitive by default, so paths are compared that way.
pub fn is_within(path: &str, folder: &str) -> bool {
    let p = path.to_lowercase();
    let f = folder.to_lowercase();
    p == f
        || p.starts_with(&if f == "/" {
            "/".to_string()
        } else {
            format!("{f}/")
        })
}

pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

pub fn name_matches(name: &str, words: &[String]) -> bool {
    if words.is_empty() {
        return false;
    }
    let plain: String = name
        .to_lowercase()
        .chars()
        .map(|c| if "-_.".contains(c) { ' ' } else { c })
        .collect();
    words.iter().all(|w| plain.contains(w.as_str()))
}

/// "Invoice-Register-FY2026.xlsx": a file's name (`^[^\s/\\]+\.[A-Za-z0-9]{1,6}$`), so
/// only names are compared.
pub fn is_file_name(text: &str) -> bool {
    let Some((stem, ext)) = text.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && !stem
            .chars()
            .any(|c| c.is_whitespace() || c == '/' || c == '\\')
        && (1..=6).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
}

fn join(folder: &str, name: &str) -> String {
    if folder == "/" {
        format!("/{name}")
    } else {
        format!("{folder}/{name}")
    }
}

fn parent(path: &str) -> String {
    match path.rfind('/') {
        Some(0) => "/".into(),
        Some(i) => path[..i].into(),
        None => ".".into(),
    }
}

fn file_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

fn is_dir(path: &str) -> bool {
    fs::metadata(path).is_ok_and(|m| m.is_dir())
}

fn is_symlink(path: &str) -> bool {
    fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

/// A folder's entries' names, or why they couldn't be read, worded for the model.
fn names_in(folder: &str) -> Result<Vec<String>, String> {
    let entries = fs::read_dir(folder).map_err(|e| match e.kind() {
        std::io::ErrorKind::PermissionDenied => {
            "this Mac doesn't give permission to see it".to_string()
        }
        std::io::ErrorKind::NotFound => "it isn't there".to_string(),
        _ => e.to_string(),
    })?;
    Ok(entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect())
}

fn optional_arg(call: &Call, name: &str) -> Option<String> {
    call.args
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| match v {
            Arg::Text(s) => s.clone(),
            Arg::Integer(n) => n.to_string(),
        })
}

fn arg(call: &Call, name: &str) -> Result<String, String> {
    optional_arg(call, name).ok_or(format!("Missing argument: {name}"))
}

/// A whole number, sent as a number or as text ("2").
fn int_arg(call: &Call, name: &str) -> Option<i64> {
    let number: f64 = optional_arg(call, name)?.trim().parse().ok()?;
    number.is_finite().then(|| number.round() as i64)
}

/// PDFs, Word, RTF and OpenDocument text through the Mac's own readers (PDFKit, and
/// AppKit's document readers), as the app read them.
#[cfg(target_os = "macos")]
mod documents {
    use objc2::rc::{Allocated, Retained, autoreleasepool};
    use objc2::runtime::{AnyClass, AnyObject};
    use objc2::{ClassType, msg_send};
    use objc2_foundation::{NSDictionary, NSString, NSURL};

    #[link(name = "PDFKit", kind = "framework")]
    unsafe extern "C" {}
    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    fn url(path: &str) -> Retained<NSURL> {
        NSURL::fileURLWithPath(&NSString::from_str(path))
    }

    pub fn pdf_text(path: &str) -> Option<String> {
        autoreleasepool(|_| {
            let class = AnyClass::get(c"PDFDocument")?;
            let url = url(path);
            unsafe {
                let new: Allocated<AnyObject> = msg_send![class, alloc];
                let document: Option<Retained<AnyObject>> = msg_send![new, initWithURL: &*url];
                let text: Option<Retained<NSString>> = msg_send![&*document?, string];
                Some(text?.to_string())
            }
        })
    }

    pub fn rich_text(path: &str) -> Option<String> {
        autoreleasepool(|_| {
            let url = url(path);
            let options = NSDictionary::<NSString, AnyObject>::new();
            unsafe {
                let new: Allocated<AnyObject> =
                    msg_send![objc2_foundation::NSAttributedString::class(), alloc];
                let text: Option<Retained<AnyObject>> = msg_send![
                    new,
                    initWithURL: &*url,
                    options: &*options,
                    documentAttributes: std::ptr::null_mut::<*mut AnyObject>(),
                    error: std::ptr::null_mut::<*mut AnyObject>()
                ];
                let string: Retained<NSString> = msg_send![&*text?, string];
                Some(string.to_string())
            }
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod documents {
    pub fn pdf_text(_path: &str) -> Option<String> {
        None
    }
    pub fn rich_text(_path: &str) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    //! `LocalToolsTests` from the app, where these tools were before, on a home folder made
    //! for each test.
    use super::*;

    struct Home(String);

    impl Drop for Home {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn home() -> Home {
        let base = std::env::temp_dir().join(format!("LocalTools-{:016x}", crate::random_u64()));
        let home = Home(standardized(&base.to_string_lossy()));
        for (file, text) in [
            (
                "Documents/notes.txt",
                "The code word for Friday is PELICAN.",
            ),
            ("Documents/Invoices/acme-november.pdf.txt", "not a pdf"),
            ("Projects/vireo/README.md", "# Vireo\nA chat app."),
            (
                "Projects/vireo/docs/secret.txt",
                "Remember: the code word is HERON.",
            ),
            ("Projects/garden/plan.md", "Plant tomatoes in May."),
            ("Library/Keychains/login.txt", "secret"),
            ("Projects/vireo/node_modules/dep/index.js", "HERON"),
        ] {
            make(&home, file, text);
        }
        home
    }

    fn make(home: &Home, relative: &str, text: &str) {
        let path = format!("{}/{relative}", home.0);
        fs::create_dir_all(parent(&path)).unwrap();
        fs::write(path, text).unwrap();
    }

    fn tools(home: &Home, access: Access) -> Files {
        Files::new(&home.0, access)
    }

    fn call(tool: &str, args: &[(&str, &str)]) -> Call {
        Call {
            tool: tool.into(),
            args: args
                .iter()
                .map(|(k, v)| (k.to_string(), Arg::Text(v.to_string())))
                .collect(),
        }
    }

    fn run(files: &Files, tool: &str, args: &[(&str, &str)]) -> String {
        files.run(&call(tool, args)).unwrap()
    }

    fn failure(files: &Files, tool: &str, args: &[(&str, &str)]) -> String {
        files.run(&call(tool, args)).unwrap_err()
    }

    // ---- Paths and access

    #[test]
    fn relative_paths_start_at_home() {
        let h = home();
        let t = tools(&h, Access::Home);
        let text = run(&t, "read_file", &[("path", "Documents/notes.txt")]);
        assert_eq!(text, "The code word for Friday is PELICAN.");
        assert_eq!(
            run(&t, "read_file", &[("path", "~/Documents/notes.txt")]),
            text
        );
        assert_eq!(
            run(&t, "read_file", &[("path", "\"Documents/notes.txt\"")]),
            text
        );
    }

    #[test]
    fn home_access_stops_outside_home_and_in_library() {
        let h = home();
        let t = tools(&h, Access::Home);
        let outside = failure(&t, "read_file", &[("path", "/etc/hosts")]);
        assert!(outside.contains("outside the user's folder"), "{outside}");
        let library = failure(&t, "read_file", &[("path", "Library/Keychains/login.txt")]);
        assert!(library.contains("~/Library"), "{library}");
        let climbing = failure(&t, "read_file", &[("path", "Documents/../../x")]);
        assert!(climbing.contains("outside the user's folder"), "{climbing}");
    }

    #[test]
    fn chosen_folders_are_the_only_ones_reached() {
        let h = home();
        let chosen = tools(&h, Access::Folders(vec![format!("{}/Projects/", h.0)]));
        let inside = run(&chosen, "read_file", &[("path", "Projects/garden/plan.md")]);
        assert_eq!(inside, "Plant tomatoes in May.");
        let elsewhere = failure(&chosen, "read_file", &[("path", "Documents/notes.txt")]);
        assert!(
            elsewhere.contains("outside the folders this chat may access"),
            "{elsewhere}"
        );
        let none = failure(
            &tools(&h, Access::Folders(vec![])),
            "list_directory",
            &[("path", "Projects")],
        );
        assert!(none.contains("(none)"), "{none}");
    }

    // macOS only: files.rs is the Mac's server-run tools (/tmp is /private/tmp there, /etc
    // is there, Git is looked for where the Mac keeps it). On Windows the app runs these
    // tools itself (PcTools); files.rs on Windows is a step of its own (HANDOFF.md).
    #[cfg(target_os = "macos")]
    #[test]
    fn full_access_reaches_anywhere() {
        let h = home();
        let text = run(&tools(&h, Access::Full), "file_exists", &[("path", "/etc")]);
        assert!(text.contains(r#""exists":true"#), "{text}");
    }

    #[test]
    fn access_comes_from_the_request() {
        assert_eq!(
            Access::from_request(Some("home"), vec![]),
            Some(Access::Home)
        );
        assert_eq!(
            Access::from_request(Some("full"), vec![]),
            Some(Access::Full)
        );
        assert_eq!(
            Access::from_request(Some("folders"), vec!["/a".into()]),
            Some(Access::Folders(vec!["/a".into()]))
        );
        assert_eq!(Access::from_request(Some("chat"), vec![]), None);
        assert_eq!(Access::from_request(None, vec![]), None);
    }

    // macOS only: files.rs is the Mac's server-run tools (/tmp is /private/tmp there, /etc
    // is there, Git is looked for where the Mac keeps it). On Windows the app runs these
    // tools itself (PcTools); files.rs on Windows is a step of its own (HANDOFF.md).
    #[cfg(target_os = "macos")]
    #[test]
    fn paths_are_standardized_like_the_macs() {
        assert_eq!(
            standardized("/Users/me/Documents/../Desktop/"),
            "/Users/me/Desktop"
        );
        assert_eq!(standardized("/Users//me/./x"), "/Users/me/x");
        assert_eq!(standardized("/a/../../b"), "/b");
        assert_eq!(standardized("/"), "/");
        assert_eq!(standardized("/private/tmp/x"), "/tmp/x");
        assert!(is_within("/Users/Me/Documents", "/users/me"));
        assert!(!is_within("/Users/meg", "/Users/me"));
        assert!(is_within("/etc", "/"));
    }

    // ---- Files

    #[test]
    fn missing_files_and_folders_say_so() {
        let h = home();
        let t = tools(&h, Access::Home);
        assert_eq!(
            failure(&t, "read_file", &[("path", "Documents/nope.txt")]),
            format!("File not found: {}/Documents/nope.txt", h.0)
        );
        let folder = failure(&t, "read_file", &[("path", "Documents")]);
        assert!(folder.contains("is a folder"), "{folder}");
        assert_eq!(failure(&t, "read_file", &[]), "Missing argument: path");
    }

    #[test]
    fn documents_it_cant_read_say_so() {
        let h = home();
        make(&h, "Documents/budget.xlsx", "PK");
        make(&h, "Documents/photo.bin", "");
        fs::write(
            format!("{}/Documents/photo.bin", h.0),
            [0xff, 0xfe, 0x00, 0x80],
        )
        .unwrap();
        let t = tools(&h, Access::Home);
        let xlsx = failure(&t, "read_file", &[("path", "Documents/budget.xlsx")]);
        assert!(xlsx.contains("can't read yet"), "{xlsx}");
        let binary = failure(&t, "read_file", &[("path", "Documents/photo.bin")]);
        assert_eq!(
            binary,
            "photo.bin is not a text file. read_file can read text files, PDFs and Word documents."
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn word_documents_and_pdfs_are_read_as_text() {
        let h = home();
        let t = tools(&h, Access::Home);
        make(
            &h,
            "Documents/letter.rtf",
            r"{\rtf1\ansi Dear Sam, the box is BLUE.}",
        );
        let rtf = run(&t, "read_file", &[("path", "Documents/letter.rtf")]);
        assert!(rtf.contains("the box is BLUE."), "{rtf}");
        // A Word document, made by the Mac's own converter.
        let docx = format!("{}/Documents/letter.docx", h.0);
        let made = Command::new("/usr/bin/textutil")
            .args(["-convert", "docx", "-output", &docx])
            .arg(format!("{}/Documents/letter.rtf", h.0))
            .status();
        if made.is_ok_and(|s| s.success()) {
            let text = run(&t, "read_file", &[("path", "Documents/letter.docx")]);
            assert!(text.contains("the box is BLUE."), "{text}");
        }
        fs::write(
            format!("{}/Documents/scan.pdf", h.0),
            pdf("The total is 42 dollars"),
        )
        .unwrap();
        let text = run(&t, "read_file", &[("path", "Documents/scan.pdf")]);
        assert!(text.contains("The total is 42 dollars"), "{text}");
        make(&h, "Documents/broken.pdf", "not a pdf");
        let broken = failure(&t, "read_file", &[("path", "Documents/broken.pdf")]);
        assert!(
            broken.contains("Couldn't read the text of broken.pdf"),
            "{broken}"
        );
    }

    /// A one-page PDF with a line of text.
    #[cfg(target_os = "macos")]
    fn pdf(line: &str) -> Vec<u8> {
        let content = format!("BT /F1 18 Tf 72 720 Td ({line}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut out = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, o) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.push_str(&format!("{} 0 obj\n{o}\nendobj\n", i + 1));
        }
        let xref = out.len();
        out.push_str(&format!(
            "xref\n0 {}\n0000000000 65535 f \n",
            objects.len() + 1
        ));
        for o in offsets {
            out.push_str(&format!("{o:010} 00000 n \n"));
        }
        out.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        ));
        out.into_bytes()
    }

    #[test]
    fn listing_puts_folders_first_and_goes_deeper_when_asked() {
        let h = home();
        let t = tools(&h, Access::Home);
        assert_eq!(
            run(&t, "list_directory", &[("path", "Projects/vireo")]),
            "[dir]  docs\n[dir]  node_modules\n[file] README.md"
        );
        // Below the folder asked for, packages are left out.
        let deep = t
            .run(&Call {
                tool: "list_directory".into(),
                args: vec![
                    ("path".into(), Arg::Text("Projects".into())),
                    ("depth".into(), Arg::Integer(2)),
                ],
            })
            .unwrap();
        assert_eq!(
            deep,
            format!(
                "{}/Projects:\n[dir]  garden\n  [file] plan.md\n[dir]  vireo\n  [dir]  docs\n  [file] README.md",
                h.0
            )
        );
        // Depth as text counts too.
        let text = run(
            &t,
            "list_directory",
            &[("path", "Projects"), ("depth", "2")],
        );
        assert_eq!(text, deep);
        fs::create_dir_all(format!("{}/Empty", h.0)).unwrap();
        assert_eq!(
            run(&t, "list_directory", &[("path", "Empty")]),
            "(empty directory)"
        );
        assert_eq!(
            failure(&t, "list_directory", &[("path", "Nowhere")]),
            format!("Folder not found: {}/Nowhere", h.0)
        );
    }

    #[test]
    fn file_exists_answers_as_json() {
        let h = home();
        let t = tools(&h, Access::Home);
        assert_eq!(
            run(&t, "file_exists", &[("path", "Documents/notes.txt")]),
            format!(
                r#"{{"exists":true,"path":{},"type":"file"}}"#,
                crate::json::string(&format!("{}/Documents/notes.txt", h.0))
            )
        );
        assert_eq!(
            run(&t, "file_exists", &[("path", "Documents/budget.xlsx")]),
            format!(
                r#"{{"exists":false,"path":{},"type":null}}"#,
                crate::json::string(&format!("{}/Documents/budget.xlsx", h.0))
            )
        );
        assert_eq!(
            run(&t, "file_exists", &[("path", "Documents")]),
            format!(
                r#"{{"exists":true,"path":{},"type":"directory"}}"#,
                crate::json::string(&format!("{}/Documents", h.0))
            )
        );
    }

    #[test]
    fn search_finds_text_in_files_and_skips_packages() {
        let h = home();
        let t = tools(&h, Access::Home);
        assert_eq!(
            run(
                &t,
                "search_files",
                &[("pattern", "code word"), ("directory", "Projects")]
            ),
            format!(
                "1 match for \"code word\":\n\n{}/Projects/vireo/docs/secret.txt:1: Remember: the code word is HERON.",
                h.0
            )
        );
        let heron = run(
            &t,
            "search_files",
            &[("pattern", "HERON"), ("directory", "Projects")],
        );
        assert!(!heron.contains("node_modules"), "{heron}");
        // And with home access, nothing in ~/Library.
        let secret = run(
            &t,
            "search_files",
            &[("pattern", "secret"), ("directory", "~")],
        );
        assert!(!secret.contains("Library"), "{secret}");
    }

    #[test]
    fn search_matches_every_word_of_a_name_and_folders() {
        let h = home();
        let t = tools(&h, Access::Home);
        let found = run(
            &t,
            "search_files",
            &[("pattern", "acme november"), ("directory", "~")],
        );
        assert!(
            found.contains("Invoices/acme-november.pdf.txt (file name matches)"),
            "{found}"
        );
        let folder = run(
            &t,
            "search_files",
            &[("pattern", "invoices"), ("directory", "Documents")],
        );
        assert!(
            folder.contains(&format!(
                "{}/Documents/Invoices/ (folder name matches)",
                h.0
            )),
            "{folder}"
        );
        assert_eq!(
            run(
                &t,
                "search_files",
                &[("pattern", "zebra"), ("directory", "Documents")]
            ),
            format!("No matches found for \"zebra\" in {}/Documents", h.0)
        );
        let md = run(
            &t,
            "search_files",
            &[
                ("pattern", "a"),
                ("directory", "Projects"),
                ("file_ext", ".MD"),
            ],
        );
        assert!(md.contains("plan.md") && !md.contains("secret.txt"), "{md}");
    }

    #[test]
    fn a_file_name_is_searched_by_name_only() {
        assert!(is_file_name("Invoice-Register-FY2026.xlsx"));
        assert!(!is_file_name("code word"));
        assert!(!is_file_name("a/b.txt"));
        let h = home();
        let t = tools(&h, Access::Home);
        // "plan.md" is a file's name, so a file that only mentions it isn't a match.
        make(&h, "Documents/todo.txt", "Write plan.md");
        let named = run(
            &t,
            "search_files",
            &[("pattern", "plan.md"), ("directory", "Documents")],
        );
        assert_eq!(
            named,
            format!("No matches found for \"plan.md\" in {}/Documents", h.0)
        );
        let text = run(
            &t,
            "search_files",
            &[("pattern", "Write plan"), ("directory", "Documents")],
        );
        assert!(text.contains("todo.txt:1: Write plan.md"), "{text}");
    }

    #[test]
    fn other_tools_are_refused() {
        let h = home();
        assert_eq!(
            failure(
                &tools(&h, Access::Home),
                "delete_file",
                &[("path", "Documents/notes.txt")]
            ),
            "delete_file can't run on this Mac."
        );
    }

    // ---- Git

    // macOS only: files.rs is the Mac's server-run tools (/tmp is /private/tmp there, /etc
    // is there, Git is looked for where the Mac keeps it). On Windows the app runs these
    // tools itself (PcTools); files.rs on Windows is a step of its own (HANDOFF.md).
    #[cfg(target_os = "macos")]
    #[test]
    fn git_status_and_diff_of_a_project() {
        let h = home();
        let t = tools(&h, Access::Home);
        let project = format!("{}/Projects/vireo", h.0);
        for args in [
            vec!["init", "-q"],
            vec!["add", "."],
            vec![
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "commit",
                "-qm",
                "start",
            ],
        ] {
            let all: Vec<&str> = ["-C", project.as_str()].into_iter().chain(args).collect();
            let (code, output) = git(&all, &project).unwrap();
            assert_eq!(code, 0, "{output}");
        }
        assert_eq!(
            run(&t, "git_status", &[("path", "Projects/vireo")]),
            format!(r#"{{"ok":true,"path":"{project}","status":"clean"}}"#)
        );
        make(
            &h,
            "Projects/vireo/README.md",
            "# Vireo\nA chat app for the Mac.",
        );
        let changed = run(&t, "git_status", &[("path", "Projects/vireo/docs")]);
        assert!(changed.contains(r#""status":"M README.md""#), "{changed}");
        let diff = run(&t, "git_diff", &[("path", "Projects/vireo/README.md")]);
        assert!(diff.contains("+A chat app for the Mac."), "{diff}");
        assert_eq!(
            run(
                &t,
                "git_diff",
                &[("path", "Projects/vireo/docs/secret.txt")]
            ),
            format!("No tracked Git diff for {project}/docs/secret.txt")
        );
        assert_eq!(
            failure(&t, "git_status", &[("path", "Documents")]),
            format!("No Git repository found for {}/Documents", h.0)
        );
        assert_eq!(
            failure(&t, "git_status", &[("path", "Nowhere")]),
            format!("Project directory not found: {}/Nowhere", h.0)
        );
    }
}
