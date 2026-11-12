//! A stand-in PC: a few folders and files in memory, and the app's tools over them,
//! answering as the app's `PcTools` does. A model may take any sensible route (listing a
//! folder before reading, or searching): each step gets a real answer, and every run gets
//! the same ones. The tools that change files change it, so a case can check what the
//! reply left; a folder is there while it holds a file.

use std::collections::BTreeMap;

use crate::json::Value;

/// The home folder relative paths start from, as the app's do.
pub const HOME: &str = r"C:\Users\me";

/// "Invoice-Register-FY2026.xlsx": a file's name (the app's `FileNamePattern`,
/// `^[^\s/\\]+\.[A-Za-z0-9]{1,6}$`), searched for by name only.
fn is_file_name(s: &str) -> bool {
    let Some((stem, ext)) = s.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && !stem
            .chars()
            .any(|c| c.is_whitespace() || c == '/' || c == '\\')
        && (1..=6).contains(&ext.len())
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
}

#[derive(Default, Clone, Debug)]
pub struct Pc {
    /// Files by full path (lowercased for lookup), with their path as written and text.
    files: BTreeMap<String, (String, String)>,
    /// `git status --short` per project folder (lowercased).
    git: BTreeMap<String, String>,
    /// Tools whose approvals the stand-in user declines.
    declines: Vec<String>,
    /// `git diff` per project folder (lowercased), where a case sets one.
    diffs: BTreeMap<String, String>,
    /// Workbooks by path (lowercased): their one sheet's rows, the first a header.
    sheets: BTreeMap<String, Vec<Vec<String>>>,
    /// A project's tests, build and linter, by folder (lowercased) and action: the
    /// command, its exit code and its output.
    commands: BTreeMap<(String, String), (String, i32, String)>,
}

fn key(path: &str) -> String {
    path.to_lowercase()
}

/// A path as the tools take it: `/` or `\`, full or relative to the home folder, with no
/// trailing separator.
fn full(path: &str) -> String {
    let p = path.trim().trim_matches('"').replace('/', "\\");
    let p = p.trim_end_matches('\\');
    if p.len() >= 2 && p.as_bytes()[1] == b':' {
        p.to_string()
    } else if p.is_empty() || p == "." || p == "~" {
        HOME.to_string()
    } else {
        format!(
            "{HOME}\\{}",
            p.trim_start_matches(".\\").trim_start_matches("~\\")
        )
    }
}

impl Pc {
    pub fn file(mut self, path: &str, text: &str) -> Self {
        self.files
            .insert(key(path), (path.to_string(), text.to_string()));
        self
    }

    pub fn git(mut self, folder: &str, status: &str) -> Self {
        self.git.insert(key(folder), status.to_string());
        self
    }

    /// A workbook with one sheet ("Sheet1") of these rows, the first its header. It's also a
    /// file, so listing and searching find it, and read_file gives the sheet, as the app's
    /// read_file describes a workbook.
    pub fn sheet(mut self, path: &str, rows: &[&[&str]]) -> Self {
        let rows: Vec<Vec<String>> = rows
            .iter()
            .map(|r| r.iter().map(|c| c.to_string()).collect())
            .collect();
        let text = sheet_text(&rows);
        self.sheets.insert(key(path), rows);
        self.file(path, &text)
    }

    /// What a project's tests, build or linter (`action`: "tests", "build", "linter") run
    /// and give, as the app's RunProjectCommandAsync reports it.
    pub fn command(
        mut self,
        folder: &str,
        action: &str,
        command: &str,
        exit: i32,
        output: &str,
    ) -> Self {
        self.commands.insert(
            (key(folder), action.to_string()),
            (command.to_string(), exit, output.to_string()),
        );
        self
    }

    /// The project's `git diff` (otherwise a one-line change of a port).
    pub fn diff(mut self, folder: &str, diff: &str) -> Self {
        self.diffs.insert(key(folder), diff.to_string());
        self
    }

    /// The stand-in user declines this tool's approvals (they approve all others).
    pub fn declining(mut self, tool: &str) -> Self {
        self.declines.push(tool.to_string());
        self
    }

    /// Whether the stand-in user approves a call to this tool. Remembering is always
    /// declined: a fact kept from one case would be in every later case's instructions.
    pub fn approves(&self, tool: &str) -> bool {
        !matches!(tool, "remember" | "forget") && !self.declines.iter().any(|t| t == tool)
    }

    /// A file's text, by its full path (any case).
    pub fn text(&self, path: &str) -> Option<&str> {
        self.files.get(&key(path)).map(|(_, t)| t.as_str())
    }

    fn is_folder(&self, path: &str) -> bool {
        let prefix = format!("{}\\", key(path));
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    /// A folder's files and folders (folders with a trailing `\`), sorted.
    fn entries(&self, folder: &str) -> Vec<String> {
        let prefix = format!("{}\\", key(folder));
        let mut entries: Vec<String> = self
            .files
            .values()
            .filter(|(p, _)| key(p).starts_with(&prefix))
            .map(|(p, _)| {
                let rest = &p[prefix.len()..];
                match rest.split_once('\\') {
                    Some((folder, _)) => format!("{folder}\\"),
                    None => rest.to_string(),
                }
            })
            .collect();
        entries.dedup();
        entries
    }

    /// A missing path's error, as the app's `NotFound` gives it: the nearest folder above
    /// it that's there (in the home folder), what that holds, the names there nearest the
    /// one asked for, and files of the same name further down.
    fn not_found(&self, path: &str, first: String) -> String {
        let home = key(HOME);
        let mut folder = path.to_string();
        let mut tail: Vec<String> = Vec::new();
        loop {
            let Some((parent, name)) = folder.rsplit_once('\\') else {
                return first;
            };
            let parent_key = key(parent);
            if !(parent_key == home || parent_key.starts_with(&format!("{home}\\"))) {
                return first;
            }
            tail.insert(0, name.to_string());
            folder = parent.to_string();
            if self.is_folder(&folder) {
                break;
            }
        }
        // Folders first, then files, each by name ignoring case.
        let mut entries = self.entries(&folder);
        entries.sort_by_key(|e| (!e.ends_with('\\'), e.to_lowercase()));
        let names: Vec<&str> = entries.iter().map(|e| e.trim_end_matches('\\')).collect();
        let below = format!("{}\\", key(&folder));
        let asked = key(tail.last().map_or("", String::as_str));
        let found: Vec<&str> = self
            .files
            .iter()
            .filter(|(k, _)| k.starts_with(&below) && k.rsplit('\\').next() == Some(asked.as_str()))
            .map(|(_, (p, _))| p.as_str())
            .take(3)
            .collect();
        not_found_text(
            &first,
            &folder,
            &tail,
            &entries,
            &closest(&tail, &names),
            &found,
        )
    }

    /// A file or folder moved, as the app's `Move`: refused onto something already there
    /// (a change of case only is the same item) or into a folder that isn't.
    fn move_item(&mut self, source: &str, destination: &str) -> Result<(), String> {
        let is_file = self.files.contains_key(&key(source));
        if !is_file && !self.is_folder(source) {
            return Err(self.not_found(source, format!("File not found: {source}")));
        }
        let same = key(source) == key(destination);
        if !same && (self.files.contains_key(&key(destination)) || self.is_folder(destination)) {
            return Err(format!("Destination already exists: {destination}"));
        }
        if let Some((parent, _)) = destination.rsplit_once('\\')
            && key(parent) != key(HOME)
            && !self.is_folder(parent)
        {
            return Err(format!("Folder not found: {parent}"));
        }
        let moved: Vec<String> = if is_file {
            vec![key(source)]
        } else {
            let prefix = format!("{}\\", key(source));
            self.files
                .keys()
                .filter(|k| k.starts_with(&prefix))
                .cloned()
                .collect()
        };
        for k in moved {
            let (path, text) = self.files.remove(&k).unwrap();
            let new_path = format!("{destination}{}", &path[source.len()..]);
            self.files.insert(key(&new_path), (new_path, text));
        }
        Ok(())
    }

    /// Runs one of the app's tools: its output, or why it failed.
    pub fn run(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let arg = |name: &str| {
            args.str(name)
                .map(full)
                .ok_or(format!("{tool} needs {name}"))
        };
        match tool {
            "read_file" => {
                let path = arg("path")?;
                match self.files.get(&key(&path)) {
                    Some((_, text)) => Ok(text.clone()),
                    None if self.is_folder(&path) => {
                        Err(format!("{path} is a folder, not a file."))
                    }
                    None => Err(self.not_found(&path, format!("File not found: {path}"))),
                }
            }
            "list_directory" => {
                let path = arg("path")?;
                if !self.is_folder(&path) {
                    return Err(self.not_found(&path, format!("Folder not found: {path}")));
                }
                Ok(self.entries(&path).join("\n"))
            }
            // As the app's SearchFiles: a file or folder name matches when it holds every
            // word of the search ("*.pdf" finds a.pdf, "STEM camp" finds Stem-Camp.docx),
            // with - _ . read as spaces; unless the search is itself a file name, lines
            // inside files that hold the whole search match too.
            "search_files" => {
                let folder = arg("directory")?;
                let raw = args.str("pattern").unwrap_or("").trim().to_string();
                let pattern = raw.to_lowercase();
                let words: Vec<String> = pattern
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .filter(|w| !w.is_empty())
                    .map(str::to_string)
                    .collect();
                let names_only = is_file_name(&raw);
                let name_matches = |name: &str| {
                    let plain: String = name
                        .to_lowercase()
                        .chars()
                        .map(|c| if matches!(c, '-' | '_' | '.') { ' ' } else { c })
                        .collect();
                    !words.is_empty() && words.iter().all(|w| plain.contains(w.as_str()))
                };
                let prefix = format!("{}\\", key(&folder));
                let mut results = Vec::new();
                let mut folders_seen = std::collections::BTreeSet::new();
                for (k, (p, text)) in &self.files {
                    if !k.starts_with(&prefix) {
                        continue;
                    }
                    // The folders between the search's folder and the file.
                    let rel = &p[prefix.len()..];
                    let mut so_far = p[..prefix.len() - 1].to_string();
                    for part in rel
                        .split('\\')
                        .collect::<Vec<_>>()
                        .split_last()
                        .map_or(&[][..], |(_, dirs)| dirs)
                    {
                        so_far = format!("{so_far}\\{part}");
                        if folders_seen.insert(so_far.to_lowercase()) && name_matches(part) {
                            results.push(format!("{so_far}\\ (folder name matches)"));
                        }
                    }
                    let name = p.rsplit('\\').next().unwrap_or(p);
                    if name_matches(name) {
                        results.push(format!("{p} (file name matches)"));
                    } else if !names_only {
                        for (n, line) in text.lines().enumerate() {
                            if line.to_lowercase().contains(&pattern) {
                                results.push(format!("{p}:{}: {}", n + 1, line.trim()));
                            }
                        }
                    }
                }
                Ok(if results.is_empty() {
                    format!("No matches found for \"{raw}\" in {folder}")
                } else {
                    let n = results.len();
                    format!(
                        "{n} match{} for \"{raw}\":\n\n{}",
                        if n == 1 { "" } else { "es" },
                        results.join("\n")
                    )
                })
            }
            "file_exists" => {
                let path = arg("path")?;
                Ok(if self.files.contains_key(&key(&path)) {
                    format!("{path} is a file.")
                } else if self.is_folder(&path) {
                    format!("{path} is a folder.")
                } else {
                    format!("{path} doesn't exist.")
                })
            }
            "git_status" | "git_diff" => {
                let path = arg("path")?;
                let project = self
                    .git
                    .iter()
                    .find(|(folder, _)| key(&path).starts_with(folder.as_str()))
                    .map(|(folder, status)| (folder.clone(), status.clone()));
                match (tool, project) {
                    (_, None) => Err(format!("{path} isn't in a Git project.")),
                    ("git_status", Some((_, status))) => Ok(status),
                    (_, Some((folder, _))) => Ok(self.diffs.get(&folder).cloned().unwrap_or_else(|| {
                        "diff --git a/src/main.rs b/src/main.rs\n-    let port = 3000;\n+    let port = 8080;".into()
                    })),
                }
            }
            // The changes, as the app's WriteFileAsync, DeleteFile, RenameFile and MoveFile
            // answer (the app's journal, which undoes them, has no part here).
            "write_file" => {
                let path = arg("path")?;
                if self.is_folder(&path) {
                    return Err(format!("{path} is a folder, not a file."));
                }
                let content = args.str("content").unwrap_or("").to_string();
                self.files.insert(key(&path), (path.clone(), content));
                Ok(format!("File written successfully: {path}"))
            }
            "delete_file" => {
                let path = arg("path")?;
                if self.is_folder(&path) {
                    return Err(format!(
                        "{path} is a folder. delete_file deletes files only."
                    ));
                }
                if self.files.remove(&key(&path)).is_none() {
                    return Err(self.not_found(&path, format!("File not found: {path}")));
                }
                Ok(format!("File moved to the Recycle Bin: {path}"))
            }
            "rename_file" => {
                let source = arg("path")?;
                let new_name = args.str("new_name").unwrap_or("").trim().to_string();
                if new_name.is_empty()
                    || new_name == "."
                    || new_name == ".."
                    || new_name.contains(['\\', '/'])
                {
                    return Err("New name must be a single file or folder name.".into());
                }
                let parent = source
                    .rsplit_once('\\')
                    .map_or(HOME, |(p, _)| p)
                    .to_string();
                let destination = format!("{parent}\\{new_name}");
                self.move_item(&source, &destination)?;
                let name = |p: &str| p.rsplit('\\').next().unwrap_or(p).to_string();
                Ok(format!(
                    "Renamed successfully: {} -> {}",
                    name(&source),
                    name(&destination)
                ))
            }
            "move_file" => {
                let source = arg("source")?;
                let destination = arg("destination")?;
                self.move_item(&source, &destination)?;
                Ok(format!("Moved successfully: {source} -> {destination}"))
            }
            "read_spreadsheet" => {
                let path = arg("path")?;
                match self.sheets.get(&key(&path)) {
                    Some(rows) => Ok(sheet_text(rows)),
                    None if self.files.contains_key(&key(&path)) => {
                        Err(format!("{path} isn't an Excel workbook (.xlsx)."))
                    }
                    None => Err(self.not_found(&path, format!("File not found: {path}"))),
                }
            }
            // As the app's CreateWordDocumentAsync: ".docx" added where it's missing, and
            // the path given back (the server words the result). Its text is the title,
            // then the Markdown, so a case can read what was written.
            "create_docx" => {
                let mut path = arg("path")?;
                if !path.to_lowercase().ends_with(".docx") {
                    path.push_str(".docx");
                }
                if self.is_folder(&path) {
                    return Err(format!("{path} is a folder."));
                }
                let text = format!(
                    "{}\n\n{}",
                    args.str("title").unwrap_or(""),
                    args.str("content").unwrap_or("")
                );
                self.files.insert(key(&path), (path.clone(), text));
                Ok(path)
            }
            "run_tests" | "run_build" | "run_linter" => {
                let path = arg("path")?;
                let action = &tool["run_".len()..];
                let action = if action == "tests" { "tests" } else { action };
                let found = self.commands.iter().find(|((folder, a), _)| {
                    a == action && key(&path).starts_with(folder.as_str())
                });
                match found {
                    None if !self.is_folder(&path) => {
                        Err(self.not_found(&path, format!("Folder not found: {path}")))
                    }
                    None => Err(format!(
                        "No {action} command found for {path}: no project file there sets one up."
                    )),
                    Some(((folder, _), (command, exit, output))) => {
                        let folder = self
                            .files
                            .values()
                            .map(|(p, _)| p)
                            .find(|p| key(p).starts_with(&format!("{folder}\\")))
                            .map_or(path.clone(), |p| p[..folder.len()].to_string());
                        let words: Vec<String> =
                            command.split(' ').map(crate::json::string).collect();
                        Ok(format!(
                            "{{\"ok\":{},\"path\":{},\"command\":[{}],\"exitCode\":{exit},\"output\":{}}}",
                            *exit == 0,
                            crate::json::string(&folder),
                            words.join(","),
                            crate::json::string(output)
                        ))
                    }
                }
            }
            // As the app's CheckChange: the change made on a copy, so its errors are the
            // change's own, and nothing here changes.
            "check_change" => {
                let tool = args.str("tool").unwrap_or("").to_string();
                let change = args.get("args").cloned().unwrap_or(Value::Null);
                if !crate::cases::CHANGE_TOOLS.contains(&tool.as_str()) {
                    return Err(format!("{tool} can't be checked on this PC."));
                }
                self.clone().run(&tool, &change).map(|_| "ok".to_string())
            }
            other => Err(format!("This PC doesn't run {other}.")),
        }
    }
}

/// A sheet as the app's `Spreadsheet.Read` gives it: the sheet names, the range, then
/// "Row | A | B" and a line a row.
fn sheet_text(rows: &[Vec<String>]) -> String {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let column = |i: usize| ((b'A' + i as u8) as char).to_string();
    let range = format!("A1:{}{}", column(width - 1), rows.len().max(1));
    let mut text = format!(
        "Sheets: Sheet1\n\nSheet \"Sheet1\", {range} (used range {range})\nRow | {}\n",
        (0..width).map(column).collect::<Vec<_>>().join(" | ")
    );
    for (i, row) in rows.iter().enumerate() {
        text.push_str(&format!("{} | {}\n", i + 1, row.join(" | ")));
    }
    text
}

/// The words of a name, for comparing names: lowercase runs of letters or of digits, a
/// file's extension left out ("Report_2026.pdf" is "report", "2026").
fn words(name: &str, is_file: bool) -> Vec<String> {
    let stem = match name.rsplit_once('.') {
        Some((stem, _)) if is_file && !stem.is_empty() => stem,
        _ => name,
    };
    let mut words = Vec::new();
    let mut word = String::new();
    for c in stem.to_lowercase().chars() {
        let breaks = word
            .chars()
            .last()
            .is_some_and(|last| last.is_ascii_digit() != c.is_ascii_digit());
        if (!c.is_alphanumeric() || breaks) && !word.is_empty() {
            words.push(std::mem::take(&mut word));
        }
        if c.is_alphanumeric() {
            word.push(c);
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// Up to three of `names` sharing the most words with the missing part of the path
/// (`tail`: its folders and its name), as the app's `Closest`.
fn closest<'a>(tail: &[String], names: &[&'a str]) -> Vec<&'a str> {
    let asked: Vec<String> = tail
        .iter()
        .enumerate()
        .flat_map(|(i, part)| words(part, i + 1 == tail.len()))
        .collect();
    let mut scored: Vec<(usize, &str)> = names
        .iter()
        .map(|n| {
            let have = words(n, true);
            (asked.iter().filter(|w| have.contains(w)).count(), *n)
        })
        .filter(|(score, _)| *score > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
    scored.into_iter().take(3).map(|(_, n)| n).collect()
}

/// How many of a folder's items a missing path's error lists.
const MAX_HINT_ENTRIES: usize = 30;

/// The error's text, the same as the app's `NotFound`.
fn not_found_text(
    first: &str,
    folder: &str,
    tail: &[String],
    entries: &[String],
    closest: &[&str],
    found: &[&str],
) -> String {
    let mut lines = vec![first.to_string()];
    if tail.len() > 1 {
        lines.push(format!("There's no folder {folder}\\{} either.", tail[0]));
    }
    lines.push(if entries.is_empty() {
        format!("{folder} is empty.")
    } else if entries.len() > MAX_HINT_ENTRIES {
        format!(
            "{folder} holds {} items, the first {MAX_HINT_ENTRIES}: {}",
            entries.len(),
            entries[..MAX_HINT_ENTRIES].join(", ")
        )
    } else {
        format!("{folder} holds: {}", entries.join(", "))
    });
    if !closest.is_empty() {
        let paths: Vec<String> = closest.iter().map(|n| format!("{folder}\\{n}")).collect();
        lines.push(format!("Closest to what was asked: {}", paths.join(", ")));
    }
    if !found.is_empty() {
        lines.push(format!(
            "Also named {}: {}",
            tail.last().map_or("", String::as_str),
            found.join(", ")
        ));
    }
    // Facts only, no advice: a last line telling the model to go on with one of these
    // ("…use it and say so; if not, tell the user what's there", or "Go on with the one
    // that fits the request, if one plainly does") scored lower, measured 2026-11-11.
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(pairs: &[(&str, &str)]) -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
                .collect(),
        )
    }

    #[test]
    fn answers_like_the_apps_tools() {
        let mut pc = Pc::default()
            .file(r"C:\Users\me\Documents\notes.txt", "hello")
            .file(r"C:\Users\me\Documents\Invoices\a.pdf", "A")
            .git(r"C:\Users\me\Projects\x", " M main.rs");
        let mut run = |tool: &str, a: &[(&str, &str)]| pc.run(tool, &args(a));
        assert_eq!(
            run("read_file", &[("path", "Documents/notes.txt")]),
            Ok("hello".into()),
            "relative to home"
        );
        assert_eq!(
            run("read_file", &[("path", r"c:\users\me\documents\NOTES.txt")]),
            Ok("hello".into())
        );
        assert!(
            run("read_file", &[("path", r"C:\Users\me\Documents")])
                .unwrap_err()
                .contains("folder")
        );
        assert_eq!(
            run("list_directory", &[("path", r"C:\Users\me\Documents\")]),
            Ok(["Invoices\\", "notes.txt"].join("\n"))
        );
        assert_eq!(
            run("search_files", &[("pattern", "a.pdf"), ("directory", ".")]),
            Ok("1 match for \"a.pdf\":\n\nC:\\Users\\me\\Documents\\Invoices\\a.pdf (file name matches)".into())
        );
        // As the app: every word in the name ("*.pdf" is the word "pdf"), folders too,
        // and lines inside files for a search that isn't a file name.
        let found = run(
            "search_files",
            &[("pattern", "*.pdf"), ("directory", "Documents")],
        )
        .unwrap();
        assert!(
            found.contains(r"Invoices\a.pdf (file name matches)"),
            "{found}"
        );
        let found = run(
            "search_files",
            &[("pattern", "invoices"), ("directory", ".")],
        )
        .unwrap();
        assert!(
            found.contains(r"C:\Users\me\Documents\Invoices\ (folder name matches)"),
            "{found}"
        );
        let found = run("search_files", &[("pattern", "hello"), ("directory", ".")]).unwrap();
        assert!(found.contains(r"notes.txt:1: hello"), "{found}");
        assert!(
            run(
                "search_files",
                &[("pattern", "nothing here"), ("directory", ".")]
            )
            .unwrap()
            .starts_with("No matches")
        );
        assert!(
            super::is_file_name("Invoice-Register-FY2026.xlsx")
                && !super::is_file_name("README invoices")
        );
        assert!(
            run("file_exists", &[("path", r"C:\Users\me\nope.txt")])
                .unwrap()
                .contains("doesn't exist")
        );
        assert_eq!(
            run("git_status", &[("path", r"C:\Users\me\Projects\x")]),
            Ok(" M main.rs".into())
        );
        assert!(run("git_status", &[("path", r"C:\Users\me")]).is_err());
    }

    #[test]
    fn a_missing_path_says_what_is_there() {
        let mut pc = Pc::default()
            .file(r"C:\Users\me\Documents\meeting-notes.txt", "Tuesday")
            .file(r"C:\Users\me\Documents\notes.txt", "eggs")
            .file(r"C:\Users\me\Documents\report-2025.txt", "1.2")
            .file(r"C:\Users\me\Documents\report-2026.txt", "1.5")
            .file(r"C:\Users\me\Documents\Old\plan.md", "old plan");
        let mut run = |tool: &str, path: &str| pc.run(tool, &args(&[("path", path)]));
        assert_eq!(
            run("read_file", r"Documents\Notes\meeting.txt").unwrap_err(),
            [
                r"File not found: C:\Users\me\Documents\Notes\meeting.txt",
                r"There's no folder C:\Users\me\Documents\Notes either.",
                r"C:\Users\me\Documents holds: Old\, meeting-notes.txt, notes.txt, report-2025.txt, report-2026.txt",
                r"Closest to what was asked: C:\Users\me\Documents\meeting-notes.txt, C:\Users\me\Documents\notes.txt",
            ]
            .join("\n")
        );
        // Words apart from letters and digits, the extension left out.
        let err = run("read_file", r"Documents\2026_report.pdf").unwrap_err();
        assert!(
            err.contains(r"asked: C:\Users\me\Documents\report-2026.txt, C:\Users\me\Documents\report-2025.txt"),
            "{err}"
        );
        assert!(
            run("read_file", r"Documents\report2026")
                .unwrap_err()
                .contains(r"asked: C:\Users\me\Documents\report-2026.txt,")
        );
        // The same name further down.
        let err = run("read_file", r"Documents\plan.md").unwrap_err();
        assert!(
            err.contains(r"Also named plan.md: C:\Users\me\Documents\Old\plan.md"),
            "{err}"
        );
        assert!(
            !run("list_directory", r"Documents\Reports")
                .unwrap_err()
                .contains("Closest to what was asked"),
            "a folder's name is one word: reports isn't report"
        );
        // Nothing past the home folder.
        assert_eq!(
            run("read_file", r"D:\nope\x.txt").unwrap_err(),
            r"File not found: D:\nope\x.txt"
        );
    }

    #[test]
    fn changes_change_it_as_the_apps_do() {
        let mut pc = Pc::default()
            .file(r"C:\Users\me\Documents\draft.txt", "draft")
            .file(r"C:\Users\me\Documents\Invoices\globex.pdf", "G")
            .file(r"C:\Users\me\Downloads\acme.pdf", "A")
            .declining("delete_file");
        let mut run = |tool: &str, a: &[(&str, &str)]| pc.run(tool, &args(a));
        assert_eq!(
            run(
                "write_file",
                &[("path", "Documents/new.txt"), ("content", "hi")]
            ),
            Ok(r"File written successfully: C:\Users\me\Documents\new.txt".into())
        );
        assert_eq!(
            run(
                "rename_file",
                &[("path", r"Documents\draft.txt"), ("new_name", "final.txt")]
            ),
            Ok("Renamed successfully: draft.txt -> final.txt".into())
        );
        assert!(
            run(
                "rename_file",
                &[("path", r"Documents\final.txt"), ("new_name", r"..\x.txt")]
            )
            .is_err()
        );
        assert_eq!(
            run("move_file", &[("source", r"Downloads\acme.pdf"), ("destination", r"Documents\Invoices\acme.pdf")]),
            Ok(r"Moved successfully: C:\Users\me\Downloads\acme.pdf -> C:\Users\me\Documents\Invoices\acme.pdf".into())
        );
        assert!(
            run(
                "move_file",
                &[
                    ("source", r"Documents\new.txt"),
                    ("destination", r"Documents\final.txt")
                ]
            )
            .unwrap_err()
            .starts_with("Destination already exists")
        );
        assert!(
            run(
                "move_file",
                &[
                    ("source", r"Documents\new.txt"),
                    ("destination", r"Documents\Nowhere\new.txt")
                ]
            )
            .unwrap_err()
            .starts_with("Folder not found")
        );
        // A folder moves with what it holds.
        run(
            "move_file",
            &[
                ("source", r"Documents\Invoices"),
                ("destination", r"Documents\Bills"),
            ],
        )
        .unwrap();
        assert!(
            run("delete_file", &[("path", r"Documents\Bills")])
                .unwrap_err()
                .contains("deletes files only")
        );
        assert_eq!(
            run("delete_file", &[("path", r"Documents\new.txt")]),
            Ok(r"File moved to the Recycle Bin: C:\Users\me\Documents\new.txt".into())
        );
        // A check changes nothing, and fails as the change would.
        let check = |change: &str| crate::json::parse(change).unwrap();
        assert_eq!(
            pc.run(
                "check_change",
                &check(r#"{"tool":"delete_file","args":{"path":"Documents\\final.txt"}}"#)
            ),
            Ok("ok".into())
        );
        let err = pc
            .run(
                "check_change",
                &check(r#"{"tool":"move_file","args":{"source":"Downloads\\Acme invoice","destination":"Documents\\Bills\\a.pdf"}}"#),
            )
            .unwrap_err();
        assert!(
            err.starts_with(r"File not found: C:\Users\me\Downloads\Acme invoice"),
            "{err}"
        );
        assert_eq!(pc.text(r"C:\Users\me\Documents\final.txt"), Some("draft"));
        assert_eq!(pc.text(r"c:\users\me\documents\bills\acme.pdf"), Some("A"));
        assert!(pc.text(r"C:\Users\me\Documents\draft.txt").is_none());
        assert!(pc.text(r"C:\Users\me\Documents\new.txt").is_none());
        assert!(!pc.approves("delete_file") && pc.approves("write_file"));
    }
}
