//! A stand-in PC: a few folders and files in memory, and the app's read-only tools over
//! them, answering as the app's `PcTools` does. A model may take any sensible route
//! (listing a folder before reading, or searching): each step gets a real answer, and
//! every run gets the same ones.

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

#[derive(Default, Clone)]
pub struct Pc {
    /// Files by full path (lowercased for lookup), with their path as written and text.
    files: BTreeMap<String, (String, String)>,
    /// `git status --short` per project folder (lowercased).
    git: BTreeMap<String, String>,
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

    fn is_folder(&self, path: &str) -> bool {
        let prefix = format!("{}\\", key(path));
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    /// Runs one of the app's tools: its output, or why it failed.
    pub fn run(&self, tool: &str, args: &Value) -> Result<String, String> {
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
                    None => Err(format!("No such file: {path}")),
                }
            }
            "list_directory" => {
                let path = arg("path")?;
                if !self.is_folder(&path) {
                    return Err(format!("No such folder: {path}"));
                }
                let prefix = format!("{}\\", key(&path));
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
                Ok(entries.join("\n"))
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
                    .map(|(_, status)| status.clone());
                match (tool, project) {
                    (_, None) => Err(format!("{path} isn't in a Git project.")),
                    ("git_status", Some(status)) => Ok(status),
                    (_, Some(_)) => Ok("diff --git a/src/main.rs b/src/main.rs\n-    let port = 3000;\n+    let port = 8080;".into()),
                }
            }
            other => Err(format!("This PC doesn't run {other}.")),
        }
    }
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
        let pc = Pc::default()
            .file(r"C:\Users\me\Documents\notes.txt", "hello")
            .file(r"C:\Users\me\Documents\Invoices\a.pdf", "A")
            .git(r"C:\Users\me\Projects\x", " M main.rs");
        let run = |tool: &str, a: &[(&str, &str)]| pc.run(tool, &args(a));
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
}
