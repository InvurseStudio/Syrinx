//! A stand-in PC: a few folders and files in memory, and the app's read-only tools over
//! them, answering as the app's `PcTools` does. A model may take any sensible route
//! (listing a folder before reading, or searching): each step gets a real answer, and
//! every run gets the same ones.

use std::collections::BTreeMap;

use crate::json::Value;

/// The home folder relative paths start from, as the app's do.
pub const HOME: &str = r"C:\Users\me";

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
            "search_files" => {
                let folder = arg("directory")?;
                let pattern = args.str("pattern").unwrap_or("").to_lowercase();
                let prefix = format!("{}\\", key(&folder));
                let hits: Vec<&str> = self
                    .files
                    .iter()
                    .filter(|(k, (p, text))| {
                        k.starts_with(&prefix)
                            && (p.to_lowercase().contains(&pattern)
                                || text.to_lowercase().contains(&pattern))
                    })
                    .map(|(_, (p, _))| p.as_str())
                    .collect();
                Ok(if hits.is_empty() {
                    format!("Nothing in {folder} matches {pattern:?}.")
                } else {
                    hits.join("\n")
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
            Ok(r"C:\Users\me\Documents\Invoices\a.pdf".into())
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
