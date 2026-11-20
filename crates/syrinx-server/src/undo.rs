//! What a reply changed on this Mac, and how to reverse it ("Undo this run"), as Vireo 0.9's
//! server kept it (`undo_journal.py`), in the same shape and words.
//!
//! Each reply that changes files gets a folder under `undo/` in this server's data folder
//! (`store::data_dir`): its manifest and private copies of what it overwrote. Moves and
//! renames need only the two paths, and a deleted file is in the Trash. Undoing goes newest
//! first and refuses to touch a file that has changed since the reply left it, so it can't
//! lose the user's work. Files the reply made go to the Trash on undo, never away for good.
//! Copies are readable by this user only, and are kept for `KEEP_DAYS` within
//! `MAX_TOTAL_BYTES`.
//!
//! The journal is on disk, so "Undo this run" works after the reply, and after the server
//! has restarted, until the copies are swept.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use crate::developer::sha256_hex;
use crate::json::{self, Value};

pub const KEEP_DAYS: u64 = 14;
const MAX_FILE_BYTES: u64 = 100 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;
const MANIFEST: &str = "manifest.json";
const CHANGED: &str = "It changed after the agent last touched it, so it was left alone.";
/// The most of a shell command the journal keeps, in characters.
const MAX_SHELL: usize = 200;

/// Replies still running, which can't be undone until they end.
static BUSY: Mutex<Option<HashSet<String>>> = Mutex::new(None);
/// One undo at a time.
static UNDOING: Mutex<()> = Mutex::new(());

/// Where the journals are: `VIREO_UNDO_DIR` (the eval's), else `undo` in the data folder.
pub fn root() -> PathBuf {
    std::env::var_os("VIREO_UNDO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::store::data_dir().join("undo"))
}

/// A new run id: 32 hex characters, as 0.9's were.
pub fn new_run_id() -> String {
    format!("{:016x}{:016x}", crate::random_u64(), crate::random_u64() ^ 0x9e37_79b9_7f4a_7c15)
}

fn is_run_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

pub fn set_busy(run_id: &str, busy: bool) {
    let mut set = BUSY.lock().unwrap();
    let set = set.get_or_insert_with(HashSet::new);
    if busy {
        set.insert(run_id.to_string());
    } else {
        set.remove(run_id);
    }
}

fn busy() -> HashSet<String> {
    BUSY.lock().unwrap().clone().unwrap_or_default()
}

/// The file's hash, or `None` when there's no file there.
fn file_hash(path: &str) -> Option<String> {
    if Path::new(path).is_dir() {
        return None;
    }
    fs::read(path).ok().map(|bytes| sha256_hex(&bytes))
}

/// How to tell a moved file hasn't changed: its hash, or for a very large file its size and
/// time, which survive a move. A folder has none: moving one back overwrites nothing, and
/// Finder keeps touching what's inside it.
fn fingerprint(path: &str) -> Option<String> {
    let info = fs::metadata(path).ok()?;
    if info.is_dir() {
        return None;
    }
    if info.len() <= MAX_FILE_BYTES {
        return file_hash(path);
    }
    let modified = info
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    Some(format!("{}:{modified}", info.len()))
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Step {
    /// `created`, `overwritten`, `moved` or `trashed`.
    action: String,
    /// The file (for a move, where it was).
    path: String,
    /// Where a moved file went.
    destination: String,
    /// Where a deleted file is in the Trash, if the Trash said.
    trash_path: String,
    /// The copy of an overwritten file, in the run's folder.
    backup: Option<String>,
    size: u64,
    before: Option<String>,
    after: Option<String>,
    fingerprint: Option<String>,
    done: bool,
}

impl Step {
    fn to_json(&self) -> String {
        let opt = |v: &Option<String>| v.as_deref().map_or("null".to_string(), json::string);
        format!(
            "{{\"action\":{},\"path\":{},\"destination\":{},\"trashPath\":{},\"backup\":{},\"size\":{},\"before\":{},\"after\":{},\"fingerprint\":{},\"done\":{}}}",
            json::string(&self.action),
            json::string(&self.path),
            json::string(&self.destination),
            json::string(&self.trash_path),
            opt(&self.backup),
            self.size,
            opt(&self.before),
            opt(&self.after),
            opt(&self.fingerprint),
            self.done
        )
    }

    fn from_json(v: &Value) -> Option<Step> {
        let opt = |k: &str| v.str(k).map(str::to_string);
        Some(Step {
            action: v.str("action")?.to_string(),
            path: v.str("path")?.to_string(),
            destination: v.str("destination").unwrap_or("").to_string(),
            trash_path: v.str("trashPath").unwrap_or("").to_string(),
            backup: opt("backup"),
            size: v.num("size").unwrap_or(0.0) as u64,
            before: opt("before"),
            after: opt("after"),
            fingerprint: opt("fingerprint"),
            done: matches!(v.get("done"), Some(Value::Bool(true))),
        })
    }
}

/// A write under way: what `after_write` needs to finish its step.
#[derive(Debug)]
pub struct Pending(Step);

/// One reply's changes. Nothing lands on disk until the reply changes something.
#[derive(Debug)]
pub struct Journal {
    root: PathBuf,
    pub run_id: String,
    chat_id: String,
    steps: Vec<Step>,
    shell: Vec<String>,
}

/// What an undo did, as the app shows it.
pub struct Outcome {
    pub message: String,
    pub finished: bool,
}

/// Why a change couldn't be reversed. `final`: trying again can't help, so it isn't
/// offered again.
struct CantUndo {
    reason: String,
    last: bool,
}

fn cant(reason: impl Into<String>) -> CantUndo {
    CantUndo { reason: reason.into(), last: false }
}

fn cant_ever(reason: impl Into<String>) -> CantUndo {
    CantUndo { reason: reason.into(), last: true }
}

impl Journal {
    pub fn new(root: PathBuf, run_id: &str, chat_id: &str) -> Journal {
        Journal {
            root,
            run_id: run_id.to_string(),
            chat_id: chat_id.to_string(),
            steps: Vec::new(),
            shell: Vec::new(),
        }
    }

    fn folder(&self) -> PathBuf {
        self.root.join(&self.run_id)
    }

    pub fn has_changes(&self) -> bool {
        !self.steps.is_empty()
    }

    pub fn load(root: PathBuf, run_id: &str) -> Option<Journal> {
        if !is_run_id(run_id) {
            return None;
        }
        let mut journal = Journal::new(root, run_id, "");
        let text = fs::read_to_string(journal.folder().join(MANIFEST)).ok()?;
        let data = json::parse(&text).ok()?;
        journal.chat_id = data.str("chatId").unwrap_or("").to_string();
        journal.steps = data.array("steps").iter().map(Step::from_json).collect::<Option<_>>()?;
        journal.shell = data
            .array("shell")
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        Some(journal)
    }

    fn copied_bytes(&self) -> u64 {
        self.steps.iter().map(|s| s.size).sum()
    }

    fn prepare(&self) -> Result<(), String> {
        let folder = self.folder();
        if folder.exists() {
            return Ok(());
        }
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        private(&self.root);
        let mut keep = busy();
        keep.insert(self.run_id.clone());
        sweep(&self.root, &keep);
        fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        private(&folder);
        Ok(())
    }

    fn save(&self) {
        if self.prepare().is_err() {
            return;
        }
        let steps: Vec<String> = self.steps.iter().map(Step::to_json).collect();
        let shell: Vec<String> = self.shell.iter().map(|s| json::string(s)).collect();
        let text = format!(
            "{{\"chatId\":{},\"steps\":[{}],\"shell\":[{}]}}",
            json::string(&self.chat_id),
            steps.join(","),
            shell.join(",")
        );
        let temp = self.folder().join(".manifest.writing");
        if fs::write(&temp, text).is_ok() {
            let _ = fs::rename(temp, self.folder().join(MANIFEST));
        }
    }

    fn add(&mut self, step: Step) {
        self.steps.push(step);
        self.save();
    }

    /// Before a write that makes or replaces `path`: keeps a copy of what's there. Failing
    /// to keep one stops the write, leaving the file alone. A file too large to copy is
    /// still written, and undo can only say so.
    pub fn before_write(&mut self, path: &str) -> Result<Pending, String> {
        let mut step = Step {
            action: "created".into(),
            path: path.to_string(),
            ..Default::default()
        };
        let info = fs::metadata(path).ok();
        if let Some(info) = info.filter(|i| i.is_file()) {
            step.action = "overwritten".into();
            let size = info.len();
            if size <= MAX_FILE_BYTES && self.copied_bytes() + size <= MAX_TOTAL_BYTES {
                self.prepare()
                    .map_err(|e| format!("Couldn't keep a copy of {path} to undo with ({e}), so it was left as it is."))?;
                let name = format!("{}.copy", self.steps.len());
                let backup = self.folder().join(&name);
                let bytes = fs::read(path)
                    .map_err(|e| format!("Couldn't keep a copy of {path} to undo with ({e}), so it was left as it is."))?;
                write_private(&backup, &bytes)
                    .map_err(|e| format!("Couldn't keep a copy of {path} to undo with ({e}), so it was left as it is."))?;
                step.before = Some(sha256_hex(&bytes));
                step.backup = Some(name);
                step.size = size;
            }
        }
        Ok(Pending(step))
    }

    /// After the write, whether it worked or not: a write that failed partway is still
    /// recorded, because the copy is then the only good version.
    pub fn after_write(&mut self, pending: Pending) {
        let mut step = pending.0;
        if step.action == "overwritten" && step.backup.is_none() {
            return self.add(step); // too large to copy
        }
        step.after = file_hash(&step.path);
        if step.after == step.before {
            // Nothing changed, or nothing was made.
            if let Some(backup) = &step.backup {
                let _ = fs::remove_file(self.folder().join(backup));
            }
            return;
        }
        self.add(step);
    }

    pub fn moved(&mut self, source: &str, destination: &str) {
        self.add(Step {
            action: "moved".into(),
            path: source.to_string(),
            destination: destination.to_string(),
            fingerprint: fingerprint(destination),
            ..Default::default()
        });
    }

    pub fn trashed(&mut self, path: &str, trash_path: &str) {
        self.add(Step {
            action: "trashed".into(),
            path: path.to_string(),
            trash_path: trash_path.to_string(),
            ..Default::default()
        });
    }

    /// A shell command ran: it can't be undone, but undoing says so.
    pub fn note_shell(&mut self, command: &str) {
        self.shell.push(command.chars().take(MAX_SHELL).collect());
        if self.has_changes() {
            self.save();
        }
    }

    /// Reverses what's left, newest first. A change that can't be reversed stays listed,
    /// so the user can sort it out and undo again. Once nothing is left to try, the run's
    /// folder goes. `trash` moves a file the reply made to the Trash.
    pub fn undo(&mut self, trash: &dyn Fn(&str) -> Result<(), String>) -> Outcome {
        let folder = self.folder();
        let mut undone = 0;
        let mut not_undone: Vec<(String, String)> = Vec::new();
        for step in self.steps.iter_mut().rev() {
            if step.done {
                continue;
            }
            match reverse(step, &folder, trash) {
                Ok(()) => {
                    undone += 1;
                    step.done = true;
                }
                Err(why) => {
                    not_undone.push((step.path.clone(), why.reason));
                    step.done = why.last;
                }
            }
        }
        let finished = self.steps.iter().all(|s| s.done);
        if finished {
            let _ = fs::remove_dir_all(&folder);
        } else {
            self.save();
        }
        Outcome {
            message: summary(undone, &not_undone, &self.shell),
            finished,
        }
    }
}

fn reverse(step: &Step, folder: &Path, trash: &dyn Fn(&str) -> Result<(), String>) -> Result<(), CantUndo> {
    let os = |e: std::io::Error| cant(format!("It couldn't be restored: {e}."));
    match step.action.as_str() {
        "overwritten" => {
            let Some(backup) = &step.backup else {
                return Err(cant_ever("It was too large to keep a copy of."));
            };
            let backup = folder.join(backup);
            let bytes = fs::read(&backup).ok().filter(|b| Some(sha256_hex(b)) == step.before);
            let Some(bytes) = bytes else {
                return Err(cant_ever("Its saved copy is missing or damaged."));
            };
            let path = Path::new(&step.path);
            if path.exists() && file_hash(&step.path) != step.after {
                return Err(cant(CHANGED));
            }
            if let Some(dir) = path.parent() {
                fs::create_dir_all(dir).map_err(os)?;
            }
            fs::write(path, &bytes).map_err(os)?;
            if file_hash(&step.path) != step.before {
                return Err(cant("The restored file doesn't match its saved copy."));
            }
            Ok(())
        }
        "created" => {
            if !Path::new(&step.path).exists() {
                return Ok(()); // already gone
            }
            if file_hash(&step.path) != step.after {
                return Err(cant(CHANGED));
            }
            trash(&step.path).map_err(|e| cant(format!("It couldn't be moved to the Trash: {e}.")))
        }
        "moved" => {
            let (source, destination) = (Path::new(&step.path), Path::new(&step.destination));
            if !destination.exists() {
                return Err(cant(format!("It is no longer at {}.", step.destination)));
            }
            if source.exists() {
                return Err(cant(format!("Something else is now at {}.", step.path)));
            }
            if fingerprint(&step.destination) != step.fingerprint {
                return Err(cant(CHANGED));
            }
            if let Some(dir) = source.parent() {
                fs::create_dir_all(dir).map_err(os)?;
            }
            fs::rename(destination, source).map_err(os)
        }
        "trashed" => {
            let path = Path::new(&step.path);
            if path.exists() {
                return Err(cant(format!("Something else is now at {}.", step.path)));
            }
            if step.trash_path.is_empty() {
                return Err(cant_ever("It's in the Trash: put it back from there."));
            }
            if let Some(dir) = path.parent() {
                fs::create_dir_all(dir).map_err(os)?;
            }
            // Tried rather than checked: macOS may hide what's in the Trash from this process.
            match fs::rename(&step.trash_path, path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    Err(cant_ever("It is no longer in the Trash."))
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => Err(cant(format!(
                    "Vireo wasn't allowed to take it out of the Trash, where it is named {}.",
                    file_name(&step.trash_path)
                ))),
                Err(e) => Err(os(e)),
            }
        }
        other => Err(cant_ever(format!("Vireo doesn't know how to undo \"{other}\"."))),
    }
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().into_owned())
}

fn changes(count: usize) -> String {
    if count == 1 { "1 change".into() } else { format!("{count} changes") }
}

fn summary(undone: usize, not_undone: &[(String, String)], shell: &[String]) -> String {
    let mut lines = vec![if undone > 0 {
        format!("Undid {}.", changes(undone))
    } else {
        "No changes were undone.".to_string()
    }];
    if !not_undone.is_empty() {
        lines.push(format!("Couldn't undo {}:", changes(not_undone.len())));
        lines.extend(not_undone.iter().map(|(path, why)| format!("{}: {why}", file_name(path))));
    }
    if !shell.is_empty() {
        lines.push(if shell.len() == 1 {
            "A shell command was also run, which can't be undone:".to_string()
        } else {
            format!("{} shell commands were also run, which can't be undone:", shell.len())
        });
        lines.extend(shell.iter().cloned());
    }
    lines.join("\n")
}

/// Undoes a run from its journal on disk: the reply for `POST /api/chat/runs/{id}/undo`,
/// as (status, body).
pub fn undo_run(run_id: &str, trash: &dyn Fn(&str) -> Result<(), String>) -> (u16, String) {
    if busy().contains(run_id) {
        return (409, detail("This reply is still running. Stop it first, then undo."));
    }
    let _one = UNDOING.lock().unwrap();
    let Some(mut journal) = Journal::load(root(), run_id) else {
        return (
            404,
            detail(&format!("These changes can no longer be undone. Vireo keeps them for {KEEP_DAYS} days.")),
        );
    };
    let outcome = journal.undo(trash);
    (
        200,
        format!(
            "{{\"runId\":{},\"message\":{},\"finished\":{}}}",
            json::string(run_id),
            json::string(&outcome.message),
            outcome.finished
        ),
    )
}

fn detail(text: &str) -> String {
    format!("{{\"detail\":{}}}", json::string(text))
}

/// Drops runs past `KEEP_DAYS`, then the oldest until they fit `MAX_TOTAL_BYTES`.
fn sweep(root: &Path, keep: &HashSet<String>) {
    let Ok(entries) = fs::read_dir(root) else { return };
    let cutoff = SystemTime::now() - Duration::from_secs(KEEP_DAYS * 86_400);
    let mut sized = Vec::new();
    for entry in entries.flatten() {
        let folder = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !folder.is_dir() || keep.contains(&name) {
            continue;
        }
        let manifest = folder.join(MANIFEST);
        let stamp = fs::metadata(if manifest.exists() { &manifest } else { &folder })
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if stamp < cutoff {
            let _ = fs::remove_dir_all(&folder);
        } else {
            let size: u64 = fs::read_dir(&folder)
                .map(|d| d.flatten().filter_map(|f| f.metadata().ok()).map(|m| m.len()).sum())
                .unwrap_or(0);
            sized.push((stamp, size, folder));
        }
    }
    let mut total: u64 = sized.iter().map(|(_, size, _)| size).sum();
    sized.sort_by_key(|(stamp, _, _)| *stamp);
    for (_, size, folder) in sized {
        if total <= MAX_TOTAL_BYTES {
            break;
        }
        let _ = fs::remove_dir_all(&folder);
        total -= size;
    }
}

#[cfg(unix)]
fn private(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn private(_path: &Path) {}

/// Writes a copy only this user can read.
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(PathBuf);

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn dir() -> Dir {
        let d = std::env::temp_dir().join(format!("undo-{:016x}", crate::random_u64()));
        fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    fn at(d: &Dir, name: &str) -> String {
        d.0.join(name).to_string_lossy().into_owned()
    }

    /// Moves a file into the test's own trash folder.
    fn trash_into(d: &Dir) -> impl Fn(&str) -> Result<(), String> + '_ {
        move |path: &str| {
            let bin = d.0.join("trash");
            fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
            fs::rename(path, bin.join(file_name(path))).map_err(|e| e.to_string())
        }
    }

    #[test]
    fn overwrite_create_move_and_delete_are_undone_newest_first() {
        let d = dir();
        let root = d.0.join("undo");
        let id = new_run_id();
        assert!(is_run_id(&id));
        let mut j = Journal::new(root.clone(), &id, "chat-1");
        let notes = at(&d, "notes.txt");
        fs::write(&notes, "old").unwrap();

        let p = j.before_write(&notes).unwrap();
        fs::write(&notes, "new").unwrap();
        j.after_write(p);

        let made = at(&d, "made.md");
        let p = j.before_write(&made).unwrap();
        fs::write(&made, "hello").unwrap();
        j.after_write(p);

        let moved = at(&d, "moved.md");
        fs::rename(&made, &moved).unwrap();
        j.moved(&made, &moved);

        let gone = at(&d, "gone.txt");
        let in_trash = at(&d, "trashed-gone.txt");
        fs::write(&in_trash, "bye").unwrap();
        j.trashed(&gone, &in_trash);
        j.note_shell("ls -la");

        // Back from disk, as after a restart.
        let mut j = Journal::load(root.clone(), &id).unwrap();
        let outcome = j.undo(&trash_into(&d));
        assert!(outcome.finished, "{}", outcome.message);
        assert!(outcome.message.starts_with("Undid 4 changes."), "{}", outcome.message);
        assert!(outcome.message.contains("ls -la"));
        assert_eq!(fs::read_to_string(&notes).unwrap(), "old");
        assert!(!Path::new(&moved).exists() && !Path::new(&made).exists());
        assert!(d.0.join("trash/made.md").exists());
        assert_eq!(fs::read_to_string(&gone).unwrap(), "bye");
        assert!(!root.join(&id).exists());
        assert!(Journal::load(root, &id).is_none());
    }

    #[test]
    fn a_file_changed_since_is_left_alone_and_offered_again() {
        let d = dir();
        let root = d.0.join("undo");
        let id = new_run_id();
        let mut j = Journal::new(root.clone(), &id, "");
        let notes = at(&d, "notes.txt");
        fs::write(&notes, "old").unwrap();
        let p = j.before_write(&notes).unwrap();
        fs::write(&notes, "new").unwrap();
        j.after_write(p);
        fs::write(&notes, "the user's own edit").unwrap();

        let outcome = j.undo(&trash_into(&d));
        assert!(!outcome.finished);
        assert!(outcome.message.contains(CHANGED), "{}", outcome.message);
        assert_eq!(fs::read_to_string(&notes).unwrap(), "the user's own edit");

        // Put back as the agent left it, and it undoes.
        fs::write(&notes, "new").unwrap();
        let mut j = Journal::load(root, &id).unwrap();
        assert!(j.undo(&trash_into(&d)).finished);
        assert_eq!(fs::read_to_string(&notes).unwrap(), "old");
    }

    #[test]
    fn a_write_that_changes_nothing_leaves_no_journal() {
        let d = dir();
        let root = d.0.join("undo");
        let mut j = Journal::new(root.clone(), &new_run_id(), "");
        let notes = at(&d, "notes.txt");
        fs::write(&notes, "same").unwrap();
        let p = j.before_write(&notes).unwrap();
        fs::write(&notes, "same").unwrap();
        j.after_write(p);
        assert!(!j.has_changes());
        j.note_shell("echo hi");
        assert!(!j.folder().join(MANIFEST).exists());
    }

    #[test]
    fn unknown_or_malformed_runs_are_not_found() {
        assert!(Journal::load(std::env::temp_dir(), "../../etc").is_none());
        assert!(!is_run_id("ABCDEF0123456789abcdef0123456789"));
    }
}
