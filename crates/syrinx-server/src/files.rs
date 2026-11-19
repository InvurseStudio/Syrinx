//! The file tools this server runs on the machine it's on, Mac or PC: reading, listing and
//! searching files, and Git's status and diff. When the app and the server share a machine
//! the server reaches the files itself rather than asking the app (`client_tool_request`,
//! which stays for an app that runs them, as the Windows app does). The answers are worded
//! as the Windows app's `PcTools` and the eval's stand-in PC (`syrinx-eval`'s `pc.rs`) word
//! theirs, so the model reads them the same way.
//!
//! What a chat may reach is its access level, sent with each chat: the home folder without
//! `~/Library` (on Windows, without AppData), the folders the user chose (sent with it), or
//! the whole machine. Paths are the machine's own: `/` on the Mac, drive letters and `\` on
//! Windows, compared ignoring case on both. PDFs and Word documents are read with the Mac's
//! own readers there, and with the server's (`attach/`) elsewhere.

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
/// How long a shell command may run, and the most of its output a result holds.
pub const SHELL_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_SHELL_OUTPUT: usize = 20_000;
/// Dependency and package folders, which a search or a deep listing walks past.
const SKIPPED_FOLDERS: [&str; 4] = ["node_modules", "venv", "__pycache__", "site-packages"];
/// Read with the Mac's document readers rather than as text.
const RICH_TEXT_EXTENSIONS: [&str; 5] = ["docx", "doc", "rtf", "rtfd", "odt"];
/// Documents these tools can't read yet.
const UNREAD_DOCUMENTS: [&str; 5] = ["xls", "pptx", "pages", "numbers", "key"];
/// Excel workbooks, read by `attach::xlsx`.
const WORKBOOKS: [&str; 2] = ["xlsx", "xlsm"];

/// Where a deleted file goes, as this machine calls it.
const TRASH: &str = if cfg!(target_os = "macos") {
    "Trash"
} else {
    "Recycle Bin"
};

fn exists(path: &str) -> bool {
    fs::symlink_metadata(path).is_ok()
}

/// The checks a move makes, without moving: nothing is at the destination already (a
/// change of capitals alone is the same item), and its folder is there.
fn can_move_to(source: &str, destination: &str) -> Result<(), String> {
    if !source.eq_ignore_ascii_case(destination) && exists(destination) {
        return Err(format!("Destination already exists: {destination}"));
    }
    let folder = parent(destination);
    if !is_dir(&folder) {
        return Err(format!("Folder not found: {folder}"));
    }
    Ok(())
}

/// Opens the workbook at `file` (`size` bytes).
fn workbook(file: &str, size: u64) -> Result<crate::attach::xlsx::Workbook, String> {
    let name = file_name(file);
    if size > crate::attach::xlsx::MAX_FILE {
        return Err(format!(
            "{name} is too large to read ({} MB). The most is 30 MB.",
            size >> 20
        ));
    }
    let data = fs::read(file).map_err(|_| format!("Couldn't read {file}."))?;
    crate::attach::xlsx::open(&data).map_err(|e| format!("Couldn't read {name}: {e}"))
}

/// The machine's path separator.
#[cfg(windows)]
pub const SEP: char = '\\';
#[cfg(not(windows))]
pub const SEP: char = '/';

/// What the answers call this machine.
#[cfg(target_os = "macos")]
const MACHINE: &str = "Mac";
#[cfg(not(target_os = "macos"))]
const MACHINE: &str = "PC";

/// How much of the machine a chat may reach (the app's `AgentAccess`).
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
    /// Where deleted and replaced files go instead of the system's Trash (tests).
    trash_folder: Option<String>,
}

impl Files {
    pub fn new(home: &str, access: Access) -> Files {
        let access = match access {
            Access::Folders(f) => {
                Access::Folders(f.iter().map(|f| standardized(&long_form(f))).collect())
            }
            a => a,
        };
        Files {
            home: standardized(&long_form(home)),
            access,
            trash_folder: None,
        }
    }

    /// Deleted and replaced files go into `folder`, not the system's Trash.
    #[cfg(test)]
    pub fn with_trash(mut self, folder: &str) -> Files {
        self.trash_folder = Some(standardized(folder));
        self
    }

    /// For the user this server runs as: `HOME`, or on Windows `USERPROFILE`.
    pub fn for_user(access: Access) -> Files {
        let home = if cfg!(windows) {
            std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME"))
        } else {
            std::env::var("HOME")
        };
        Files::new(&home.unwrap_or_else(|_| SEP.to_string()), access)
    }

    /// `~/Library` holds other apps' data and sign-ins, so home access leaves it out, as
    /// the Windows app leaves out AppData (which this leaves out on Windows).
    fn library(&self) -> String {
        join(&self.home, if cfg!(windows) { "AppData" } else { "Library" })
    }

    /// Runs one of the tools: its answer, or what went wrong, worded for the model.
    pub fn run(&self, call: &Call) -> Result<String, String> {
        match call.tool.as_str() {
            "read_file" => self.read_file(&arg(call, "path")?),
            "list_directory" => self.list_directory(&arg(call, "path")?, int_arg(call, "depth")),
            "read_spreadsheet" => self.read_spreadsheet(&arg(call, "path")?),
            "file_exists" => self.file_exists(&arg(call, "path")?),
            "search_files" => self.search_files(
                &arg(call, "pattern")?,
                &arg(call, "directory")?,
                optional_arg(call, "file_ext"),
            ),
            "git_status" => self.git_status(&arg(call, "path")?),
            "git_diff" => self.git_diff(&arg(call, "path")?),
            "write_file" => self.write_file(&arg(call, "path")?, &arg(call, "content")?),
            "rename_file" => self.rename_file(&arg(call, "path")?, &arg(call, "new_name")?),
            "move_file" => self.move_file(&arg(call, "source")?, &arg(call, "destination")?),
            "delete_file" => self.delete_file(&arg(call, "path")?),
            "create_docx" => self.create_docx(
                &arg(call, "path")?,
                &optional_arg(call, "title").unwrap_or_default(),
                &arg(call, "content")?,
            ),
            "fill_docx_template" => self.fill_docx_template(
                &arg(call, "template")?,
                &arg(call, "path")?,
                call.json("values").or_else(|| call.text("values")).unwrap_or(""),
            ),
            "create_pptx" => self.create_pptx(
                &arg(call, "path")?,
                &optional_arg(call, "title").unwrap_or_default(),
                call.json("slides").or_else(|| call.text("slides")).unwrap_or(""),
            ),
            "write_spreadsheet" => self.write_spreadsheet(
                &arg(call, "path")?,
                &optional_arg(call, "start_cell").unwrap_or_default(),
                call.json("rows").or_else(|| call.text("rows")).unwrap_or(""),
            ),
            "edit_file" => self.edit_file(
                &arg(call, "path")?,
                &arg(call, "old_text")?,
                &optional_arg(call, "new_text").unwrap_or_default(),
            ),
            "create_pdf" => self.create_pdf(
                &arg(call, "path")?,
                &optional_arg(call, "title").unwrap_or_default(),
                &arg(call, "content")?,
            ),
            "run_shell_command" => self.run_shell_command(&arg(call, "command")?),
            other => Err(format!("{other} can't run on this {MACHINE}.")),
        }
    }

    // ---- Changes -------------------------------------------------------------------------
    //
    // As the Windows app's PcTools makes them, with the same answers. Nothing is deleted
    // for good: a deleted file, and the old version of one that's replaced, go to the Trash.

    /// Whether a change could be made now, without making it (the app's `check_change`):
    /// the server asks before it asks the user to approve one, so nobody approves a change
    /// to a file that isn't there. `Ok` for a tool that changes no file.
    pub fn check(&self, call: &Call) -> Result<(), String> {
        match call.tool.as_str() {
            "write_file" => self.writable(&arg(call, "path")?).map(|_| ()),
            // The workbook is written out but not saved: rows it would refuse (a header
            // written over, rows already there) are said before anyone is asked to approve.
            "write_spreadsheet" => self
                .spreadsheet_written(
                    &arg(call, "path")?,
                    &optional_arg(call, "start_cell").unwrap_or_default(),
                    call.json("rows").or_else(|| call.text("rows")).unwrap_or(""),
                )
                .map(|_| ()),
            "create_docx" => self.office_target(&arg(call, "path")?, ".docx").map(|_| ()),
            "create_pptx" => self.office_target(&arg(call, "path")?, ".pptx").map(|_| ()),
            "create_pdf" => self.office_target(&arg(call, "path")?, ".pdf").map(|_| ()),
            "edit_file" => self
                .edited(
                    &arg(call, "path")?,
                    &arg(call, "old_text")?,
                    &optional_arg(call, "new_text").unwrap_or_default(),
                )
                .map(|_| ()),
            "run_shell_command" => self.shell_allowed(&arg(call, "command")?),
            "fill_docx_template" => self
                .template_and_target(&arg(call, "template")?, &arg(call, "path")?)
                .map(|_| ()),
            "delete_file" => self.deletable(&arg(call, "path")?).map(|_| ()),
            "rename_file" => self
                .rename_target(&arg(call, "path")?, &arg(call, "new_name")?)
                .map(|_| ()),
            "move_file" => self
                .move_target(&arg(call, "source")?, &arg(call, "destination")?)
                .map(|_| ()),
            _ => Ok(()),
        }
    }

    fn writable(&self, path: &str) -> Result<String, String> {
        let file = self.resolve(path)?;
        if is_dir(&file) {
            return Err(format!("{file} is a folder, not a file."));
        }
        Ok(file)
    }

    fn deletable(&self, path: &str) -> Result<String, String> {
        let file = self.resolve(path)?;
        if is_dir(&file) {
            return Err(format!(
                "{file} is a folder. delete_file deletes files only."
            ));
        }
        if !exists(&file) {
            return Err(format!("File not found: {file}"));
        }
        Ok(file)
    }

    /// A rename's source and where it goes, checked.
    fn rename_target(&self, path: &str, new_name: &str) -> Result<(String, String), String> {
        let source = self.resolve(path)?;
        if !exists(&source) {
            return Err(format!("File not found: {source}"));
        }
        let name = new_name.trim();
        if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
            return Err("New name must be a single file or folder name.".into());
        }
        let destination = self.resolve(&join(&parent(&source), name))?;
        can_move_to(&source, &destination)?;
        Ok((source, destination))
    }

    fn move_target(&self, source: &str, destination: &str) -> Result<(String, String), String> {
        let source = self.resolve(source)?;
        let destination = self.resolve(destination)?;
        if !exists(&source) {
            return Err(format!("File not found: {source}"));
        }
        can_move_to(&source, &destination)?;
        Ok((source, destination))
    }

    fn write_file(&self, path: &str, content: &str) -> Result<String, String> {
        let file = self.writable(path)?;
        self.replace(&file, content.as_bytes())?;
        Ok(format!("File written successfully: {file}"))
    }

    /// Writes `bytes` as `file`, its folder made if need be. The new file is written
    /// beside it first, so a failed write leaves the old one whole, and the old one goes
    /// to the Trash, not away.
    fn replace(&self, file: &str, bytes: &[u8]) -> Result<(), String> {
        fs::create_dir_all(parent(file))
            .map_err(|e| format!("Couldn't make the folder for {file}: {e}"))?;
        let temp = format!("{file}.vireo-writing");
        fs::write(&temp, bytes).map_err(|e| format!("Couldn't write {file}: {e}"))?;
        if exists(file)
            && let Err(e) = self.to_trash(file)
        {
            let _ = fs::remove_file(&temp);
            return Err(format!(
                "Couldn't keep the old version of {file} ({e}), so it was left as it is."
            ));
        }
        fs::rename(&temp, file).map_err(|e| format!("Couldn't write {file}: {e}"))?;
        if fs::read(file).ok().as_deref() != Some(bytes) {
            return Err(format!("File verification failed after writing: {file}"));
        }
        Ok(())
    }

    /// The Office file a path names: its extension added when it's left off.
    fn office_target(&self, path: &str, extension: &str) -> Result<String, String> {
        let mut file = self.resolve(path.trim())?;
        if !file.to_lowercase().ends_with(extension) {
            file.push_str(extension);
        }
        if is_dir(&file) {
            return Err(format!("{file} is a folder."));
        }
        Ok(file)
    }

    /// A template that's there, and where its filled copy goes: a file name on its own
    /// means beside the template, and never over the template itself.
    fn template_and_target(&self, template: &str, path: &str) -> Result<(String, String), String> {
        let template = self.resolve(template)?;
        if !exists(&template) || is_dir(&template) {
            return Err(format!("File not found: {template}"));
        }
        let lower = template.to_lowercase();
        if !(lower.ends_with(".docx") || lower.ends_with(".dotx")) {
            return Err(format!(
                "{} isn't a Word template (.docx or .dotx).",
                file_name(&template)
            ));
        }
        let target = path.trim();
        let bare = !target.is_empty() && !target.contains(['/', '\\']) && !target.starts_with('~');
        let file = if bare {
            self.office_target(&join(&parent(&template), target), ".docx")?
        } else {
            self.office_target(target, ".docx")?
        };
        if file.eq_ignore_ascii_case(&template) {
            return Err(
                "Save the filled document under a new name, so the template stays as it is.".into(),
            );
        }
        Ok((template, file))
    }

    /// A Word document from a title and Markdown (`attach::docx_write`).
    fn create_docx(&self, path: &str, title: &str, content: &str) -> Result<String, String> {
        let file = self.office_target(path, ".docx")?;
        self.replace(&file, &crate::attach::docx_write::create(title, content))?;
        Ok(format!("Word document saved: {file}"))
    }

    /// A copy of a Word template with its {{placeholders}} filled in.
    fn fill_docx_template(&self, template: &str, path: &str, values: &str) -> Result<String, String> {
        use crate::json::Value;
        const SHAPE: &str = "values must name each placeholder and its text.";
        let (template, file) = self.template_and_target(template, path)?;
        let mut parsed = crate::json::parse(values).map_err(|_| SHAPE.to_string())?;
        if let Value::String(inner) = &parsed {
            parsed = crate::json::parse(inner).map_err(|_| SHAPE.to_string())?;
        }
        let Value::Object(map) = parsed else {
            return Err(SHAPE.into());
        };
        let values: Vec<(String, String)> = map
            .iter()
            .map(|(k, v)| {
                let text = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_json(),
                };
                (k.clone(), text)
            })
            .collect();
        let data = fs::read(&template).map_err(|_| format!("Couldn't read {template}."))?;
        let (bytes, result) = crate::attach::docx_write::fill(&data, &values)
            .map_err(|e| format!("Couldn't fill {}: {e}", file_name(&template)))?;
        self.replace(&file, &bytes)?;
        let mut lines = vec![format!("Document saved successfully: {file}")];
        if !result.filled.is_empty() {
            lines.push(format!("Filled: {}.", result.filled.join(", ")));
        }
        if !result.left_empty.is_empty() {
            let names: Vec<String> = result.left_empty.iter().map(|n| format!("{{{{{n}}}}}")).collect();
            lines.push(format!(
                "Left as they were, with no value given: {}.",
                names.join(", ")
            ));
        }
        if !result.not_in_template.is_empty() {
            lines.push(format!(
                "Not in the template, so not used: {}.",
                result.not_in_template.join(", ")
            ));
        }
        Ok(lines.join("\n"))
    }

    /// A slide deck: a title slide, then a slide per entry (`attach::pptx_write`).
    fn create_pptx(&self, path: &str, title: &str, slides: &str) -> Result<String, String> {
        use crate::attach::pptx_write;
        let file = self.office_target(path, ".pptx")?;
        let slides = pptx_write::slides_from(slides)?;
        self.replace(&file, &pptx_write::create(title, &slides))?;
        let titles: Vec<&str> = slides.iter().map(|s| s.title.as_str()).collect();
        Ok(format!(
            "Slide deck saved successfully: {file}\n{} slides: the title slide, then {}.",
            slides.len() + 1,
            titles.join(", ")
        ))
    }

    /// Rows of values into a workbook's first sheet, from a cell or after its last row;
    /// the workbook is made when there's none (`attach::xlsx_write`).
    fn write_spreadsheet(&self, path: &str, start: &str, rows: &str) -> Result<String, String> {
        let (file, bytes, message) = self.spreadsheet_written(path, start, rows)?;
        self.replace(&file, &bytes)?;
        Ok(message)
    }

    /// The workbook as `write_spreadsheet` would save it: its path, bytes and answer.
    fn spreadsheet_written(
        &self,
        path: &str,
        start: &str,
        rows: &str,
    ) -> Result<(String, Vec<u8>, String), String> {
        use crate::attach::xlsx_write;
        let file = self.office_target(path, ".xlsx")?;
        let start = xlsx_write::Start::parse(start)?;
        let rows = xlsx_write::rows_from(rows)?;
        let existing = if exists(&file) {
            Some(fs::read(&file).map_err(|_| format!("Couldn't read {file}."))?)
        } else {
            None
        };
        let (bytes, message) = xlsx_write::write(&file, existing.as_deref(), start, &rows)?;
        Ok((file, bytes, message))
    }

    /// A file's text with `old` replaced by `new`, where `old` is in it once: the file, its
    /// new text, and how many lines went for how many.
    fn edited(&self, path: &str, old: &str, new: &str) -> Result<(String, String, String), String> {
        let file = self.writable(path)?;
        if !exists(&file) {
            return Err(format!("File not found: {file}. To make a new file, use write_file."));
        }
        let size = fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
        if size > MAX_READ_BYTES {
            return Err(format!("{file} is too large to edit here (over 1 MB)."));
        }
        let bytes = fs::read(&file).map_err(|e| format!("Couldn't read {file}: {e}"))?;
        let text = String::from_utf8(bytes)
            .map_err(|_| format!("{file} isn't a text file, so it can't be edited by replacing text."))?;
        if old.is_empty() {
            return Err("old_text is empty: copy the text to replace from the file.".into());
        }
        // A file with Windows line ends is matched with them.
        let (old, new) = if text.contains("\r\n") && !old.contains("\r\n") {
            (old.replace('\n', "\r\n"), new.replace('\n', "\r\n"))
        } else {
            (old.to_string(), new.to_string())
        };
        let found = text.matches(old.as_str()).count();
        if found == 0 {
            let trimmed = old.trim();
            let hint = if !trimmed.is_empty() && trimmed != old && text.contains(trimmed) {
                " It is there without the spaces or lines around it: copy it exactly."
            } else {
                " Read the file again and copy the text exactly, spaces and line breaks included."
            };
            return Err(format!("The text to replace isn't in {file}.{hint}"));
        }
        if found > 1 {
            return Err(format!(
                "The text to replace is in {file} {found} times. Include more of the text around it, so it names one place."
            ));
        }
        // A model sometimes sends the whole new file, or more of it, as new_text with only a
        // line as old_text, which would leave the lines around it twice (Gemma 4 E4B, Nov 19:
        // "- eggs" for the whole list). A line next to the match that new_text repeats, and
        // old_text doesn't hold, is refused.
        let at = text.find(old.as_str()).unwrap_or(0);
        let before = text[..at].trim_end_matches(['\r', '\n']);
        let after = text[at + old.len()..].trim_start_matches(['\r', '\n']);
        let neighbours = [
            before.rsplit('\n').next().unwrap_or(""),
            after.split('\n').next().unwrap_or(""),
        ];
        if let Some(line) = neighbours.iter().map(|l| l.trim_end_matches('\r')).find(|line| {
            line.trim().chars().count() >= 3 && new.contains(*line) && !old.contains(*line)
        }) {
            return Err(format!(
                "new_text repeats \"{}\", which is next to old_text in the file and would then be there twice. new_text replaces old_text only: give just the text that goes in its place, or put that line in old_text too.",
                line.trim()
            ));
        }
        let changed = text.replacen(old.as_str(), new.as_str(), 1);
        let lines = |t: &str| if t.is_empty() { 0 } else { t.lines().count() };
        let said = format!(
            "{} line{} replaced with {} line{}",
            lines(&old),
            if lines(&old) == 1 { "" } else { "s" },
            lines(&new),
            if lines(&new) == 1 { "" } else { "s" }
        );
        Ok((file, changed, said))
    }

    fn edit_file(&self, path: &str, old: &str, new: &str) -> Result<String, String> {
        let (file, changed, said) = self.edited(path, old, new)?;
        self.replace(&file, changed.as_bytes())?;
        Ok(format!("File edited successfully: {file} ({said})."))
    }

    /// A PDF from a title and Markdown (`attach::pdf_write`).
    fn create_pdf(&self, path: &str, title: &str, content: &str) -> Result<String, String> {
        let file = self.office_target(path, ".pdf")?;
        let (bytes, missing) = crate::attach::pdf_write::create(title, content);
        self.replace(&file, &bytes)?;
        let mut said = format!("PDF saved: {file}");
        if !missing.is_empty() {
            let shown: String = missing.iter().take(8).collect::<Vec<_>>().iter().map(|c| format!(" {c}")).collect();
            said.push_str(&format!(
                "\nThe PDF's fonts don't have some characters, so they show as \"?\":{shown}. Say so to the user if it matters."
            ));
        }
        Ok(said)
    }

    /// Shell commands reach the whole machine, so a chat that may reach only the folders
    /// the user chose can't run them.
    fn shell_allowed(&self, command: &str) -> Result<(), String> {
        if command.trim().is_empty() {
            return Err("The command is empty.".into());
        }
        if matches!(self.access, Access::Folders(_)) {
            return Err(
                "This chat may reach only the folders the user chose, so it can't run shell commands. The user can give it more access in the chat's settings."
                    .into(),
            );
        }
        Ok(())
    }

    /// Runs a command in the user's shell (zsh on the Mac, `cmd` on Windows) from the home
    /// folder, stopping it, and what it started, after `SHELL_TIMEOUT`.
    fn run_shell_command(&self, command: &str) -> Result<String, String> {
        self.shell_allowed(command)?;
        let mut shell = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", command]);
            c
        } else if Path::new("/bin/zsh").exists() {
            let mut c = Command::new("/bin/zsh");
            c.args(["-l", "-c", command]);
            c
        } else {
            let mut c = Command::new("/bin/sh");
            c.args(["-c", command]);
            c
        };
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Its own process group, so a timeout stops what it started too.
            shell.process_group(0);
        }
        let (mut reader, writer) = std::io::pipe().map_err(|e| format!("Couldn't run the command: {e}"))?;
        let errors = writer.try_clone().map_err(|e| format!("Couldn't run the command: {e}"))?;
        let mut child = shell
            .current_dir(&self.home)
            .stdin(Stdio::null())
            .stdout(writer)
            .stderr(errors)
            .spawn()
            .map_err(|e| format!("Couldn't run the command: {e}"))?;
        // The command holds the pipe's write ends too: they go, so reading ends with it.
        drop(shell);
        let collected = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = reader.read_to_end(&mut bytes);
            bytes
        });
        let deadline = Instant::now() + SHELL_TIMEOUT;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() > deadline => {
                    #[cfg(unix)]
                    let _ = Command::new("/bin/kill").args(["-9", &format!("-{}", child.id())]).status();
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "The command was stopped after {} seconds. Run a quicker one, or ask the user to run it in Terminal.",
                        SHELL_TIMEOUT.as_secs()
                    ));
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => return Err(format!("Couldn't run the command: {e}")),
            }
        };
        // A command that left something running in the background keeps the pipe open:
        // the output is what came within a moment of it finishing.
        let bytes = {
            let waited = Instant::now();
            while !collected.is_finished() && waited.elapsed() < Duration::from_secs(2) {
                std::thread::sleep(Duration::from_millis(20));
            }
            if collected.is_finished() {
                collected.join().unwrap_or_default()
            } else {
                b"(It left something running in the background, so its output wasn't collected.)".to_vec()
            }
        };
        let output = String::from_utf8_lossy(&bytes);
        let output = output.trim();
        let output = if output.chars().count() > MAX_SHELL_OUTPUT {
            let head: String = output.chars().take(MAX_SHELL_OUTPUT / 2).collect();
            let tail: Vec<char> = output.chars().rev().take(MAX_SHELL_OUTPUT / 2).collect();
            let tail: String = tail.into_iter().rev().collect();
            format!("{head}\n[... the middle of the output is left out ...]\n{tail}")
        } else {
            output.to_string()
        };
        let code = status.code().unwrap_or(-1);
        if code != 0 {
            return Err(format!(
                "The command exited with status {code}: {}",
                if output.is_empty() { "no output" } else { &output }
            ));
        }
        Ok(if output.is_empty() {
            "The command finished with no output.".to_string()
        } else {
            output
        })
    }

    fn rename_file(&self, path: &str, new_name: &str) -> Result<String, String> {
        let (source, destination) = self.rename_target(path, new_name)?;
        fs::rename(&source, &destination)
            .map_err(|e| format!("Couldn't rename {source}: {e}"))?;
        Ok(format!(
            "Renamed successfully: {} -> {}",
            file_name(&source),
            file_name(&destination)
        ))
    }

    fn move_file(&self, source: &str, destination: &str) -> Result<String, String> {
        let (source, destination) = self.move_target(source, destination)?;
        fs::rename(&source, &destination).map_err(|e| format!("Couldn't move {source}: {e}"))?;
        Ok(format!("Moved successfully: {source} -> {destination}"))
    }

    fn delete_file(&self, path: &str) -> Result<String, String> {
        let file = self.deletable(path)?;
        self.to_trash(&file)
            .map_err(|e| format!("Couldn't move {file} to the {TRASH}: {e}"))?;
        Ok(format!("File moved to the {TRASH}: {file}"))
    }

    /// Moves a file to the Trash (in tests, to their own folder).
    fn to_trash(&self, file: &str) -> Result<(), String> {
        match &self.trash_folder {
            Some(folder) => {
                fs::create_dir_all(folder).map_err(|e| e.to_string())?;
                let mut target = join(folder, &file_name(file));
                let mut n = 1;
                while exists(&target) {
                    n += 1;
                    target = join(folder, &format!("{} {n}", file_name(file)));
                }
                fs::rename(file, target).map_err(|e| e.to_string())
            }
            None if documents::trash(file) => Ok(()),
            None => Err(format!("this {MACHINE}'s {TRASH} didn't take it")),
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
        } else if let Some(rest) = expanded
            .strip_prefix("~/")
            .or_else(|| expanded.strip_prefix("~\\"))
        {
            expanded = rest;
        }
        let relative = !expanded.is_empty()
            && !is_absolute(expanded)
            && !expanded.starts_with(['/', '\\']);
        let full = standardized(&if is_absolute(expanded) {
            expanded.to_string()
        } else if cfg!(windows) && (expanded.starts_with('/') || expanded.starts_with('\\')) {
            // The root of the home folder's drive.
            format!("{}{expanded}", self.home.get(..2).unwrap_or(""))
        } else if expanded.is_empty() {
            self.home.clone()
        } else {
            join(&self.home, expanded)
        });
        match &self.access {
            Access::Full => Ok(full),
            Access::Folders(folders) => {
                if folders.iter().any(|f| is_within(&full, f)) {
                    return Ok(full);
                }
                // A relative path that isn't under the home folder's own copy of a chosen
                // folder is looked for inside the chosen folders, as the user writes one
                // ("Admin/CCA-Form-Template.docx", from the folder they work in). 0.9
                // resolved them so; here `list_directory Admin` failed as outside the
                // chat's folders and the model spent a dozen calls finding its way (Nov 15
                // test gauntlet, 6.1 and 12.3).
                if relative {
                    match Self::within_folders(folders, expanded) {
                        Ok(Some(found)) => return Ok(found),
                        Ok(None) => {}
                        Err(several) => return Err(several),
                    }
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
                        "{full} is in {}, which these tools can't reach.",
                        if cfg!(windows) { "AppData" } else { "~/Library" }
                    ));
                }
                Ok(full)
            }
        }
    }

    /// A relative path inside the folders a chat may reach: where it exists, or failing
    /// that where the folder it would go in exists (a file about to be written). `None`
    /// when neither (the path stays outside the chat's folders, as before); `Err` when it
    /// is in more than one of them.
    fn within_folders(folders: &[String], relative: &str) -> Result<Option<String>, String> {
        let candidates: Vec<String> = folders
            .iter()
            .map(|f| standardized(&join(f, relative)))
            .filter(|c| folders.iter().any(|f| is_within(c, f)))
            .collect();
        let pick = |found: Vec<&String>| -> Result<Option<String>, String> {
            match found.as_slice() {
                [] => Ok(None),
                [one] => Ok(Some((*one).clone())),
                many => Err(format!(
                    "\"{relative}\" is in more than one of this chat's folders: {}. Give the full path of the one you mean.",
                    many.iter().map(|m| m.as_str()).collect::<Vec<_>>().join(", ")
                )),
            }
        };
        if let Some(found) = pick(candidates.iter().filter(|c| exists(c)).collect())? {
            return Ok(Some(found));
        }
        let parent_exists = |c: &&String| {
            std::path::Path::new(c.as_str())
                .parent()
                .and_then(|p| p.to_str())
                .is_some_and(exists)
        };
        pick(candidates.iter().filter(parent_exists).collect())
    }

    /// The folder a file is in, and what else is in it that this chat may reach, by name
    /// (a folder's ends with "/"), in name order, hidden things left out. `None` when the
    /// file isn't there.
    pub fn beside(&self, path: &str) -> Option<(String, Vec<String>)> {
        let full = self.resolve(path).ok()?;
        if !exists(&full) || is_dir(&full) {
            return None;
        }
        let folder = parent(&full);
        let mut names: Vec<String> = fs::read_dir(&folder)
            .ok()?
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let path = join(&folder, &name);
                (!name.starts_with('.') && path != full && self.reachable(&path))
                    .then(|| if is_dir(&path) { format!("{name}/") } else { name })
            })
            .collect();
        names.sort();
        Some((folder, names))
    }

    /// Whether a path this chat may reach is a folder (`false`: a file); `None` when it
    /// isn't there or isn't the chat's to reach.
    pub fn is_folder(&self, path: &str) -> Option<bool> {
        let full = self.resolve(path).ok()?;
        exists(&full).then(|| is_dir(&full))
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
        if WORKBOOKS.contains(&ext.as_str()) {
            return Ok(workbook(&file, meta.len())?.describe());
        }
        if UNREAD_DOCUMENTS.contains(&ext.as_str()) {
            return Err(format!(
                "{name} is a document this {MACHINE}'s tools can't read yet. read_file can read text files, PDFs, Word documents and Excel workbooks."
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
                "{name} is not a text file. read_file can read text files, PDFs, Word documents and Excel workbooks."
            )
        })?;
        Ok(text
            .strip_prefix('\u{FEFF}')
            .map(str::to_string)
            .unwrap_or(text))
    }

    /// An Excel workbook's sheet names and the cells of its first sheet.
    fn read_spreadsheet(&self, path: &str) -> Result<String, String> {
        let file = self.resolve(path)?;
        let Ok(meta) = fs::metadata(&file) else {
            return Err(format!("File not found: {file}"));
        };
        let name = file_name(&file);
        let ext = Path::new(&file)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if meta.is_dir() || !WORKBOOKS.contains(&ext.as_str()) {
            return Err(format!(
                "{name} isn't an Excel workbook (.xlsx). Use read_file for other files."
            ));
        }
        Ok(workbook(&file, meta.len())?.read())
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
                        results.push(format!("{full}{SEP} (folder name matches)"));
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

/// A folder as Windows spells it in full, its 8.3 short names (`RUNNER~1`) expanded, as
/// Git and the file dialogs give paths: so a path Git reports is found within the home
/// folder or a chosen one. Links and junctions are left as they are. Unchanged if it isn't
/// there, and everywhere but Windows.
#[cfg(windows)]
fn long_form(path: &str) -> String {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLongPathNameW(short: *const u16, long: *mut u16, len: u32) -> u32;
    }
    if !path.contains('~') {
        return path.to_string();
    }
    let wide: Vec<u16> = path.encode_utf16().chain([0]).collect();
    let mut buffer = vec![0u16; 1024];
    let n = unsafe { GetLongPathNameW(wide.as_ptr(), buffer.as_mut_ptr(), buffer.len() as u32) } as usize;
    if n == 0 || n >= buffer.len() {
        return path.to_string();
    }
    String::from_utf16_lossy(&buffer[..n])
}

#[cfg(not(windows))]
fn long_form(path: &str) -> String {
    path.to_string()
}

/// Runs Git with its output and errors together, giving up after `GIT_TIMEOUT`.
pub fn git(arguments: &[&str], folder: &str) -> Result<(i32, String), String> {
    let Some(git) = git_program() else {
        return Err(if cfg!(target_os = "macos") {
            "Git isn't installed on this Mac. Installing Xcode's command line tools adds it."
                .into()
        } else {
            "Git isn't installed on this PC, or isn't on its PATH.".into()
        });
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

/// Where Git is: on the Mac, where Xcode's tools or Homebrew put it; elsewhere, on the PATH
/// (and Git for Windows' own folder).
fn git_program() -> Option<String> {
    if cfg!(target_os = "macos") {
        return [
            "/usr/bin/git",
            "/opt/homebrew/bin/git",
            "/usr/local/bin/git",
        ]
        .into_iter()
        .find(|p| Path::new(p).is_file())
        .map(str::to_string);
    }
    let name = if cfg!(windows) { "git.exe" } else { "git" };
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .chain(
            cfg!(windows)
                .then(|| std::path::PathBuf::from(r"C:\Program Files\Git\cmd\git.exe")),
        )
        .find(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned())
}

// ---- Helpers ----------------------------------------------------------------------------

/// Whether a path names a place by itself: `/…` on the Mac; on Windows a drive (`C:\…`)
/// or a network share (`\\server\share\…`).
pub fn is_absolute(path: &str) -> bool {
    if cfg!(windows) {
        let b = path.as_bytes();
        (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':')
            || path.starts_with("\\\\")
            || path.starts_with("//")
    } else {
        path.starts_with('/')
    }
}

/// A path with `.` and `..` worked out and no doubled or trailing separator. On Windows,
/// `/` read as `\`, and the drive letter upper case (`C:\Users\me`).
#[cfg(windows)]
pub fn standardized(path: &str) -> String {
    let p = path.replace('/', "\\");
    let b = p.as_bytes();
    let (prefix, rest): (String, &str) =
        if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
            (format!("{}:\\", p[..1].to_uppercase()), &p[2..])
        } else if let Some(unc) = p.strip_prefix("\\\\") {
            let mut parts = unc.splitn(3, '\\');
            let (server, share) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
            (format!("\\\\{server}\\{share}\\"), parts.next().unwrap_or(""))
        } else {
            (String::new(), p.as_str())
        };
    let absolute = !prefix.is_empty();
    let mut parts: Vec<&str> = Vec::new();
    for part in rest.split('\\') {
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
    let joined = parts.join("\\");
    if absolute {
        if joined.is_empty() {
            prefix
        } else {
            format!("{prefix}{joined}")
        }
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// A path with `.` and `..` worked out, no doubled or trailing `/`, and `/private` left off
/// where the rest names the same place (`/var`, `/tmp`), as the Mac's
/// `standardizingPath` writes it.
#[cfg(not(windows))]
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
    // /private/tmp is /tmp, and so on. Whether or not the path itself is there yet: a
    // file about to be written in a chosen folder must come out inside that folder.
    if let Some(rest) = result.strip_prefix("/private/")
        && let Some(d) = ["var", "tmp", "etc"]
            .iter()
            .find(|d| rest == **d || rest.starts_with(&format!("{d}/")))
        && Path::new(&format!("/{d}")).exists()
    {
        return format!("/{rest}");
    }
    result
}

/// The Mac's disks are case-insensitive by default, and Windows' always are, so paths are
/// compared that way.
pub fn is_within(path: &str, folder: &str) -> bool {
    let p = path.to_lowercase();
    let f = folder.to_lowercase();
    p == f
        || p.starts_with(&if f.ends_with(SEP) {
            f.clone()
        } else {
            format!("{f}{SEP}")
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
    if folder.ends_with(SEP) {
        format!("{folder}{name}")
    } else {
        format!("{folder}{SEP}{name}")
    }
}

fn parent(path: &str) -> String {
    match path.rfind(SEP) {
        // The root: "/", or a drive's "C:\".
        Some(i) if i == 0 || path[..i].ends_with(':') => path[..=i].into(),
        Some(i) => path[..i].into(),
        None => ".".into(),
    }
}

fn file_name(path: &str) -> String {
    path.rsplit([SEP, '/']).next().unwrap_or(path).to_string()
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
            format!("this {MACHINE} doesn't give permission to see it")
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
            Arg::Json(j) => j.clone(),
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

    /// To the user's Trash, as Finder's Move to Trash: it can be put back from there.
    pub fn trash(path: &str) -> bool {
        autoreleasepool(|_| {
            let class = AnyClass::get(c"NSFileManager")?;
            let url = url(path);
            unsafe {
                let manager: Option<Retained<AnyObject>> = msg_send![class, defaultManager];
                let done: bool = msg_send![
                    &*manager?,
                    trashItemAtURL: &*url,
                    resultingItemURL: std::ptr::null_mut::<*mut AnyObject>(),
                    error: std::ptr::null_mut::<*mut AnyObject>()
                ];
                Some(done)
            }
        })
        .unwrap_or(false)
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

/// Off the Mac, PDFs and Word documents through the server's own readers (`attach/`, the
/// ones attachments use); RTF and OpenDocument aren't read yet.
#[cfg(not(target_os = "macos"))]
mod documents {
    pub fn pdf_text(path: &str) -> Option<String> {
        let bytes = std::fs::read(path).ok()?;
        crate::attach::pdf::text(&bytes).ok()
    }
    /// The server makes no changes on this machine: the app does, to its Recycle Bin.
    pub fn trash(_path: &str) -> bool {
        false
    }
    pub fn rich_text(path: &str) -> Option<String> {
        if !path.to_lowercase().ends_with(".docx") {
            return None;
        }
        let bytes = std::fs::read(path).ok()?;
        crate::attach::docx::text(&bytes).ok()
    }
}

#[cfg(all(test, not(windows)))]
mod tests {
    //! `LocalToolsTests` from the app, where these tools were before, on a home folder made
    //! for each test. Written in the Mac's paths; `windows_tests` has Windows'.
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
        Files::new(&home.0, access).with_trash(&format!("{}/.test-trash", home.0))
    }

    #[test]
    fn an_edit_replaces_the_text_named_once_and_keeps_the_old_file() {
        let h = home();
        make(&h, "Documents/todo.md", "- milk\n- eggs\n- bread\n");
        make(&h, "Documents/win.txt", "one\r\ntwo\r\nthree\r\n");
        let t = tools(&h, Access::Home);
        let home = &t.home;
        let said = run(&t, "edit_file", &[("path", "Documents/todo.md"), ("old_text", "- eggs\n"), ("new_text", "- eggs (a dozen)\n- butter\n")]);
        assert_eq!(said, format!("File edited successfully: {home}/Documents/todo.md (1 line replaced with 2 lines)."));
        assert_eq!(fs::read_to_string(format!("{home}/Documents/todo.md")).unwrap(), "- milk\n- eggs (a dozen)\n- butter\n- bread\n");
        assert_eq!(fs::read_to_string(format!("{home}/.test-trash/todo.md")).unwrap(), "- milk\n- eggs\n- bread\n");
        // Windows line ends are matched and kept.
        run(&t, "edit_file", &[("path", "Documents/win.txt"), ("old_text", "two\nthree"), ("new_text", "2\n3")]);
        assert_eq!(fs::read_to_string(format!("{home}/Documents/win.txt")).unwrap(), "one\r\n2\r\n3\r\n");
        // Text that isn't there, or is there twice, is refused before anyone approves it.
        let missing = t.check(&call("edit_file", &[("path", "Documents/todo.md"), ("old_text", "- cheese"), ("new_text", "")]));
        assert!(missing.unwrap_err().contains("isn't in"));
        // The whole list as new_text for one line of it would leave the lines around it twice.
        let whole = failure(&t, "edit_file", &[("path", "Documents/todo.md"), ("old_text", "- butter"), ("new_text", "- milk\n- eggs (a dozen)\n- cheese\n- bread")]);
        assert!(whole.contains("repeats \"- eggs (a dozen)\""), "{whole}");
        let twice = failure(&t, "edit_file", &[("path", "Documents/todo.md"), ("old_text", "- "), ("new_text", "* ")]);
        assert!(twice.contains("4 times"), "{twice}");
        let none = failure(&t, "edit_file", &[("path", "Documents/none.md"), ("old_text", "x"), ("new_text", "y")]);
        assert!(none.contains("use write_file"), "{none}");
    }

    #[test]
    fn a_pdf_is_saved_with_its_extension() {
        let h = home();
        let t = tools(&h, Access::Home);
        let home = &t.home;
        let said = run(&t, "create_pdf", &[("path", "Documents/plan"), ("title", "Plan"), ("content", "Go \u{2192} there.")]);
        assert!(said.starts_with(&format!("PDF saved: {home}/Documents/plan.pdf")), "{said}");
        assert!(said.contains("\u{2192}"));
        assert!(fs::read(format!("{home}/Documents/plan.pdf")).unwrap().starts_with(b"%PDF-1.4"));
    }

    #[test]
    fn a_shell_command_runs_from_home_and_fails_with_its_output() {
        let h = home();
        let t = tools(&h, Access::Home);
        let home = &t.home;
        assert_eq!(run(&t, "run_shell_command", &[("command", "pwd -P")]), fs::canonicalize(home).unwrap().to_string_lossy());
        let failed = failure(&t, "run_shell_command", &[("command", "echo nope >&2; exit 3")]);
        assert_eq!(failed, "The command exited with status 3: nope");
        assert_eq!(run(&t, "run_shell_command", &[("command", "true")]), "The command finished with no output.");
        // Not from a chat that may reach only some folders.
        let chosen = tools(&h, Access::Folders(vec![format!("{home}/Documents")]));
        assert!(chosen.check(&call("run_shell_command", &[("command", "ls")])).unwrap_err().contains("only the folders"));
    }

    #[test]
    fn changes_are_checked_made_and_nothing_is_lost() {
        let h = home();
        make(&h, "Documents/notes.txt", "old");
        make(&h, "Documents/Invoices/acme.pdf", "pdf");
        let t = tools(&h, Access::Home);
        let home = &t.home;
        // A new file, its folder made on the way; then replaced, the old text kept.
        let made = run(&t, "write_file", &[("path", "Documents/New/plan.md"), ("content", "one")]);
        assert_eq!(made, format!("File written successfully: {home}/Documents/New/plan.md"));
        run(&t, "write_file", &[("path", "Documents/notes.txt"), ("content", "new")]);
        assert_eq!(fs::read_to_string(format!("{home}/Documents/notes.txt")).unwrap(), "new");
        assert_eq!(fs::read_to_string(format!("{home}/.test-trash/notes.txt")).unwrap(), "old");
        assert_eq!(
            failure(&t, "write_file", &[("path", "Documents"), ("content", "x")]),
            format!("{home}/Documents is a folder, not a file.")
        );
        // Renaming and moving: the name must be free and the folder there.
        assert_eq!(
            run(&t, "rename_file", &[("path", "Documents/notes.txt"), ("new_name", "notes-2026.txt")]),
            "Renamed successfully: notes.txt -> notes-2026.txt"
        );
        assert_eq!(
            failure(&t, "rename_file", &[("path", "Documents/notes-2026.txt"), ("new_name", "a/b.txt")]),
            "New name must be a single file or folder name."
        );
        assert_eq!(
            failure(&t, "move_file", &[("source", "Documents/notes-2026.txt"), ("destination", "Documents/Invoices/acme.pdf")]),
            format!("Destination already exists: {home}/Documents/Invoices/acme.pdf")
        );
        assert_eq!(
            failure(&t, "move_file", &[("source", "Documents/notes-2026.txt"), ("destination", "Documents/Nowhere/n.txt")]),
            format!("Folder not found: {home}/Documents/Nowhere")
        );
        assert_eq!(
            run(&t, "move_file", &[("source", "Documents/notes-2026.txt"), ("destination", "Documents/Invoices/notes.txt")]),
            format!("Moved successfully: {home}/Documents/notes-2026.txt -> {home}/Documents/Invoices/notes.txt")
        );
        // Deleting: files only, and into the Trash.
        assert!(failure(&t, "delete_file", &[("path", "Documents/Invoices")]).ends_with("delete_file deletes files only."));
        assert_eq!(
            failure(&t, "delete_file", &[("path", "Documents/gone.txt")]),
            format!("File not found: {home}/Documents/gone.txt")
        );
        assert!(run(&t, "delete_file", &[("path", "Documents/Invoices/acme.pdf")]).starts_with("File moved to the "));
        assert!(!exists(&format!("{home}/Documents/Invoices/acme.pdf")));
        assert_eq!(fs::read_to_string(format!("{home}/.test-trash/acme.pdf")).unwrap(), "pdf");
        // The check says what the change itself would, and changes nothing.
        let wrong = call("delete_file", &[("path", "Documents/gone.txt")]);
        assert_eq!(t.check(&wrong), Err(format!("File not found: {home}/Documents/gone.txt")));
        assert_eq!(t.check(&call("write_file", &[("path", "Documents/x.txt"), ("content", "x")])), Ok(()));
        assert!(!exists(&format!("{home}/Documents/x.txt")));
        // A workbook: made, added to, and its old version kept.
        let made = run(
            &t,
            "write_spreadsheet",
            &[("path", "Documents/register"), ("start_cell", "A1"), ("rows", r#"[["Invoice","CAD"],["NP-1",120.5]]"#)],
        );
        assert_eq!(
            made,
            format!("Spreadsheet written successfully: {home}/Documents/register.xlsx\nWrote 4 cells to Sheet1!A1:B2 in a new workbook.")
        );
        let added = run(
            &t,
            "write_spreadsheet",
            &[("path", "Documents/register.xlsx"), ("start_cell", "append"), ("rows", r#"[["NP-2",80]]"#)],
        );
        assert!(added.ends_with("Wrote 2 cells to Sheet1!A3:B3."), "{added}");
        let read = run(&t, "read_spreadsheet", &[("path", "Documents/register.xlsx")]);
        assert!(read.ends_with("1 | Invoice | CAD\n2 | NP-1 | 120.5\n3 | NP-2 | 80"), "{read}");
        assert!(exists(&format!("{home}/.test-trash/register.xlsx")));
        // Word and PowerPoint: made, and a template filled beside itself.
        let doc = run(&t, "create_docx", &[("path", "Documents/memo"), ("title", "Memo"), ("content", "To: {{name}}\n\n- one")]);
        assert_eq!(doc, format!("Word document saved: {home}/Documents/memo.docx"));
        assert!(run(&t, "read_file", &[("path", "Documents/memo.docx")]).contains("To: {{name}}"));
        let filled = run(
            &t,
            "fill_docx_template",
            &[("template", "Documents/memo.docx"), ("path", "memo-ada.docx"), ("values", r#"{"Name": "Ada", "extra": 1}"#)],
        );
        assert_eq!(
            filled,
            format!("Document saved successfully: {home}/Documents/memo-ada.docx\nFilled: name.\nNot in the template, so not used: extra.")
        );
        assert!(run(&t, "read_file", &[("path", "Documents/memo-ada.docx")]).contains("To: Ada"));
        assert_eq!(
            failure(&t, "fill_docx_template", &[("template", "Documents/memo.docx"), ("path", "memo.docx"), ("values", "{}")]),
            "Save the filled document under a new name, so the template stays as it is."
        );
        assert_eq!(
            failure(&t, "fill_docx_template", &[("template", "Documents/none.docx"), ("path", "x.docx"), ("values", "{}")]),
            format!("File not found: {home}/Documents/none.docx")
        );
        let deck = run(
            &t,
            "create_pptx",
            &[("path", "Documents/deck.pptx"), ("title", "Forum"), ("slides", r#"[{"title":"Goals","bullets":["One"]},{"title":"Ask"}]"#)],
        );
        assert_eq!(
            deck,
            format!("Slide deck saved successfully: {home}/Documents/deck.pptx\n3 slides: the title slide, then Goals, Ask.")
        );
        // Outside what the chat may reach, nothing is changed.
        let chosen = tools(&h, Access::Folders(vec![format!("{home}/Documents/New")]));
        assert!(failure(&chosen, "write_file", &[("path", "Documents/Invoices/x.txt"), ("content", "x")]).contains("outside the folders"));
        assert!(failure(&chosen, "move_file", &[("source", "Documents/New/plan.md"), ("destination", "Documents/plan.md")]).contains("outside the folders"));
    }

    #[test]
    fn a_file_knows_what_is_beside_it() {
        let h = home();
        let home = h.0.clone();
        make(&h, "Trip/itinerary.eml", "flights");
        make(&h, "Trip/agenda.eml", "agenda");
        make(&h, "Trip/.hidden", "x");
        make(&h, "Trip/Receipts/taxi.pdf", "pdf");
        let t = tools(&h, Access::Folders(vec![format!("{home}/Trip")]));
        assert_eq!(
            t.beside("itinerary.eml"),
            Some((format!("{home}/Trip"), vec!["Receipts/".to_string(), "agenda.eml".to_string()]))
        );
        assert_eq!(t.beside("Trip"), None);
        assert_eq!(t.beside("Trip/none.eml"), None);
    }

    #[test]
    fn a_relative_path_is_looked_for_inside_the_chosen_folders() {
        let h = home();
        let home = h.0.clone();
        let work = format!("{home}/Work/Aurelia");
        fs::create_dir_all(format!("{work}/Admin")).unwrap();
        fs::write(format!("{work}/Admin/form.md"), "the form").unwrap();
        let t = tools(&h, Access::Folders(vec![work.clone()]));
        // As the user names it, from the folder they work in; the home folder's way still works.
        assert_eq!(run(&t, "read_file", &[("path", "Admin/form.md")]), "the form");
        assert_eq!(run(&t, "read_file", &[("path", "Work/Aurelia/Admin/form.md")]), "the form");
        assert_eq!(run(&t, "list_directory", &[("path", "Admin")]), "[file] form.md");
        // A new file goes where its folder is; one whose folder is nowhere stays outside.
        assert_eq!(t.resolve("Admin/new.md"), Ok(format!("{work}/Admin/new.md")));
        assert!(t.resolve("Elsewhere/new.md").unwrap_err().contains("outside the folders"));
        // No way out of the chosen folder by a relative path.
        assert!(t.resolve("Admin/../../../Documents/x.md").unwrap_err().contains("outside the folders"));
        // In two chosen folders: the model is told, not given one of them.
        let other = format!("{home}/Work/Other");
        fs::create_dir_all(format!("{other}/Admin")).unwrap();
        fs::write(format!("{other}/Admin/form.md"), "another").unwrap();
        let both = tools(&h, Access::Folders(vec![work, other]));
        assert!(both.resolve("Admin/form.md").unwrap_err().contains("more than one"));
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
        make(&h, "Documents/budget.pptx", "PK");
        make(&h, "Documents/photo.bin", "");
        fs::write(
            format!("{}/Documents/photo.bin", h.0),
            [0xff, 0xfe, 0x00, 0x80],
        )
        .unwrap();
        let t = tools(&h, Access::Home);
        let deck = failure(&t, "read_file", &[("path", "Documents/budget.pptx")]);
        assert!(deck.contains("can't read yet"), "{deck}");
        make(&h, "Documents/broken.xlsx", "PK");
        let broken = failure(&t, "read_file", &[("path", "Documents/broken.xlsx")]);
        assert!(broken.starts_with("Couldn't read broken.xlsx:"), "{broken}");
        let notes = failure(&t, "read_spreadsheet", &[("path", "Documents/budget.pptx")]);
        assert!(notes.contains("isn't an Excel workbook"), "{notes}");
        let binary = failure(&t, "read_file", &[("path", "Documents/photo.bin")]);
        assert_eq!(
            binary,
            "photo.bin is not a text file. read_file can read text files, PDFs, Word documents and Excel workbooks."
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
                "run_tests",
                &[("path", "Documents/notes.txt")]
            ),
            "run_tests can't run on this Mac."
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

#[cfg(all(test, windows))]
mod windows_tests {
    //! The tools on Windows: its paths, its home folder without AppData, and documents read
    //! with the server's own readers.
    use super::*;

    struct Home(String);

    impl Drop for Home {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn home() -> Home {
        let base = std::env::temp_dir().join(format!("FileTools-{:016x}", crate::random_u64()));
        let home = Home(standardized(&base.to_string_lossy()));
        for (file, text) in [
            (r"Documents\notes.txt", "The code word for Friday is PELICAN."),
            (r"Documents\Invoices\acme-november.txt", "Acme, CAD 1,200"),
            (r"Projects\garden\plan.md", "Plant tomatoes in May."),
            (r"AppData\Local\secret.txt", "secret"),
        ] {
            let path = join(&home.0, file);
            fs::create_dir_all(parent(&path)).unwrap();
            fs::write(path, text).unwrap();
        }
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), r"\tests\fixtures");
        for name in ["note.pdf", "report.docx"] {
            fs::copy(
                format!(r"{fixtures}\{name}"),
                join(&home.0, &format!(r"Documents\{name}")),
            )
            .unwrap();
        }
        home
    }

    fn run(files: &Files, tool: &str, args: &[(&str, &str)]) -> Result<String, String> {
        files.run(&Call {
            tool: tool.into(),
            args: args
                .iter()
                .map(|(k, v)| (k.to_string(), Arg::Text(v.to_string())))
                .collect(),
        })
    }

    #[test]
    fn paths_are_windows_paths() {
        assert_eq!(standardized(r"c:\Users\me\.\Documents\..\Desktop\"), r"C:\Users\me\Desktop");
        assert_eq!(standardized("C:/Users/me/Documents"), r"C:\Users\me\Documents");
        assert_eq!(standardized(r"C:\"), r"C:\");
        assert_eq!(standardized(r"C:\.."), r"C:\");
        assert_eq!(standardized(r"\\server\share\a\..\b"), r"\\server\share\b");
        assert!(is_absolute(r"D:\x") && is_absolute(r"\\server\share") && !is_absolute(r"Documents\x"));
        assert!(is_within(r"C:\Users\Me\documents", r"c:\users\me"));
        assert!(!is_within(r"C:\Users\meg", r"C:\Users\me"));
        assert!(is_within(r"C:\anything", r"C:\"));
        assert_eq!(parent(r"C:\Users\me"), r"C:\Users");
        assert_eq!(parent(r"C:\Users"), r"C:\");
        assert_eq!(file_name(r"C:\Users\me\notes.txt"), "notes.txt");
        assert_eq!(join(r"C:\", "Users"), r"C:\Users");
    }

    #[test]
    fn the_home_folder_without_appdata() {
        let h = home();
        let t = Files::new(&h.0, Access::Home);
        let notes = join(&h.0, r"Documents\notes.txt");
        // Relative, with either slash, from ~, and in full.
        for path in [r"Documents\notes.txt", "Documents/notes.txt", r"~\Documents\notes.txt", &notes] {
            assert_eq!(t.resolve(path), Ok(notes.clone()), "{path}");
        }
        assert!(t.resolve(r"AppData\Local\secret.txt").unwrap_err().contains("AppData"));
        assert!(t.resolve(r"C:\Windows\win.ini").unwrap_err().contains("outside the user's folder"));
        assert_eq!(
            run(&t, "read_file", &[("path", r"Documents\notes.txt")]),
            Ok("The code word for Friday is PELICAN.".into())
        );
        assert_eq!(
            run(&t, "list_directory", &[("path", "Documents")]),
            Ok("[dir]  Invoices\n[file] note.pdf\n[file] notes.txt\n[file] report.docx".into())
        );
        // Listing the home folder leaves AppData's insides out of a deep listing.
        let deep = run(&t, "list_directory", &[("path", "~"), ("depth", "3")]).unwrap();
        assert!(!deep.contains("secret.txt"), "{deep}");
        let found = run(&t, "search_files", &[("pattern", "invoices"), ("directory", "~")]).unwrap();
        assert!(
            found.contains(&format!(r"{}\Documents\Invoices\ (folder name matches)", h.0)),
            "{found}"
        );
        let found = run(&t, "search_files", &[("pattern", "tomatoes"), ("directory", "Projects")]).unwrap();
        assert!(found.contains(r"garden\plan.md:1: Plant tomatoes in May."), "{found}");
        assert_eq!(
            run(&t, "read_file", &[("path", r"Documents\nowhere.txt")]),
            Err(format!(r"File not found: {}\Documents\nowhere.txt", h.0))
        );
    }

    #[test]
    fn pdfs_and_word_documents_are_read_with_the_servers_readers() {
        let h = home();
        let t = Files::new(&h.0, Access::Home);
        let pdf = run(&t, "read_file", &[("path", r"Documents\note.pdf")]).unwrap();
        assert!(pdf.contains("The launch code is BLUE HERON 42."), "{pdf}");
        let docx = run(&t, "read_file", &[("path", r"Documents\report.docx")]).unwrap();
        assert!(docx.starts_with("Quarterly report\n"), "{docx}");
    }

    #[test]
    fn git_is_found_on_the_path() {
        if git_program().is_none() {
            return; // no Git on this PC
        }
        let h = home();
        let project = join(&h.0, r"Projects\garden");
        git(&["init", "-q"], &project).unwrap();
        let t = Files::new(&h.0, Access::Home);
        let status = run(&t, "git_status", &[("path", r"Projects\garden")]).unwrap();
        assert!(status.contains("plan.md"), "{status}");
        // The project as the tools spell it: the temporary folder's short names expanded.
        let project = join(&t.home, r"Projects\garden");
        assert!(status.contains(&crate::json::string(&project)), "{status}");
    }
}
