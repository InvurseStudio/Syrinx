//! Pages, Numbers and Keynote files, as Vireo 0.9 read them (`IWorkService.swift`): only
//! the apps read their own format reliably, so the app exports a copy as Word, Excel or
//! PowerPoint into a temporary folder, the copy is read with the Office readers, and the
//! folder is deleted. A document the user already has open is exported as it is, unsaved
//! changes included, and left open; one opened here is closed unsaved. An app this starts
//! is started hidden and quit afterwards; one the user had open is left running. macOS
//! asks once per app before Vireo may control it (Automation). On the Mac only.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct App {
    pub name: &'static str,
    bundle_id: &'static str,
    /// The export format's name in the app's scripting dictionary.
    format: &'static str,
    /// The copy's extension, and what it's called.
    pub extension: &'static str,
    pub office: &'static str,
    /// Added to the export command: Numbers' summary sheet is left out.
    options: &'static str,
}

const PAGES: App = App { name: "Pages", bundle_id: "com.apple.Pages", format: "Microsoft Word", extension: "docx", office: "Word", options: "" };
const NUMBERS: App = App {
    name: "Numbers",
    bundle_id: "com.apple.Numbers",
    format: "Microsoft Excel",
    extension: "xlsx",
    office: "Excel",
    options: " with properties {exclude summary worksheet:true}",
};
const KEYNOTE: App = App { name: "Keynote", bundle_id: "com.apple.Keynote", format: "Microsoft PowerPoint", extension: "pptx", office: "PowerPoint", options: "" };

/// Long enough for a large deck; an app stuck on a dialog is given up on.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Each app does one export at a time: otherwise the first to finish quits an app Vireo
/// started while another export is still using it.
static ONE_AT_A_TIME: [Mutex<()>; 3] = [Mutex::new(()), Mutex::new(()), Mutex::new(())];

/// The app for an extension (`pages`, `numbers`, `key`).
pub fn app_for(extension: &str) -> Option<&'static App> {
    match extension.to_ascii_lowercase().as_str() {
        "pages" => Some(&PAGES),
        "numbers" => Some(&NUMBERS),
        "key" => Some(&KEYNOTE),
        _ => None,
    }
}

/// An exported copy; its temporary folder is deleted when this is dropped.
pub struct Copy {
    pub path: PathBuf,
    folder: PathBuf,
}

impl Drop for Copy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.folder);
    }
}

/// Whether a document has a password: iWork keeps a password verifier file in it.
fn locked(path: &Path) -> bool {
    if path.is_dir() {
        return path.join(".iwpv2").exists();
    }
    let Ok(data) = std::fs::read(path) else { return false };
    crate::attach::docx::Zip::open(&data).is_ok_and(|zip| zip.names().any(|n| n.ends_with(".iwpv2")))
}

/// Has the app export `path` as an Office file.
pub fn export(path: &Path) -> Result<Copy, String> {
    let file_name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let extension = path.extension().map_or_else(String::new, |e| e.to_string_lossy().into_owned());
    let (index, app) = match extension.to_ascii_lowercase().as_str() {
        "pages" => (0, &PAGES),
        "numbers" => (1, &NUMBERS),
        "key" => (2, &KEYNOTE),
        _ => return Err(format!("{file_name} isn't a Pages, Numbers or Keynote file.")),
    };
    if !cfg!(target_os = "macos") {
        return Err(format!("{file_name} needs {} on a Mac to read it.", app.name));
    }
    if !path.exists() {
        return Err(format!("File not found: {}", path.display()));
    }
    if locked(path) {
        return Err(format!(
            "{file_name} is locked with a password, so it can't be read. The user can open it in {} \
             and remove the password, or export a copy.",
            app.name
        ));
    }
    if !installed(app) {
        return Err(format!(
            "{} isn't installed on this Mac, so {file_name} can't be read. It's free in the App Store.",
            app.name
        ));
    }
    let _one = ONE_AT_A_TIME[index].lock().unwrap_or_else(|e| e.into_inner());
    let folder = std::env::temp_dir().join(format!("vireo-iwork-{:016x}", crate::random_u64()));
    std::fs::create_dir_all(&folder).map_err(|e| format!("Couldn't make a folder for the copy: {e}"))?;
    let stem = path.file_stem().map_or_else(|| "document".into(), |s| s.to_string_lossy().into_owned());
    let copy = Copy { path: folder.join(format!("{stem}.{}", app.extension)), folder };

    let was_running = running(app);
    // As the app reports it: symlinks resolved (/var is /private/var), or it isn't found.
    let input = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().into_owned();
    // The apps are sandboxed: AppleScript's `open` can't reach a file the user didn't hand
    // them ("Operation not permitted"), but opening it through Launch Services, as Finder
    // does, gives the app the file. In the background and hidden, so it doesn't take over
    // the screen. A document the user already has open is used as it is.
    let already_open = was_running && is_open(app, &input);
    if !already_open {
        let _ = Command::new("/usr/bin/open")
            .args(["-g", "-j", "-b", app.bundle_id, &input])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let opened_here = if already_open { "false" } else { "true" };
    let result = run(&script(app), &[&input, &copy.path.to_string_lossy(), opened_here], app, &file_name);
    if !was_running {
        // Quit an app Vireo started, once it's done; never one the user had open.
        let _ = Command::new("/usr/bin/osascript")
            .args(["-e", &format!("tell application id \"{}\" to quit", app.bundle_id)])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    result?;
    if !copy.path.is_file() {
        return Err(format!("{} didn't export {file_name}.", app.name));
    }
    Ok(copy)
}

fn installed(app: &App) -> bool {
    let wanted = format!("kMDItemCFBundleIdentifier == '{}'", app.bundle_id);
    let found = Command::new("/usr/bin/mdfind")
        .arg(&wanted)
        .stderr(Stdio::null())
        .output()
        .is_ok_and(|o| !o.stdout.trim_ascii().is_empty());
    found
        || Path::new(&format!("/Applications/{}.app", app.name)).exists()
        || Path::new(&format!("/System/Applications/{}.app", app.name)).exists()
}

/// Whether the user has `path` open in the app.
fn is_open(app: &App, path: &str) -> bool {
    let check = format!(
        r#"on run argv
    tell application id "{}"
        repeat with d in documents
            try
                set openPath to POSIX path of (file of d as alias)
                if openPath is (item 1 of argv) or openPath is ((item 1 of argv) & "/") then return "yes"
            end try
        end repeat
    end tell
    return "no"
end run"#,
        app.bundle_id
    );
    let Ok(mut child) = Command::new("/usr/bin/osascript")
        .args(["-", path])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    if let Some(mut input) = child.stdin.take() {
        let _ = input.write_all(check.as_bytes());
    }
    child.wait_with_output().is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "yes")
}

fn running(app: &App) -> bool {
    Command::new("/usr/bin/osascript")
        .args(["-e", &format!("application id \"{}\" is running", app.bundle_id)])
        .stderr(Stdio::null())
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "true")
}

/// Exports the document at argv 1, once the app has it open, to argv 2, and closes it
/// unsaved if argv 3 is "true" (it was opened for this). Paths come in as arguments, so
/// nothing in them is read as script.
fn script(app: &App) -> String {
    format!(
        r#"on run argv
    set inPath to item 1 of argv
    set outFile to POSIX file (item 2 of argv)
    set openedHere to ((item 3 of argv) is "true")
    tell application id "{bundle}"
        set theDoc to missing value
        repeat 60 times
            repeat with d in documents
                try
                    set openPath to POSIX path of (file of d as alias)
                    if openPath is inPath or openPath is (inPath & "/") then
                        set theDoc to contents of d
                        exit repeat
                    end if
                end try
            end repeat
            if theDoc is not missing value then exit repeat
            delay 0.5
        end repeat
        if theDoc is missing value then error "It didn't open the document." number -1728
        try
            export theDoc to outFile as {format}{options}
        on error errorText number errorNumber
            if openedHere then close theDoc saving no
            error errorText number errorNumber
        end try
        if openedHere then close theDoc saving no
    end tell
end run
"#,
        bundle = app.bundle_id,
        format = app.format,
        options = app.options
    )
}

/// Runs the script with osascript, giving up after `TIMEOUT`.
fn run(script: &str, arguments: &[&str], app: &App, file_name: &str) -> Result<(), String> {
    let mut child = Command::new("/usr/bin/osascript")
        .arg("-")
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Couldn't run osascript: {e}"))?;
    if let Some(mut input) = child.stdin.take() {
        let _ = input.write_all(script.as_bytes());
    }
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(status);
        }
        if started.elapsed() > TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut detail = String::new();
    if let Some(mut errors) = child.stderr.take() {
        use std::io::Read;
        let _ = errors.read_to_string(&mut detail);
    }
    match status {
        Some(s) if s.success() => Ok(()),
        Some(s) => Err(error_from(&detail, s.code().unwrap_or(-1), app, file_name)),
        None => Err(error_from("", 15, app, file_name)),
    }
}

/// osascript's error output, as the model can tell the user.
fn error_from(detail: &str, status: i32, app: &App, file_name: &str) -> String {
    let name = app.name;
    if detail.contains("-1743") {
        return format!(
            "Vireo isn't allowed to control {name}. The user can allow it in System Settings → \
             Privacy & Security → Automation → Vireo."
        );
    }
    if status == 15 || detail.contains("-1712") {
        return format!(
            "{name} took too long with {file_name}. It may be waiting on a dialog, such as a \
             password; the user can open the file in {name} to check."
        );
    }
    // "execution error: Pages got an error: The document can’t be opened. (-1)" → the part
    // after the app's name, without the number.
    let mut message = detail.trim();
    if let Some(at) = message.find("execution error: ") {
        message = &message[at + "execution error: ".len()..];
    }
    if let Some(at) = message.find("got an error: ") {
        message = &message[at + "got an error: ".len()..];
    }
    let message = match message.rfind(" (") {
        Some(at) if message.ends_with(')') => &message[..at],
        _ => message,
    };
    let message = if message.is_empty() { format!("error {status}") } else { message.to_string() };
    format!("{name} couldn't export {file_name}: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_said_plainly() {
        let pages = app_for("PAGES").unwrap();
        assert!(error_from("execution error: Not authorized (-1743)", 1, pages, "a.pages").contains("Automation"));
        assert!(error_from("", 15, pages, "a.pages").contains("took too long"));
        assert_eq!(
            error_from("0:12: execution error: Pages got an error: The document can’t be opened. (-1)", 1, pages, "a.pages"),
            "Pages couldn't export a.pages: The document can’t be opened."
        );
        assert!(app_for("docx").is_none());
    }

    #[test]
    fn the_script_names_the_app_and_format() {
        let s = script(app_for("numbers").unwrap());
        assert!(s.contains("tell application id \"com.apple.Numbers\""));
        assert!(s.contains("as Microsoft Excel with properties {exclude summary worksheet:true}"));
    }

    #[test]
    fn a_locked_or_missing_file_is_refused_before_any_app_opens() {
        let dir = std::env::temp_dir().join(format!("iwork-{:016x}", crate::random_u64()));
        std::fs::create_dir_all(dir.join("Locked.pages")).unwrap();
        std::fs::write(dir.join("Locked.pages/.iwpv2"), "").unwrap();
        let err = export(&dir.join("Locked.pages")).err().unwrap();
        assert!(err.contains("locked with a password"), "{err}");
        let err = export(&dir.join("Gone.key")).err().unwrap();
        assert!(err.starts_with("File not found"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
