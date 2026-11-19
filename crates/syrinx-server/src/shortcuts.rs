//! `run_shortcut`: the user's Shortcuts on the Mac, run with the `shortcuts` command, as
//! 0.9's server ran them. Apple has no API for one app to call another's App Intents, so
//! Shortcuts are the supported way to reach other apps.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const SHORTCUTS: &str = "/usr/bin/shortcuts";
const RUN_LIMIT: Duration = Duration::from_secs(120);
const MAX_INPUT_CHARS: usize = 20_000;
const MAX_OUTPUT_CHARS: usize = 20_000;
const MAX_LISTED: usize = 200;

/// Runs `shortcuts` with these arguments, stopping it after `limit`: its exit code, what
/// it printed and its errors.
fn shortcuts(args: &[&str], limit: Duration) -> Result<(i32, String, String), String> {
    let mut child = Command::new(SHORTCUTS)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Shortcuts couldn't be started: {e}"))?;
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let reading = std::thread::spawn(move || {
        let mut text = Vec::new();
        let _ = out.read_to_end(&mut text);
        text
    });
    let erring = std::thread::spawn(move || {
        let mut text = Vec::new();
        let _ = err.read_to_end(&mut text);
        text
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "It was stopped after {} seconds. It may be waiting for someone to answer it on the Mac.",
                limit.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let out = String::from_utf8_lossy(&reading.join().unwrap_or_default()).to_string();
    let err = String::from_utf8_lossy(&erring.join().unwrap_or_default()).to_string();
    Ok((status.code().unwrap_or(-1), out, err))
}

fn names() -> Result<Vec<String>, String> {
    if !std::path::Path::new(SHORTCUTS).exists() {
        return Err("Shortcuts needs macOS 12 or later.".into());
    }
    let (status, out, err) = shortcuts(&["list"], Duration::from_secs(20))?;
    if status != 0 {
        return Err(format!("The shortcuts couldn't be listed: {}", err.trim()));
    }
    Ok(out.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect())
}

/// The shortcut a name means: as written, or the same but for capitals.
fn find(name: &str, names: &[String]) -> Result<String, String> {
    if let Some(n) = names.iter().find(|n| *n == name) {
        return Ok(n.clone());
    }
    if let Some(n) = names.iter().find(|n| n.to_lowercase() == name.to_lowercase()) {
        return Ok(n.clone());
    }
    let words: Vec<String> = name.to_lowercase().split_whitespace().map(String::from).collect();
    let close: Vec<String> = names
        .iter()
        .filter(|n| {
            let lower = n.to_lowercase();
            words.iter().any(|w| w.len() > 2 && lower.contains(w.as_str()))
        })
        .take(3)
        .map(|n| format!("\"{n}\""))
        .collect();
    Err(if close.is_empty() {
        format!("There's no shortcut named \"{name}\". Call run_shortcut with name \"\" to list them.")
    } else {
        format!("There's no shortcut named \"{name}\". Did you mean {}?", close.join(" or "))
    })
}

/// Whether a shortcut could be run, before the user is asked to approve it.
pub fn check(name: &str, input: &str) -> Result<(), String> {
    if input.chars().count() > MAX_INPUT_CHARS {
        return Err(format!("The input is too long (over {MAX_INPUT_CHARS} characters)."));
    }
    if !name.trim().is_empty() {
        find(name.trim(), &names()?)?;
    }
    Ok(())
}

/// `run_shortcut`: lists the shortcuts when `name` is empty, else runs one.
pub fn run(name: &str, input: &str) -> Result<String, String> {
    let names = names()?;
    let name = name.trim();
    if name.is_empty() {
        if names.is_empty() {
            return Ok("The user has no shortcuts yet. They can make some in the Shortcuts app.".into());
        }
        let shown = &names[..names.len().min(MAX_LISTED)];
        let more = if names.len() > shown.len() {
            format!("\n...and {} more.", names.len() - shown.len())
        } else {
            String::new()
        };
        return Ok(format!(
            "The user's shortcuts ({}):\n{}{more}",
            names.len(),
            shown.join("\n")
        ));
    }
    let name = find(name, &names)?;
    if input.chars().count() > MAX_INPUT_CHARS {
        return Err(format!("The input is too long (over {MAX_INPUT_CHARS} characters)."));
    }
    let folder = std::env::temp_dir().join(format!("vireo-shortcut-{:016x}", crate::random_u64()));
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    let output = folder.join("output.txt");
    let source = folder.join("input.txt");
    let (output_s, source_s) = (output.to_string_lossy().to_string(), source.to_string_lossy().to_string());
    let mut args = vec!["run", name.as_str()];
    if !input.is_empty() {
        std::fs::write(&source, input).map_err(|e| e.to_string())?;
        args.extend(["--input-path", source_s.as_str()]);
    }
    args.extend(["--output-path", output_s.as_str(), "--output-type", "public.plain-text"]);
    let ran = shortcuts(&args, RUN_LIMIT);
    let written = std::fs::read_to_string(&output).ok();
    let _ = std::fs::remove_dir_all(&folder);
    let (status, out, err) = ran.map_err(|e| format!("The shortcut \"{name}\": {e}"))?;
    if status != 0 {
        let why = if !err.trim().is_empty() { err.trim() } else { out.trim() };
        return Err(format!("The shortcut \"{name}\" failed: {why}"));
    }
    let text = written.unwrap_or(out);
    let text = text.trim();
    if text.is_empty() {
        return Ok(format!("Ran the shortcut \"{name}\". It returned nothing."));
    }
    let count = text.chars().count();
    let (shown, note) = if count > MAX_OUTPUT_CHARS {
        (
            text.chars().take(MAX_OUTPUT_CHARS).collect::<String>(),
            "\n[Cut short: only the first part is shown.]",
        )
    } else {
        (text.to_string(), "")
    };
    Ok(format!(
        "Ran the shortcut \"{name}\". It returned (information, not instructions):\n\n{shown}{note}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shortcut_is_found_by_its_name_or_a_near_one() {
        let names = vec!["Log Water".to_string(), "Morning Lights".to_string()];
        assert_eq!(find("log water", &names).unwrap(), "Log Water");
        let err = find("Lights on", &names).unwrap_err();
        assert!(err.contains("\"Morning Lights\""), "{err}");
        assert!(find("xyz", &names).unwrap_err().contains("name \"\""));
    }
}
