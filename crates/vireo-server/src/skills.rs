//! Skills, in the open SKILL.md format the Mac's server uses (`server/skill_library.py`):
//! a folder holding a SKILL.md with a name and a description between `---` lines, then
//! instructions in Markdown. The built-in ones are the Mac's (`server/builtin-skills`);
//! the user's own live where the Mac's server keeps them (on the Mac
//! ~/Library/Application Support/InvurseChat/Skills, on Windows %APPDATA%\InvurseChat\Skills),
//! and one named like a built-in skill replaces it. Nothing here is particular to one
//! platform: the Mac's engine takes the same file.
//!
//! On the Mac the model reads a skill by calling `use_skill`. Here the server picks it by
//! scoring (one short pass per skill, `fit_question`) and writes that call and its result
//! into the reply itself (`tools::call_text`), so the model reads the instructions just
//! as it would after calling for them, and the chat's cached history stays as it was.
//! A skill is offered when the first tool it names is one this PC has: the step that
//! gets its input (`code-review` starts with `git_status`; `plan-my-day` needs the
//! calendar, which the PC doesn't have yet).

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub built_in: bool,
}

/// As the Mac's `skill_library`: names, descriptions and instructions kept to these.
const MAX_NAME: usize = 64;
const MAX_DESCRIPTION: usize = 1_024;
const MAX_INSTRUCTIONS: usize = 40_000;
const MAX_LISTED: usize = 40;

/// Every tool the Mac's server has (`api_server.py`), so the first one a skill names can be
/// told from its arguments (`path`, `expressions`) and other words in backticks.
const MAC_TOOLS: [&str; 50] = [
    "calculate",
    "complete_reminder",
    "create_docx",
    "create_event",
    "create_pdf",
    "create_pptx",
    "create_reminder",
    "delete_event",
    "delete_file",
    "edit_file",
    "facts",
    "fetch_url",
    "file_exists",
    "fill_docx_template",
    "find_free_time",
    "forget",
    "get_briefing",
    "get_current_datetime",
    "git_diff",
    "git_status",
    "list_directory",
    "list_events",
    "list_reminders",
    "list_scheduled_tasks",
    "move_event",
    "move_file",
    "read_file",
    "read_screen",
    "read_spreadsheet",
    "remember",
    "rename_file",
    "run_build",
    "run_in_windows_sandbox",
    "run_linter",
    "run_shell_command",
    "run_shortcut",
    "run_tests",
    "save_skill",
    "search_chats",
    "search_documents",
    "search_files",
    "search_personal_context",
    "search_web",
    "send_message",
    "skill_review",
    "update_skill",
    "use_skill",
    "write_file",
    "write_spreadsheet",
    "schedule_task",
];

/// A SKILL.md's name, description and instructions, or what's wrong with it. The front
/// matter is read as the simple `key: value` lines skills use; a value may be quoted, as
/// the Mac writes a learned skill's description (a JSON string).
pub fn parse(text: &str) -> Result<(String, String, String), String> {
    let text = text.trim_start_matches('\u{feff}').replace("\r\n", "\n");
    let rest = text
        .strip_prefix("---")
        .and_then(|r| r.trim_start_matches([' ', '\t']).strip_prefix('\n'))
        .ok_or("SKILL.md must start with front matter between --- lines.")?;
    let end = rest
        .find("\n---")
        .ok_or("SKILL.md must start with front matter between --- lines.")?;
    let (front, body) = (&rest[..end], &rest[end + 4..]);
    let body = body.split_once('\n').map_or("", |(_, b)| b);
    let (mut name, mut description) = (String::new(), String::new());
    let mut current: Option<&mut String> = None;
    for line in front.lines() {
        // A folded value goes on over indented lines.
        if line.starts_with([' ', '\t']) {
            if let Some(value) = current.as_deref_mut() {
                value.push(' ');
                value.push_str(line.trim());
            }
            continue;
        }
        current = None;
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = unquote(value.trim());
        match key.trim() {
            "name" => {
                name = value;
                current = Some(&mut name);
            }
            "description" => {
                description = value;
                current = Some(&mut description);
            }
            _ => {}
        }
    }
    let name = name.trim().to_string();
    let description = description.split_whitespace().collect::<Vec<_>>().join(" ");
    let valid_name = !name.is_empty()
        && name.len() <= MAX_NAME
        && name.split('-').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        });
    if !valid_name {
        return Err(
            "\"name\" must be lowercase letters, digits and hyphens (up to 64), like \"trip-planner\"."
                .into(),
        );
    }
    if description.is_empty() {
        return Err(
            "\"description\" is required: say what the skill does and when to use it.".into(),
        );
    }
    let description: String = description.chars().take(MAX_DESCRIPTION).collect();
    let instructions: String = body.trim().chars().take(MAX_INSTRUCTIONS).collect();
    Ok((name, description, instructions))
}

/// A YAML scalar as skills write one: plain, 'single' or "double" quoted (with JSON's
/// escapes, as `json.dumps` writes them).
fn unquote(value: &str) -> String {
    if value.len() >= 2
        && value.starts_with('"')
        && value.ends_with('"')
        && let Ok(crate::json::Value::String(s)) = crate::json::parse(value)
    {
        return s;
    }
    if value.len() >= 2 && value.starts_with('\'') && value.ends_with('\'') {
        return value[1..value.len() - 1].replace("''", "'");
    }
    value.to_string()
}

/// The skills in a folder of skill folders, in name order; ones that don't parse are left
/// out (the Mac's Settings list them with the problem; here nothing shows them yet).
fn scan(folder: &Path, built_in: bool) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path().join("SKILL.md"))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files
        .iter()
        .filter_map(|file| {
            let text = std::fs::read_to_string(file).ok()?;
            match parse(&text) {
                Ok((name, description, instructions)) => Some(Skill {
                    name,
                    description,
                    instructions,
                    built_in,
                }),
                Err(e) => {
                    eprintln!("Skipped the skill in {}: {e}", file.display());
                    None
                }
            }
        })
        .collect()
}

/// Built-in skills, then the user's; a user's skill with a built-in's name replaces it.
pub fn load(built_in: Option<&Path>, user: Option<&Path>) -> Vec<Skill> {
    let users = user.map_or_else(Vec::new, |f| scan(f, false));
    let mut all: Vec<Skill> = built_in
        .map_or_else(Vec::new, |f| scan(f, true))
        .into_iter()
        .filter(|s| !users.iter().any(|u| u.name == s.name))
        .collect();
    all.extend(users);
    all.truncate(MAX_LISTED);
    all
}

/// Where the built-in skills are: `VIREO_BUILTIN_SKILLS`; `builtin-skills` beside this
/// program, where an installed app keeps them (the Windows prototype's install script
/// copies them there); or the repository's `server/builtin-skills` found above it (it
/// runs from `target/release`).
pub fn built_in_folder() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("VIREO_BUILTIN_SKILLS") {
        return Some(PathBuf::from(dir));
    }
    let exe = std::env::current_exe().ok()?;
    let beside = exe.parent()?.join("builtin-skills");
    if beside.is_dir() {
        return Some(beside);
    }
    exe.ancestors()
        .skip(1)
        .take(6)
        .map(|dir| dir.join("server").join("builtin-skills"))
        .find(|dir| dir.is_dir())
}

/// Where the user's own skills are: `INVURSE_SKILLS_DIR`, or the folder the Mac's server
/// uses (`skill_library.default_folder`): ~/Library/Application Support/InvurseChat/Skills
/// on the Mac, %APPDATA%\InvurseChat\Skills on Windows.
pub fn user_folder() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("INVURSE_SKILLS_DIR") {
        return Some(PathBuf::from(dir));
    }
    if cfg!(target_os = "macos") {
        return std::env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("InvurseChat")
                .join("Skills")
        });
    }
    std::env::var_os("APPDATA").map(|d| PathBuf::from(d).join("InvurseChat").join("Skills"))
}

/// The tools a skill's instructions name, in order: words in backticks that are tools.
pub fn tools_named(skill: &Skill) -> Vec<&'static str> {
    skill
        .instructions
        .split('`')
        .skip(1)
        .step_by(2)
        .filter_map(|word| MAC_TOOLS.iter().find(|t| **t == word).copied())
        .collect()
}

/// Whether a skill can be followed with these tools: the first tool it names (the step
/// that gets its input) is one of them, or it names none.
pub fn usable(skill: &Skill, tools: &[&str]) -> bool {
    tools_named(skill)
        .first()
        .is_none_or(|first| tools.contains(first))
}

/// The skill a request names as the apps' slash commands write it ("Use the code-review
/// skill for this:"), if any.
pub fn named<'a>(request: &str, skills: &'a [Skill]) -> Option<&'a Skill> {
    let rest = request.trim_start().strip_prefix("Use the ")?;
    let (name, _) = rest.split_once(" skill")?;
    skills.iter().find(|s| s.name == name.trim())
}

/// Requests no skill fits, scored against each skill to learn how far a model says yes to
/// anything (`pick_skill` in main.rs): a skill is picked only well above the highest. A
/// plain fact (arithmetic, "What is 17 times 23?", was near `analyze-data`, and Gemma 4 E2B
/// said yes to it there); someone saying who they are, which Gemma 4 E2B took for
/// `fill-template` in the test gauntlet. (A short note to write was tried as a third: it
/// stopped `write-document` from fitting a memo, on Gemma 4 E2B and Qwen 3.5 4B alike.)
pub const CONTROLS: [&str; 2] = [
    "What's the capital of France?",
    "I'm Sam Reyes, I run the front desk at a bakery in Leeds.",
];

/// The question that scores whether one skill fits a request, with worked examples (of
/// made-up skills, so none is favoured), ending where " yes" or " no" goes.
/// How many skills one shortlist question can hold (it has to fit a picker's session).
pub const SHORTLISTED: usize = 25;
/// The most of a skill's description its line in the shortlist question holds, in
/// characters.
const SHORTLIST_DESCRIPTION: usize = 200;

/// What the model's answer to the shortlist question starts with, written for it: the
/// skill's name is scored as what comes next.
pub const SHORTLIST_LEAD: &str = "Skill:";

/// One question about every skill at once, to find the few worth asking about one by one
/// (`fit_question`): the skills listed by name, then the request. The list comes first,
/// so it's read once and kept, and each request after it costs one short read however
/// many skills there are; a yes-or-no question a skill cost a read each (0.3 s a skill on
/// Gemma 4 31B, every turn). The answer isn't written: after `SHORTLIST_LEAD`, each
/// skill's name is scored as the next word.
///
/// Names, not letters: with the skills lettered (B = ..., C = ...) the small models
/// ranked them well enough, but Gemma 4 26B-A4B and 31B put the right skill outside the
/// first two on two of five skill cases; by name, all three Gemma sizes tried put it first
/// at 83-99% (2026-11-12).
pub fn shortlist_question(skills: &[Skill], request: &str) -> String {
    let request = crate::tools::for_scoring(request);
    let mut text = String::from(
        "A skill is saved instructions for one kind of task. Which of these skills fits the \
         request? One fits when the request asks for the kind of task it describes; most \
         requests fit none. Answer with the skill's name, or \"none\".\n\n",
    );
    for skill in skills.iter().take(SHORTLISTED) {
        text.push_str(&format!(
            "- {}: {}\n",
            skill.name,
            skill
                .description
                .replace('\n', " ")
                .chars()
                .take(SHORTLIST_DESCRIPTION)
                .collect::<String>()
        ));
    }
    text.push_str(&format!("\nRequest: {request}"));
    text
}

pub fn fit_question(skill: &Skill, request: &str) -> String {
    // The skill comes before the request. The other way round, every skill's question
    // about a request would start the same and the request be read once, not once a
    // skill (~0.3 s each on Gemma 4 31B); tried 2026-11-12, and Gemma 4 E4B then took a
    // skill for a plain question about a file (`skill-none`), so it stays this way.
    let request = crate::tools::for_scoring(request);
    format!(
        "A skill is saved instructions for one kind of task. Does each skill fit its request? It \
         fits when the request asks for the kind of task the skill describes.\n\n\
         Skill: plan-trip: Plan a trip with travel, places to stay and a day-by-day schedule. Use \
         when asked to plan travel.\nRequest: Put together four days in Lisbon for me in May.\n\
         Fits: yes\n\n\
         Skill: plan-trip: Plan a trip with travel, places to stay and a day-by-day schedule. Use \
         when asked to plan travel.\nRequest: What time is it in Lisbon?\nFits: no\n\n\
         Skill: write-invoice: Write an invoice from hours and rates. Use when asked to bill a \
         client.\nRequest: Make Northwind's invoice from the hours in Documents/hours.csv.\n\
         Fits: yes\n\n\
         Skill: write-invoice: Write an invoice from hours and rates. Use when asked to bill a \
         client.\nRequest: Open invoice-7.txt and tell me who it's for.\nFits: no\n\n\
         Skill: {}: {}\nRequest: {request}\nFits:",
        skill.name, skill.description
    )
}

/// The `use_skill` result the model reads, as the Mac's `SkillLibrary.read` gives it.
/// The skill's text is followed by a line saying to start: written into the reply by the
/// server (the model didn't ask for it), a small model otherwise describes the steps it
/// would take and asks whether to go on (Gemma 4 E2B in the test gauntlet).
pub fn instructions(skill: &Skill) -> String {
    format!(
        "Skill \"{}\":\n\n{}\n\n{START}",
        skill.name,
        skill.instructions.trim_end()
    )
}

/// What follows a skill's text in its `use_skill` result.
pub const START: &str = "Do the task now, following these steps: start with step 1 and call \
                         the tool it names. Don't describe the steps or ask whether to go on.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shortlist_question_lists_each_skill_by_name() {
        let skill = |name: &str, description: &str| Skill {
            name: name.into(),
            description: description.into(),
            instructions: String::new(),
            built_in: true,
        };
        let skills = [
            skill("analyze-data", "Answer a question from a spreadsheet."),
            skill("write-memo", &"long ".repeat(100)),
        ];
        let q = shortlist_question(&skills, "What's the total in budget.xlsx?");
        assert!(q.contains("\n\n- analyze-data: Answer a question from a spreadsheet.\n- write-memo: long long"));
        assert!(q.ends_with("\n\nRequest: What's the total in budget.xlsx?"));
        let line = q.lines().find(|l| l.starts_with("- write-memo")).unwrap();
        assert!(line.chars().count() <= 220);
    }

    fn skill(name: &str, instructions: &str) -> Skill {
        Skill {
            name: name.into(),
            description: "d".into(),
            instructions: instructions.into(),
            built_in: true,
        }
    }

    #[test]
    fn reads_skill_files_as_the_macs_library_does() {
        let (name, description, body) = parse(
            "---\r\nname: trip-planner\r\ndescription: \"Plan trips: flights, \\\"hotels\\\".\"\r\n---\r\n\r\n# Plan\r\n\r\n1. Ask.\r\n",
        )
        .unwrap();
        assert_eq!(name, "trip-planner");
        assert_eq!(description, "Plan trips: flights, \"hotels\".");
        assert_eq!(body, "# Plan\n\n1. Ask.");
        let (_, folded, _) =
            parse("---\nname: a-b\ndescription: one\n  two\nother: x\n---\nText").unwrap();
        assert_eq!(folded, "one two");
        assert!(parse("no front matter").is_err());
        assert!(parse("---\nname: Bad Name\ndescription: x\n---\n").is_err());
        assert!(parse("---\nname: ok\n---\n").is_err());
    }

    #[test]
    fn the_macs_built_in_skills_load_and_six_suit_this_pc() {
        let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../server/builtin-skills");
        let all = load(Some(&folder), None);
        assert_eq!(
            all.len(),
            13,
            "{:?}",
            all.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        let pc = [
            "read_file",
            "list_directory",
            "search_files",
            "file_exists",
            "git_status",
            "git_diff",
            "write_file",
            "get_current_datetime",
        ];
        let usable: Vec<&str> = all
            .iter()
            .filter(|s| usable(s, &pc))
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(
            usable,
            [
                "apply-policy",
                "code-review",
                "fill-template",
                "process-invoices",
                "summarize-document",
                "translate-document"
            ]
        );
        let review = all.iter().find(|s| s.name == "code-review").unwrap();
        assert_eq!(
            tools_named(review)[..3],
            ["git_status", "git_diff", "read_file"]
        );
    }

    #[test]
    fn a_users_skill_replaces_a_built_in_one() {
        let dir = std::env::temp_dir().join(format!("vireo-skills-{}", std::process::id()));
        for (folder, name) in [
            ("builtin/a", "same"),
            ("builtin/b", "other"),
            ("user/x", "same"),
        ] {
            let d = dir.join(folder);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(
                d.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: from {folder}\n---\nSteps"),
            )
            .unwrap();
        }
        let all = load(Some(&dir.join("builtin")), Some(&dir.join("user")));
        let _ = std::fs::remove_dir_all(&dir);
        let names: Vec<(&str, &str)> = all
            .iter()
            .map(|s| (s.name.as_str(), s.description.as_str()))
            .collect();
        assert_eq!(
            names,
            [("other", "from builtin/b"), ("same", "from user/x")]
        );
    }

    #[test]
    fn slash_commands_name_their_skill() {
        let skills = [skill("code-review", "")];
        assert_eq!(
            named("Use the code-review skill for this:\n\nmy repo", &skills).map(|s| &s.name[..]),
            Some("code-review")
        );
        assert!(named("Use the other skill for this:", &skills).is_none());
        assert!(named("Review my code", &skills).is_none());
        assert!(usable(&skill("x", "Nothing named"), &[]));
        assert!(!usable(
            &skill("x", "Call `list_events`, then `read_file`."),
            &["read_file"]
        ));
    }
}
