//! The harder cases: several steps, long results, near names, follow-ups that lean on
//! earlier turns, and recovering from a wrong path. Worded apart from the easy ones, so
//! a model can't pass them on what it learned there.

use crate::cases::{Case, need};
use crate::pc::Pc;

/// A log of about 6,000 characters: routine lines every few minutes, three errors at
/// different times, the one asked about at 14:32.
fn long_log() -> String {
    let mut out = String::new();
    for i in 0..140 {
        let (h, m) = (9 + i * 5 / 60, i * 5 % 60);
        let line = match (h, m) {
            (9, 10) => "09:10:04 ERROR upload worker: TIMEOUT after 30 s".to_string(),
            (16, 5) => "16:05:41 ERROR login: AUTH_FAILED for user sam".to_string(),
            _ => format!(
                "{h:02}:{m:02}:00 INFO heartbeat ok, queue length {}",
                (i * 7) % 13
            ),
        };
        out.push_str(&line);
        out.push('\n');
        if (h, m) == (14, 30) {
            out.push_str("14:32:17 ERROR backup job: DISK_FULL on drive D:\n");
        }
    }
    out
}

pub fn all() -> Vec<Case> {
    let docs = r"C:\Users\me\Documents";
    let projects = r"C:\Users\me\Projects";
    let logs = r"C:\Users\me\Logs";
    let notes_history = vec![
        ("user", format!(r"Read {docs}\notes.txt for me.")),
        (
            "assistant",
            "The notes say: Shopping: eggs, flour, milk. The code word for Friday is PELICAN."
                .to_string(),
        ),
    ];
    let notes_pc = || {
        Pc::default()
            .file(
                &format!(r"{docs}\notes.txt"),
                "Shopping: eggs, flour, milk.\nThe code word for Friday is PELICAN.\n",
            )
            .file(
                &format!(r"{docs}\plan.md"),
                "# Plan\nShip the new app in December.\n",
            )
    };
    vec![
        Case {
            id: "three-steps",
            title: "Finds a project, then asks its Git",
            history: vec![],
            prompt: format!(
                r"One of my projects in {projects} has a README that mentions invoices. Which project is it, and which files have I changed in it?"
            ),
            pc: Pc::default()
                .file(
                    &format!(r"{projects}\billing\README.md"),
                    "# Billing\nSends invoices every month.",
                )
                .file(
                    &format!(r"{projects}\billing\src\invoice.rs"),
                    "fn send() {}",
                )
                .file(
                    &format!(r"{projects}\garden\README.md"),
                    "# Garden\nWhat to plant and when.",
                )
                .file(
                    &format!(r"{projects}\notes-app\README.md"),
                    "# Notes\nA place for notes.",
                )
                .git(&format!(r"{projects}\billing"), " M src/invoice.rs")
                .git(&format!(r"{projects}\garden"), " M plan.md"),
            check: |r| {
                need(
                    r.calls.iter().any(|c| {
                        (c.tool == "git_status" || c.tool == "git_diff")
                            && c.points_at("path", "billing")
                    }),
                    "didn't ask Git about billing",
                )?;
                need(r.says(&["billing"]), "didn't name the project")?;
                need(r.says(&["invoice.rs"]), "didn't name the changed file")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "long-log",
            title: "Finds one line in a long log",
            history: vec![],
            prompt: format!(r"What went wrong at 14:32 according to {logs}\app.log?"),
            pc: Pc::default().file(&format!(r"{logs}\app.log"), &long_log()),
            check: |r| {
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "app.log")),
                    "didn't read the log",
                )?;
                // In its own words is fine: "the disk on drive D: was full".
                need(
                    r.says(&["DISK_FULL"]) || (r.says(&["disk", "drive"]) && r.says(&["full"])),
                    "didn't find the 14:32 error",
                )
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "find-by-content",
            title: "Finds which file holds a code",
            history: vec![],
            prompt: format!(r"Which of the log files in {logs} mentions ERR-4471?"),
            pc: (1..=8).fold(Pc::default(), |pc, n| {
                let text = if n == 2 {
                    "worker started\nERR-4471: could not reach the print server\n".to_string()
                } else {
                    format!("worker {n} started\nall jobs done\n")
                };
                pc.file(&format!(r"{logs}\worker-{n}.log"), &text)
            }),
            check: |r| {
                need(!r.calls.is_empty(), "didn't look")?;
                need(r.says(&["worker-2"]), "didn't name worker-2.log")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "which-report",
            title: "Picks the right one of two files",
            history: vec![],
            prompt: format!(r"Open my 2026 report in {docs} and tell me the revenue it gives."),
            pc: Pc::default()
                .file(
                    &format!(r"{docs}\report-2025.txt"),
                    "Annual report 2025\nRevenue: $1.2 million\n",
                )
                .file(
                    &format!(r"{docs}\report-2026.txt"),
                    "Annual report 2026\nRevenue: $1.5 million\n",
                )
                .file(&format!(r"{docs}\budget.txt"), "Budget: $900,000"),
            check: |r| {
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "report-2026.txt")),
                    "didn't read report-2026.txt",
                )?;
                need(r.says(&["1.5"]), "didn't give $1.5 million")?;
                need(!r.says(&["1.2 million"]), "gave 2025's revenue")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "relative-path",
            title: "Reads a path relative to home",
            history: vec![],
            prompt: "What's on my list in Documents/todo.txt?".into(),
            pc: Pc::default().file(
                &format!(r"{docs}\todo.txt"),
                "- call the bank\n- buy stamps\n- water the ferns\n",
            ),
            check: |r| {
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "todo.txt")),
                    "didn't read todo.txt",
                )?;
                need(
                    r.says(&["stamps"]) && r.says(&["ferns"]),
                    "didn't give the list",
                )
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "count",
            title: "Counts files of one kind",
            history: vec![],
            prompt: format!(r"How many PDF files are there in {docs}\Invoices?"),
            pc: Pc::default()
                .file(&format!(r"{docs}\Invoices\acme-june.pdf"), "A")
                .file(&format!(r"{docs}\Invoices\acme-july.pdf"), "A")
                .file(&format!(r"{docs}\Invoices\globex-july.pdf"), "G")
                .file(&format!(r"{docs}\Invoices\totals.xlsx"), "T")
                .file(&format!(r"{docs}\Invoices\readme.txt"), "R"),
            check: |r| {
                need(!r.calls.is_empty(), "didn't look")?;
                need(r.says(&["3", "three"]), "didn't say three")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "follow-up-file",
            title: "Follows up on an earlier file",
            history: notes_history.clone(),
            prompt: "And what does plan.md in the same folder say?".into(),
            pc: notes_pc(),
            check: |r| {
                need(
                    r.calls.iter().any(|c| {
                        c.tool == "read_file" && c.points_at("path", r"documents\plan.md")
                    }),
                    "didn't read Documents\\plan.md",
                )?;
                need(r.says(&["December"]), "didn't say what the plan says")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "follow-up-no-tool",
            title: "Answers a follow-up from what it knows",
            history: notes_history,
            prompt: "Which day is that code word for?".into(),
            pc: notes_pc(),
            check: |r| {
                r.only(&[])?;
                need(r.says(&["Friday"]), "didn't say Friday")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "explain-git",
            title: "Explains a command without running it",
            history: vec![],
            prompt: "In a sentence or two, what does the git status command do?".into(),
            pc: Pc::default(),
            check: |r| {
                r.only(&[])?;
                need(
                    r.says(&["change", "modified", "staged", "working"]),
                    "didn't explain it",
                )
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "recover",
            title: "Looks around after a wrong path",
            history: vec![],
            prompt: format!(r"Read {docs}\Notes\meeting.txt and tell me when the next meeting is."),
            pc: Pc::default()
                .file(
                    &format!(r"{docs}\meeting-notes.txt"),
                    "Next meeting: Tuesday at 10am, room 4.\n",
                )
                .file(&format!(r"{docs}\notes.txt"), "Shopping: eggs."),
            check: |r| {
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "meeting-notes.txt")),
                    "never found meeting-notes.txt",
                )?;
                need(r.says(&["Tuesday"]), "didn't say Tuesday")
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "deadline",
            title: "Compares a file's date with today",
            history: vec![],
            prompt: format!(r"Has the deadline in {projects}\launch\project.txt already passed?"),
            pc: Pc::default().file(
                &format!(r"{projects}\launch\project.txt"),
                "Project: website launch\nDeadline: October 30, 2026\nOwner: Sam\n",
            ),
            check: |r| {
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "project.txt")),
                    "didn't read project.txt",
                )?;
                need(
                    !r.called("get_current_datetime").is_empty(),
                    "didn't check today's date",
                )?;
                need(
                    r.says(&["passed", "past", "yes", "overdue", "already"]),
                    "didn't say it has passed",
                )
            },
            hard: true,
            changes: false,
            skills: false,
        },
        Case {
            id: "git-diff",
            title: "Reads what a change was",
            history: vec![],
            prompt: format!(r"What did I change in src\main.rs in my project at {projects}\vireo?"),
            pc: Pc::default()
                .file(
                    &format!(r"{projects}\vireo\src\main.rs"),
                    "fn main() { let port = 8080; }",
                )
                .git(&format!(r"{projects}\vireo"), " M src/main.rs"),
            check: |r| {
                need(
                    r.calls.iter().any(|c| c.tool == "git_diff"),
                    "didn't look at the diff",
                )?;
                need(r.says(&["8080"]), "didn't say the port became 8080")
            },
            hard: true,
            changes: false,
            skills: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_long_log_is_long_and_has_its_line() {
        let log = super::long_log();
        assert!(
            (5000..8000).contains(&log.len()),
            "{} characters",
            log.len()
        );
        assert_eq!(log.matches("DISK_FULL").count(), 1);
        assert!(log.contains("14:32:17 ERROR backup job: DISK_FULL"));
    }

    #[test]
    fn twelve_hard_cases_with_their_own_ids() {
        let all = crate::cases::all();
        assert_eq!(all.iter().filter(|c| !c.changes && !c.skills).count(), 24);
        assert_eq!(all.iter().filter(|c| c.hard).count(), 12);
        let mut ids: Vec<_> = all.iter().map(|c| c.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), all.len());
    }
}
