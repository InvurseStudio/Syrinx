//! The cases about skills: each of the three the PC can follow (`summarize-document`,
//! `translate-document`, `code-review`) asked for in plain words, one asked for by an
//! app's slash command, and a request near a skill that shouldn't take one. Each checks
//! the skill picked (the server's use_skill), the work done and, where the skill sets a
//! shape for the answer, that the answer has it.

use crate::cases::{Case, need};
use crate::pc::Pc;

const MINUTES: &str = "Board meeting, 4 November 2026\n\
Present: Ana Ruiz (chair), Ben Cole, Priya Shah, Tom Ellis\n\n\
1. Budget. The board approved the 2027 budget of $48,000 for the community garden, \
up from $41,500. Ben noted the water bill rose 18% this year.\n\
2. Volunteers. Priya reported 26 regular volunteers, down from 31. The board agreed to \
run a sign-up day on 6 December.\n\
3. Tool shed. Tom presented two quotes for a new shed: $3,200 and $4,750. No decision; \
Tom will get a third quote.\n\n\
Actions:\n\
- Priya: organise the sign-up day by 6 December.\n\
- Tom: a third shed quote before the next meeting.\n\
- Ben: ask the city about the water rate by 30 November.\n\n\
Next meeting: 2 December 2026.\n";

const CART_DIFF: &str = "diff --git a/src/cart.py b/src/cart.py\n\
--- a/src/cart.py\n\
+++ b/src/cart.py\n\
@@ -10,9 +10,12 @@ def line_total(price, quantity, discount):\n\
-    if quantity > 0:\n\
-        return price * quantity * (1 - discount)\n\
-    return 0\n\
+    if quantity <= 0:\n\
+        return 0\n\
+    total = price * quantity * (1 - discount)\n\
+    # members get their discount\n\
+    total = total * (1 - discount)\n\
+    return round(total, 2)\n";

pub fn all() -> Vec<Case> {
    let docs = r"C:\Users\me\Documents";
    vec![
        Case {
            id: "skill-summarize",
            title: "Summarizes minutes with the skill",
            history: vec![],
            prompt: format!(r"Can you summarize {docs}\board-minutes.txt for me?"),
            pc: Pc::default().file(&format!(r"{docs}\board-minutes.txt"), MINUTES),
            check: |r| {
                need(
                    r.skills == ["summarize-document"],
                    "didn't use the summarize-document skill",
                )?;
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "board-minutes.txt")),
                    "didn't read the minutes",
                )?;
                need(r.says(&["48,000", "48000", "48k"]), "left out the $48,000 budget")?;
                need(
                    r.says(&["sign-up", "signup", "sign up"]) && r.says(&["quote"]),
                    "left out the actions",
                )?;
                // The skill's shape: its headings, in its words.
                need(
                    r.says(&["in short"]) && r.says(&["key points"]),
                    "didn't follow the skill's layout",
                )
            },
            hard: false,
            changes: false,
            skills: true,
            more: false,
        },
        Case {
            id: "skill-translate",
            title: "Translates a file into a new one",
            history: vec![],
            prompt: format!(
                r"Translate {docs}\welcome.txt into Spanish and save it as welcome-es.txt in the same folder."
            ),
            pc: Pc::default().file(
                &format!(r"{docs}\welcome.txt"),
                "Welcome to the garden!\nPlease close the gate behind you.\nThe tools are in the green shed.\n",
            ),
            check: |r| {
                need(
                    r.skills == ["translate-document"],
                    "didn't use the translate-document skill",
                )?;
                let text = r
                    .pc
                    .text(r"C:\Users\me\Documents\welcome-es.txt")
                    .unwrap_or("")
                    .to_lowercase();
                need(!text.is_empty(), "didn't write welcome-es.txt")?;
                need(
                    text.contains("bienvenid") && (text.contains("puerta") || text.contains("verja") || text.contains("portón")),
                    "welcome-es.txt isn't the Spanish text",
                )?;
                need(
                    r.pc.text(r"C:\Users\me\Documents\welcome.txt").is_some_and(|t| t.starts_with("Welcome")),
                    "changed the original",
                )
            },
            hard: false,
            changes: false,
            skills: true,
            more: false,
        },
        Case {
            id: "skill-review",
            title: "Reviews uncommitted code with the skill",
            history: vec![],
            prompt: r"Could you review the changes I haven't committed yet in C:\Users\me\Projects\shop?"
                .into(),
            pc: Pc::default()
                .file(r"C:\Users\me\Projects\shop\src\cart.py", "def line_total(price, quantity, discount):\n")
                .git(r"C:\Users\me\Projects\shop", " M src/cart.py")
                .diff(r"C:\Users\me\Projects\shop", CART_DIFF),
            check: |r| {
                need(r.skills == ["code-review"], "didn't use the code-review skill")?;
                need(!r.called("git_diff").is_empty(), "didn't read the diff")?;
                need(
                    r.says(&["twice", "two times", "double", "second time", "applied again", "again"])
                        && r.says(&["discount"]),
                    "missed the discount applied twice",
                )
            },
            hard: false,
            changes: false,
            skills: true,
            more: false,
        },
        Case {
            id: "skill-slash",
            title: "Uses the skill a slash command names",
            history: vec![],
            prompt: format!(
                "Use the summarize-document skill for this:\n\n{docs}\\board-minutes.txt"
            ),
            pc: Pc::default().file(&format!(r"{docs}\board-minutes.txt"), MINUTES),
            check: |r| {
                need(
                    r.skills == ["summarize-document"],
                    "didn't use the skill the command named",
                )?;
                need(
                    !r.called("read_file").is_empty(),
                    "didn't read the minutes",
                )?;
                need(r.says(&["48,000", "48000", "48k"]), "left out the $48,000 budget")
            },
            hard: false,
            changes: false,
            skills: true,
            more: false,
        },
        Case {
            id: "skill-none",
            title: "Reads a file without a skill",
            history: vec![],
            prompt: format!(r"How long does the bread bake, going by {docs}\bread.txt?"),
            pc: Pc::default().file(
                &format!(r"{docs}\bread.txt"),
                "Simple loaf\n500 g flour, 350 ml water, 10 g salt, 7 g yeast.\nRise 1 hour. Bake at 220 C for 35 minutes.\n",
            ),
            check: |r| {
                need(
                    r.skills.is_empty(),
                    "took a skill for a plain question about a file",
                )?;
                need(r.says(&["35"]), "didn't say 35 minutes")
            },
            hard: false,
            changes: false,
            skills: true,
            more: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::Run;

    #[test]
    fn five_skill_cases_that_fail_when_nothing_is_done() {
        let cases = all();
        assert_eq!(cases.len(), 5);
        for case in &cases {
            let idle = Run {
                answer: "Done.".into(),
                pc: case.pc.clone(),
                ..Run::default()
            };
            assert!((case.check)(&idle).is_err(), "{}", case.id);
        }
    }
}
