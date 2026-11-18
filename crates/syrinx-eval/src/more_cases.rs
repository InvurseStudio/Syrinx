//! The cases for the tools added after the first ones: `calculate` (the server's),
//! `read_spreadsheet`, `create_docx` and `run_tests` (the Windows app's), and the two
//! skills they let the PC follow, `analyze-data` and `write-document`. Counted apart, so
//! the earlier categories keep their numbers.

use crate::cases::{Case, need};
use crate::pc::Pc;

pub fn all() -> Vec<Case> {
    let docs = r"C:\Users\me\Documents";
    vec![
        Case {
            id: "calc-tip",
            title: "Works out a tip with calculate",
            history: vec![],
            prompt:
                "My dinner bill is $86.40. What's an 18% tip, and what's the total with the tip?"
                    .into(),
            pc: Pc::default(),
            check: |r| {
                need(!r.called("calculate").is_empty(), "didn't use calculate")?;
                need(r.says(&["15.55"]), "didn't give the $15.55 tip")?;
                need(r.says(&["101.95"]), "didn't give the $101.95 total")
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
        Case {
            id: "sheet-sales",
            title: "Answers from a spreadsheet, with the skill",
            history: vec![],
            prompt: format!(
                r"Which region had the highest Q3 sales in {docs}\sales-2026.xlsx, and what's the Q3 total across all regions?"
            ),
            pc: Pc::default().sheet(
                &format!(r"{docs}\sales-2026.xlsx"),
                &[
                    &["Region", "Q2", "Q3"],
                    &["North", "45,100", "48,250"],
                    &["South", "52,800", "61,930"],
                    &["East", "41,020", "39,480"],
                    &["West", "50,300", "57,115"],
                ],
            ),
            check: |r| {
                need(
                    r.calls.iter().any(|c| {
                        matches!(c.tool.as_str(), "read_spreadsheet" | "read_file")
                            && c.points_at("path", "sales-2026.xlsx")
                    }),
                    "didn't read the workbook",
                )?;
                need(r.says(&["South"]), "didn't say South")?;
                need(
                    r.says(&["206,775", "206775"]),
                    "didn't give the 206,775 total",
                )?;
                need(
                    r.skills == ["analyze-data"],
                    "didn't use the analyze-data skill",
                )
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
        Case {
            id: "memo-docx",
            title: "Writes a memo as a Word document, with the skill",
            history: vec![],
            prompt: format!(
                r"Write a short memo to the team saying the office is closed December 24 to 26, and save it as {docs}\holiday-closing.docx."
            ),
            pc: Pc::default().file(&format!(r"{docs}\notes.txt"), "x"),
            check: |r| {
                let text =
                    r.pc.text(r"C:\Users\me\Documents\holiday-closing.docx")
                        .unwrap_or("")
                        .to_lowercase();
                need(!text.is_empty(), "didn't save holiday-closing.docx")?;
                need(
                    text.contains("24") && text.contains("26") && text.contains("december"),
                    "the memo doesn't give the dates",
                )?;
                need(
                    r.asked.iter().any(|t| t == "create_docx"),
                    "saved it without asking",
                )?;
                need(
                    r.skills == ["write-document"],
                    "didn't use the write-document skill",
                )
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
        Case {
            id: "run-tests",
            title: "Runs a project's tests and names the failure",
            history: vec![],
            prompt: r"Run the tests in C:\Users\me\Projects\shop and tell me which one failed."
                .into(),
            pc: Pc::default()
                .file(
                    r"C:\Users\me\Projects\shop\pyproject.toml",
                    "[project]\nname = \"shop\"\n",
                )
                .file(
                    r"C:\Users\me\Projects\shop\tests\test_cart.py",
                    "def test_total(): ...\n",
                )
                .command(
                    r"C:\Users\me\Projects\shop",
                    "tests",
                    "python -m pytest",
                    1,
                    "============ test session starts ============\n\
                     collected 14 items\n\n\
                     tests/test_cart.py ......F.......\n\n\
                     ================= FAILURES ==================\n\
                     ___________ test_discount_applied_once ___________\n\
                     >       assert line_total(10.00, 2, 0.1) == 18.00\n\
                     E       assert 16.2 == 18.0\n\n\
                     tests/test_cart.py:41: AssertionError\n\
                     ========= 1 failed, 13 passed in 0.42s ==========",
                ),
            check: |r| {
                need(!r.called("run_tests").is_empty(), "didn't run the tests")?;
                need(
                    r.says(&["test_discount_applied_once"]),
                    "didn't name the failing test",
                )
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::Run;

    #[test]
    fn four_cases_that_fail_when_nothing_is_done() {
        let cases = all();
        assert_eq!(cases.len(), 4);
        assert!(
            cases
                .iter()
                .all(|c| c.more && !c.hard && !c.changes && !c.skills)
        );
        for case in &cases {
            let idle = Run {
                answer: "Done.".into(),
                pc: case.pc.clone(),
                ..Run::default()
            };
            assert!((case.check)(&idle).is_err(), "{}", case.id);
        }
    }

    #[test]
    fn the_stand_in_pc_runs_the_new_tools_as_the_app_does() {
        let args = |pairs: &[(&str, &str)]| {
            crate::json::Value::Object(
                pairs
                    .iter()
                    .map(|(k, v)| (k.to_string(), crate::json::Value::String(v.to_string())))
                    .collect(),
            )
        };
        let mut sheet = all()[1].pc.clone();
        let read = sheet
            .run(
                "read_spreadsheet",
                &args(&[("path", "Documents/sales-2026.xlsx")]),
            )
            .unwrap();
        assert!(read.starts_with("Sheets: Sheet1\n\nSheet \"Sheet1\", A1:C5 (used range A1:C5)\nRow | A | B | C\n1 | Region | Q2 | Q3\n"), "{read}");
        assert!(
            sheet
                .run("read_spreadsheet", &args(&[("path", "Documents/x.xlsx")]))
                .is_err()
        );

        let mut memo = all()[2].pc.clone();
        assert_eq!(
            memo.run(
                "create_docx",
                &args(&[
                    ("path", "Documents/memo"),
                    ("title", "Memo"),
                    ("content", "# Hi")
                ])
            ),
            Ok(r"C:\Users\me\Documents\memo.docx".into())
        );
        assert_eq!(
            memo.text(r"C:\Users\me\Documents\memo.docx"),
            Some("Memo\n\n# Hi")
        );

        let mut shop = all()[3].pc.clone();
        let tests = shop
            .run("run_tests", &args(&[("path", r"Projects\shop")]))
            .unwrap();
        assert!(
            tests.starts_with(r#"{"ok":false,"path":"C:\\Users\\me\\Projects\\shop","command":["python","-m","pytest"],"exitCode":1,"#),
            "{tests}"
        );
        assert!(
            shop.run("run_build", &args(&[("path", r"Projects\shop")]))
                .unwrap_err()
                .starts_with("No build command")
        );
    }
}
