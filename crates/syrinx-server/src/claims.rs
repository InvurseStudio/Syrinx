//! A check of what an answer says it did. A model sometimes reports a change it never
//! made: in three test-gauntlet runs in a row (Nov 13-15, 11.2) a task's reply said "I
//! have added the following row" to a tracker with no call that wrote to one. The answer
//! can't be taken back once it's shown, so a note follows it.

/// What kinds of change an answer may claim, each with the words that name it in a
/// sentence, the tools that make it, and how the note names it.
const KINDS: [(&[&str], &[&str], &str); 5] = [
    (
        &["row", "tracker", "register", "spreadsheet", "workbook", ".xlsx", "sheet"],
        &["write_spreadsheet", "write_file", "edit_file"],
        "a spreadsheet",
    ),
    (
        &["calendar", "event"],
        &["create_event", "move_event", "delete_event"],
        "the calendar",
    ),
    (&["reminder"], &["create_reminder", "complete_reminder"], "a reminder"),
    (&["scheduled task", "the task"], &["schedule_task"], "a task"),
    (
        &["file", "folder", "word document", ".docx", ".md", ".pptx", ".pdf", ".txt", ".csv"],
        &[
            "write_file",
            "edit_file",
            "create_pdf",
            "write_spreadsheet",
            "create_docx",
            "fill_docx_template",
            "create_pptx",
            "rename_file",
            "move_file",
            "delete_file",
        ],
        "a file",
    ),
];

/// Ways an answer says a change is made.
const CLAIMS: [&str; 34] = [
    "i have added", "i've added", "i added", "i have appended", "i've appended", "i appended",
    "i have logged", "i've logged", "i logged", "i have recorded", "i've recorded", "i recorded",
    "i have updated", "i've updated", "i updated", "i have saved", "i've saved", "i saved",
    "i have created", "i've created", "i created", "i have scheduled", "i've scheduled",
    "has been added", "has been updated", "has been saved", "has been logged",
    "has been recorded", "has been created", "have been added", "have been logged",
    "have been saved", "have been recorded", "have been updated",
];

/// Words that make a sentence about another time, or about what could be done.
const NOT_NOW: [&str; 14] = [
    "already", "earlier", "previously", "before", "last time", " will ", "i'll", "would",
    " can ", "could", "should", " if ", " once ", "not ",
];

/// What an answer says was changed though no tool changed it: `done` holds the tools
/// that ran without an error, in this reply and earlier in the chat. `None` when every
/// claim is backed, or the answer makes none.
pub fn unbacked(answer: &str, done: &[String]) -> Option<&'static str> {
    // A sentence ends at a full stop with a space after it: "memo.docx" is one word.
    let answer = answer.to_lowercase().replace(". ", "\n");
    for sentence in answer.split(['!', '?', '\n']) {
        let sentence = format!(" {sentence} ");
        if !CLAIMS.iter().any(|c| sentence.contains(c))
            || NOT_NOW.iter().any(|w| sentence.contains(w))
        {
            continue;
        }
        // The first kind the sentence names is the one it claims.
        if let Some((_, tools, what)) = KINDS
            .iter()
            .find(|(words, _, _)| words.iter().any(|w| names(&sentence, w)))
        {
            if !tools.iter().any(|t| done.iter().any(|d| d == t)) {
                return Some(what);
            }
        }
    }
    None
}

/// Whether a sentence names a kind's word as a word of its own: "row" or "rows", not the
/// "row" in "tomorrow" (Nov 19: a calendar event made "tomorrow" was flagged as a
/// spreadsheet claimed). A word starting with "." (".xlsx") is a file's ending, matched as is.
fn names(sentence: &str, word: &str) -> bool {
    if word.starts_with('.') {
        return sentence.contains(word);
    }
    let mut from = 0;
    while let Some(at) = sentence[from..].find(word) {
        let start = from + at;
        let end = start + word.len();
        let before = sentence[..start].chars().next_back();
        let after: String = sentence[end..].chars().take(2).collect();
        let starts = before.is_none_or(|c| !c.is_alphanumeric());
        let ends = after.is_empty()
            || !after.chars().next().unwrap().is_alphanumeric()
            || after == "s"
            || (after.starts_with('s') && !after.chars().nth(1).unwrap().is_alphanumeric());
        if starts && ends {
            return true;
        }
        from = start + word.len().max(1);
    }
    false
}

/// What follows such an answer.
pub fn note(what: &str) -> String {
    format!(
        "\n\n**Check:** this reply says {what} was changed, but no tool changed one in this chat, so that has not been done. Ask again to have it done."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(tools: &[&str]) -> Vec<String> {
        tools.iter().map(|t| t.to_string()).collect()
    }

    #[test]
    fn a_change_no_tool_made_is_caught() {
        // The gauntlet's case: an event made, a tracker row claimed.
        let said = "The request is eligible. I have added the following row to the Donations Tracker:\n| Fire Brigade | 5,840 |";
        assert_eq!(unbacked(said, &done(&["read_spreadsheet", "create_event"])), Some("a spreadsheet"));
        assert_eq!(unbacked(said, &done(&["write_spreadsheet"])), None);
        assert_eq!(unbacked("I've saved the memo as Admin/memo.docx.", &done(&[])), Some("a file"));
        assert_eq!(unbacked("The meeting has been added to your calendar.", &done(&["write_file"])), Some("the calendar"));
        assert!(note("a file").contains("this reply says a file was changed"));
        // "tomorrow" has "row" in it, but names no spreadsheet.
        let event = "I've added the event 'Vireo test' to your calendar for tomorrow.";
        assert_eq!(unbacked(event, &done(&["create_event"])), None);
        assert_eq!(unbacked("I've added two rows to the tracker.", &done(&[])), Some("a spreadsheet"));
        assert_eq!(unbacked("I've added it to budget.xlsx.", &done(&[])), Some("a spreadsheet"));
    }

    #[test]
    fn talk_of_other_times_and_of_sums_is_left_alone() {
        let none = done(&[]);
        for said in [
            "I have added the two subtotals: 8,440.",
            "I've drafted the email below.",
            "I have not added the row to the tracker, since it needs approval.",
            "The rows were already there, so nothing has been added to the register again.",
            "Once you approve, the row will be added to the tracker.",
            "I can add this to the register if you like.",
            "Seats needed: 29.",
        ] {
            assert_eq!(unbacked(said, &none), None, "{said}");
        }
    }
}
