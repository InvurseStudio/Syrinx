//! The Markdown a model writes for a document, split into blocks and styled runs: the
//! same headings, lists, tables, quotes, code and rules as the Windows app's
//! `MarkdownBlocks` (and the Mac's `document_writer.py` before it) make of it, so a Word
//! document built here is laid out as one built there.

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Heading { level: usize, text: String },
    Paragraph(String),
    /// A list item: its nesting depth (0 at the top) and, for a numbered one, its number.
    Bullet { depth: usize, text: String },
    Numbered { depth: usize, number: u32, text: String },
    Quote(String),
    Code(String),
    /// A table's rows, its header first, and each column's alignment.
    Table { rows: Vec<Vec<String>>, alignments: Vec<Alignment> },
    Rule,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alignment {
    Left,
    Center,
    Right,
}

/// A piece of text and its styles; `link` is the address it links to.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Run {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub code: bool,
    pub link: Option<String>,
}

pub fn parse(content: &str) -> Vec<Block> {
    let content = content.replace("\r\n", "\n");
    let lines: Vec<&str> = content.split('\n').collect();
    let mut blocks = Vec::new();
    // A paragraph's lines so far, as written.
    let mut paragraph: Vec<&str> = Vec::new();
    fn flush(paragraph: &mut Vec<&str>, blocks: &mut Vec<Block>) {
        if !paragraph.is_empty() {
            blocks.push(Block::Paragraph(join_lines(paragraph)));
            paragraph.clear();
        }
    }
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let stripped = line.trim();
        if stripped.starts_with("```") {
            flush(&mut paragraph, &mut blocks);
            let mut code = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("```") {
                code.push(lines[i]);
                i += 1;
            }
            blocks.push(Block::Code(code.join("\n")));
            i += 1;
            continue;
        }
        if stripped.is_empty() {
            flush(&mut paragraph, &mut blocks);
            i += 1;
            continue;
        }
        if stripped.contains('|') && i + 1 < lines.len() {
            let header = cells(stripped);
            let separator = cells(lines[i + 1]);
            if separator.len() == header.len() && separator.iter().all(|c| is_separator(c)) {
                flush(&mut paragraph, &mut blocks);
                let alignments = separator
                    .iter()
                    .map(|c| {
                        if c.len() > 1 && c.starts_with(':') && c.ends_with(':') {
                            Alignment::Center
                        } else if c.ends_with(':') {
                            Alignment::Right
                        } else {
                            Alignment::Left
                        }
                    })
                    .collect();
                let mut rows = vec![header.clone()];
                i += 2;
                while i < lines.len() && lines[i].contains('|') && !lines[i].trim().is_empty() {
                    let mut row: Vec<String> = cells(lines[i]).into_iter().take(header.len()).collect();
                    row.resize(header.len(), String::new());
                    rows.push(row);
                    i += 1;
                }
                blocks.push(Block::Table { rows, alignments });
                continue;
            }
        }
        if is_rule(stripped) {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::Rule);
        } else if let Some((level, text)) = heading(stripped) {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::Heading { level, text });
        } else if let Some((indent, number, text)) = numbered_item(line) {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::Numbered { depth: depth(indent), number, text });
        } else if let Some((indent, text)) = bullet_item(line) {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::Bullet { depth: depth(indent), text });
        } else if stripped.starts_with('>') {
            flush(&mut paragraph, &mut blocks);
            blocks.push(Block::Quote(stripped.trim_start_matches('>').trim().to_string()));
        } else {
            paragraph.push(line);
        }
        i += 1;
    }
    flush(&mut paragraph, &mut blocks);
    blocks
}

/// A document that opens with a top-level heading uses it as its title, rather than
/// showing the title and then nearly the same heading again.
pub fn split_title(title: &str, mut blocks: Vec<Block>) -> (String, Vec<Block>) {
    let first = blocks.iter().position(|b| *b != Block::Rule);
    if let Some(first) = first
        && let Block::Heading { level: 1, text } = &blocks[first]
    {
        let title = plain(text);
        blocks.drain(..=first);
        return (title, blocks);
    }
    (title.to_string(), blocks)
}

/// A paragraph's lines: wrapped prose is joined with spaces, but a field line ("To: CFO",
/// "**Date:** 4 November"), or a line after a Markdown hard break (two spaces or a
/// backslash at its end), starts a new line, kept as a line break.
fn join_lines(lines: &[&str]) -> String {
    let mut text = String::new();
    for (i, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if i > 0 {
            let previous = lines[i - 1];
            let hard = previous.ends_with("  ") || previous.trim_end().ends_with('\\');
            text.push(if hard || is_field_line(line) { '\n' } else { ' ' });
        }
        text.push_str(match line.strip_suffix('\\') {
            Some(rest) => rest.trim_end(),
            None => line,
        });
    }
    text
}

/// `# Title`, `## Section ##`: one to six #, a space, the text (closing #s left off).
fn heading(stripped: &str) -> Option<(usize, String)> {
    let level = stripped.chars().take_while(|c| *c == '#').count();
    let rest = &stripped[level..];
    if !(1..=6).contains(&level) || !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let text = rest.trim().trim_end_matches('#').trim_end();
    Some((level, text.to_string()))
}

fn indent_of(line: &str) -> (&str, &str) {
    let at = line.len() - line.trim_start().len();
    line.split_at(at)
}

/// `- item`, `* item`, `+ item`, `• item`: the indent before it and its text.
fn bullet_item(line: &str) -> Option<(&str, String)> {
    let (indent, rest) = indent_of(line);
    let mut chars = rest.chars();
    let mark = chars.next()?;
    let after = chars.as_str();
    if !matches!(mark, '-' | '*' | '+' | '•') || !after.starts_with(char::is_whitespace) {
        return None;
    }
    Some((indent, after.trim().to_string()))
}

/// `1. item`, `2) item`: the indent, the number and the text.
fn numbered_item(line: &str) -> Option<(&str, u32, String)> {
    let (indent, rest) = indent_of(line);
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    let after = &rest[digits..];
    if digits == 0 || !(after.starts_with('.') || after.starts_with(')')) {
        return None;
    }
    let text = &after[1..];
    if !text.starts_with(char::is_whitespace) {
        return None;
    }
    Some((indent, rest[..digits].parse().unwrap_or(0), text.trim().to_string()))
}

/// `---`, `* * *`, `___`: three or more of one mark, with only spaces between.
fn is_rule(stripped: &str) -> bool {
    let Some(mark) = stripped.chars().next().filter(|c| matches!(c, '-' | '*' | '_')) else {
        return false;
    };
    stripped.chars().all(|c| c == mark || c.is_whitespace())
        && stripped.chars().filter(|c| *c == mark).count() >= 3
}

/// A line that stands on its own inside a paragraph, as in a memo header or an address
/// block: a short label and a colon ("To:", "**Date:**"), then a space or the line's end.
fn is_field_line(line: &str) -> bool {
    let mut rest = line;
    for mark in ["**", "__"] {
        if let Some(r) = rest.strip_prefix(mark) {
            rest = r;
            break;
        }
    }
    let mut chars = rest.char_indices();
    if !chars.next().is_some_and(|(_, c)| c.is_ascii_alphabetic()) {
        return false;
    }
    let mut label = 0;
    for (i, c) in chars {
        if c == ':' {
            let mut after = &rest[i + 1..];
            for mark in ["**", "__"] {
                if let Some(r) = after.strip_prefix(mark) {
                    after = r;
                    break;
                }
            }
            return after.is_empty() || after.starts_with(char::is_whitespace);
        }
        label += 1;
        let allowed = c.is_alphanumeric() || matches!(c, '_' | ' ' | '&' | '/' | '(' | ')' | '.' | '\'' | '-');
        if !allowed || label > 30 {
            return false;
        }
    }
    false
}

fn is_separator(cell: &str) -> bool {
    let dashes = cell.trim_start_matches(':').trim_end_matches(':');
    let colons = cell.len() - dashes.len();
    !dashes.is_empty()
        && dashes.chars().all(|c| c == '-')
        && colons <= 2
        && (colons < 2 || (cell.starts_with(':') && cell.ends_with(':')))
}

/// A table row's cells: split on bars that aren't escaped or in `code`.
fn cells(line: &str) -> Vec<String> {
    let mut row = line.trim();
    row = row.strip_prefix('|').unwrap_or(row);
    if row.ends_with('|') && !row.ends_with("\\|") {
        row = &row[..row.len() - 1];
    }
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut in_code = false;
    let mut chars = row.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&'|') {
            cell.push('|');
            chars.next();
        } else if c == '|' && !in_code {
            cells.push(cell.trim().to_string());
            cell.clear();
        } else {
            if c == '`' {
                in_code = !in_code;
            }
            cell.push(c);
        }
    }
    cells.push(cell.trim().to_string());
    cells
}

fn depth(indent: &str) -> usize {
    (indent.replace('\t', "    ").len() / 2).min(3)
}

/// Text split into runs with their styles: `code`, **bold**, *italic* and [links](to).
pub fn runs(text: &str) -> Vec<Run> {
    let mut out = Vec::new();
    walk(&text.chars().collect::<Vec<_>>(), &Run::default(), &mut out);
    out
}

/// The text without its marks.
pub fn plain(text: &str) -> String {
    runs(text).into_iter().map(|r| r.text).collect()
}

fn add(piece: &[char], style: &Run, out: &mut Vec<Run>) {
    let mut text = String::with_capacity(piece.len());
    let mut i = 0;
    while i < piece.len() {
        // A backslash before a mark is the mark itself (not in code).
        if !style.code
            && piece[i] == '\\'
            && i + 1 < piece.len()
            && matches!(piece[i + 1], '\\' | '`' | '*' | '_' | '[' | ']' | '#' | '|')
        {
            i += 1;
        }
        text.push(piece[i]);
        i += 1;
    }
    if !text.is_empty() {
        out.push(Run { text, ..style.clone() });
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The index after a mark that closes at the first `close` from `from`, when everything
/// before it passes `inside` and there's at least one character of it.
fn closing(s: &[char], from: usize, close: &[char], inside: impl Fn(char) -> bool) -> Option<usize> {
    let mut i = from;
    while i < s.len() {
        if s[i..].starts_with(close) {
            return (i > from).then_some(i);
        }
        if !inside(s[i]) {
            return None;
        }
        i += 1;
    }
    None
}

fn walk(s: &[char], style: &Run, out: &mut Vec<Run>) {
    let mut done = 0;
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];
        // A mark after a backslash is only itself.
        if rest[0] == '\\' && rest.len() > 1 {
            i += 2;
            continue;
        }
        // What starts here, tried in this order: code, bold, italic, a link. Each gives
        // the styled part's start and end and where the scan goes on.
        let found = if rest[0] == '`' {
            closing(s, i + 1, &['`'], |c| c != '`').map(|end| {
                (Run { code: true, ..style.clone() }, i + 1, end, end + 1, false)
            })
        } else if rest.starts_with(&['*', '*']) {
            closing(s, i + 2, &['*', '*'], |c| c != '*')
                .map(|end| (Run { bold: true, ..style.clone() }, i + 2, end, end + 2, true))
        } else if rest.starts_with(&['_', '_']) {
            closing(s, i + 2, &['_', '_'], |c| c != '_')
                .map(|end| (Run { bold: true, ..style.clone() }, i + 2, end, end + 2, true))
        } else {
            None
        };
        let found = found.or_else(|| {
            let first_ok = |mark: char| {
                rest.len() > 1 && rest[1] != mark && !rest[1].is_whitespace()
            };
            if rest[0] == '*' && first_ok('*') {
                closing(s, i + 1, &['*'], |c| c != '*')
                    .map(|end| (Run { italic: true, ..style.clone() }, i + 1, end, end + 1, true))
            } else if rest[0] == '_' && first_ok('_') && (i == 0 || !is_word(s[i - 1])) {
                closing(s, i + 1, &['_'], |c| c != '_')
                    .filter(|end| s.get(end + 1).is_none_or(|c| !is_word(*c)))
                    .map(|end| (Run { italic: true, ..style.clone() }, i + 1, end, end + 1, true))
            } else if rest[0] == '[' {
                // [text](address): no ] in the text, no ) or space in the address.
                closing(s, i + 1, &[']'], |c| c != ']').and_then(|text_end| {
                    if s.get(text_end + 1) != Some(&'(') {
                        return None;
                    }
                    let to = closing(s, text_end + 2, &[')'], |c| c != ')' && !c.is_whitespace())?;
                    let link: String = s[text_end + 2..to].iter().collect();
                    Some((Run { link: Some(link), ..style.clone() }, i + 1, text_end, to + 1, true))
                })
            } else {
                None
            }
        });
        match found {
            Some((inner, start, end, next, nested)) => {
                add(&s[done..i], style, out);
                if nested {
                    walk(&s[start..end], &inner, out);
                } else {
                    add(&s[start..end], &inner, out);
                }
                i = next;
                done = next;
            }
            None => i += 1,
        }
    }
    add(&s[done..], style, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str) -> Run {
        Run { text: text.into(), ..Run::default() }
    }

    #[test]
    fn blocks_of_every_kind() {
        let blocks = parse(
            "# Proposal\n\nFirst line\nwrapped on.\n\n## Costs ##\n- one\n  - nested\n1. first\n2) second\n\n> quoted\n\n---\n```\nlet x = 1;\n  y\n```\n| Item | CAD |\n|:--|--:|\n| A \\| B | 12 |\n| short |\nafter",
        );
        assert_eq!(
            blocks,
            vec![
                Block::Heading { level: 1, text: "Proposal".into() },
                Block::Paragraph("First line wrapped on.".into()),
                Block::Heading { level: 2, text: "Costs".into() },
                Block::Bullet { depth: 0, text: "one".into() },
                Block::Bullet { depth: 1, text: "nested".into() },
                Block::Numbered { depth: 0, number: 1, text: "first".into() },
                Block::Numbered { depth: 0, number: 2, text: "second".into() },
                Block::Quote("quoted".into()),
                Block::Rule,
                Block::Code("let x = 1;\n  y".into()),
                Block::Table {
                    rows: vec![
                        vec!["Item".into(), "CAD".into()],
                        vec!["A | B".into(), "12".into()],
                        vec!["short".into(), "".into()],
                    ],
                    alignments: vec![Alignment::Left, Alignment::Right],
                },
                Block::Paragraph("after".into()),
            ]
        );
    }

    #[test]
    fn a_memos_fields_keep_their_lines_and_a_first_heading_is_the_title() {
        let blocks = parse("To: Helen Achebe\nFrom: Maya\n**Date:** 12 November\nSee the notes  \nbelow, and\nthe rest.");
        assert_eq!(
            blocks,
            vec![Block::Paragraph(
                "To: Helen Achebe\nFrom: Maya\n**Date:** 12 November See the notes\nbelow, and the rest.".into()
            )]
        );
        let (title, rest) = split_title("Given", parse("---\n# **Real** title\ntext"));
        assert_eq!(title, "Real title");
        assert_eq!(rest, vec![Block::Paragraph("text".into())]);
        let (title, rest) = split_title("Given", parse("## Not a title\ntext"));
        assert_eq!((title.as_str(), rest.len()), ("Given", 2));
        // Not a heading, a list or a rule.
        assert_eq!(parse("#hashtag\n-dash\n--"), vec![Block::Paragraph("#hashtag -dash --".into())]);
        assert_eq!(parse("* * *"), vec![Block::Rule]);
        assert_eq!(parse("* item"), vec![Block::Bullet { depth: 0, text: "item".into() }]);
    }

    #[test]
    fn runs_with_their_styles() {
        assert_eq!(
            runs("Pay **CAD 12,085.35** by *Friday*, see `GL 6410` or [the guide](https://x.test/a)."),
            vec![
                run("Pay "),
                Run { bold: true, ..run("CAD 12,085.35") },
                run(" by "),
                Run { italic: true, ..run("Friday") },
                run(", see "),
                Run { code: true, ..run("GL 6410") },
                run(" or "),
                Run { link: Some("https://x.test/a".into()), ..run("the guide") },
                run("."),
            ]
        );
        // Marks inside bold, underscores inside words, escapes, and marks left open.
        assert_eq!(
            runs("__bold *and* more__ snake_case_name \\*kept\\* 2 * 3 * 4 **open"),
            vec![
                Run { bold: true, ..run("bold ") },
                Run { bold: true, italic: true, ..run("and") },
                Run { bold: true, ..run(" more") },
                run(" snake_case_name *kept* 2 * 3 * 4 **open"),
            ]
        );
        assert_eq!(plain("**Total:** `x_y` _done_"), "Total: x_y done");
    }
}
