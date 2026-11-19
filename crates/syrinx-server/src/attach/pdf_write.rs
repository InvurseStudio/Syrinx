//! Writing a PDF from a title and the model's Markdown (`create_pdf`), laid out as the
//! Word documents `docx_write` makes: US Letter with the same margins, a title, headings,
//! lists, quotes, code, tables and rules. It uses the fonts every PDF reader has
//! (Helvetica and Courier), so nothing is embedded; their character set is Windows-1252,
//! and a character outside it shows as "?" (`create` says which were).

use super::markdown::{self, Block, Run};

/// US Letter in points, with `docx_write`'s margins (0.9" at the sides, 0.8" top and bottom).
const PAGE_WIDTH: f32 = 612.0;
const PAGE_HEIGHT: f32 = 792.0;
const SIDE: f32 = 64.8;
const END: f32 = 57.6;
const TEXT_WIDTH: f32 = PAGE_WIDTH - 2.0 * SIDE;
const BODY: f32 = 11.0;
const CODE: f32 = 9.5;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Font {
    Regular,
    Bold,
    Italic,
    BoldItalic,
    Mono,
}

impl Font {
    fn name(self) -> &'static str {
        match self {
            Font::Regular => "F1",
            Font::Bold => "F2",
            Font::Italic => "F3",
            Font::BoldItalic => "F4",
            Font::Mono => "F5",
        }
    }

    fn of(run: &Run, bold: bool) -> Font {
        match (run.code, run.bold || bold, run.italic) {
            (true, _, _) => Font::Mono,
            (_, true, true) => Font::BoldItalic,
            (_, true, false) => Font::Bold,
            (_, false, true) => Font::Italic,
            _ => Font::Regular,
        }
    }

    /// A character's width, in thousandths of the font's size (the fonts' AFM metrics).
    fn width(self, byte: u8) -> f32 {
        const REGULAR: [u16; 95] = [
            278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556,
            556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722,
            667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667,
            944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556, 556, 278, 556, 556,
            222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500,
            334, 260, 334, 584,
        ];
        const BOLD: [u16; 95] = [
            278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556,
            556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611, 975, 722, 722, 722, 722,
            667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667,
            944, 667, 667, 611, 333, 278, 333, 584, 556, 333, 556, 611, 556, 611, 556, 333, 611, 611,
            278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500,
            389, 280, 389, 584,
        ];
        let table = match self {
            Font::Mono => return 600.0,
            Font::Bold | Font::BoldItalic => &BOLD,
            Font::Regular | Font::Italic => &REGULAR,
        };
        match byte {
            32..=126 => f32::from(table[usize::from(byte - 32)]),
            // Bullets and dashes; accented letters are about an average letter's width.
            0x95 => 350.0,
            0x96 => 556.0,
            0x97 => 1000.0,
            0x85 => 1000.0,
            _ => 556.0,
        }
    }
}

/// A character in Windows-1252 (the fonts' WinAnsiEncoding), if it's there.
fn win_ansi(c: char) -> Option<u8> {
    let code = c as u32;
    match code {
        0x20..=0x7e | 0xa0..=0xff => Some(code as u8),
        _ => Some(match c {
            '\u{20ac}' => 0x80,
            '\u{201a}' => 0x82,
            '\u{201e}' => 0x84,
            '\u{2026}' => 0x85,
            '\u{2020}' => 0x86,
            '\u{2021}' => 0x87,
            '\u{2030}' => 0x89,
            '\u{0160}' => 0x8a,
            '\u{2039}' => 0x8b,
            '\u{0152}' => 0x8c,
            '\u{017d}' => 0x8e,
            '\u{2018}' => 0x91,
            '\u{2019}' => 0x92,
            '\u{201c}' => 0x93,
            '\u{201d}' => 0x94,
            '\u{2022}' => 0x95,
            '\u{2013}' => 0x96,
            '\u{2014}' => 0x97,
            '\u{2122}' => 0x99,
            '\u{0161}' => 0x9a,
            '\u{203a}' => 0x9b,
            '\u{0153}' => 0x9c,
            '\u{017e}' => 0x9e,
            '\u{0178}' => 0x9f,
            '\t' => b' ',
            '\n' => b'\n',
            '\u{2212}' => b'-',
            _ => return None,
        }),
    }
}

/// A piece of a line: its text (Windows-1252) in one font.
#[derive(Clone, Debug)]
struct Piece {
    bytes: Vec<u8>,
    font: Font,
    size: f32,
    link: bool,
}

/// A line laid out: its pieces, and where it starts.
struct Line {
    pieces: Vec<Piece>,
    x: f32,
    /// A list item's bullet or number, and where it goes.
    mark: Option<(f32, Piece)>,
}

struct Writer {
    pages: Vec<Vec<u8>>,
    page: Vec<u8>,
    y: f32,
    /// Characters the fonts don't have, each once.
    missing: Vec<char>,
}

impl Writer {
    fn new() -> Writer {
        Writer {
            pages: Vec::new(),
            page: Vec::new(),
            y: PAGE_HEIGHT - END,
            missing: Vec::new(),
        }
    }

    fn encode(&mut self, text: &str) -> Vec<u8> {
        text.chars()
            .filter(|c| *c != '\r')
            .map(|c| {
                win_ansi(c).unwrap_or_else(|| {
                    if !self.missing.contains(&c) {
                        self.missing.push(c);
                    }
                    b'?'
                })
            })
            .collect()
    }

    fn new_page(&mut self) {
        let page = std::mem::take(&mut self.page);
        self.pages.push(page);
        self.y = PAGE_HEIGHT - END;
    }

    /// Room for `height` more, or a new page.
    fn room(&mut self, height: f32) {
        if self.y - height < END && self.y < PAGE_HEIGHT - END {
            self.new_page();
        }
    }

    fn gap(&mut self, points: f32) {
        if self.y < PAGE_HEIGHT - END {
            self.y -= points;
        }
    }

    /// Runs wrapped into lines `width` wide, starting at `x` (`first_x` for the first).
    fn lay_out(&mut self, runs: &[Run], size: f32, bold: bool, x: f32, first_x: f32, width: f32) -> Vec<Line> {
        let mut lines: Vec<Line> = Vec::new();
        let mut line = Line { pieces: Vec::new(), x: first_x, mark: None };
        let mut used = 0.0;
        let room = |line: &Line| width - (line.x - x);
        for run in runs {
            let font = Font::of(run, bold);
            let size = if font == Font::Mono { size * 0.9 } else { size };
            let bytes = self.encode(&run.text);
            // Words, each with the spaces after it.
            let mut words: Vec<Vec<u8>> = Vec::new();
            let mut word = Vec::new();
            for &b in &bytes {
                if b == b'\n' {
                    words.push(std::mem::take(&mut word));
                    words.push(vec![b'\n']);
                    continue;
                }
                word.push(b);
                if b == b' ' {
                    words.push(std::mem::take(&mut word));
                }
            }
            if !word.is_empty() {
                words.push(word);
            }
            for word in words {
                if word == b"\n" {
                    lines.push(std::mem::replace(&mut line, Line { pieces: Vec::new(), x, mark: None }));
                    used = 0.0;
                    continue;
                }
                let measure = |w: &[u8]| w.iter().map(|&b| font.width(b)).sum::<f32>() * size / 1000.0;
                let trimmed = trimmed_end(&word);
                if used > 0.0 && used + measure(&trimmed) > room(&line) {
                    lines.push(std::mem::replace(&mut line, Line { pieces: Vec::new(), x, mark: None }));
                    used = 0.0;
                }
                // A word longer than the line is broken where it must be.
                let mut word = word;
                while used == 0.0 && measure(&trimmed_end(&word)) > room(&line) && word.len() > 1 {
                    let mut fit = 1;
                    while fit < word.len() && measure(&word[..fit + 1]) <= room(&line) {
                        fit += 1;
                    }
                    let rest = word.split_off(fit);
                    push_piece(&mut line, word, font, size, run.link.is_some());
                    lines.push(std::mem::replace(&mut line, Line { pieces: Vec::new(), x, mark: None }));
                    word = rest;
                }
                used += measure(&word);
                push_piece(&mut line, word, font, size, run.link.is_some());
            }
        }
        if !line.pieces.is_empty() || lines.is_empty() {
            lines.push(line);
        }
        lines
    }

    /// Writes lines down the page, `leading` apart.
    fn write_lines(&mut self, lines: Vec<Line>, leading: f32) {
        for line in lines {
            self.room(leading);
            self.y -= leading;
            let mark = line.mark.as_ref().map(|(x, piece)| (*x, piece));
            let pieces = mark.into_iter().chain(
                line.pieces
                    .iter()
                    .scan(line.x, |x, piece| {
                        let at = *x;
                        *x += piece.bytes.iter().map(|&b| piece.font.width(b)).sum::<f32>() * piece.size / 1000.0;
                        Some((at, piece))
                    }),
            );
            for (x, piece) in pieces {
                let text: Vec<u8> = piece.bytes.clone();
                if text.is_empty() {
                    continue;
                }
                let colour = if piece.link { "0.05 0.33 0.65 rg" } else { "0 0 0 rg" };
                self.page.extend_from_slice(
                    format!("BT {colour} /{} {:.1} Tf {x:.2} {:.2} Td (", piece.font.name(), piece.size, self.y)
                        .as_bytes(),
                );
                escape_into(&text, &mut self.page);
                self.page.extend_from_slice(b") Tj ET\n");
            }
        }
    }

    fn paragraph(&mut self, text: &str, size: f32, bold: bool, x: f32, leading: f32) {
        let runs = markdown::runs(text);
        let lines = self.lay_out(&runs, size, bold, x, x, TEXT_WIDTH - (x - SIDE));
        self.write_lines(lines, leading);
    }

    fn rule(&mut self) {
        self.room(12.0);
        self.y -= 6.0;
        self.page.extend_from_slice(
            format!("0.75 G 0.5 w {SIDE:.2} {y:.2} m {:.2} {y:.2} l S\n", PAGE_WIDTH - SIDE, y = self.y).as_bytes(),
        );
        self.y -= 6.0;
    }

    fn table(&mut self, rows: &[Vec<String>]) {
        let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
        if columns == 0 {
            return;
        }
        let width = TEXT_WIDTH / columns as f32;
        const PAD: f32 = 4.0;
        const LEADING: f32 = 13.0;
        for (r, row) in rows.iter().enumerate() {
            let cells: Vec<Vec<Line>> = (0..columns)
                .map(|c| {
                    let text = row.get(c).map_or("", String::as_str);
                    let x = SIDE + c as f32 * width + PAD;
                    self.lay_out(&markdown::runs(text), 10.0, r == 0, x, x, width - 2.0 * PAD)
                })
                .collect();
            let height = cells.iter().map(Vec::len).max().unwrap_or(1) as f32 * LEADING + 2.0 * PAD;
            self.room(height);
            let top = self.y;
            for lines in cells {
                self.y = top - PAD;
                self.write_lines(lines, LEADING);
            }
            self.y = top - height;
            // The row's lines, grey, as the Word documents' tables have.
            let mut lines = format!("0.75 G 0.5 w {SIDE:.2} {top:.2} m {:.2} {top:.2} l S\n", PAGE_WIDTH - SIDE);
            lines.push_str(&format!("{SIDE:.2} {y:.2} m {:.2} {y:.2} l S\n", PAGE_WIDTH - SIDE, y = self.y));
            for c in 0..=columns {
                let x = SIDE + c as f32 * width;
                lines.push_str(&format!("{x:.2} {top:.2} m {x:.2} {:.2} l S\n", self.y));
            }
            self.page.extend_from_slice(lines.as_bytes());
        }
        self.y -= 6.0;
    }

    fn finish(mut self, title: &str) -> Vec<u8> {
        if !self.page.is_empty() || self.pages.is_empty() {
            let page = std::mem::take(&mut self.page);
            self.pages.push(page);
        }
        let count = self.pages.len();
        // Page numbers, at the foot of each page.
        if count > 1 {
            for (i, page) in self.pages.iter_mut().enumerate() {
                let label = format!("{}", i + 1);
                let x = PAGE_WIDTH / 2.0 - label.len() as f32 * 2.5;
                page.extend_from_slice(format!("BT 0.4 g /F1 9.0 Tf {x:.2} {:.2} Td ({label}) Tj ET\n", END / 2.0).as_bytes());
            }
        }
        // Objects: 1 the catalog, 2 the pages, 3-7 the fonts, 8 the info, then a page and
        // its contents for each page.
        let mut objects: Vec<Vec<u8>> = Vec::new();
        objects.push(b"<< /Type /Catalog /Pages 2 0 R >>".to_vec());
        let kids: Vec<String> = (0..count).map(|i| format!("{} 0 R", 9 + 2 * i)).collect();
        objects.push(format!("<< /Type /Pages /Kids [{}] /Count {count} >>", kids.join(" ")).into_bytes());
        for base in ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier"] {
            objects.push(
                format!("<< /Type /Font /Subtype /Type1 /BaseFont /{base} /Encoding /WinAnsiEncoding >>").into_bytes(),
            );
        }
        objects.push(format!("<< /Title {} /Producer (Vireo) >>", text_string(title)).into_bytes());
        for (i, content) in self.pages.iter().enumerate() {
            objects.push(
                format!(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] /Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R /F4 6 0 R /F5 7 0 R >> >> /Contents {} 0 R >>",
                    10 + 2 * i
                )
                .into_bytes(),
            );
            let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
            stream.extend_from_slice(content);
            stream.extend_from_slice(b"\nendstream");
            objects.push(stream);
        }
        let mut out = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
        let mut offsets = Vec::with_capacity(objects.len());
        for (i, object) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(object);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
        for offset in offsets {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R /Info 8 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }
}

fn trimmed_end(word: &[u8]) -> Vec<u8> {
    let end = word.iter().rposition(|&b| b != b' ').map_or(0, |p| p + 1);
    word[..end].to_vec()
}

fn push_piece(line: &mut Line, bytes: Vec<u8>, font: Font, size: f32, link: bool) {
    if let Some(last) = line.pieces.last_mut()
        && last.font == font
        && last.size == size
        && last.link == link
    {
        last.bytes.extend(bytes);
        return;
    }
    line.pieces.push(Piece { bytes, font, size, link });
}

/// Bytes in a PDF literal string: brackets and backslashes escaped.
fn escape_into(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        if matches!(b, b'(' | b')' | b'\\') {
            out.push(b'\\');
        }
        out.push(b);
    }
}

/// Text for the document's information (its title): UTF-16, as PDF readers read it.
fn text_string(text: &str) -> String {
    let mut hex = String::from("<FEFF");
    for unit in text.encode_utf16() {
        hex.push_str(&format!("{unit:04X}"));
    }
    hex.push('>');
    hex
}

/// A PDF from a title and Markdown, and the characters its fonts didn't have. A document
/// that opens with a top-level heading takes that as its title, as `docx_write` does.
pub fn create(title: &str, content: &str) -> (Vec<u8>, Vec<char>) {
    let (title, blocks) = markdown::split_title(title, markdown::parse(content));
    let mut w = Writer::new();
    if !title.trim().is_empty() {
        let runs = vec![Run { text: title.trim().to_string(), ..Run::default() }];
        let lines = w.lay_out(&runs, 22.0, true, SIDE, SIDE, TEXT_WIDTH);
        w.write_lines(lines, 27.0);
        w.gap(10.0);
    }
    for block in &blocks {
        match block {
            Block::Heading { level, text } => {
                let (size, before) = match level {
                    1 => (17.0, 12.0),
                    2 => (14.5, 10.0),
                    3 => (12.5, 8.0),
                    _ => (11.0, 6.0),
                };
                // A heading isn't left alone at the foot of a page.
                w.room(size * 1.3 + before + 30.0);
                w.gap(before);
                w.paragraph(text, size, true, SIDE, size * 1.3);
                w.gap(4.0);
            }
            Block::Paragraph(text) => {
                w.paragraph(text, BODY, false, SIDE, BODY * 1.4);
                w.gap(7.0);
            }
            Block::Bullet { depth, text } | Block::Numbered { depth, text, .. } => {
                let indent = SIDE + 18.0 * (*depth as f32 + 1.0);
                let mark = match block {
                    Block::Numbered { number, .. } => format!("{number}."),
                    _ => "\u{2022}".to_string(),
                };
                let mark = Piece { bytes: w.encode(&mark), font: Font::Regular, size: BODY, link: false };
                let mut lines =
                    w.lay_out(&markdown::runs(text), BODY, false, indent, indent, TEXT_WIDTH - (indent - SIDE));
                if let Some(first) = lines.first_mut() {
                    first.mark = Some((indent - 14.0, mark));
                }
                w.write_lines(lines, BODY * 1.4);
                w.gap(2.0);
            }
            Block::Quote(text) => {
                let runs: Vec<Run> = markdown::runs(text)
                    .into_iter()
                    .map(|r| Run { italic: true, ..r })
                    .collect();
                let x = SIDE + 18.0;
                let lines = w.lay_out(&runs, BODY, false, x, x, TEXT_WIDTH - 18.0);
                w.write_lines(lines, BODY * 1.4);
                w.gap(7.0);
            }
            Block::Code(code) => {
                let runs = vec![Run { text: code.clone(), code: true, ..Run::default() }];
                let lines = w.lay_out(&runs, CODE / 0.9, false, SIDE + 8.0, SIDE + 8.0, TEXT_WIDTH - 16.0);
                w.write_lines(lines, CODE * 1.3);
                w.gap(8.0);
            }
            Block::Table { rows, .. } => {
                w.gap(4.0);
                w.table(rows);
            }
            Block::Rule => w.rule(),
        }
    }
    let missing = std::mem::take(&mut w.missing);
    (w.finish(&title), missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(pdf: &[u8]) -> String {
        String::from_utf8_lossy(pdf).to_string()
    }

    #[test]
    fn a_document_is_a_whole_pdf_with_its_text() {
        let (pdf, missing) = create(
            "Trip plan",
            "## Days\n\nWe leave on **Friday** (early).\n\n- Pack \u{201c}light\u{201d}\n- Book the caf\u{e9}\n\n| Day | Cost |\n|---|---|\n| Fri | $120 |\n\n---\n\nDone \u{2192} home.",
        );
        let text = text_of(&pdf);
        assert!(text.starts_with("%PDF-1.4"));
        assert!(text.ends_with("%%EOF\n"));
        assert!(text.contains("(Trip plan) Tj"));
        assert!(text.contains("(Days) Tj"));
        assert!(text.contains("(Friday) Tj"), "bold run kept apart");
        assert!(text.contains("\\(early\\)."));
        assert!(text.contains("(Fri) Tj"));
        assert_eq!(missing, vec!['\u{2192}']);
        // Every object the cross-reference names starts where it says.
        let xref_at: usize = text.rsplit("startxref\n").next().unwrap().lines().next().unwrap().parse().unwrap();
        let xref = &pdf[xref_at..];
        assert!(xref.starts_with(b"xref\n"));
        let entries: Vec<usize> = String::from_utf8_lossy(xref)
            .lines()
            .skip(3)
            .take_while(|l| l.ends_with(" n "))
            .map(|l| l[..10].parse().unwrap())
            .collect();
        for (i, offset) in entries.iter().enumerate() {
            assert!(pdf[*offset..].starts_with(format!("{} 0 obj", i + 1).as_bytes()));
        }
    }

    #[test]
    fn a_long_document_runs_onto_more_pages_and_long_words_break() {
        let content = format!("{}\n\n{}", "A sentence that fills some of a line. ".repeat(400), "x".repeat(300));
        let (pdf, _) = create("Long", &content);
        let text = text_of(&pdf);
        let pages = text.matches("/Type /Page /Parent").count();
        assert!(pages >= 3, "{pages} pages");
        assert!(text.contains(&format!("/Count {pages}")));
    }

    /// Writes a sample to look at: `SYRINX_PDF_OUT=/path/x.pdf cargo test -p syrinx-server sample_pdf -- --ignored`.
    #[test]
    #[ignore]
    fn sample_pdf() {
        let content = "Notes for **Friday's** offsite, with *italic*, `code` and a [link](https://example.com).\n\n## Agenda\n\n1. Arrive at 9:00\n2. Review the \u{201c}Q3\u{201d} numbers \u{2014} all of them, including the long tail of small accounts that nobody looks at\n   - Revenue\n   - Costs\n3. Lunch\n\n> Keep it short.\n\n```\nfn main() {\n    println!(\"hi\");\n}\n```\n\n| Item | Owner | Cost |\n|---|---|---|\n| Venue | Sam | $1,200 |\n| Food for everyone who comes, with vegetarian options | Alex | $640 |\n\n---\n\nThat's all.";
        let (pdf, _) = create("Offsite plan", content);
        std::fs::write(std::env::var("SYRINX_PDF_OUT").unwrap(), pdf).unwrap();
    }
}
