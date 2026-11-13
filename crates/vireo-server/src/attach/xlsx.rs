//! Excel (.xlsx, .xlsm) cells as text. The file is a ZIP of XML: `xl/workbook.xml` names
//! the sheets, its relationships say which file each is, `xl/sharedStrings.xml` holds the
//! text cells point at, `xl/styles.xml` says which number formats are dates, and each
//! sheet's file has its rows of cells.
//!
//! A sheet is written as the Windows app's `Spreadsheet.cs` writes it, so a model sees
//! the same thing on both machines: a line naming the sheet and its used range, the
//! column letters, then a row a line with its number and its cells between bars. A cell
//! reads as Excel shows it (text, numbers without float noise, dates as 2026-11-12,
//! TRUE/FALSE; a formula's saved result, or the formula when none was saved).

use super::docx::{Zip, unescape_into};

/// What one read returns at most, so a big sheet doesn't fill the model's context.
pub const MAX_ROWS: usize = 200;
pub const MAX_COLUMNS: u32 = 40;
/// `read_file` shows this much of each sheet of a big workbook (a small one, whole).
pub const OVERVIEW_ROWS: usize = 30;
/// The largest workbook read, as a file.
pub const MAX_FILE: u64 = 30 << 20;

#[derive(Debug)]
pub struct Sheet {
    pub name: String,
    /// The rows that have values, in order: the row's number (from 1) and its cells, each
    /// a column (from 1) and its text.
    pub rows: Vec<(u32, Vec<(u32, String)>)>,
    /// A chart sheet has no cells.
    pub chart: bool,
}

#[derive(Debug)]
pub struct Workbook {
    pub sheets: Vec<Sheet>,
}

/// Reads a workbook's sheets, or says why it can't, worded for the model.
pub fn open(data: &[u8]) -> Result<Workbook, String> {
    if data.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        return Err(
            "This workbook is password-protected or in the old .xls format, so its cells \
             can't be read. Save it as an unprotected .xlsx."
                .into(),
        );
    }
    let workbook = |e: String| e.replace(".docx", "workbook");
    let zip = Zip::open(data).map_err(workbook)?;
    let part = |name: &str| -> Result<Option<String>, String> {
        Ok(zip
            .read(name)
            .map_err(workbook)?
            .map(|b| String::from_utf8_lossy(&b).into_owned()))
    };
    let book = part("xl/workbook.xml")?
        .ok_or("This isn't an Excel workbook (xl/workbook.xml is missing).")?;
    let relationships = part("xl/_rels/workbook.xml.rels")?.unwrap_or_default();
    let shared = part("xl/sharedStrings.xml")?
        .map(|x| shared_strings(&x))
        .unwrap_or_default();
    let dates = part("xl/styles.xml")?
        .map(|x| date_styles(&x))
        .unwrap_or_default();

    // Each relationship's id and the file it names.
    let mut targets = Vec::new();
    for piece in Pieces::new(&relationships) {
        if let Piece::Open("Relationship", attrs, _) = piece
            && let (Some(id), Some(target)) = (attribute(attrs, "Id"), attribute(attrs, "Target"))
        {
            let target = unescaped(target);
            let file = match target.strip_prefix('/') {
                Some(absolute) => absolute.to_string(),
                None => format!("xl/{target}"),
            };
            targets.push((id.to_string(), file));
        }
    }
    let mut sheets = Vec::new();
    for piece in Pieces::new(&book) {
        let Piece::Open("sheet", attrs, _) = piece else {
            continue;
        };
        let name = unescaped(attribute(attrs, "name").unwrap_or(""));
        let file = attribute(attrs, "r:id")
            .or_else(|| attribute(attrs, "id"))
            .and_then(|id| targets.iter().find(|t| t.0 == id))
            .map(|t| t.1.clone());
        let xml = match &file {
            Some(f) if f.contains("worksheets/") => part(f)?,
            _ => None,
        };
        sheets.push(match xml {
            Some(xml) => Sheet {
                name,
                rows: sheet_rows(&xml, &shared, &dates),
                chart: false,
            },
            None => Sheet {
                name,
                rows: Vec::new(),
                chart: true,
            },
        });
    }
    if sheets.is_empty() {
        return Err("This workbook has no sheets.".into());
    }
    Ok(Workbook { sheets })
}

impl Workbook {
    fn names(&self) -> String {
        self.sheets
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// `read_file`: every sheet's name and rows. A small workbook comes whole; of a big
    /// one, the start of each sheet.
    pub fn describe(&self) -> String {
        let total: usize = self.sheets.iter().map(|s| s.rows.len()).sum();
        let rows = if total <= MAX_ROWS { MAX_ROWS } else { OVERVIEW_ROWS };
        let n = self.sheets.len();
        let mut text = format!(
            "Excel workbook with {n} sheet{}: {}\n",
            if n == 1 { "" } else { "s" },
            self.names()
        );
        for sheet in &self.sheets {
            text.push('\n');
            text.push_str(&sheet.text(rows));
        }
        text.trim_end().to_string()
    }

    /// `read_spreadsheet`: the sheets' names, then the first sheet's cells.
    pub fn read(&self) -> String {
        format!("Sheets: {}\n\n{}", self.names(), self.sheets[0].text(MAX_ROWS))
            .trim_end()
            .to_string()
    }
}

impl Sheet {
    pub fn text(&self, max_rows: usize) -> String {
        if self.chart {
            return format!("Sheet \"{}\" is a chart sheet, with no cells.\n", self.name);
        }
        if self.rows.is_empty() {
            return format!("Sheet \"{}\" is empty.\n", self.name);
        }
        let columns = self.rows.iter().flat_map(|r| r.1.iter().map(|c| c.0));
        let (first, last) = (columns.clone().min().unwrap(), columns.max().unwrap());
        let (top, bottom) = (self.rows[0].0, self.rows[self.rows.len() - 1].0);
        let used = format!(
            "{}{top}:{}{bottom}",
            column_name(first),
            column_name(last)
        );
        let last_shown = last.min(first + MAX_COLUMNS - 1);
        let mut text = format!("Sheet \"{}\", {used} (used range {used})\nRow", self.name);
        for c in first..=last_shown {
            text.push_str(" | ");
            text.push_str(&column_name(c));
        }
        text.push('\n');
        for (row, cells) in self.rows.iter().take(max_rows) {
            text.push_str(&row.to_string());
            for c in first..=last_shown {
                text.push_str(" | ");
                if let Some(cell) = cells.iter().find(|cell| cell.0 == c) {
                    text.push_str(&cell.1);
                }
            }
            text.push('\n');
        }
        if self.rows.len() > max_rows {
            text.push_str(&format!(
                "…{} more rows with values, up to row {bottom}.\n",
                self.rows.len() - max_rows
            ));
        }
        if last > last_shown {
            text.push_str(&format!(
                "…columns after {} left out.\n",
                column_name(last_shown)
            ));
        }
        text
    }
}

// ---- The parts' XML --------------------------------------------------------------------

/// A piece of XML: a tag opened (its name without a namespace prefix, its attributes,
/// and whether it closes itself), a tag closed, or the text between tags.
pub(super) enum Piece<'a> {
    Open(&'a str, &'a str, bool),
    Close(&'a str),
    Text(&'a str),
}

pub(super) struct Pieces<'a> {
    rest: &'a str,
}

impl<'a> Pieces<'a> {
    pub(super) fn new(xml: &'a str) -> Self {
        Self { rest: xml }
    }
}

fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

impl<'a> Iterator for Pieces<'a> {
    type Item = Piece<'a>;

    fn next(&mut self) -> Option<Piece<'a>> {
        loop {
            if self.rest.is_empty() {
                return None;
            }
            if !self.rest.starts_with('<') {
                let end = self.rest.find('<').unwrap_or(self.rest.len());
                let text = &self.rest[..end];
                self.rest = &self.rest[end..];
                return Some(Piece::Text(text));
            }
            if let Some(body) = self.rest.strip_prefix("<![CDATA[") {
                let end = body.find("]]>").unwrap_or(body.len());
                self.rest = body.get(end + 3..).unwrap_or("");
                return Some(Piece::Text(&body[..end]));
            }
            let end = self.rest.find('>')?;
            let tag = &self.rest[1..end];
            self.rest = &self.rest[end + 1..];
            if tag.starts_with('?') || tag.starts_with('!') {
                continue; // the declaration, a comment
            }
            if let Some(name) = tag.strip_prefix('/') {
                return Some(Piece::Close(local(name.trim())));
            }
            let (tag, empty) = match tag.strip_suffix('/') {
                Some(t) => (t, true),
                None => (tag, false),
            };
            let split = tag.find(char::is_whitespace).unwrap_or(tag.len());
            return Some(Piece::Open(local(&tag[..split]), &tag[split..], empty));
        }
    }
}

/// An attribute's value in a tag's attributes (` r="A1" t="s"`), still escaped.
pub(super) fn attribute<'t>(attrs: &'t str, key: &str) -> Option<&'t str> {
    let mut from = 0;
    while let Some(found) = attrs[from..].find(key) {
        let at = from + found;
        let after = at + key.len();
        // The whole name, not the end of a longer one ("id" in "r:id").
        let starts = at == 0 || attrs[..at].ends_with(char::is_whitespace);
        if starts && attrs[after..].starts_with('=') {
            let quote = attrs[after + 1..].chars().next()?;
            let value = &attrs[after + 2..];
            return Some(&value[..value.find(quote)?]);
        }
        from = after;
    }
    None
}

pub(super) fn unescaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    unescape_into(&mut out, text);
    out
}

/// Line breaks and bars would break the row layout.
fn clean(text: &str) -> String {
    text.replace('\r', "")
        .replace('\n', " ⏎ ")
        .replace('|', "¦")
        .trim()
        .to_string()
}

/// `xl/sharedStrings.xml`: each `<si>`'s text, its runs joined (phonetic guides left out).
fn shared_strings(xml: &str) -> Vec<String> {
    let mut strings = Vec::new();
    let mut current = String::new();
    let (mut in_text, mut phonetic) = (false, false);
    for piece in Pieces::new(xml) {
        match piece {
            Piece::Open("si", _, empty) => {
                current.clear();
                if empty {
                    strings.push(String::new());
                }
            }
            Piece::Close("si") => strings.push(clean(&current)),
            Piece::Open("rPh", _, false) => phonetic = true,
            Piece::Close("rPh") => phonetic = false,
            Piece::Open("t", _, false) => in_text = true,
            Piece::Close("t") => in_text = false,
            Piece::Text(text) if in_text && !phonetic => unescape_into(&mut current, text),
            _ => {}
        }
    }
    strings
}

/// `xl/styles.xml`: the cell styles (by index) whose number format is a date. Excel's
/// built-in date formats, or a custom one with days, months, years, hours or seconds
/// outside quotes and brackets.
fn date_styles(xml: &str) -> Vec<bool> {
    let mut custom_dates = Vec::new();
    let mut styles = Vec::new();
    let mut in_cell_styles = false;
    for piece in Pieces::new(xml) {
        match piece {
            Piece::Open("numFmt", attrs, _) => {
                if let (Some(id), Some(code)) = (
                    attribute(attrs, "numFmtId").and_then(|v| v.parse::<u32>().ok()),
                    attribute(attrs, "formatCode"),
                ) && is_date_format(&unescaped(code))
                {
                    custom_dates.push(id);
                }
            }
            Piece::Open("cellXfs", _, false) => in_cell_styles = true,
            Piece::Close("cellXfs") => in_cell_styles = false,
            Piece::Open("xf", attrs, _) if in_cell_styles => {
                let id = attribute(attrs, "numFmtId")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(0);
                let built_in = matches!(id, 14..=22 | 27..=36 | 45..=47 | 50..=58);
                styles.push(built_in || custom_dates.contains(&id));
            }
            _ => {}
        }
    }
    styles
}

fn is_date_format(code: &str) -> bool {
    let (mut quoted, mut bracketed, mut escaped) = (false, false, false);
    for c in code.chars() {
        if escaped {
            escaped = false;
        } else if quoted {
            quoted = c != '"';
        } else if bracketed {
            bracketed = c != ']';
        } else {
            match c {
                '"' => quoted = true,
                '[' => bracketed = true,
                '\\' => escaped = true,
                'd' | 'm' | 'y' | 'h' | 's' | 'D' | 'M' | 'Y' | 'H' | 'S' => return true,
                _ => {}
            }
        }
    }
    false
}

/// A sheet's rows that have values, with each cell's text.
fn sheet_rows(xml: &str, shared: &[String], dates: &[bool]) -> Vec<(u32, Vec<(u32, String)>)> {
    let mut rows: Vec<(u32, Vec<(u32, String)>)> = Vec::new();
    // Where the next row and cell are when they don't say (some writers leave `r` out).
    let (mut next_row, mut next_column) = (1u32, 1u32);
    let mut row: (u32, Vec<(u32, String)>) = (0, Vec::new());
    // The cell being read: its column, type, style, and what's been read of it.
    let mut cell: Option<(u32, &str, Option<usize>)> = None;
    let (mut value, mut formula, mut inline) = (String::new(), String::new(), String::new());
    let mut reading = "";
    for piece in Pieces::new(xml) {
        match piece {
            Piece::Open("row", attrs, empty) => {
                let number = attribute(attrs, "r")
                    .and_then(|r| r.parse().ok())
                    .unwrap_or(next_row);
                next_row = number + 1;
                next_column = 1;
                row = (number, Vec::new());
                let _ = empty;
            }
            Piece::Close("row") => {
                if !row.1.is_empty() {
                    row.1.sort_by_key(|c| c.0);
                    rows.push(std::mem::take(&mut row));
                }
            }
            Piece::Open("c", attrs, empty) => {
                let column = attribute(attrs, "r")
                    .and_then(column_of)
                    .unwrap_or(next_column);
                next_column = column + 1;
                if !empty {
                    cell = Some((
                        column,
                        attribute(attrs, "t").unwrap_or("n"),
                        attribute(attrs, "s").and_then(|s| s.parse().ok()),
                    ));
                    value.clear();
                    formula.clear();
                    inline.clear();
                }
            }
            Piece::Close("c") => {
                if let Some((column, kind, style)) = cell.take() {
                    let is_date = style.and_then(|s| dates.get(s)).copied().unwrap_or(false);
                    let text = cell_text(kind, value.trim(), &formula, &inline, shared, is_date);
                    if !text.is_empty() {
                        row.1.push((column, text));
                    }
                }
            }
            Piece::Open(name @ ("v" | "f" | "t"), _, false) if cell.is_some() => reading = name,
            Piece::Close("v" | "f" | "t") => reading = "",
            Piece::Text(text) => match reading {
                "v" => unescape_into(&mut value, text),
                "f" => unescape_into(&mut formula, text),
                "t" => unescape_into(&mut inline, text),
                _ => {}
            },
            _ => {}
        }
    }
    rows.sort_by_key(|r| r.0);
    rows
}

/// A cell as Excel would show it.
fn cell_text(
    kind: &str,
    raw: &str,
    formula: &str,
    inline: &str,
    shared: &[String],
    is_date: bool,
) -> String {
    match kind {
        "s" => raw
            .parse::<usize>()
            .ok()
            .and_then(|i| shared.get(i))
            .cloned()
            .unwrap_or_default(),
        "inlineStr" => clean(inline),
        "b" => if raw == "1" { "TRUE" } else { "FALSE" }.to_string(),
        // A formula written without a saved result shows as the formula.
        _ if raw.is_empty() => {
            if formula.is_empty() {
                String::new()
            } else {
                format!("={formula}")
            }
        }
        "n" => match raw.parse::<f64>() {
            Ok(n) if n.is_finite() => {
                if is_date && n > 0.0 && n < 2_958_466.0 {
                    date_text(n)
                } else {
                    number_text(n)
                }
            }
            _ => clean(raw),
        },
        _ => clean(raw),
    }
}

/// A number to 15 significant digits, the most a cell holds exactly: 0.1 + 0.2 is saved
/// as 0.30000000000000004 and shows as 0.3.
pub fn number_text(n: f64) -> String {
    let rounded: f64 = format!("{n:.14e}").parse().unwrap_or(n);
    let text = format!("{rounded}");
    if text == "-0" { "0".into() } else { text }
}

/// Excel's day number (days since 1899-12-30, the time of day its fraction) as
/// 2026-11-12, or 2026-11-12 09:30 when it has a time.
fn date_text(serial: f64) -> String {
    let mut days = serial.floor() as i64;
    let mut minutes = ((serial - serial.floor()) * 1440.0).round() as i64;
    if minutes >= 1440 {
        days += 1;
        minutes = 0;
    }
    // Days since 1970-01-01, then the civil date (Howard Hinnant's algorithm).
    let z = days - 25_569 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    if minutes == 0 {
        format!("{year:04}-{month:02}-{day:02}")
    } else {
        format!(
            "{year:04}-{month:02}-{day:02} {:02}:{:02}",
            minutes / 60,
            minutes % 60
        )
    }
}

/// "A", "Z", "AA": a column's letters from its number (from 1).
pub fn column_name(mut column: u32) -> String {
    let mut name = Vec::new();
    while column > 0 {
        column -= 1;
        name.push(b'A' + (column % 26) as u8);
        column /= 26;
    }
    name.reverse();
    String::from_utf8(name).unwrap_or_default()
}

/// The column of a cell reference ("AB12" is 28).
pub(super) fn column_of(reference: &str) -> Option<u32> {
    let mut column = 0u32;
    for c in reference.chars().take_while(char::is_ascii_alphabetic) {
        column = column.checked_mul(26)? + (c.to_ascii_uppercase() as u32 - 'A' as u32 + 1);
    }
    (column > 0).then_some(column)
}

#[cfg(test)]
mod tests {
    use super::super::docx::tests::stored_zip;
    use super::*;

    const BOOK: &str = r#"<?xml version="1.0"?><workbook xmlns:r="x"><sheets>
        <sheet name="Stock &amp; Orders" sheetId="1" r:id="rId1"/>
        <sheet name="Chart" sheetId="2" r:id="rId2"/>
        <sheet name="Blank" sheetId="3" r:id="rId3"/></sheets></workbook>"#;
    const RELS: &str = r#"<Relationships>
        <Relationship Id="rId1" Type="t" Target="worksheets/sheet1.xml"/>
        <Relationship Id="rId2" Type="t" Target="chartsheets/sheet1.xml"/>
        <Relationship Id="rId3" Type="t" Target="/xl/worksheets/sheet3.xml"/></Relationships>"#;
    const STRINGS: &str = r#"<sst><si><t>Item</t></si><si><r><t>Hood</t></r><r><t>ies</t></r></si>
        <si><t>a | b
c</t><rPh><t>guide</t></rPh></si></sst>"#;
    const STYLES: &str = r##"<styleSheet><numFmts><numFmt numFmtId="164" formatCode="yyyy\-mm\-dd"/>
        <numFmt numFmtId="165" formatCode="#,##0 &quot;days&quot;"/></numFmts>
        <cellStyleXfs><xf numFmtId="14"/></cellStyleXfs>
        <cellXfs><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="165"/></cellXfs>
        </styleSheet>"##;
    const SHEET: &str = r#"<worksheet><sheetData>
        <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Qty</t></is></c><c r="D1" t="str"><v>Due</v></c></row>
        <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>0.30000000000000004</v></c><c r="C2" s="3"><v>12</v></c><c r="D2" s="1"><v>46338</v></c></row>
        <row r="3"><c r="A3" t="s"><v>2</v></c><c r="B3"><f>SUM(B1:B2)</f></c><c r="C3" t="b"><v>1</v></c><c r="D3" s="2"><v>46338.395833333336</v></c></row>
        <row r="4"><c r="A4" s="1"/></row>
        <row r="5"><c r="B5"><f>B2*2</f><v>-1250.5</v></c></row>
        </sheetData></worksheet>"#;

    fn workbook() -> Vec<u8> {
        stored_zip(&[
            ("xl/workbook.xml", BOOK.as_bytes()),
            ("xl/_rels/workbook.xml.rels", RELS.as_bytes()),
            ("xl/sharedStrings.xml", STRINGS.as_bytes()),
            ("xl/styles.xml", STYLES.as_bytes()),
            ("xl/worksheets/sheet1.xml", SHEET.as_bytes()),
            ("xl/worksheets/sheet3.xml", b"<worksheet><sheetData/></worksheet>"),
        ])
    }

    #[test]
    fn a_workbook_reads_as_excel_shows_it() {
        let book = open(&workbook()).unwrap();
        assert_eq!(
            book.read(),
            "Sheets: Stock & Orders, Chart, Blank\n\n\
             Sheet \"Stock & Orders\", A1:D5 (used range A1:D5)\n\
             Row | A | B | C | D\n\
             1 | Item | Qty |  | Due\n\
             2 | Hoodies | 0.3 | 12 | 2026-11-12\n\
             3 | a ¦ b ⏎ c | =SUM(B1:B2) | TRUE | 2026-11-12 09:30\n\
             5 |  | -1250.5 |  | "
                .trim_end()
        );
        let described = book.describe();
        assert!(described.starts_with(
            "Excel workbook with 3 sheets: Stock & Orders, Chart, Blank\n\nSheet \"Stock & Orders\""
        ));
        assert!(described.contains("Sheet \"Chart\" is a chart sheet, with no cells."));
        assert!(described.ends_with("Sheet \"Blank\" is empty."));
    }

    #[test]
    fn a_long_or_wide_sheet_is_cut_with_a_note() {
        let mut xml = String::from("<worksheet><sheetData>");
        for r in 1..=250 {
            xml.push_str(&format!("<row r=\"{r}\"><c r=\"A{r}\"><v>{r}</v></c></row>"));
        }
        xml.push_str("<row r=\"251\"><c r=\"AZ251\"><v>9</v></c></row></sheetData></worksheet>");
        let rows = sheet_rows(&xml, &[], &[]);
        let sheet = Sheet {
            name: "Big".into(),
            rows,
            chart: false,
        };
        let text = sheet.text(MAX_ROWS);
        assert!(text.starts_with("Sheet \"Big\", A1:AZ251 (used range A1:AZ251)\nRow | A | B"));
        assert!(text.contains("\n200 | 200 |"));
        assert!(!text.contains("\n201 |"));
        assert!(text.contains("…51 more rows with values, up to row 251.\n"));
        assert!(text.ends_with("…columns after AN left out.\n"));
        // A big workbook's overview shows the start of each sheet.
        let book = Workbook { sheets: vec![sheet] };
        assert!(book.describe().contains("…221 more rows with values"));
    }

    #[test]
    fn cells_without_references_and_odd_files() {
        let rows = sheet_rows(
            "<worksheet><sheetData><row><c><v>1</v></c><c><v>2</v></c></row>\
             <row><c t=\"e\"><v>#DIV/0!</v></c></row></sheetData></worksheet>",
            &[],
            &[],
        );
        assert_eq!(
            rows,
            vec![
                (1, vec![(1, "1".to_string()), (2, "2".to_string())]),
                (2, vec![(1, "#DIV/0!".to_string())])
            ]
        );
        assert!(open(&[0xD0, 0xCF, 0x11, 0xE0, 0, 0]).unwrap_err().contains("old .xls"));
        assert!(open(b"not a zip").is_err());
        let no_book = stored_zip(&[("word/document.xml", b"<w:document/>")]);
        assert!(open(&no_book).unwrap_err().contains("isn't an Excel workbook"));
    }

    /// Prints a real workbook as `read_file` gives it, every row:
    /// `VIREO_XLSX=<file> cargo test -p vireo-server print_a_workbook -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn print_a_workbook() {
        let path = std::env::var("VIREO_XLSX").expect("VIREO_XLSX names the workbook");
        let book = open(&std::fs::read(path).unwrap()).unwrap();
        for sheet in &book.sheets {
            println!("{}", sheet.text(usize::MAX));
        }
    }

    #[test]
    fn numbers_dates_and_columns() {
        assert_eq!(number_text(1234.5), "1234.5");
        assert_eq!(number_text(21470.0), "21470");
        assert_eq!(number_text(-0.0), "0");
        assert_eq!(number_text(12085.349999999999), "12085.35");
        assert_eq!(date_text(45658.0), "2025-01-01");
        assert_eq!(date_text(60.0 + 1.0), "1900-03-01");
        assert_eq!(column_name(1), "A");
        assert_eq!(column_name(26), "Z");
        assert_eq!(column_name(27), "AA");
        assert_eq!(column_of("AB12"), Some(28));
        assert!(is_date_format("d-mmm-yy"));
        assert!(!is_date_format("#,##0.00"));
        assert!(!is_date_format("0 \"hours\""));
        assert!(!is_date_format("[Red]0.0"));
    }
}
