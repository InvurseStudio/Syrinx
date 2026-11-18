//! Writing cells into an Excel workbook (.xlsx), as the Windows app's `Spreadsheet.Write`
//! does: rows of values from a cell of the first sheet, or after its last row, into a
//! workbook that's there or a new one.
//!
//! An existing workbook is changed as little as can be: only the first sheet's
//! `<sheetData>` is rewritten, with the cells written put among the ones already there
//! (each keeping the style it had, and a new cell taking the style of the one above it,
//! so a row added to a register looks like the rows before it); every other part of the
//! file (styles, other sheets, charts, widths) is copied as it is. Excel is told to work
//! the formulas out again when it opens the file, and its list of formula cells
//! (`calcChain.xml`), which the write may have made wrong, is left out. The file is
//! written as a ZIP without compression.

use super::docx::{Zip, stored_zip};
use super::xlsx::{self, Piece, Pieces, attribute, column_name, column_of, unescaped};
use crate::json::Value;

const MAX_COLUMN: u32 = 16_384;
const MAX_ROW: u32 = 1_048_576;

/// Where the rows go: from a cell, or after the sheet's last row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Start {
    Cell { column: u32, row: u32 },
    Append,
}

impl Start {
    /// "A1", "c12", "append" (or nothing, for A1).
    pub fn parse(text: &str) -> Result<Start, String> {
        let t = text.trim();
        if t.is_empty() {
            return Ok(Start::Cell { column: 1, row: 1 });
        }
        if t.eq_ignore_ascii_case("append") {
            return Ok(Start::Append);
        }
        let letters = t.chars().take_while(char::is_ascii_alphabetic).count();
        let row = t[letters..].parse::<u32>().ok().filter(|r| (1..=MAX_ROW).contains(r));
        match (column_of(t).filter(|c| *c <= MAX_COLUMN && letters > 0), row) {
            (Some(column), Some(row)) => Ok(Start::Cell { column, row }),
            _ => Err(format!("\"{text}\" isn't a cell like A1 (or \"append\").")),
        }
    }
}

/// The rows as lists of values, from the tool's `rows` argument (a JSON list of lists; a
/// model sometimes sends the JSON inside a string).
pub fn rows_from(json: &str) -> Result<Vec<Vec<Value>>, String> {
    const SHAPE: &str = "rows must be a list of rows, each a list of values.";
    let mut value = crate::json::parse(json).map_err(|_| SHAPE.to_string())?;
    if let Value::String(inner) = &value {
        value = crate::json::parse(inner).map_err(|_| SHAPE.to_string())?;
    }
    let Value::Array(rows) = value else {
        return Err(SHAPE.into());
    };
    rows.into_iter()
        .enumerate()
        .map(|(r, row)| match row {
            Value::Array(cells) => Ok(cells),
            _ => Err(format!("Row {} isn't a list of values.", r + 1)),
        })
        .collect()
}

fn blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.trim().is_empty(),
        _ => false,
    }
}

fn shown(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => xlsx::number_text(*n),
        Value::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        other => other.to_json(),
    }
}

/// The workbook with `rows` written from `start`: its new bytes and what to tell the
/// model. `existing` is the workbook as it is, or `None` to make one. `path` is for the
/// answer's wording only.
pub fn write(
    path: &str,
    existing: Option<&[u8]>,
    start: Start,
    rows: &[Vec<Value>],
) -> Result<(Vec<u8>, String), String> {
    if rows.is_empty() || rows.iter().all(|row| row.iter().all(blank)) {
        return Err("Every row to write is empty. Give the values to write.".into());
    }
    let (parts, sheet_file, sheet_name) = match existing {
        Some(data) => existing_parts(data)?,
        None => (new_parts(), "xl/worksheets/sheet1.xml".to_string(), "Sheet1".to_string()),
    };
    let sheet_xml = parts
        .iter()
        .find(|p| p.0 == sheet_file)
        .map(|p| String::from_utf8_lossy(&p.1).into_owned())
        .ok_or("This workbook's first sheet has no cells in it.")?;
    let mut sheet = SheetXml::parse(&sheet_xml)?;
    let last_row = sheet.rows.iter().filter(|r| !r.cells.is_empty()).map(|r| r.number).max();
    let (column, row) = match start {
        Start::Cell { column, row } => (column, row),
        Start::Append => (1, last_row.map_or(1, |r| r + 1)),
    };
    // Added after the last row: a row the sheet already has, the same in every cell, isn't
    // added again. A job run twice wrote its rows twice (Nov 15 test gauntlet, 10.3: six
    // invoices put in the register a second time, an hour after the first).
    let mut already: Vec<u32> = Vec::new();
    let rows: Vec<&Vec<Value>> = match (start, existing) {
        (Start::Append, Some(data)) => {
            let book = xlsx::open(data)?;
            let have = book.sheets.first().map_or(&[][..], |s| s.rows.as_slice());
            rows.iter()
                .filter(|row| match same_row(have, row) {
                    Some(number) => {
                        already.push(number);
                        false
                    }
                    None => true,
                })
                .collect()
        }
        _ => rows.iter().collect(),
    };
    let left_out = match already.len() {
        0 => String::new(),
        n => format!(
            "{n} row{} already in the sheet, the same in every cell (row{} {})",
            if n == 1 { " is" } else { "s are" },
            if n == 1 { "" } else { "s" },
            already.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
        ),
    };
    if rows.iter().all(|row| row.iter().all(blank)) {
        return Err(format!(
            "Nothing was added: {left_out}. Don't add them again; say they were already there."
        ));
    }
    if let Some(data) = existing {
        refuse_header_overwrite(data, &sheet_name, column, row, rows[0], last_row)?;
    }

    let (mut written, mut width) = (0usize, 0usize);
    for (r, values) in rows.iter().enumerate() {
        width = width.max(values.len());
        for (c, value) in values.iter().enumerate() {
            if matches!(value, Value::Null) {
                continue;
            }
            let (at_column, at_row) = (column + c as u32, row + r as u32);
            if at_column > MAX_COLUMN || at_row > MAX_ROW {
                return Err(format!(
                    "The rows go past the edge of the sheet at {}{at_row}.",
                    column_name(at_column)
                ));
            }
            sheet.set(at_column, at_row, value);
            written += 1;
        }
    }

    let new_sheet = sheet.to_xml();
    let mut out: Vec<(String, Vec<u8>)> = Vec::with_capacity(parts.len());
    for (name, data) in parts {
        // Excel's list of formula cells may no longer be right: it makes another.
        if name == "xl/calcChain.xml" {
            continue;
        }
        let data = if name == sheet_file {
            new_sheet.clone().into_bytes()
        } else if name == "xl/workbook.xml" {
            recalculated(&String::from_utf8_lossy(&data)).into_bytes()
        } else if name == "xl/_rels/workbook.xml.rels" || name == "[Content_Types].xml" {
            without_calc_chain(&String::from_utf8_lossy(&data)).into_bytes()
        } else {
            data
        };
        out.push((name, data));
    }
    let files: Vec<(&str, &[u8])> = out.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    let area = format!(
        "{}{row}:{}{}",
        column_name(column),
        column_name(column + width.max(1) as u32 - 1),
        row + rows.len() as u32 - 1
    );
    let message = format!(
        "Spreadsheet written successfully: {path}\nWrote {written} cell{} to {sheet_name}!{area}{}.{}",
        if written == 1 { "" } else { "s" },
        if existing.is_some() { "" } else { " in a new workbook" },
        if left_out.is_empty() {
            String::new()
        } else {
            format!("\nNot added again: {left_out}.")
        }
    );
    Ok((stored_zip(&files), message))
}

/// The sheet's row that holds just what `row` would: the same cells filled, each with the
/// same text or the same number (120.5 and "120.50" are one amount).
fn same_row(have: &[(u32, Vec<(u32, String)>)], row: &[Value]) -> Option<u32> {
    let new: Vec<(u32, String)> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| !blank(v))
        .map(|(c, v)| (c as u32 + 1, shown(v)))
        .collect();
    if new.is_empty() {
        return None;
    }
    let number = |text: &str| text.trim().replace(',', "").parse::<f64>().ok();
    let same = |a: &str, b: &str| {
        a.trim() == b.trim()
            || matches!((number(a), number(b)), (Some(x), Some(y)) if (x - y).abs() <= 1e-9 * x.abs().max(y.abs()))
    };
    have.iter()
        .find(|(_, cells)| {
            let mut old: Vec<&(u32, String)> =
                cells.iter().filter(|c| !c.1.trim().is_empty()).collect();
            old.sort_by_key(|c| c.0);
            old.len() == new.len()
                && old.iter().zip(&new).all(|(o, n)| o.0 == n.0 && same(&o.1, &n.1))
        })
        .map(|r| r.0)
}

/// Refuses new rows written over a sheet's header row, which is almost certainly a
/// mistake (the Mac's `check_rows`, from the second gauntlet: a tracker's header written
/// over), and says where the first empty row is.
fn refuse_header_overwrite(
    data: &[u8],
    sheet_name: &str,
    column: u32,
    row: u32,
    first: &[Value],
    last_row: Option<u32>,
) -> Result<(), String> {
    let Some(last_row) = last_row.filter(|r| *r > 1) else {
        return Ok(());
    };
    if row != 1 {
        return Ok(());
    }
    let book = xlsx::open(data)?;
    let Some((_, top)) = book.sheets[0].rows.iter().find(|r| r.0 == 1) else {
        return Ok(());
    };
    let header: Vec<String> = (column..column + first.len() as u32)
        .map(|c| top.iter().find(|cell| cell.0 == c).map_or(String::new(), |cell| cell.1.clone()))
        .collect();
    let is_header = header.iter().any(|v| !v.trim().is_empty())
        && !header.iter().any(|v| v.trim().parse::<f64>().is_ok());
    let header_words: Vec<String> = header
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.trim().to_lowercase())
        .collect();
    let new_words: Vec<String> = first
        .iter()
        .filter(|v| !blank(v))
        .map(|v| shown(v).trim().to_lowercase())
        .collect();
    if is_header && header_words != new_words {
        return Err(format!(
            "Row 1 of \"{sheet_name}\" is its header ({}); writing there would replace it. To add rows, set start_cell to {}{}, the first empty row.",
            header.iter().filter(|v| !v.is_empty()).cloned().collect::<Vec<_>>().join(", "),
            column_name(column),
            last_row + 1
        ));
    }
    Ok(())
}

/// An existing workbook's files, its first sheet's file and that sheet's name.
#[allow(clippy::type_complexity)]
fn existing_parts(data: &[u8]) -> Result<(Vec<(String, Vec<u8>)>, String, String), String> {
    if data.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        return Err(
            "This workbook is password-protected or in the old .xls format, so it can't be \
             written to. Save it as an unprotected .xlsx."
                .into(),
        );
    }
    let workbook = |e: String| e.replace(".docx", "workbook");
    let zip = Zip::open(data).map_err(workbook)?;
    let mut parts = Vec::new();
    for name in zip.names() {
        // Folders are entries too, with nothing in them.
        if name.ends_with('/') {
            continue;
        }
        if let Some(bytes) = zip.read(name).map_err(workbook)? {
            parts.push((name.to_string(), bytes));
        }
    }
    let text = |name: &str| {
        parts
            .iter()
            .find(|p| p.0 == name)
            .map(|p| String::from_utf8_lossy(&p.1).into_owned())
    };
    let book = text("xl/workbook.xml")
        .ok_or("This isn't an Excel workbook (xl/workbook.xml is missing).")?;
    let relationships = text("xl/_rels/workbook.xml.rels").unwrap_or_default();
    let (mut name, mut id) = (String::new(), String::new());
    for piece in Pieces::new(&book) {
        if let Piece::Open("sheet", attrs, _) = piece {
            name = unescaped(attribute(attrs, "name").unwrap_or(""));
            id = attribute(attrs, "r:id")
                .or_else(|| attribute(attrs, "id"))
                .unwrap_or("")
                .to_string();
            break;
        }
    }
    let mut file = None;
    for piece in Pieces::new(&relationships) {
        if let Piece::Open("Relationship", attrs, _) = piece
            && attribute(attrs, "Id") == Some(id.as_str())
            && let Some(target) = attribute(attrs, "Target")
        {
            let target = unescaped(target);
            file = Some(match target.strip_prefix('/') {
                Some(absolute) => absolute.to_string(),
                None => format!("xl/{target}"),
            });
        }
    }
    match file {
        Some(f) if f.contains("worksheets/") => Ok((parts, f, name)),
        _ => Err(format!("Sheet \"{name}\" is a chart sheet, with no cells to write to.")),
    }
}

/// A new workbook's files: one sheet, "Sheet1".
fn new_parts() -> Vec<(String, Vec<u8>)> {
    const MAIN: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
    const RELS: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    const PACKAGE: &str = "http://schemas.openxmlformats.org/package/2006";
    const HEAD: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
    let part = |name: &str, body: String| (name.to_string(), format!("{HEAD}{body}").into_bytes());
    vec![
        part(
            "[Content_Types].xml",
            format!(
                "<Types xmlns=\"{PACKAGE}/content-types\">\
                 <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
                 <Default Extension=\"xml\" ContentType=\"application/xml\"/>\
                 <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>\
                 <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>\
                 </Types>"
            ),
        ),
        part(
            "_rels/.rels",
            format!(
                "<Relationships xmlns=\"{PACKAGE}/relationships\">\
                 <Relationship Id=\"rId1\" Type=\"{RELS}/officeDocument\" Target=\"xl/workbook.xml\"/>\
                 </Relationships>"
            ),
        ),
        part(
            "xl/workbook.xml",
            format!(
                "<workbook xmlns=\"{MAIN}\" xmlns:r=\"{RELS}\"><sheets>\
                 <sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>"
            ),
        ),
        part(
            "xl/_rels/workbook.xml.rels",
            format!(
                "<Relationships xmlns=\"{PACKAGE}/relationships\">\
                 <Relationship Id=\"rId1\" Type=\"{RELS}/worksheet\" Target=\"worksheets/sheet1.xml\"/>\
                 </Relationships>"
            ),
        ),
        part(
            "xl/worksheets/sheet1.xml",
            format!("<worksheet xmlns=\"{MAIN}\"><sheetData/></worksheet>"),
        ),
    ]
}

/// `workbook.xml` telling Excel to work every formula out when it opens the file: the
/// formulas written here have no saved result, and others' results may be out of date.
fn recalculated(xml: &str) -> String {
    if let Some(at) = xml.find("<calcPr") {
        let end = at + xml[at..].find('>').unwrap_or(0);
        let tag = &xml[at..end];
        if tag.contains("fullCalcOnLoad") {
            return xml.to_string();
        }
        let insert = if tag.ends_with('/') { end - 1 } else { end };
        return format!("{} fullCalcOnLoad=\"1\"{}", &xml[..insert], &xml[insert..]);
    }
    // calcPr comes after these, in this order, and before anything else that follows.
    for name in ["definedNames", "externalReferences", "functionGroups", "sheets"] {
        let (closed, empty) = (format!("</{name}>"), format!("<{name}/>"));
        let at = xml
            .find(&closed)
            .map(|i| i + closed.len())
            .or_else(|| xml.find(&empty).map(|i| i + empty.len()));
        if let Some(at) = at {
            return format!("{}<calcPr fullCalcOnLoad=\"1\"/>{}", &xml[..at], &xml[at..]);
        }
    }
    xml.to_string()
}

/// A relationships or content-types file without its entry for `calcChain.xml`.
fn without_calc_chain(xml: &str) -> String {
    let Some(found) = xml.find("calcChain.xml") else {
        return xml.to_string();
    };
    let (Some(start), Some(end)) = (xml[..found].rfind('<'), xml[found..].find('>')) else {
        return xml.to_string();
    };
    format!("{}{}", &xml[..start], &xml[found + end + 1..])
}

// ---- A sheet's XML, taken apart and put back -------------------------------------------

struct Cell {
    column: u32,
    /// Its `s="…"` attribute's number, if it has a style.
    style: Option<String>,
    /// The whole element as it was, or as written.
    xml: String,
}

struct Row {
    number: u32,
    /// The row's attributes as they were, without `r` and `spans`.
    attrs: String,
    cells: Vec<Cell>,
}

struct SheetXml {
    before: String,
    rows: Vec<Row>,
    after: String,
}

/// Each `name` element in `xml`, not nested: its attributes, what's inside it, and the
/// whole of it.
fn elements<'a>(xml: &'a str, name: &str) -> Vec<(&'a str, &'a str, &'a str)> {
    let open = format!("<{name}");
    let close = format!("</{name}>");
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = xml[from..].find(&open).map(|i| from + i) {
        let rest = &xml[at + open.len()..];
        // "<c" is also how "<col" and "<cols" start.
        if !rest.starts_with(|c: char| c.is_whitespace() || c == '>' || c == '/') {
            from = at + open.len();
            continue;
        }
        let Some(tag_end) = rest.find('>') else { break };
        let attrs = rest[..tag_end].trim_end_matches('/');
        let after_tag = at + open.len() + tag_end + 1;
        if rest[..tag_end].ends_with('/') {
            found.push((attrs, "", &xml[at..after_tag]));
            from = after_tag;
            continue;
        }
        let Some(end) = xml[after_tag..].find(&close).map(|i| after_tag + i) else {
            break;
        };
        found.push((attrs, &xml[after_tag..end], &xml[at..end + close.len()]));
        from = end + close.len();
    }
    found
}

/// A tag's attributes without the one called `key`.
fn without_attribute(attrs: &str, key: &str) -> String {
    match attribute(attrs, key) {
        Some(value) => {
            let whole = format!("{key}=\"{value}\"");
            let other = format!("{key}='{value}'");
            attrs.replacen(&whole, "", 1).replacen(&other, "", 1)
        }
        None => attrs.to_string(),
    }
}

fn escaped(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

impl SheetXml {
    fn parse(xml: &str) -> Result<SheetXml, String> {
        const ODD: &str = "This workbook's first sheet is laid out in a way Vireo can't write to.";
        // An empty sheet has `<sheetData/>`.
        if let Some(at) = xml.find("<sheetData/>") {
            return Ok(SheetXml {
                before: xml[..at].to_string(),
                rows: Vec::new(),
                after: xml[at + "<sheetData/>".len()..].to_string(),
            });
        }
        let start = xml.find("<sheetData").ok_or(ODD)?;
        let body_start = start + xml[start..].find('>').ok_or(ODD)? + 1;
        let end = xml[body_start..].find("</sheetData>").map(|i| body_start + i).ok_or(ODD)?;
        let mut rows = Vec::new();
        let mut next_row = 1u32;
        for (attrs, inner, _) in elements(&xml[body_start..end], "row") {
            let number = attribute(attrs, "r").and_then(|r| r.parse().ok()).unwrap_or(next_row);
            next_row = number + 1;
            let mut cells = Vec::new();
            let mut next_column = 1u32;
            for (cell_attrs, _, whole) in elements(inner, "c") {
                let column = attribute(cell_attrs, "r").and_then(column_of).unwrap_or(next_column);
                next_column = column + 1;
                // A cell without its reference gets one, now that cells may go between.
                let xml = if attribute(cell_attrs, "r").is_some() {
                    whole.to_string()
                } else {
                    whole.replacen("<c", &format!("<c r=\"{}{number}\"", column_name(column)), 1)
                };
                cells.push(Cell {
                    column,
                    style: attribute(cell_attrs, "s").map(str::to_string),
                    xml,
                });
            }
            let attrs = without_attribute(&without_attribute(attrs, "r"), "spans");
            rows.push(Row {
                number,
                attrs: attrs.split_whitespace().collect::<Vec<_>>().join(" "),
                cells,
            });
        }
        Ok(SheetXml {
            // The size Excel noted no longer holds; it works it out again.
            before: without_dimension(&xml[..start]),
            rows,
            after: xml[end + "</sheetData>".len()..].to_string(),
        })
    }

    /// Writes `value` at a cell: over what's there, keeping its style, or as a new cell
    /// with the style of the cell above it (below a header row).
    fn set(&mut self, column: u32, number: u32, value: &Value) {
        let above = (number > 2)
            .then(|| {
                self.rows
                    .iter()
                    .find(|r| r.number == number - 1)
                    .and_then(|r| r.cells.iter().find(|c| c.column == column))
                    .and_then(|c| c.style.clone())
            })
            .flatten();
        let at = match self.rows.iter().position(|r| r.number >= number) {
            Some(i) if self.rows[i].number == number => i,
            Some(i) => {
                self.rows.insert(i, Row { number, attrs: String::new(), cells: Vec::new() });
                i
            }
            None => {
                self.rows.push(Row { number, attrs: String::new(), cells: Vec::new() });
                self.rows.len() - 1
            }
        };
        let row = &mut self.rows[at];
        let existing = row.cells.iter().position(|c| c.column == column);
        let style = match existing {
            Some(i) => row.cells[i].style.clone(),
            None => above,
        };
        let reference = format!("{}{number}", column_name(column));
        let s = style.as_ref().map_or(String::new(), |s| format!(" s=\"{s}\""));
        let xml = match value {
            Value::Number(n) => format!("<c r=\"{reference}\"{s}><v>{}</v></c>", xlsx::number_text(*n)),
            Value::Bool(b) => format!("<c r=\"{reference}\"{s} t=\"b\"><v>{}</v></c>", u8::from(*b)),
            Value::String(text) if text.len() > 1 && text.starts_with('=') => {
                format!("<c r=\"{reference}\"{s}><f>{}</f></c>", escaped(&text[1..]))
            }
            other => format!(
                "<c r=\"{reference}\"{s} t=\"inlineStr\"><is><t xml:space=\"preserve\">{}</t></is></c>",
                escaped(&shown(other))
            ),
        };
        let cell = Cell { column, style, xml };
        match existing {
            Some(i) => row.cells[i] = cell,
            None => {
                let at = row.cells.iter().position(|c| c.column > column).unwrap_or(row.cells.len());
                row.cells.insert(at, cell);
            }
        }
    }

    fn to_xml(&self) -> String {
        let mut xml = String::with_capacity(self.before.len() + self.after.len() + 4096);
        xml.push_str(&self.before);
        xml.push_str("<sheetData>");
        for row in &self.rows {
            xml.push_str(&format!("<row r=\"{}\"", row.number));
            if !row.attrs.is_empty() {
                xml.push(' ');
                xml.push_str(&row.attrs);
            }
            xml.push('>');
            for cell in &row.cells {
                xml.push_str(&cell.xml);
            }
            xml.push_str("</row>");
        }
        xml.push_str("</sheetData>");
        xml.push_str(&self.after);
        xml
    }
}

fn without_dimension(xml: &str) -> String {
    match xml.find("<dimension") {
        Some(at) => match xml[at..].find('>') {
            Some(end) => format!("{}{}", &xml[..at], &xml[at + end + 1..]),
            None => xml.to_string(),
        },
        None => xml.to_string(),
    }
}

/// The text in an XML fragment, for tests.
#[cfg(test)]
fn text_of(xml: &str) -> String {
    let mut out = String::new();
    for piece in Pieces::new(xml) {
        if let Piece::Text(t) = piece {
            super::docx::unescape_into(&mut out, t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(json: &str) -> Vec<Vec<Value>> {
        rows_from(json).unwrap()
    }

    #[test]
    fn a_new_workbook_reads_back_as_written() {
        let rows = values(r#"[["Item","Qty","Due"],["Hoodies & caps",12,"2026-11-12"],["Total","=SUM(B2:B2)",true]]"#);
        let (bytes, message) = write("/x/stock.xlsx", None, Start::parse("A1").unwrap(), &rows).unwrap();
        assert_eq!(
            message,
            "Spreadsheet written successfully: /x/stock.xlsx\nWrote 9 cells to Sheet1!A1:C3 in a new workbook."
        );
        let book = xlsx::open(&bytes).unwrap();
        assert_eq!(
            book.read(),
            "Sheets: Sheet1\n\nSheet \"Sheet1\", A1:C3 (used range A1:C3)\nRow | A | B | C\n\
             1 | Item | Qty | Due\n2 | Hoodies & caps | 12 | 2026-11-12\n3 | Total | =SUM(B2:B2) | TRUE"
        );
    }

    #[test]
    fn rows_are_added_to_a_workbook_that_keeps_everything_else() {
        let sheet = r#"<worksheet><dimension ref="A1:C2"/><sheetViews/><sheetData><row r="1" spans="1:3" ht="20"><c r="A1" s="1" t="inlineStr"><is><t>Invoice</t></is></c><c r="B1" s="1" t="inlineStr"><is><t>CAD</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>NP-1</t></is></c><c r="B2" s="4"><v>120.5</v></c><c r="C2" s="7"><f>B2*2</f><v>241</v></c></row></sheetData><pageMargins left="0.7"/></worksheet>"#;
        let book = r#"<workbook xmlns:r="x"><sheets><sheet name="Register" sheetId="1" r:id="rId1"/></sheets><definedNames/></workbook>"#;
        let rels = r#"<Relationships><Relationship Id="rId1" Type="t" Target="worksheets/sheet1.xml"/><Relationship Id="rId9" Type="t" Target="calcChain.xml"/></Relationships>"#;
        let types = r#"<Types><Override PartName="/xl/calcChain.xml" ContentType="c"/><Override PartName="/xl/styles.xml" ContentType="s"/></Types>"#;
        let existing = stored_zip(&[
            ("[Content_Types].xml", types.as_bytes()),
            ("xl/workbook.xml", book.as_bytes()),
            ("xl/_rels/workbook.xml.rels", rels.as_bytes()),
            ("xl/styles.xml", b"<styleSheet/>"),
            ("xl/calcChain.xml", b"<calcChain/>"),
            ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
        ]);
        let rows = values(r#"[["NP-2", 80, null], ["NP-3", "=B2+1"]]"#);
        let (bytes, message) = write("r.xlsx", Some(&existing), Start::Append, &rows).unwrap();
        assert_eq!(message, "Spreadsheet written successfully: r.xlsx\nWrote 4 cells to Register!A3:C4.");
        let zip = Zip::open(&bytes).unwrap();
        let part = |name: &str| String::from_utf8(zip.read(name).unwrap().unwrap()).unwrap();
        let new_sheet = part("xl/worksheets/sheet1.xml");
        // What was there is as it was (but for the size note and the row's span hint)...
        assert!(new_sheet.starts_with("<worksheet><sheetViews/><sheetData><row r=\"1\" ht=\"20\"><c r=\"A1\" s=\"1\""));
        assert!(new_sheet.contains(r#"<c r="C2" s="7"><f>B2*2</f><v>241</v></c>"#));
        assert!(new_sheet.ends_with("</sheetData><pageMargins left=\"0.7\"/></worksheet>"));
        // ...and the new cells take the style of the cell above them.
        assert!(new_sheet.contains(r#"<row r="3"><c r="A3" t="inlineStr"><is><t xml:space="preserve">NP-2</t></is></c><c r="B3" s="4"><v>80</v></c></row>"#));
        assert!(new_sheet.contains(r#"<c r="B4" s="4"><f>B2+1</f></c>"#));
        assert_eq!(text_of(&new_sheet).matches("NP-").count(), 3);
        // Excel works the formulas out again, and its old list of them is gone.
        assert!(part("xl/workbook.xml").contains("<definedNames/><calcPr fullCalcOnLoad=\"1\"/>"));
        assert!(zip.read("xl/calcChain.xml").unwrap().is_none());
        assert!(!part("xl/_rels/workbook.xml.rels").contains("calcChain"));
        assert!(!part("[Content_Types].xml").contains("calcChain"));
        assert_eq!(part("xl/styles.xml"), "<styleSheet/>");

        // Added again, a row that's already there stays one row; a new one beside it goes in.
        let again = values(r#"[["NP-2", "80.00"], ["NP-4", 15]]"#);
        let (once, message) = write("r.xlsx", Some(&bytes), Start::Append, &again).unwrap();
        assert_eq!(
            message,
            "Spreadsheet written successfully: r.xlsx\nWrote 2 cells to Register!A5:B5.\nNot added again: 1 row is already in the sheet, the same in every cell (row 3)."
        );
        let zip = Zip::open(&once).unwrap();
        let sheet = String::from_utf8(zip.read("xl/worksheets/sheet1.xml").unwrap().unwrap()).unwrap();
        assert_eq!(text_of(&sheet).matches("NP-2").count(), 1);
        assert_eq!(
            write("r.xlsx", Some(&bytes), Start::Append, &values(r#"[["NP-2", 80]]"#)).unwrap_err(),
            "Nothing was added: 1 row is already in the sheet, the same in every cell (row 3). Don't add them again; say they were already there."
        );
        // Written at a cell, the rows go where they're put.
        assert!(write("r.xlsx", Some(&bytes), Start::parse("a3").unwrap(), &values(r#"[["NP-2", 80]]"#)).is_ok());

        // A cell written over keeps its style; a cell written between two goes between.
        let (bytes, _) = write("r.xlsx", Some(&bytes), Start::parse("b2").unwrap(), &values("[[99]]")).unwrap();
        let zip = Zip::open(&bytes).unwrap();
        let sheet = String::from_utf8(zip.read("xl/worksheets/sheet1.xml").unwrap().unwrap()).unwrap();
        assert!(sheet.contains(r#"<c r="A2" t="inlineStr"><is><t>NP-1</t></is></c><c r="B2" s="4"><v>99</v></c><c r="C2""#));
    }

    /// Adds a row to a copy of a real workbook, for checking with another reader:
    /// `VIREO_XLSX=<in> VIREO_XLSX_OUT=<out> cargo test -p syrinx-server append_to_a_real -- --ignored`.
    #[test]
    #[ignore]
    fn append_to_a_real_workbook() {
        let data = std::fs::read(std::env::var("VIREO_XLSX").unwrap()).unwrap();
        let rows = values(r#"[["VIREO-TEST", 1234.5, "=1+1", true, null, "text & <more>"]]"#);
        let (bytes, message) = write("out.xlsx", Some(&data), Start::Append, &rows).unwrap();
        println!("{message}");
        std::fs::write(std::env::var("VIREO_XLSX_OUT").unwrap(), bytes).unwrap();
    }

    #[test]
    fn mistakes_are_refused_with_what_to_do() {
        let (existing, _) = write("t.xlsx", None, Start::Append, &values(r#"[["Name","Amount"],["Ada",5]]"#)).unwrap();
        // Rows over the header.
        let over = write("t.xlsx", Some(&existing), Start::parse("A1").unwrap(), &values(r#"[["Bo",7]]"#));
        assert_eq!(
            over.unwrap_err(),
            "Row 1 of \"Sheet1\" is its header (Name, Amount); writing there would replace it. To add rows, set start_cell to A3, the first empty row."
        );
        // The same header again is fine.
        assert!(write("t.xlsx", Some(&existing), Start::parse("A1").unwrap(), &values(r#"[["Name","Amount"]]"#)).is_ok());
        assert_eq!(
            write("t.xlsx", None, Start::Append, &values(r#"[["", null], [" "]]"#)).unwrap_err(),
            "Every row to write is empty. Give the values to write."
        );
        assert!(Start::parse("12").is_err());
        assert!(Start::parse("A0").is_err());
        assert_eq!(Start::parse("  ").unwrap(), Start::Cell { column: 1, row: 1 });
        assert_eq!(Start::parse("Append").unwrap(), Start::Append);
        assert_eq!(Start::parse("AB12").unwrap(), Start::Cell { column: 28, row: 12 });
        assert_eq!(rows_from("{}").unwrap_err(), "rows must be a list of rows, each a list of values.");
        assert_eq!(rows_from("[[1],2]").unwrap_err(), "Row 2 isn't a list of values.");
        assert_eq!(rows_from(r#""[[1,2]]""#).unwrap().len(), 1);
    }
}
