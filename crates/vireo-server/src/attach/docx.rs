//! Word (.docx) text: the file is a ZIP, and the text is in `word/document.xml`, in
//! `<w:t>` runs inside `<w:p>` paragraphs. Headings come out as Markdown headings, list
//! items with a dash and table rows with their cells between bars, so a model still sees
//! the document's shape.

use super::inflate;

/// The most a file inside the ZIP may unpack to.
const MAX_UNPACKED: usize = 256 << 20;

pub fn text(data: &[u8]) -> Result<String, String> {
    if data.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        // An OLE file: Word's own old format, or a .docx locked with a password.
        return Err(
            "This Word file is password-protected or in the old .doc format, so its \
                    text can't be read. Save it as an unprotected .docx or a PDF and attach that."
                .into(),
        );
    }
    let zip = Zip::open(data)?;
    let xml = zip
        .read("word/document.xml")?
        .ok_or("This .docx has no document in it (word/document.xml is missing).")?;
    Ok(document_text(&String::from_utf8_lossy(&xml)))
}

// ---- ZIP -------------------------------------------------------------------------------

pub struct Zip<'a> {
    data: &'a [u8],
    /// Name, method, CRC, compressed and full size, the local header's offset.
    entries: Vec<(String, u16, u32, usize, usize, usize)>,
}

fn u16_at(d: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(i..i + 2)?.try_into().ok()?))
}

fn u32_at(d: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(i..i + 4)?.try_into().ok()?))
}

const BROKEN: &str = "This .docx isn't a readable ZIP file: it may be damaged.";

impl<'a> Zip<'a> {
    pub fn open(data: &'a [u8]) -> Result<Self, String> {
        // The end record is in the last 64 KB (its comment can be that long).
        let from = data.len().saturating_sub(22 + 65_535);
        let end = (from..data.len().saturating_sub(21))
            .rev()
            .find(|&i| u32_at(data, i) == Some(0x0605_4b50))
            .ok_or(BROKEN)?;
        let count = u16_at(data, end + 10).ok_or(BROKEN)? as usize;
        let mut at = u32_at(data, end + 16).ok_or(BROKEN)? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(data, at) != Some(0x0201_4b50) {
                return Err(BROKEN.into());
            }
            let get16 = |o| u16_at(data, at + o).ok_or(BROKEN);
            let get32 = |o| u32_at(data, at + o).ok_or(BROKEN);
            let (method, crc) = (get16(10)?, get32(16)?);
            let (packed, size) = (get32(20)? as usize, get32(24)? as usize);
            let (name_len, extra, comment) = (
                get16(28)? as usize,
                get16(30)? as usize,
                get16(32)? as usize,
            );
            let offset = get32(42)? as usize;
            let name = data.get(at + 46..at + 46 + name_len).ok_or(BROKEN)?;
            entries.push((
                String::from_utf8_lossy(name).into_owned(),
                method,
                crc,
                packed,
                size,
                offset,
            ));
            at += 46 + name_len + extra + comment;
        }
        Ok(Self { data, entries })
    }

    /// The file called `name`, unpacked and checked; None if there's none.
    pub fn read(&self, name: &str) -> Result<Option<Vec<u8>>, String> {
        let Some(&(_, method, crc, packed, size, offset)) =
            self.entries.iter().find(|e| e.0 == name)
        else {
            return Ok(None);
        };
        let d = self.data;
        if u32_at(d, offset) != Some(0x0403_4b50) {
            return Err(BROKEN.into());
        }
        let start = offset
            + 30
            + u16_at(d, offset + 26).ok_or(BROKEN)? as usize
            + u16_at(d, offset + 28).ok_or(BROKEN)? as usize;
        let raw = d.get(start..start + packed).ok_or(BROKEN)?;
        if size > MAX_UNPACKED {
            return Err(format!("{name} in this .docx is too large to read."));
        }
        let bytes = match method {
            0 => raw.to_vec(),
            8 => inflate::inflate(raw, MAX_UNPACKED)
                .map_err(|e| format!("{name} in this .docx can't be unpacked: {e}."))?,
            m => {
                return Err(format!(
                    "{name} in this .docx is packed in a way Vireo can't read ({m})."
                ));
            }
        };
        if inflate::crc32(&bytes) != crc {
            return Err(format!(
                "{name} in this .docx is damaged (its checksum is wrong)."
            ));
        }
        Ok(Some(bytes))
    }
}

// ---- The document's XML ----------------------------------------------------------------

/// The text of WordprocessingML: paragraphs a line each, tabs and breaks kept.
pub fn document_text(xml: &str) -> String {
    let mut out = String::new();
    let mut in_text = false;
    // Inside <mc:Fallback>, which repeats what <mc:Choice> already has.
    let mut fallback = 0usize;
    let mut cell_depth = 0usize;
    // What starts the paragraph under way, written before its first text.
    let mut prefix = String::new();
    let mut paragraph_started = false;
    let mut rest = xml;
    while let Some(open) = rest.find('<') {
        if in_text && fallback == 0 {
            push_text(&mut out, &mut prefix, &mut paragraph_started, &rest[..open]);
        }
        let Some(close) = rest[open..].find('>') else {
            break;
        };
        let tag = &rest[open + 1..open + close];
        rest = &rest[open + close + 1..];
        let closing = tag.starts_with('/');
        let self_closing = tag.ends_with('/');
        let body = tag.trim_start_matches('/').trim_end_matches('/');
        let name = body.split(|c: char| c.is_whitespace()).next().unwrap_or("");
        match (name, closing) {
            ("mc:Fallback", false) if !self_closing => fallback += 1,
            ("mc:Fallback", true) => fallback = fallback.saturating_sub(1),
            _ if fallback > 0 => {}
            ("w:t", false) => in_text = !self_closing,
            ("w:t", true) => in_text = false,
            ("w:p", false) if !self_closing => {
                prefix.clear();
                paragraph_started = false;
            }
            ("w:pStyle", false) => {
                let style = attribute(body, "w:val").unwrap_or("");
                let level = if style.eq_ignore_ascii_case("Title") {
                    1
                } else {
                    style
                        .strip_prefix("Heading")
                        .or_else(|| style.strip_prefix("heading"))
                        .and_then(|n| n.trim().parse::<usize>().ok())
                        .unwrap_or(0)
                };
                if (1..=6).contains(&level) {
                    prefix = format!("{} ", "#".repeat(level));
                } else if style.starts_with("List") {
                    // "List Bullet", "List Number", "List Paragraph".
                    prefix = "- ".into();
                }
            }
            ("w:numPr", false) if prefix.is_empty() => prefix = "- ".into(),
            ("w:tab", false) => push_text(&mut out, &mut prefix, &mut paragraph_started, "\t"),
            ("w:br" | "w:cr", false) => out.push('\n'),
            ("w:noBreakHyphen", false) => {
                push_text(&mut out, &mut prefix, &mut paragraph_started, "-")
            }
            ("w:p", true) => {
                if cell_depth > 0 {
                    if !out.ends_with([' ', '\n']) && !out.ends_with("| ") {
                        out.push(' ');
                    }
                } else {
                    trim_end_spaces(&mut out);
                    out.push('\n');
                }
            }
            ("w:tc", false) if !self_closing => cell_depth += 1,
            ("w:tc", true) => {
                cell_depth = cell_depth.saturating_sub(1);
                trim_end_spaces(&mut out);
                out.push_str(" | ");
            }
            ("w:tr", false) if !self_closing => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("| ");
            }
            ("w:tr", true) => {
                trim_end_spaces(&mut out);
                out.push('\n');
            }
            ("w:tbl", true) => out.push('\n'),
            _ => {}
        }
    }
    super::tidy(&out)
}

fn push_text(out: &mut String, prefix: &mut String, started: &mut bool, text: &str) {
    if text.is_empty() {
        return;
    }
    if !*started {
        *started = true;
        out.push_str(prefix);
    }
    unescape_into(out, text);
}

fn trim_end_spaces(out: &mut String) {
    while out.ends_with([' ', '\t']) {
        out.pop();
    }
}

/// An attribute's value in a tag's body (`w:pStyle w:val="Heading1"`).
fn attribute<'t>(body: &'t str, key: &str) -> Option<&'t str> {
    let at = body.find(&format!("{key}="))? + key.len() + 1;
    let quote = body[at..].chars().next()?;
    let value = &body[at + 1..];
    Some(&value[..value.find(quote)?])
}

/// XML's escapes: the five named ones and numeric characters.
pub fn unescape_into(out: &mut String, text: &str) {
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let Some(semi) = rest.find(';').filter(|&s| s <= 10) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..semi];
        let c = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        match c {
            Some(c) => {
                out.push(c);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A ZIP of stored files, as a test .docx.
    pub fn stored_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in files {
            let offset = out.len() as u32;
            let crc = inflate::crc32(data);
            let header = |sig: u32, central_entry: bool| {
                let mut h = Vec::new();
                h.extend_from_slice(&sig.to_le_bytes());
                if central_entry {
                    h.extend_from_slice(&20u16.to_le_bytes()); // made by
                }
                h.extend_from_slice(&20u16.to_le_bytes()); // needed
                h.extend_from_slice(&0u16.to_le_bytes()); // flags
                h.extend_from_slice(&0u16.to_le_bytes()); // stored
                h.extend_from_slice(&0u32.to_le_bytes()); // time, date
                h.extend_from_slice(&crc.to_le_bytes());
                h.extend_from_slice(&(data.len() as u32).to_le_bytes());
                h.extend_from_slice(&(data.len() as u32).to_le_bytes());
                h.extend_from_slice(&(name.len() as u16).to_le_bytes());
                h.extend_from_slice(&0u16.to_le_bytes()); // extra
                if central_entry {
                    h.extend_from_slice(&[0; 6]); // comment, disk, internal attributes
                    h.extend_from_slice(&0u32.to_le_bytes()); // external attributes
                    h.extend_from_slice(&offset.to_le_bytes());
                }
                h.extend_from_slice(name.as_bytes());
                h
            };
            out.extend(header(0x0403_4b50, false));
            out.extend_from_slice(data);
            central.extend(header(0x0201_4b50, true));
        }
        let at = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&at.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn reads_a_word_file_from_textutil() {
        // `textutil -convert docx` of a small HTML page: deflated, with a list.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/report.docx"
        ))
        .unwrap();
        let text = text(&data).unwrap();
        assert!(
            text.starts_with("Quarterly report\nSales rose 12% & costs fell.\n"),
            "{text}"
        );
        assert!(text.contains("Second <point>"), "{text}");
    }

    #[test]
    fn keeps_headings_lists_and_tables() {
        let xml = r#"<w:document><w:body>
            <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Plan</w:t></w:r></w:p>
            <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/></w:numPr></w:pPr><w:r><w:t>Buy milk</w:t></w:r></w:p>
            <w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr><w:r><w:t>Call Sam</w:t></w:r></w:p>
            <w:p><w:r><w:t xml:space="preserve">Tab</w:t><w:tab/><w:t>bed &amp; </w:t><w:t>done&#x2019;s</w:t></w:r></w:p>
            <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Total</w:t></w:r></w:p></w:tc></w:tr>
            <w:tr><w:tc><w:p><w:r><w:t>North</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>4,200</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
            <w:p><mc:AlternateContent><mc:Choice><w:t>Box</w:t></mc:Choice><mc:Fallback><w:t>Box</w:t></mc:Fallback></mc:AlternateContent></w:p>
            <w:p><w:r><w:delText>gone</w:delText><w:t>After</w:t></w:r></w:p>
            </w:body></w:document>"#;
        let zip = stored_zip(&[("word/document.xml", xml.as_bytes())]);
        assert_eq!(
            text(&zip).unwrap(),
            "## Plan\n- Buy milk\n- Call Sam\nTab\tbed & done\u{2019}s\n| Region | Total |\n| North | 4,200 |\n\nBox\nAfter"
        );
    }

    #[test]
    fn says_what_is_wrong_with_a_bad_file() {
        assert!(
            text(b"not a zip")
                .unwrap_err()
                .contains("isn't a readable ZIP")
        );
        let no_document = stored_zip(&[("other.xml", b"<x/>")]);
        assert!(text(&no_document).unwrap_err().contains("no document"));
        let mut damaged = stored_zip(&[("word/document.xml", b"<w:t>hi</w:t>")]);
        let at = damaged.windows(2).position(|w| w == b"hi").unwrap();
        damaged[at] = b'H';
        assert!(text(&damaged).unwrap_err().contains("checksum"));
        assert!(
            text(&[0xD0, 0xCF, 0x11, 0xE0, 0, 0])
                .unwrap_err()
                .contains("password-protected")
        );
    }
}
