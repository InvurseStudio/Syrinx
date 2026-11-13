//! Writing Word documents (.docx): a new one from a title and the model's Markdown
//! (`create`), and a template's `{{placeholders}}` filled in (`fill`). The Windows app's
//! `WordDocument` makes the same documents with the same plain style (Aptos, grey table
//! lines, US Letter with the Mac's margins), so a document looks alike from either machine.
//! Each is written as a ZIP without compression.

use super::docx::{Zip, stored_zip, unescape_into};
use super::markdown::{self, Alignment, Block, Run};

const WORD: &str = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"";
const RELATIONSHIPS: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PACKAGE: &str = "http://schemas.openxmlformats.org/package/2006";
const HEAD: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>";

/// US Letter, with the Mac's margins: 0.8" top and bottom, 0.9" at the sides (twentieths
/// of a point).
const PAGE_WIDTH: usize = 12240;
const PAGE_HEIGHT: usize = 15840;
const SIDE_MARGIN: usize = 1296;
const END_MARGIN: usize = 1152;
const TEXT_WIDTH: usize = PAGE_WIDTH - 2 * SIDE_MARGIN;

pub fn escaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            // Control characters aren't allowed in XML at all.
            c if (c as u32) < 0x20 && !matches!(c, '\t' | '\n' | '\r') => {}
            c => out.push(c),
        }
    }
    out
}

// ---- A new document --------------------------------------------------------------------

/// A document's hyperlinks, each a relationship of `word/document.xml`.
#[derive(Default)]
struct Links(Vec<String>);

impl Links {
    /// The relationship id for a web or mail address; None for anything else, which stays
    /// as its text.
    fn id(&mut self, address: &str) -> Option<String> {
        let lower = address.to_lowercase();
        if !["http://", "https://", "mailto:"].iter().any(|p| lower.starts_with(p)) {
            return None;
        }
        self.0.push(address.to_string());
        Some(format!("rIdLink{}", self.0.len()))
    }
}

fn text_run(run: &Run, bold: bool, style: Option<&str>) -> String {
    let mut xml = String::from("<w:r><w:rPr>");
    if let Some(style) = style.or(run.code.then_some("InlineCode")) {
        xml.push_str(&format!("<w:rStyle w:val=\"{style}\"/>"));
    }
    if bold || run.bold {
        xml.push_str("<w:b/>");
    }
    if run.italic {
        xml.push_str("<w:i/>");
    }
    xml.push_str("</w:rPr>");
    // A paragraph's kept line breaks ("To: CFO" under "From: ...") become breaks in the run.
    for (i, line) in run.text.split('\n').enumerate() {
        if i > 0 {
            xml.push_str("<w:br/>");
        }
        if !line.is_empty() {
            xml.push_str(&format!("<w:t xml:space=\"preserve\">{}</w:t>", escaped(line)));
        }
    }
    xml.push_str("</w:r>");
    xml
}

/// A paragraph in a style, with `properties` after the style in its properties.
fn paragraph(style: Option<&str>, properties: &str, text: &str, bold: bool, links: &mut Links) -> String {
    let mut xml = String::from("<w:p><w:pPr>");
    if let Some(style) = style {
        xml.push_str(&format!("<w:pStyle w:val=\"{style}\"/>"));
    }
    xml.push_str(properties);
    xml.push_str("</w:pPr>");
    for run in markdown::runs(text) {
        match run.link.as_deref().and_then(|address| links.id(address)) {
            Some(id) => xml.push_str(&format!(
                "<w:hyperlink r:id=\"{id}\" w:history=\"1\">{}</w:hyperlink>",
                text_run(&run, bold, Some("Hyperlink"))
            )),
            None => xml.push_str(&text_run(&run, bold, None)),
        }
    }
    xml.push_str("</w:p>");
    xml
}

fn table(rows: &[Vec<String>], alignments: &[Alignment], links: &mut Links) -> String {
    let columns = rows[0].len();
    // Columns share the width by how much text they hold.
    let weights: Vec<usize> = (0..columns)
        .map(|c| {
            rows.iter()
                .map(|r| markdown::plain(&r[c]).chars().count())
                .max()
                .unwrap_or(0)
                .clamp(6, 40)
        })
        .collect();
    let total: usize = weights.iter().sum();
    let widths: Vec<usize> = weights.iter().map(|w| TEXT_WIDTH * w / total).collect();
    let mut xml = String::from(
        "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"5000\" w:type=\"pct\"/>\
         <w:jc w:val=\"center\"/><w:tblLook w:val=\"04A0\" w:firstRow=\"1\" w:noVBand=\"1\"/></w:tblPr><w:tblGrid>",
    );
    for w in &widths {
        xml.push_str(&format!("<w:gridCol w:w=\"{w}\"/>"));
    }
    xml.push_str("</w:tblGrid>");
    for (r, row) in rows.iter().enumerate() {
        xml.push_str("<w:tr>");
        if r == 0 {
            xml.push_str("<w:trPr><w:tblHeader/></w:trPr>");
        }
        for (c, cell) in row.iter().enumerate() {
            xml.push_str(&format!("<w:tc><w:tcPr><w:tcW w:w=\"{}\" w:type=\"dxa\"/>", widths[c]));
            if r == 0 {
                xml.push_str("<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F3F4F6\"/>");
            }
            xml.push_str("</w:tcPr>");
            let align = match alignments.get(c) {
                Some(Alignment::Center) => "<w:jc w:val=\"center\"/>",
                Some(Alignment::Right) => "<w:jc w:val=\"right\"/>",
                _ => "",
            };
            xml.push_str(&paragraph(Some("TableText"), align, cell, r == 0, links));
            xml.push_str("</w:tc>");
        }
        xml.push_str("</w:tr>");
    }
    xml.push_str("</w:tbl><w:p/>");
    xml
}

/// A new document from a title and Markdown: its bytes. A document that opens with a
/// top-level heading takes that as its title.
pub fn create(title: &str, content: &str) -> Vec<u8> {
    let (title, blocks) = markdown::split_title(title, markdown::parse(content));
    let mut links = Links::default();
    let mut body = String::new();
    if !title.trim().is_empty() {
        let run = Run { text: title.clone(), ..Run::default() };
        body.push_str(&format!(
            "<w:p><w:pPr><w:pStyle w:val=\"Title\"/></w:pPr>{}</w:p>",
            text_run(&run, false, None)
        ));
    }
    // Each run of list items is a list of its own, so its numbering starts again: a
    // list's kind (false a bulleted one) and the number it starts at.
    let mut lists: Vec<(bool, u32)> = Vec::new();
    // The lists of the run of items being written: a bulleted one's id, a numbered one's.
    let mut current: (Option<usize>, Option<usize>) = (None, None);
    for block in &blocks {
        if !matches!(block, Block::Bullet { .. } | Block::Numbered { .. }) {
            current = (None, None);
        }
        match block {
            Block::Heading { level, text } => {
                let style = format!("Heading{}", (*level).min(4));
                body.push_str(&paragraph(Some(&style), "", text, false, &mut links));
            }
            Block::Bullet { depth, text } | Block::Numbered { depth, text, .. } => {
                let (numbered, start) = match block {
                    Block::Numbered { number, .. } => (true, (*number).max(1)),
                    _ => (false, 1),
                };
                let slot = if numbered { &mut current.1 } else { &mut current.0 };
                let id = *slot.get_or_insert_with(|| {
                    lists.push((numbered, start));
                    lists.len()
                });
                let numbering = format!("<w:numPr><w:ilvl w:val=\"{depth}\"/><w:numId w:val=\"{id}\"/></w:numPr>");
                body.push_str(&paragraph(Some("ListParagraph"), &numbering, text, false, &mut links));
            }
            Block::Quote(text) => body.push_str(&paragraph(Some("Quote"), "", text, false, &mut links)),
            Block::Code(code) => {
                body.push_str("<w:p><w:pPr><w:pStyle w:val=\"Code\"/></w:pPr><w:r>");
                for (i, line) in code.split('\n').enumerate() {
                    if i > 0 {
                        body.push_str("<w:br/>");
                    }
                    body.push_str(&format!("<w:t xml:space=\"preserve\">{}</w:t>", escaped(line)));
                }
                body.push_str("</w:r></w:p>");
            }
            Block::Rule => body.push_str(
                "<w:p><w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"6\" w:space=\"1\" w:color=\"D0D7DE\"/></w:pBdr></w:pPr></w:p>",
            ),
            Block::Table { rows, alignments } => body.push_str(&table(rows, alignments, &mut links)),
            Block::Paragraph(text) => body.push_str(&paragraph(None, "", text, false, &mut links)),
        }
    }
    body.push_str(&format!(
        "<w:sectPr><w:pgSz w:w=\"{PAGE_WIDTH}\" w:h=\"{PAGE_HEIGHT}\"/><w:pgMar w:top=\"{END_MARGIN}\" w:right=\"{SIDE_MARGIN}\" \
         w:bottom=\"{END_MARGIN}\" w:left=\"{SIDE_MARGIN}\" w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/></w:sectPr>"
    ));
    let document = format!(
        "{HEAD}<w:document {WORD} xmlns:r=\"{RELATIONSHIPS}\"><w:body>{body}</w:body></w:document>"
    );

    let mut relationships = format!(
        "{HEAD}<Relationships xmlns=\"{PACKAGE}/relationships\">\
         <Relationship Id=\"rIdStyles\" Type=\"{RELATIONSHIPS}/styles\" Target=\"styles.xml\"/>\
         <Relationship Id=\"rIdSettings\" Type=\"{RELATIONSHIPS}/settings\" Target=\"settings.xml\"/>"
    );
    if !lists.is_empty() {
        relationships.push_str(&format!(
            "<Relationship Id=\"rIdNumbering\" Type=\"{RELATIONSHIPS}/numbering\" Target=\"numbering.xml\"/>"
        ));
    }
    for (i, address) in links.0.iter().enumerate() {
        relationships.push_str(&format!(
            "<Relationship Id=\"rIdLink{}\" Type=\"{RELATIONSHIPS}/hyperlink\" Target=\"{}\" TargetMode=\"External\"/>",
            i + 1,
            escaped(address)
        ));
    }
    relationships.push_str("</Relationships>");

    const KIND: &str = "application/vnd.openxmlformats-officedocument.wordprocessingml";
    let mut types = format!(
        "{HEAD}<Types xmlns=\"{PACKAGE}/content-types\">\
         <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
         <Default Extension=\"xml\" ContentType=\"application/xml\"/>\
         <Override PartName=\"/word/document.xml\" ContentType=\"{KIND}.document.main+xml\"/>\
         <Override PartName=\"/word/styles.xml\" ContentType=\"{KIND}.styles+xml\"/>\
         <Override PartName=\"/word/settings.xml\" ContentType=\"{KIND}.settings+xml\"/>\
         <Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
    );
    if !lists.is_empty() {
        types.push_str(&format!(
            "<Override PartName=\"/word/numbering.xml\" ContentType=\"{KIND}.numbering+xml\"/>"
        ));
    }
    types.push_str("</Types>");
    let package = format!(
        "{HEAD}<Relationships xmlns=\"{PACKAGE}/relationships\">\
         <Relationship Id=\"rId1\" Type=\"{RELATIONSHIPS}/officeDocument\" Target=\"word/document.xml\"/>\
         <Relationship Id=\"rId2\" Type=\"{PACKAGE}/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>\
         </Relationships>"
    );
    let core = core_properties(&title);
    let styles = styles();
    let settings = format!(
        "{HEAD}<w:settings {WORD}><w:compat><w:compatSetting w:name=\"compatibilityMode\" \
         w:uri=\"http://schemas.microsoft.com/office/word\" w:val=\"15\"/></w:compat></w:settings>"
    );
    let numbering = numbering(&lists);
    let mut files: Vec<(&str, &[u8])> = vec![
        ("[Content_Types].xml", types.as_bytes()),
        ("_rels/.rels", package.as_bytes()),
        ("docProps/core.xml", core.as_bytes()),
        ("word/document.xml", document.as_bytes()),
        ("word/_rels/document.xml.rels", relationships.as_bytes()),
        ("word/styles.xml", styles.as_bytes()),
        ("word/settings.xml", settings.as_bytes()),
    ];
    if !lists.is_empty() {
        files.push(("word/numbering.xml", numbering.as_bytes()));
    }
    stored_zip(&files)
}

/// The package's title, as Word shows it in a file's details.
pub fn core_properties(title: &str) -> String {
    format!(
        "{HEAD}<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" \
         xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>{}</dc:title></cp:coreProperties>",
        escaped(title)
    )
}

/// Bullets and numbers for four levels, then one list per run of items.
fn numbering(lists: &[(bool, u32)]) -> String {
    let levels = |level: &dyn Fn(usize) -> (String, String)| -> String {
        (0..4)
            .map(|i| {
                let (format, text) = level(i);
                format!(
                    "<w:lvl w:ilvl=\"{i}\"><w:start w:val=\"1\"/><w:numFmt w:val=\"{format}\"/><w:lvlText w:val=\"{text}\"/>\
                     <w:lvlJc w:val=\"left\"/><w:pPr><w:ind w:left=\"{}\" w:hanging=\"360\"/></w:pPr></w:lvl>",
                    720 + 360 * i
                )
            })
            .collect()
    };
    let bullets = ["•", "◦", "▪", "•"];
    let numbers = ["decimal", "lowerLetter", "lowerRoman", "decimal"];
    let mut xml = format!("{HEAD}<w:numbering {WORD}>");
    xml.push_str("<w:abstractNum w:abstractNumId=\"0\"><w:multiLevelType w:val=\"hybridMultilevel\"/>");
    xml.push_str(&levels(&|i| ("bullet".into(), bullets[i].into())));
    xml.push_str("</w:abstractNum><w:abstractNum w:abstractNumId=\"1\"><w:multiLevelType w:val=\"hybridMultilevel\"/>");
    xml.push_str(&levels(&|i| (numbers[i].into(), format!("%{}.", i + 1))));
    xml.push_str("</w:abstractNum>");
    for (n, (numbered, start)) in lists.iter().enumerate() {
        xml.push_str(&format!(
            "<w:num w:numId=\"{}\"><w:abstractNumId w:val=\"{}\"/>",
            n + 1,
            u8::from(*numbered)
        ));
        // Overriding where each level starts is what makes Word start this list again.
        for i in 0..4 {
            xml.push_str(&format!(
                "<w:lvlOverride w:ilvl=\"{i}\"><w:startOverride w:val=\"{}\"/></w:lvlOverride>",
                if i == 0 { *start } else { 1 }
            ));
        }
        xml.push_str("</w:num>");
    }
    xml.push_str("</w:numbering>");
    xml
}

fn styles() -> String {
    let heading = |level: usize, half_points: usize, before: usize| {
        format!(
            "<w:style w:type=\"paragraph\" w:styleId=\"Heading{level}\"><w:name w:val=\"heading {level}\"/><w:basedOn w:val=\"Normal\"/>\
             <w:next w:val=\"Normal\"/><w:uiPriority w:val=\"9\"/><w:qFormat/>\
             <w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before=\"{before}\" w:after=\"80\"/><w:outlineLvl w:val=\"{}\"/></w:pPr>\
             <w:rPr><w:b/><w:bCs/><w:color w:val=\"1F2328\"/><w:sz w:val=\"{half_points}\"/><w:szCs w:val=\"{half_points}\"/></w:rPr></w:style>",
            level - 1
        )
    };
    let grey = "w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"D0D7DE\"";
    format!(
        "{HEAD}<w:styles {WORD}>\
         <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\" w:eastAsia=\"Aptos\" w:cs=\"Aptos\"/>\
         <w:sz w:val=\"22\"/><w:szCs w:val=\"22\"/><w:lang w:val=\"en-US\"/></w:rPr></w:rPrDefault>\
         <w:pPrDefault><w:pPr><w:spacing w:after=\"120\" w:line=\"276\" w:lineRule=\"auto\"/></w:pPr></w:pPrDefault></w:docDefaults>\
         <w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:qFormat/>\
         <w:rPr><w:color w:val=\"1F2328\"/></w:rPr></w:style>\
         <w:style w:type=\"character\" w:default=\"1\" w:styleId=\"DefaultParagraphFont\"><w:name w:val=\"Default Paragraph Font\"/>\
         <w:uiPriority w:val=\"1\"/><w:semiHidden/><w:unhideWhenUsed/></w:style>\
         <w:style w:type=\"table\" w:default=\"1\" w:styleId=\"TableNormal\"><w:name w:val=\"Normal Table\"/><w:semiHidden/><w:unhideWhenUsed/>\
         <w:tblPr><w:tblInd w:w=\"0\" w:type=\"dxa\"/><w:tblCellMar><w:top w:w=\"0\" w:type=\"dxa\"/><w:left w:w=\"108\" w:type=\"dxa\"/>\
         <w:bottom w:w=\"0\" w:type=\"dxa\"/><w:right w:w=\"108\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"Title\"><w:name w:val=\"Title\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>\
         <w:uiPriority w:val=\"10\"/><w:qFormat/><w:pPr><w:spacing w:after=\"240\" w:line=\"240\" w:lineRule=\"auto\"/></w:pPr>\
         <w:rPr><w:b/><w:bCs/><w:kern w:val=\"28\"/><w:sz w:val=\"48\"/><w:szCs w:val=\"48\"/></w:rPr></w:style>\
         {}{}{}{}\
         <w:style w:type=\"paragraph\" w:styleId=\"ListParagraph\"><w:name w:val=\"List Paragraph\"/><w:basedOn w:val=\"Normal\"/>\
         <w:uiPriority w:val=\"34\"/><w:qFormat/><w:pPr><w:spacing w:after=\"60\"/><w:ind w:left=\"720\"/><w:contextualSpacing/></w:pPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"Quote\"><w:name w:val=\"Quote\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>\
         <w:uiPriority w:val=\"29\"/><w:qFormat/><w:pPr><w:pBdr><w:left w:val=\"single\" w:sz=\"18\" w:space=\"8\" w:color=\"D0D7DE\"/></w:pBdr>\
         <w:ind w:left=\"360\"/></w:pPr><w:rPr><w:i/><w:iCs/><w:color w:val=\"57606A\"/></w:rPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"Code\"><w:name w:val=\"Code\"/><w:basedOn w:val=\"Normal\"/><w:qFormat/>\
         <w:pPr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F3F4F6\"/><w:spacing w:before=\"80\" w:after=\"200\" w:line=\"240\" w:lineRule=\"auto\"/>\
         <w:ind w:left=\"120\" w:right=\"120\"/></w:pPr>\
         <w:rPr><w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\" w:cs=\"Consolas\"/><w:sz w:val=\"19\"/><w:szCs w:val=\"19\"/></w:rPr></w:style>\
         <w:style w:type=\"paragraph\" w:styleId=\"TableText\"><w:name w:val=\"Table Text\"/><w:basedOn w:val=\"Normal\"/>\
         <w:pPr><w:spacing w:before=\"40\" w:after=\"40\" w:line=\"240\" w:lineRule=\"auto\"/></w:pPr>\
         <w:rPr><w:sz w:val=\"20\"/><w:szCs w:val=\"20\"/></w:rPr></w:style>\
         <w:style w:type=\"character\" w:styleId=\"InlineCode\"><w:name w:val=\"Inline Code\"/><w:basedOn w:val=\"DefaultParagraphFont\"/>\
         <w:rPr><w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\" w:cs=\"Consolas\"/><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F3F4F6\"/></w:rPr></w:style>\
         <w:style w:type=\"character\" w:styleId=\"Hyperlink\"><w:name w:val=\"Hyperlink\"/><w:basedOn w:val=\"DefaultParagraphFont\"/>\
         <w:uiPriority w:val=\"99\"/><w:unhideWhenUsed/><w:rPr><w:color w:val=\"0969DA\"/><w:u w:val=\"single\"/></w:rPr></w:style>\
         <w:style w:type=\"table\" w:styleId=\"TableGrid\"><w:name w:val=\"Table Grid\"/><w:basedOn w:val=\"TableNormal\"/><w:uiPriority w:val=\"39\"/>\
         <w:tblPr><w:tblBorders><w:top {grey}/><w:left {grey}/><w:bottom {grey}/><w:right {grey}/><w:insideH {grey}/><w:insideV {grey}/>\
         </w:tblBorders></w:tblPr></w:style>\
         </w:styles>",
        heading(1, 34, 280),
        heading(2, 28, 240),
        heading(3, 24, 200),
        heading(4, 22, 160),
    )
}

// ---- Filling a template ----------------------------------------------------------------

/// What filling a template did: the placeholders filled, the ones in the template with
/// no value given (left as they were), and the values given for names it doesn't have.
#[derive(Debug, PartialEq)]
pub struct Filled {
    pub filled: Vec<String>,
    pub left_empty: Vec<String>,
    pub not_in_template: Vec<String>,
}

/// A copy of `template` with each `{{placeholder}}` replaced by its value, in the body,
/// headers and footers. A placeholder split across runs (as Word often stores typed text)
/// keeps the first run's formatting; a value's line breaks become line breaks.
pub fn fill(template: &[u8], values: &[(String, String)]) -> Result<(Vec<u8>, Filled), String> {
    if template.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        return Err(
            "This template is password-protected or in the old .doc format. Save it as an \
             unprotected .docx."
                .into(),
        );
    }
    let zip = Zip::open(template)?;
    let lookup: Vec<(String, &str)> = values
        .iter()
        .map(|(k, v)| (k.trim().to_lowercase(), v.as_str()))
        .collect();
    let mut filled: Vec<String> = Vec::new();
    let mut left_empty: Vec<String> = Vec::new();
    let note = |list: &mut Vec<String>, name: &str| {
        if !list.iter().any(|n| n.eq_ignore_ascii_case(name)) {
            list.push(name.to_string());
        }
    };
    let mut parts: Vec<(String, Vec<u8>)> = Vec::new();
    let mut has_document = false;
    for name in zip.names() {
        if name.ends_with('/') {
            continue;
        }
        let Some(mut bytes) = zip.read(name)? else { continue };
        let text_part = name == "word/document.xml"
            || (name.starts_with("word/header") || name.starts_with("word/footer")) && name.ends_with(".xml");
        if text_part {
            has_document |= name == "word/document.xml";
            let xml = String::from_utf8_lossy(&bytes).into_owned();
            let new = fill_part(&xml, &mut |placeholder| {
                match lookup.iter().find(|(k, _)| *k == placeholder.to_lowercase()) {
                    Some((_, value)) => {
                        note(&mut filled, placeholder);
                        Some(value.to_string())
                    }
                    None => {
                        note(&mut left_empty, placeholder);
                        None
                    }
                }
            });
            bytes = new.into_bytes();
        } else if name == "[Content_Types].xml" {
            // A template (.dotx) saved as a document is a document.
            bytes = String::from_utf8_lossy(&bytes)
                .replace("wordprocessingml.template.main+xml", "wordprocessingml.document.main+xml")
                .into_bytes();
        }
        parts.push((name.to_string(), bytes));
    }
    if !has_document {
        return Err("The template has no document in it.".into());
    }
    let mut not_in_template: Vec<String> = values
        .iter()
        .map(|(k, _)| k.trim().to_string())
        .filter(|k| !filled.iter().any(|f| f.eq_ignore_ascii_case(k)))
        .collect();
    let by_name = |a: &String, b: &String| a.to_lowercase().cmp(&b.to_lowercase());
    filled.sort_by(by_name);
    left_empty.sort_by(by_name);
    not_in_template.sort_by(by_name);
    let files: Vec<(&str, &[u8])> = parts.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
    Ok((stored_zip(&files), Filled { filled, left_empty, not_in_template }))
}

/// The placeholder names in a text, in order, without repeats.
#[cfg(test)]
pub fn placeholders(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut names: Vec<String> = Vec::new();
    for (_, _, name) in placeholder_spans(&chars) {
        if !names.iter().any(|n| n.eq_ignore_ascii_case(&name)) {
            names.push(name);
        }
    }
    names
}

/// Each `{{ name }}` in the text: where it starts and ends (in characters) and its name.
fn placeholder_spans(chars: &[char]) -> Vec<(usize, usize, String)> {
    let mut found = Vec::new();
    let mut i = 0;
    while i + 1 < chars.len() {
        if chars[i] == '{' && chars[i + 1] == '{' {
            // The name: anything but braces, up to the next "}}".
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '{' && chars[j] != '}' {
                j += 1;
            }
            let name: String = chars[i + 2..j].iter().collect();
            if j + 1 < chars.len() && chars[j] == '}' && chars[j + 1] == '}' && !name.trim().is_empty() {
                found.push((i, j + 2, name.trim().to_string()));
                i = j + 2;
                continue;
            }
        }
        i += 1;
    }
    found
}

/// One of a paragraph's `<w:t>` elements: where its opening tag and its text are in the
/// part's XML, and the text itself.
struct TextPiece {
    tag: (usize, usize),
    content: (usize, usize),
    text: Vec<char>,
    changed: bool,
    /// Further lines put after this text (a value with line breaks): each a `<w:br/>` and
    /// a text of its own, in the same run.
    more: Vec<String>,
}

/// A part's XML with its paragraphs' placeholders replaced by what `value` gives for each
/// name (None leaves one as it is).
fn fill_part(xml: &str, value: &mut dyn FnMut(&str) -> Option<String>) -> String {
    // The texts of each paragraph, innermost: a text box's paragraphs are their own.
    let mut paragraphs: Vec<Vec<TextPiece>> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let mut at = 0;
    while let Some(tag_start) = xml[at..].find('<').map(|i| at + i) {
        let Some(tag_end) = xml[tag_start..].find('>').map(|i| tag_start + i + 1) else {
            break;
        };
        let tag = &xml[tag_start..tag_end];
        at = tag_end;
        let named = |name: &str| tag == format!("<{name}>") || tag.starts_with(&format!("<{name} "));
        if named("w:p") && !tag.ends_with("/>") {
            paragraphs.push(Vec::new());
            open.push(paragraphs.len() - 1);
        } else if tag == "</w:p>" {
            open.pop();
        } else if named("w:t") && !tag.ends_with("/>") {
            let Some(end) = xml[tag_end..].find("</w:t>").map(|i| tag_end + i) else {
                break;
            };
            if let Some(&p) = open.last() {
                let mut text = String::new();
                unescape_into(&mut text, &xml[tag_end..end]);
                paragraphs[p].push(TextPiece {
                    tag: (tag_start, tag_end),
                    content: (tag_end, end),
                    text: text.chars().collect(),
                    changed: false,
                    more: Vec::new(),
                });
            }
            at = end;
        }
    }

    let mut edits: Vec<((usize, usize), String)> = Vec::new();
    for texts in &mut paragraphs {
        if texts.is_empty() {
            continue;
        }
        let whole: Vec<char> = texts.iter().flat_map(|t| t.text.iter().copied()).collect();
        let mut starts = Vec::with_capacity(texts.len());
        let mut total = 0;
        for t in texts.iter() {
            starts.push(total);
            total += t.text.len();
        }
        // Last first, so earlier positions stay right.
        for (start, end, name) in placeholder_spans(&whole).into_iter().rev() {
            let Some(new) = value(&name) else { continue };
            let new = new.replace("\r\n", "\n");
            let mut lines = new.split('\n');
            let first_line: Vec<char> = lines.next().unwrap_or("").chars().collect();
            let rest: Vec<String> = lines.map(str::to_string).collect();
            for i in (0..texts.len()).rev() {
                let (piece_start, piece_end) = (starts[i], starts[i] + texts[i].text.len());
                if piece_end <= start || piece_start >= end {
                    continue;
                }
                let from = start.max(piece_start) - piece_start;
                let to = end.min(piece_end) - piece_start;
                let t = &mut texts[i];
                let after: Vec<char> = t.text[to..].to_vec();
                t.text.truncate(from);
                t.changed = true;
                // The text the placeholder starts in takes the value; the others lose
                // their part of it.
                if start >= piece_start && start < piece_end {
                    t.text.extend(&first_line);
                    if rest.is_empty() {
                        t.text.extend(after);
                    } else {
                        // What followed the placeholder goes after the value's last line.
                        let mut more = rest.clone();
                        let last = more.len() - 1;
                        more[last].push_str(&after.iter().collect::<String>());
                        more.append(&mut t.more);
                        t.more = more;
                    }
                } else {
                    t.text.extend(after);
                }
            }
        }
        for t in texts.iter().filter(|t| t.changed) {
            let mut content = escaped(&t.text.iter().collect::<String>());
            for line in &t.more {
                content.push_str(&format!("</w:t><w:br/><w:t xml:space=\"preserve\">{}", escaped(line)));
            }
            edits.push((t.content, content));
            edits.push((t.tag, "<w:t xml:space=\"preserve\">".to_string()));
        }
    }
    edits.sort_by_key(|e| e.0.0);
    let mut out = String::with_capacity(xml.len() + 256);
    let mut done = 0;
    for ((start, end), text) in edits {
        out.push_str(&xml[done..start]);
        out.push_str(&text);
        done = end;
    }
    out.push_str(&xml[done..]);
    out
}

#[cfg(test)]
mod tests {
    use super::super::docx;
    use super::*;

    #[test]
    fn a_new_document_reads_back_with_its_shape() {
        let bytes = create(
            "Ignored",
            "# Sponsorship proposal\n\nTo: Helen Achebe\nFrom: Maya\n\n## Costs\n\n1. Gold: **CAD 25,000**\n2. Silver\n\n- a bullet with a [link](https://example.com/a?b=1&c=2)\n\n| Level | CAD |\n|---|--:|\n| Gold | 25,000 |\n\n> Decide by *Friday*.\n\n```\nx < 1 && y\n```",
        );
        let text = docx::text(&bytes).unwrap();
        assert!(text.starts_with("# Sponsorship proposal\n"), "{text}");
        assert!(!text.contains("Ignored"));
        for expected in ["Helen Achebe", "## Costs", "Gold: CAD 25,000", "a bullet with a link", "| Level | CAD |", "| Gold | 25,000 |", "Decide by Friday.", "x < 1 && y"] {
            assert!(text.contains(expected), "no {expected:?} in:\n{text}");
        }
        let zip = Zip::open(&bytes).unwrap();
        let part = |name: &str| String::from_utf8(zip.read(name).unwrap().unwrap()).unwrap();
        let rels = part("word/_rels/document.xml.rels");
        assert!(rels.contains("Target=\"https://example.com/a?b=1&amp;c=2\" TargetMode=\"External\""));
        let document = part("word/document.xml");
        assert!(document.contains("<w:hyperlink r:id=\"rIdLink1\""));
        // The memo's lines stay lines, and the two lists are lists of their own.
        assert!(document.contains("To: Helen Achebe</w:t><w:br/><w:t xml:space=\"preserve\">From: Maya"));
        assert!(document.contains("<w:numId w:val=\"1\"/>") && document.contains("<w:numId w:val=\"2\"/>"));
        assert!(part("word/numbering.xml").contains("<w:num w:numId=\"2\"><w:abstractNumId w:val=\"0\"/>"));
        assert!(part("docProps/core.xml").contains("<dc:title>Sponsorship proposal</dc:title>"));
        // No lists, no numbering part.
        let plain = create("Note", "Just a line.");
        assert!(Zip::open(&plain).unwrap().read("word/numbering.xml").unwrap().is_none());
        assert_eq!(docx::text(&plain).unwrap(), "# Note\nJust a line.");
    }

    /// Writes sample files for other readers to open: `VIREO_OUT=<folder> [VIREO_TEMPLATE=<docx>]
    /// cargo test -p vireo-server write_samples -- --ignored`.
    #[test]
    #[ignore]
    fn write_samples() {
        let out = std::env::var("VIREO_OUT").unwrap();
        let content = "# Sponsorship proposal\n\nTo: Helen Achebe\nFrom: Maya Lindqvist\n**Date:** 12 November 2026\n\n## Recommendation\n\nGo with **Gold**, at *CAD 25,000*. See `GL 6410` and [the forum site](https://example.com/forum?year=2027&x=1).\n\n1. Booth space\n2. Two speaking slots\n   - keynote\n   - panel\n\n- A separate bullet\n\n| Level | CAD | Passes |\n|:--|--:|:-:|\n| Gold | 25,000 | 12 |\n| Silver | 15,000 | 6 |\n\n> Decide by Friday.\n\n---\n\n```\ntotal = 25000 * 1.13\n```\n\nÉtudes & <angles> stay as written.";
        std::fs::write(format!("{out}/sample.docx"), create("Ignored", content)).unwrap();
        let slides = super::super::pptx_write::slides_from(
            r#"[{"title":"Why Gold","bullets":["Booth & two slots","  Keynote","  Panel","CAD 25,000"]},{"title":"Timeline","bullets":["Decide by Friday"]},{"title":"Questions"}]"#,
        )
        .unwrap();
        std::fs::write(format!("{out}/sample.pptx"), super::super::pptx_write::create("Forum 2027: sponsorship", &slides)).unwrap();
        if let Ok(template) = std::env::var("VIREO_TEMPLATE") {
            let data = std::fs::read(&template).unwrap();
            let text = docx::text(&data).unwrap();
            let names = placeholders(&text);
            println!("placeholders: {names:?}");
            let values: Vec<(String, String)> = names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), if i == 0 { "Line one\nLine two".to_string() } else { format!("VALUE-{i} & <ok>") }))
                .collect();
            let (bytes, result) = fill(&data, &values).unwrap();
            println!("{result:?}");
            let name = std::path::Path::new(&template).file_name().unwrap().to_string_lossy().into_owned();
            std::fs::write(format!("{out}/filled-{name}"), bytes).unwrap();
        }
    }

    fn template(document: &str, header: &str) -> Vec<u8> {
        stored_zip(&[
            ("[Content_Types].xml", b"<Types><Override ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml\"/></Types>"),
            ("word/document.xml", document.as_bytes()),
            ("word/header1.xml", header.as_bytes()),
            ("word/styles.xml", b"<w:styles/>"),
        ])
    }

    #[test]
    fn a_template_is_filled_keeping_its_formatting() {
        // A placeholder split over three runs, one in a table inside a text box, one with
        // no value, and one in the header.
        let document = "<w:document><w:body>\
            <w:p><w:r><w:rPr><w:b/></w:rPr><w:t>Dear {{</w:t></w:r><w:r><w:t>cli</w:t></w:r><w:r><w:rPr><w:i/></w:rPr><w:t>ent}}, re {{ Amount }} &amp; {{missing}}</w:t></w:r></w:p>\
            <w:p><w:r><w:t>Outer {{date}}</w:t><w:pict><w:txbxContent><w:p><w:r><w:t xml:space=\"preserve\">Inner {{client}} </w:t></w:r></w:p></w:txbxContent></w:pict></w:r></w:p>\
            <w:p><w:r><w:tab/><w:t>{{address}} end</w:t></w:r></w:p>\
            </w:body></w:document>";
        let header = "<w:hdr><w:p><w:r><w:t>Ref {{REF}}</w:t></w:r></w:p></w:hdr>";
        let values: Vec<(String, String)> = [
            ("client", "Acme & Sons"),
            (" amount ", "CAD <1,250>"),
            ("date", "12 November 2026"),
            ("address", "1 Main St\nTurin\nItaly"),
            ("ref", "R-7"),
            ("unused", "x"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let (bytes, result) = fill(&template(document, header), &values).unwrap();
        assert_eq!(
            result,
            Filled {
                filled: vec!["address".into(), "Amount".into(), "client".into(), "date".into(), "REF".into()],
                left_empty: vec!["missing".into()],
                not_in_template: vec!["unused".into()],
            }
        );
        let zip = Zip::open(&bytes).unwrap();
        let part = |name: &str| String::from_utf8(zip.read(name).unwrap().unwrap()).unwrap();
        let filled = part("word/document.xml");
        // The value sits in the run the placeholder started in, with that run's bold.
        assert!(filled.contains("<w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">Dear Acme &amp; Sons</w:t>"), "{filled}");
        assert!(filled.contains("<w:t xml:space=\"preserve\"></w:t></w:r><w:r><w:rPr><w:i/></w:rPr><w:t xml:space=\"preserve\">, re CAD &lt;1,250&gt; &amp; {{missing}}</w:t>"), "{filled}");
        assert!(filled.contains("Outer 12 November 2026</w:t>"));
        assert!(filled.contains("Inner Acme &amp; Sons </w:t>"));
        assert!(filled.contains("<w:t xml:space=\"preserve\">1 Main St</w:t><w:br/><w:t xml:space=\"preserve\">Turin</w:t><w:br/><w:t xml:space=\"preserve\">Italy end</w:t>"), "{filled}");
        assert!(part("word/header1.xml").contains("Ref R-7"));
        assert_eq!(part("word/styles.xml"), "<w:styles/>");
        assert!(part("[Content_Types].xml").contains("document.main+xml"));
        assert_eq!(placeholders("{{a}} {{ B }} {{a}} {{}} {x} {{ c"), vec!["a", "B"]);
        assert!(fill(&stored_zip(&[("x.xml", b"<x/>")]), &values).unwrap_err().contains("no document"));
    }
}
