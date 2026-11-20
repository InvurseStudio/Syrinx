//! PowerPoint (.pptx) text, as Vireo 0.9 read a deck (`office_files.read_deck`): each
//! slide's title, its other text a line at a time, its tables' rows between bars, and its
//! speaker notes, in the deck's order. The file is a ZIP: the order is in
//! `ppt/presentation.xml` (its slide ids, through `ppt/_rels/presentation.xml.rels`), each
//! slide in `ppt/slides/slideN.xml`, and its notes in the notes slide its own relationships
//! name.

use super::docx::Zip;
use super::xlsx::{Piece, Pieces, attribute, unescaped};

/// The most slides read, so a huge deck doesn't fill the chat.
const MAX_SLIDES: usize = 300;

pub fn text(data: &[u8]) -> Result<String, String> {
    let (count, slides) = read(data)?;
    let mut out = format!(
        "PowerPoint deck with {count} slide{}",
        if count == 1 { "" } else { "s" }
    );
    for (number, slide) in slides.iter().enumerate() {
        out.push_str(&format!(
            "\n\n## Slide {}{}: {}",
            number + 1,
            if slide.hidden { " (hidden)" } else { "" },
            slide.title.as_deref().unwrap_or("(no title)")
        ));
        for line in &slide.lines {
            out.push_str(&format!("\n- {line}"));
        }
        for row in &slide.rows {
            out.push_str(&format!("\n| {} |", row.join(" | ")));
        }
        if !slide.notes.is_empty() {
            out.push_str(&format!("\nNotes: {}", slide.notes));
        }
    }
    if count > MAX_SLIDES {
        out.push_str(&format!("\n\n(The other {} slides weren't read.)", count - MAX_SLIDES));
    }
    Ok(out)
}

/// The deck as Markdown for the app's preview pane, as 0.9 showed it: a heading per slide,
/// its text, tables and notes, slides apart.
pub fn markdown(data: &[u8]) -> Result<String, String> {
    let (_, slides) = read(data)?;
    let parts: Vec<String> = slides
        .iter()
        .enumerate()
        .map(|(number, slide)| {
            let mut part = format!(
                "## {}. {}{}",
                number + 1,
                slide.title.as_deref().unwrap_or("(no title)"),
                if slide.hidden { " *(hidden)*" } else { "" }
            );
            for line in &slide.lines {
                part.push_str(&format!("\n- {line}"));
            }
            if let Some((head, rest)) = slide.rows.split_first() {
                let width = slide.rows.iter().map(Vec::len).max().unwrap_or(1);
                let cells = |row: &Vec<String>| {
                    let mut cells = row.clone();
                    cells.resize(width, String::new());
                    format!("| {} |", cells.join(" | "))
                };
                part.push_str(&format!("\n\n{}\n|{}", cells(head), " --- |".repeat(width)));
                for row in rest {
                    part.push_str(&format!("\n{}", cells(row)));
                }
            }
            if !slide.notes.is_empty() {
                part.push_str(&format!("\n\n**Notes:** {}", slide.notes));
            }
            part
        })
        .collect();
    Ok(if parts.is_empty() { "*No slides*".into() } else { parts.join("\n\n---\n\n") })
}

/// The deck's slide count, and its first `MAX_SLIDES` slides, read.
fn read(data: &[u8]) -> Result<(usize, Vec<Slide>), String> {
    if data.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        return Err(
            "This PowerPoint file is password-protected or in the old .ppt format, so its text \
             can't be read. Save it as an unprotected .pptx and try again."
                .into(),
        );
    }
    let zip = Zip::open(data)?;
    let read = |name: &str| -> Result<Option<String>, String> {
        Ok(zip.read(name)?.map(|b| String::from_utf8_lossy(&b).into_owned()))
    };
    let presentation = read("ppt/presentation.xml")?
        .ok_or("This .pptx has no presentation in it (ppt/presentation.xml is missing).")?;
    let relations = relationships(&read("ppt/_rels/presentation.xml.rels")?.unwrap_or_default());
    let mut paths: Vec<String> = slide_ids(&presentation)
        .iter()
        .filter_map(|id| relations.iter().find(|(rid, _)| rid == id))
        .map(|(_, target)| resolve("ppt", target))
        .collect();
    if paths.is_empty() {
        // No order recorded: the slide files by number.
        let mut found: Vec<(u32, String)> = zip
            .names()
            .filter_map(|n| {
                let number = n.strip_prefix("ppt/slides/slide")?.strip_suffix(".xml")?;
                Some((number.parse().ok()?, n.to_string()))
            })
            .collect();
        found.sort();
        paths = found.into_iter().map(|(_, n)| n).collect();
    }
    let mut slides = Vec::new();
    for path in paths.iter().take(MAX_SLIDES) {
        let Some(xml) = read(path)? else { continue };
        let mut slide = parse_slide(&xml);
        // The notes slide, through the slide's own relationships.
        let (folder, file) = path.rsplit_once('/').unwrap_or(("", path));
        let rels = read(&format!("{folder}/_rels/{file}.rels"))?.unwrap_or_default();
        if let Some((_, target)) = relationships(&rels).into_iter().find(|(_, t)| t.contains("notesSlide"))
            && let Some(notes) = read(&resolve(folder, &target))?
        {
            slide.notes = notes_text(&notes);
        }
        slides.push(slide);
    }
    Ok((paths.len(), slides))
}

/// The `r:id` of each slide, in the deck's order.
fn slide_ids(xml: &str) -> Vec<String> {
    Pieces::new(xml)
        .filter_map(|p| match p {
            Piece::Open("sldId", attrs, _) => attribute(attrs, "r:id").map(unescaped),
            _ => None,
        })
        .collect()
}

/// A relationships file's `(Id, Target)` pairs.
fn relationships(xml: &str) -> Vec<(String, String)> {
    Pieces::new(xml)
        .filter_map(|p| match p {
            Piece::Open("Relationship", attrs, _) => Some((
                unescaped(attribute(attrs, "Id")?),
                unescaped(attribute(attrs, "Target")?),
            )),
            _ => None,
        })
        .collect()
}

/// A relationship's target, from the folder of the part that names it.
fn resolve(folder: &str, target: &str) -> String {
    if let Some(absolute) = target.strip_prefix('/') {
        return absolute.to_string();
    }
    let mut parts: Vec<&str> = folder.split('/').filter(|p| !p.is_empty()).collect();
    for piece in target.split('/') {
        match piece {
            ".." => {
                parts.pop();
            }
            "." | "" => {}
            p => parts.push(p),
        }
    }
    parts.join("/")
}

#[derive(Debug, Default, PartialEq)]
struct Slide {
    title: Option<String>,
    lines: Vec<String>,
    rows: Vec<Vec<String>>,
    hidden: bool,
    notes: String,
}

/// One shape's text as it's read: its placeholder type and its paragraphs.
#[derive(Default)]
struct Shape {
    placeholder: Option<String>,
    paragraphs: Vec<String>,
}

/// A slide's shapes, in order: titles apart, table cells by row. Shapes in groups are
/// read where they are, as 0.9 did.
fn parse_slide(xml: &str) -> Slide {
    let mut slide = Slide::default();
    let shapes = shapes(xml, &mut slide);
    for shape in shapes {
        let is_title = matches!(shape.placeholder.as_deref(), Some("title" | "ctrTitle"));
        if is_title && slide.title.is_none() {
            let title = shape.paragraphs.join(" ").trim().to_string();
            if !title.is_empty() {
                slide.title = Some(title);
                continue;
            }
        }
        // Slide numbers, dates and footers are the layout's, not the slide's words.
        if matches!(shape.placeholder.as_deref(), Some("sldNum" | "dt" | "ftr")) {
            continue;
        }
        for paragraph in shape.paragraphs {
            for line in paragraph.lines() {
                let line = line.trim();
                if !line.is_empty() {
                    slide.lines.push(line.to_string());
                }
            }
        }
    }
    slide
}

/// Walks a slide (or notes slide): its shapes' text, and table rows into `slide.rows`.
fn shapes(xml: &str, slide: &mut Slide) -> Vec<Shape> {
    let mut shapes = Vec::new();
    let mut shape: Option<Shape> = None;
    let mut paragraph: Option<String> = None;
    let mut in_text = false;
    let mut row: Option<Vec<String>> = None;
    let mut cell: Option<String> = None;
    for piece in Pieces::new(xml) {
        match piece {
            Piece::Open("sld", attrs, _) => slide.hidden = attribute(attrs, "show") == Some("0"),
            Piece::Open("sp", _, false) => shape = Some(Shape::default()),
            Piece::Close("sp") => shapes.extend(shape.take()),
            Piece::Open("ph", attrs, _) => {
                if let Some(s) = shape.as_mut() {
                    // A placeholder with no type is the body.
                    s.placeholder = Some(attribute(attrs, "type").unwrap_or("body").to_string());
                }
            }
            Piece::Open("tr", _, false) => row = Some(Vec::new()),
            Piece::Close("tr") => {
                if let Some(r) = row.take().filter(|r| r.iter().any(|c| !c.is_empty())) {
                    slide.rows.push(r);
                }
            }
            Piece::Open("tc", _, false) => cell = Some(String::new()),
            Piece::Close("tc") => {
                if let (Some(r), Some(c)) = (row.as_mut(), cell.take()) {
                    r.push(c.split_whitespace().collect::<Vec<_>>().join(" ").replace('|', "/"));
                }
            }
            Piece::Open("p", _, false) => paragraph = Some(String::new()),
            Piece::Close("p") => {
                if let Some(p) = paragraph.take() {
                    if let Some(c) = cell.as_mut() {
                        if !c.is_empty() {
                            c.push(' ');
                        }
                        c.push_str(&p);
                    } else if let Some(s) = shape.as_mut() {
                        s.paragraphs.push(p);
                    }
                }
            }
            Piece::Open("br", _, _) => {
                if let Some(p) = paragraph.as_mut() {
                    p.push('\n');
                }
            }
            Piece::Open("t", _, empty) => in_text = !empty,
            Piece::Close("t") => in_text = false,
            Piece::Text(text) if in_text => {
                if let Some(p) = paragraph.as_mut() {
                    p.push_str(&unescaped(text));
                }
            }
            _ => {}
        }
    }
    shapes
}

/// A notes slide's notes: its body placeholder's text, on one line.
fn notes_text(xml: &str) -> String {
    let mut ignored = Slide::default();
    shapes(xml, &mut ignored)
        .into_iter()
        .filter(|s| s.placeholder.as_deref() == Some("body"))
        .flat_map(|s| s.paragraphs)
        .flat_map(|p| p.lines().map(|l| l.trim().to_string()).collect::<Vec<_>>())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attach::docx::stored_zip;
    use crate::attach::pptx_write;

    #[test]
    fn reads_back_a_deck_vireo_wrote() {
        let deck = pptx_write::create(
            "Q3 review",
            &[
                pptx_write::Slide { title: "Revenue".into(), bullets: vec!["Up 12% & rising".into(), "Two new clients".into()] },
                pptx_write::Slide { title: "Next steps".into(), bullets: vec!["Hire".into()] },
            ],
        );
        let said = text(&deck).unwrap();
        assert!(said.starts_with("PowerPoint deck with 3 slides"), "{said}");
        assert!(said.contains("## Slide 1: Q3 review"), "{said}");
        assert!(said.contains("## Slide 2: Revenue\n- Up 12% & rising\n- Two new clients"), "{said}");
        assert!(said.contains("## Slide 3: Next steps\n- Hire"), "{said}");
    }

    const P: &str = "xmlns:a=\"a\" xmlns:p=\"p\" xmlns:r=\"r\"";

    fn shape(placeholder: &str, paragraphs: &[&str]) -> String {
        let ph = if placeholder.is_empty() { String::new() } else { format!("<p:nvPr><p:ph {placeholder}/></p:nvPr>") };
        let body: String = paragraphs.iter().map(|p| format!("<a:p><a:r><a:t>{p}</a:t></a:r></a:p>")).collect();
        format!("<p:sp><p:nvSpPr>{ph}</p:nvSpPr><p:txBody>{body}</p:txBody></p:sp>")
    }

    #[test]
    fn reads_titles_tables_notes_hidden_slides_and_the_deck_order() {
        let slide1 = format!(
            "<p:sld {P} show=\"0\"><p:cSld><p:spTree>{}{}<p:grpSp>{}</p:grpSp>\
             <p:graphicFrame><a:graphic><a:graphicData><a:tbl>\
             <a:tr><a:tc><a:txBody><a:p><a:r><a:t>Item</a:t></a:r></a:p></a:txBody></a:tc>\
             <a:tc><a:txBody><a:p><a:r><a:t>Cost</a:t></a:r></a:p></a:txBody></a:tc></a:tr>\
             <a:tr><a:tc><a:txBody><a:p><a:r><a:t>Venue</a:t></a:r></a:p></a:txBody></a:tc>\
             <a:tc><a:txBody><a:p><a:r><a:t>$4,000</a:t></a:r></a:p></a:txBody></a:tc></a:tr>\
             </a:tbl></a:graphicData></a:graphic></p:graphicFrame>{}</p:spTree></p:cSld></p:sld>",
            shape("type=\"title\"", &["Budget"]),
            shape("idx=\"1\"", &["Total &lt; $10k", "Approved"]),
            shape("", &["In a group"]),
            shape("type=\"sldNum\"", &["7"]),
        );
        let slide2 = format!("<p:sld {P}><p:cSld><p:spTree>{}</p:spTree></p:cSld></p:sld>", shape("", &["Untitled words"]));
        let notes = format!(
            "<p:notes {P}><p:cSld><p:spTree>{}{}</p:spTree></p:cSld></p:notes>",
            shape("type=\"sldImg\"", &[]),
            shape("type=\"body\" idx=\"1\"", &["Say the venue", "is booked."])
        );
        let rels = |items: &[(&str, &str)]| {
            let r: String = items.iter().map(|(id, t)| format!("<Relationship Id=\"{id}\" Type=\"x\" Target=\"{t}\"/>")).collect();
            format!("<Relationships>{r}</Relationships>")
        };
        // The deck lists slide2.xml first.
        let presentation = format!("<p:presentation {P}><p:sldIdLst><p:sldId id=\"256\" r:id=\"rId9\"/><p:sldId id=\"257\" r:id=\"rId8\"/></p:sldIdLst></p:presentation>");
        let deck_rels = rels(&[("rId8", "slides/slide1.xml"), ("rId9", "slides/slide2.xml")]);
        let slide1_rels = rels(&[("rId1", "../notesSlides/notesSlide1.xml")]);
        let deck = stored_zip(&[
            ("ppt/presentation.xml", presentation.as_bytes()),
            ("ppt/_rels/presentation.xml.rels", deck_rels.as_bytes()),
            ("ppt/slides/slide1.xml", slide1.as_bytes()),
            ("ppt/slides/slide2.xml", slide2.as_bytes()),
            ("ppt/slides/_rels/slide1.xml.rels", slide1_rels.as_bytes()),
            ("ppt/notesSlides/notesSlide1.xml", notes.as_bytes()),
        ]);
        let said = text(&deck).unwrap();
        assert_eq!(
            said,
            "PowerPoint deck with 2 slides\n\n## Slide 1: (no title)\n- Untitled words\n\n\
             ## Slide 2 (hidden): Budget\n- Total < $10k\n- Approved\n- In a group\n\
             | Item | Cost |\n| Venue | $4,000 |\nNotes: Say the venue is booked."
        );
    }

    #[test]
    fn markdown_for_the_preview() {
        let deck = pptx_write::create("Q3", &[pptx_write::Slide { title: "Revenue".into(), bullets: vec!["Up".into()] }]);
        assert_eq!(markdown(&deck).unwrap(), "## 1. Q3\n\n---\n\n## 2. Revenue\n- Up");
    }

    #[test]
    fn says_why_an_old_or_locked_deck_cant_be_read() {
        assert!(text(&[0xD0, 0xCF, 0x11, 0xE0, 0, 0]).unwrap_err().contains(".ppt"));
        assert!(resolve("ppt/slides", "../notesSlides/n.xml") == "ppt/notesSlides/n.xml");
    }
}
