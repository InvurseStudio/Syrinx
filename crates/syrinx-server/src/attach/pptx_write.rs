//! Writing a PowerPoint deck (.pptx): a title slide, then a slide per entry with its
//! title and bullet points, in the plain theme the Windows app's `SlideDeck` uses (16:9,
//! Aptos, Vireo's green for bullets). Written as a ZIP without compression.

use super::docx::stored_zip;
use super::docx_write::{core_properties, escaped};
use crate::json::Value;

const NAMESPACES: &str = "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" \
     xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" \
     xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"";
const RELATIONSHIPS: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PACKAGE: &str = "http://schemas.openxmlformats.org/package/2006";
const HEAD: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>";

#[derive(Debug, Clone, PartialEq)]
pub struct Slide {
    pub title: String,
    pub bullets: Vec<String>,
}

/// The slides in the tool's `slides` argument: a JSON list of `{title, bullets}` (a model
/// sometimes sends the JSON inside a string, or a slide as its title alone).
pub fn slides_from(json: &str) -> Result<Vec<Slide>, String> {
    const SHAPE: &str = "slides must be a list of slides.";
    let mut value = crate::json::parse(json).map_err(|_| SHAPE.to_string())?;
    if let Value::String(inner) = &value {
        value = crate::json::parse(inner).map_err(|_| SHAPE.to_string())?;
    }
    let Value::Array(items) = value else {
        return Err(SHAPE.into());
    };
    let text = |v: &Value| match v {
        Value::String(s) => s.clone(),
        other => other.to_json(),
    };
    items
        .iter()
        .map(|item| match item {
            Value::String(title) => Ok(Slide { title: title.clone(), bullets: Vec::new() }),
            Value::Object(_) => Ok(Slide {
                title: item.str("title").unwrap_or("").to_string(),
                bullets: item.array("bullets").iter().map(text).collect(),
            }),
            _ => Err("Each slide needs to be an object with a title, and may have bullets, like \
                      {\"title\": \"Goals\", \"bullets\": [\"First\"]}."
                .to_string()),
        })
        .collect()
}

const GROUP_START: &str = "<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>\
     <p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>";
const COLOR_MAP: &str = "bg1=\"lt1\" tx1=\"dk1\" bg2=\"lt2\" tx2=\"dk2\" accent1=\"accent1\" accent2=\"accent2\" accent3=\"accent3\" \
     accent4=\"accent4\" accent5=\"accent5\" accent6=\"accent6\" hlink=\"hlink\" folHlink=\"folHlink\"";
const EMPTY_PARAGRAPH: &str = "<a:p><a:endParaRPr lang=\"en-US\"/></a:p>";
const NO_STYLE: &str = "<a:lstStyle/>";

/// A placeholder shape: its type and index, an optional position, and its paragraphs.
fn placeholder(
    id: u32,
    name: &str,
    kind: &str,
    position: Option<&str>,
    paragraphs: &str,
    body: &str,
    list_style: &str,
) -> String {
    let shape = match position {
        Some(p) => format!("<p:spPr><a:xfrm>{p}</a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr>"),
        None => "<p:spPr/>".to_string(),
    };
    format!(
        "<p:sp><p:nvSpPr><p:cNvPr id=\"{id}\" name=\"{name}\"/><p:cNvSpPr><a:spLocks noGrp=\"1\"/></p:cNvSpPr><p:nvPr>{kind}</p:nvPr></p:nvSpPr>\
         {shape}<p:txBody>{body}{list_style}{paragraphs}</p:txBody></p:sp>"
    )
}

fn run(text: &str, level: usize) -> String {
    let properties = if level > 0 { format!("<a:pPr lvl=\"{level}\"/>") } else { String::new() };
    format!(
        "<a:p>{properties}<a:r><a:rPr lang=\"en-US\" dirty=\"0\"/><a:t>{}</a:t></a:r></a:p>",
        escaped(text)
    )
}

fn slide_xml(shapes: &str) -> String {
    format!(
        "{HEAD}<p:sld {NAMESPACES}><p:cSld><p:spTree>{GROUP_START}{shapes}</p:spTree></p:cSld>\
         <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )
}

fn title_slide(title: &str) -> String {
    slide_xml(&placeholder(2, "Title 1", "<p:ph type=\"ctrTitle\"/>", None, &run(title, 0), "<a:bodyPr/>", NO_STYLE))
}

/// A bullet starting with two spaces is indented a level; its own mark is left off.
fn content_slide(slide: &Slide) -> String {
    let bullets: String = slide
        .bullets
        .iter()
        .filter(|b| !b.trim().is_empty())
        .map(|b| {
            let text = b.trim().trim_start_matches(['-', '•', '*']).trim();
            run(text, usize::from(b.starts_with("  ")))
        })
        .collect();
    let mut shapes = placeholder(2, "Title 1", "<p:ph type=\"title\"/>", None, &run(&slide.title, 0), "<a:bodyPr/>", NO_STYLE);
    if !bullets.is_empty() {
        shapes.push_str(&placeholder(3, "Content Placeholder 2", "<p:ph idx=\"1\"/>", None, &bullets, "<a:bodyPr/>", NO_STYLE));
    }
    slide_xml(&shapes)
}

fn master() -> String {
    let level = |n: usize, margin: usize, before: usize, mark: &str, size: usize| {
        format!(
            "<a:lvl{n}pPr marL=\"{margin}\" indent=\"-285750\"><a:spcBef><a:spcPts val=\"{before}\"/></a:spcBef>\
             <a:buClr><a:schemeClr val=\"accent1\"/></a:buClr><a:buFont typeface=\"Arial\"/><a:buChar char=\"{mark}\"/>\
             <a:defRPr sz=\"{size}\"><a:solidFill><a:schemeClr val=\"tx1\"/></a:solidFill><a:latin typeface=\"+mn-lt\"/></a:defRPr></a:lvl{n}pPr>"
        )
    };
    format!(
        "{HEAD}<p:sldMaster {NAMESPACES}><p:cSld><p:bg><p:bgRef idx=\"1001\"><a:schemeClr val=\"bg1\"/></p:bgRef></p:bg><p:spTree>{GROUP_START}{}{}</p:spTree></p:cSld>\
         <p:clrMap {COLOR_MAP}/>\
         <p:sldLayoutIdLst><p:sldLayoutId id=\"2147483649\" r:id=\"rIdLayout1\"/><p:sldLayoutId id=\"2147483650\" r:id=\"rIdLayout2\"/></p:sldLayoutIdLst>\
         <p:txStyles><p:titleStyle><a:lvl1pPr algn=\"l\"><a:defRPr sz=\"4000\" b=\"1\"><a:solidFill><a:schemeClr val=\"tx2\"/></a:solidFill>\
         <a:latin typeface=\"+mj-lt\"/></a:defRPr></a:lvl1pPr></p:titleStyle>\
         <p:bodyStyle>{}{}</p:bodyStyle>\
         <p:otherStyle><a:defPPr><a:defRPr lang=\"en-US\"/></a:defPPr></p:otherStyle></p:txStyles></p:sldMaster>",
        placeholder(
            2,
            "Title Placeholder 1",
            "<p:ph type=\"title\"/>",
            Some("<a:off x=\"838200\" y=\"365125\"/><a:ext cx=\"10515600\" cy=\"1325563\"/>"),
            EMPTY_PARAGRAPH,
            "<a:bodyPr anchor=\"ctr\"/>",
            NO_STYLE,
        ),
        placeholder(
            3,
            "Text Placeholder 2",
            "<p:ph type=\"body\" idx=\"1\"/>",
            Some("<a:off x=\"838200\" y=\"1825625\"/><a:ext cx=\"10515600\" cy=\"4351338\"/>"),
            EMPTY_PARAGRAPH,
            "<a:bodyPr/>",
            NO_STYLE,
        ),
        level(1, 285750, 1000, "•", 2400),
        level(2, 742950, 500, "–", 2000),
    )
}

fn layout(kind: &str, name: &str, shapes: &str) -> String {
    format!(
        "{HEAD}<p:sldLayout {NAMESPACES} type=\"{kind}\" preserve=\"1\"><p:cSld name=\"{name}\"><p:spTree>{GROUP_START}{shapes}</p:spTree></p:cSld>\
         <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    )
}

fn theme() -> String {
    let fill = "<a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill>";
    let line = |w: usize| format!("<a:ln w=\"{w}\">{fill}</a:ln>");
    let effect = "<a:effectStyle><a:effectLst/></a:effectStyle>";
    let color = |name: &str, value: &str| format!("<a:{name}><a:srgbClr val=\"{value}\"/></a:{name}>");
    let colors: String = [
        ("dk1", "1B2430"), ("lt1", "FFFFFF"), ("dk2", "0B1018"), ("lt2", "E8F0F1"),
        ("accent1", "1FA673"), ("accent2", "7C5CE0"), ("accent3", "2F80ED"), ("accent4", "F2994A"),
        ("accent5", "EB5757"), ("accent6", "8D9BA8"), ("hlink", "2F80ED"), ("folHlink", "7C5CE0"),
    ]
    .iter()
    .map(|(n, v)| color(n, v))
    .collect();
    format!(
        "{HEAD}<a:theme xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" name=\"Vireo\"><a:themeElements>\
         <a:clrScheme name=\"Vireo\">{colors}</a:clrScheme>\
         <a:fontScheme name=\"Vireo\">\
         <a:majorFont><a:latin typeface=\"Aptos Display\"/><a:ea typeface=\"\"/><a:cs typeface=\"\"/></a:majorFont>\
         <a:minorFont><a:latin typeface=\"Aptos\"/><a:ea typeface=\"\"/><a:cs typeface=\"\"/></a:minorFont></a:fontScheme>\
         <a:fmtScheme name=\"Vireo\"><a:fillStyleLst>{fill}{fill}{fill}</a:fillStyleLst>\
         <a:lnStyleLst>{}{}{}</a:lnStyleLst>\
         <a:effectStyleLst>{effect}{effect}{effect}</a:effectStyleLst>\
         <a:bgFillStyleLst>{fill}{fill}{fill}</a:bgFillStyleLst></a:fmtScheme>\
         </a:themeElements></a:theme>",
        line(6350),
        line(12700),
        line(19050),
    )
}

fn relationships(items: &[(&str, &str, &str)]) -> String {
    let mut xml = format!("{HEAD}<Relationships xmlns=\"{PACKAGE}/relationships\">");
    for (id, kind, target) in items {
        xml.push_str(&format!(
            "<Relationship Id=\"{id}\" Type=\"{RELATIONSHIPS}/{kind}\" Target=\"{target}\"/>"
        ));
    }
    xml.push_str("</Relationships>");
    xml
}

/// A deck: a title slide, then a slide per entry. Its bytes.
pub fn create(title: &str, slides: &[Slide]) -> Vec<u8> {
    const KIND: &str = "application/vnd.openxmlformats-officedocument.presentationml";
    let count = slides.len() + 1;
    let mut parts: Vec<(String, String)> = Vec::new();

    let mut types = format!(
        "{HEAD}<Types xmlns=\"{PACKAGE}/content-types\">\
         <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
         <Default Extension=\"xml\" ContentType=\"application/xml\"/>\
         <Override PartName=\"/ppt/presentation.xml\" ContentType=\"{KIND}.presentation.main+xml\"/>\
         <Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"{KIND}.slideMaster+xml\"/>\
         <Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"{KIND}.slideLayout+xml\"/>\
         <Override PartName=\"/ppt/slideLayouts/slideLayout2.xml\" ContentType=\"{KIND}.slideLayout+xml\"/>\
         <Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>\
         <Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
    );
    for n in 1..=count {
        types.push_str(&format!(
            "<Override PartName=\"/ppt/slides/slide{n}.xml\" ContentType=\"{KIND}.slide+xml\"/>"
        ));
    }
    types.push_str("</Types>");
    parts.push(("[Content_Types].xml".into(), types));
    parts.push((
        "_rels/.rels".into(),
        format!(
            "{HEAD}<Relationships xmlns=\"{PACKAGE}/relationships\">\
             <Relationship Id=\"rId1\" Type=\"{RELATIONSHIPS}/officeDocument\" Target=\"ppt/presentation.xml\"/>\
             <Relationship Id=\"rId2\" Type=\"{PACKAGE}/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>\
             </Relationships>"
        ),
    ));
    parts.push(("docProps/core.xml".into(), core_properties(title)));

    let slide_ids: String = (1..=count)
        .map(|n| format!("<p:sldId id=\"{}\" r:id=\"rIdSlide{n}\"/>", 255 + n))
        .collect();
    parts.push((
        "ppt/presentation.xml".into(),
        format!(
            "{HEAD}<p:presentation {NAMESPACES} saveSubsetFonts=\"1\">\
             <p:sldMasterIdLst><p:sldMasterId id=\"2147483648\" r:id=\"rIdMaster\"/></p:sldMasterIdLst>\
             <p:sldIdLst>{slide_ids}</p:sldIdLst>\
             <p:sldSz cx=\"12192000\" cy=\"6858000\"/><p:notesSz cx=\"6858000\" cy=\"9144000\"/></p:presentation>"
        ),
    ));
    let slide_targets: Vec<(String, String)> = (1..=count)
        .map(|n| (format!("rIdSlide{n}"), format!("slides/slide{n}.xml")))
        .collect();
    let mut presentation_rels: Vec<(&str, &str, &str)> = vec![
        ("rIdMaster", "slideMaster", "slideMasters/slideMaster1.xml"),
        ("rIdTheme", "theme", "theme/theme1.xml"),
    ];
    presentation_rels.extend(slide_targets.iter().map(|(id, target)| (id.as_str(), "slide", target.as_str())));
    parts.push(("ppt/_rels/presentation.xml.rels".into(), relationships(&presentation_rels)));

    parts.push(("ppt/theme/theme1.xml".into(), theme()));
    parts.push(("ppt/slideMasters/slideMaster1.xml".into(), master()));
    parts.push((
        "ppt/slideMasters/_rels/slideMaster1.xml.rels".into(),
        relationships(&[
            ("rIdLayout1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rIdLayout2", "slideLayout", "../slideLayouts/slideLayout2.xml"),
            ("rIdTheme", "theme", "../theme/theme1.xml"),
        ]),
    ));
    let title_shapes = format!(
        "{}{}",
        placeholder(
            2,
            "Title 1",
            "<p:ph type=\"ctrTitle\"/>",
            Some("<a:off x=\"1524000\" y=\"1122363\"/><a:ext cx=\"9144000\" cy=\"2387600\"/>"),
            EMPTY_PARAGRAPH,
            "<a:bodyPr anchor=\"b\"/>",
            "<a:lstStyle><a:lvl1pPr algn=\"ctr\"><a:defRPr sz=\"5400\"/></a:lvl1pPr></a:lstStyle>",
        ),
        placeholder(
            3,
            "Subtitle 2",
            "<p:ph type=\"subTitle\" idx=\"1\"/>",
            Some("<a:off x=\"1524000\" y=\"3602038\"/><a:ext cx=\"9144000\" cy=\"1655762\"/>"),
            EMPTY_PARAGRAPH,
            "<a:bodyPr/>",
            "<a:lstStyle><a:lvl1pPr marL=\"0\" indent=\"0\" algn=\"ctr\"><a:buNone/><a:defRPr sz=\"2400\"/></a:lvl1pPr></a:lstStyle>",
        ),
    );
    let content_shapes = format!(
        "{}{}",
        placeholder(2, "Title 1", "<p:ph type=\"title\"/>", None, EMPTY_PARAGRAPH, "<a:bodyPr/>", NO_STYLE),
        placeholder(3, "Content Placeholder 2", "<p:ph idx=\"1\"/>", None, EMPTY_PARAGRAPH, "<a:bodyPr/>", NO_STYLE),
    );
    let to_master = relationships(&[("rIdMaster", "slideMaster", "../slideMasters/slideMaster1.xml")]);
    parts.push(("ppt/slideLayouts/slideLayout1.xml".into(), layout("title", "Title Slide", &title_shapes)));
    parts.push(("ppt/slideLayouts/_rels/slideLayout1.xml.rels".into(), to_master.clone()));
    parts.push(("ppt/slideLayouts/slideLayout2.xml".into(), layout("obj", "Title and Content", &content_shapes)));
    parts.push(("ppt/slideLayouts/_rels/slideLayout2.xml.rels".into(), to_master));

    for n in 1..=count {
        let (xml, layout) = if n == 1 {
            (title_slide(title), "../slideLayouts/slideLayout1.xml")
        } else {
            (content_slide(&slides[n - 2]), "../slideLayouts/slideLayout2.xml")
        };
        parts.push((format!("ppt/slides/slide{n}.xml"), xml));
        parts.push((
            format!("ppt/slides/_rels/slide{n}.xml.rels"),
            relationships(&[("rIdLayout", "slideLayout", layout)]),
        ));
    }
    let files: Vec<(&str, &[u8])> = parts.iter().map(|(n, d)| (n.as_str(), d.as_bytes())).collect();
    stored_zip(&files)
}

#[cfg(test)]
mod tests {
    use super::super::docx::Zip;
    use super::*;

    #[test]
    fn a_deck_has_its_title_slide_and_a_slide_per_entry() {
        let slides = slides_from(
            r#"[{"title": "Goals & scope", "bullets": ["- First <point>", "  nested", "", 3]}, "Questions"]"#,
        )
        .unwrap();
        assert_eq!(
            slides,
            vec![
                Slide { title: "Goals & scope".into(), bullets: vec!["- First <point>".into(), "  nested".into(), "".into(), "3".into()] },
                Slide { title: "Questions".into(), bullets: vec![] },
            ]
        );
        let bytes = create("Forum 2027", &slides);
        let zip = Zip::open(&bytes).unwrap();
        let part = |name: &str| String::from_utf8(zip.read(name).unwrap().unwrap()).unwrap();
        assert!(part("ppt/slides/slide1.xml").contains("<a:t>Forum 2027</a:t>"));
        let second = part("ppt/slides/slide2.xml");
        assert!(second.contains("<a:t>Goals &amp; scope</a:t>"));
        assert!(second.contains("<a:p><a:r><a:rPr lang=\"en-US\" dirty=\"0\"/><a:t>First &lt;point&gt;</a:t></a:r></a:p><a:p><a:pPr lvl=\"1\"/>"));
        assert!(second.contains("<a:t>3</a:t>"));
        // A slide without bullets has only its title.
        assert!(!part("ppt/slides/slide3.xml").contains("Content Placeholder"));
        assert!(zip.read("ppt/slides/slide4.xml").unwrap().is_none());
        assert_eq!(part("ppt/presentation.xml").matches("<p:sldId ").count(), 3);
        assert!(part("[Content_Types].xml").contains("/ppt/slides/slide3.xml"));
        assert!(part("ppt/slides/_rels/slide1.xml.rels").contains("slideLayout1.xml"));
        assert!(part("ppt/slides/_rels/slide3.xml.rels").contains("slideLayout2.xml"));
        assert_eq!(slides_from("{}").unwrap_err(), "slides must be a list of slides.");
        assert!(slides_from("[1]").unwrap_err().starts_with("Each slide needs to be an object"));
        assert_eq!(slides_from(r#""[\"Only\"]""#).unwrap().len(), 1);
    }
}
