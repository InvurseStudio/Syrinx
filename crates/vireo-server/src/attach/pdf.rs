//! PDF text: the pages in order, each page's content streams read for the text they show,
//! through each font's map to Unicode (its ToUnicode CMap, else its encoding).
//!
//! The objects are found by scanning the file rather than through its cross-reference
//! table, so a damaged table doesn't matter; objects packed into object streams (PDF 1.5
//! and later) are unpacked too. Encrypted PDFs, and pages with no text (scans), are
//! reported rather than read.

use std::collections::HashMap;

use super::inflate;

/// The most one stream may unpack to.
const MAX_STREAM: usize = 256 << 20;
/// Form XObjects can draw other forms: this deep at most.
const MAX_DEPTH: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub enum Obj {
    Null,
    Bool(bool),
    Num(f64),
    Str(Vec<u8>),
    Name(String),
    Array(Vec<Obj>),
    Dict(Dict),
    Ref(u32),
    /// A stream: its dictionary and its bytes as stored (still filtered).
    Stream(Dict, Vec<u8>),
}

pub type Dict = HashMap<String, Obj>;

impl Obj {
    fn dict(&self) -> Option<&Dict> {
        match self {
            Obj::Dict(d) | Obj::Stream(d, _) => Some(d),
            _ => None,
        }
    }
    fn name(&self) -> Option<&str> {
        match self {
            Obj::Name(n) => Some(n),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Obj::Num(n) => Some(*n),
            _ => None,
        }
    }
}

// ---- Lexing and parsing ------------------------------------------------------------------

fn is_white(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0c' | b'\0')
}

fn is_delimiter(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

struct Parser<'a> {
    d: &'a [u8],
    pos: usize,
}

/// One token of a content stream or object.
#[derive(Debug, PartialEq)]
enum Token {
    Obj(Obj),
    ArrayStart,
    ArrayEnd,
    DictStart,
    DictEnd,
    /// A bare word: an operator, or `R`, `obj`, `stream`...
    Word(String),
}

impl<'a> Parser<'a> {
    fn new(d: &'a [u8], pos: usize) -> Self {
        Self { d, pos }
    }

    fn skip_white(&mut self) {
        while let Some(&b) = self.d.get(self.pos) {
            if is_white(b) {
                self.pos += 1;
            } else if b == b'%' {
                while self
                    .d
                    .get(self.pos)
                    .is_some_and(|&b| b != b'\n' && b != b'\r')
                {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn token(&mut self) -> Option<Token> {
        self.skip_white();
        let &b = self.d.get(self.pos)?;
        Some(match b {
            b'[' => {
                self.pos += 1;
                Token::ArrayStart
            }
            b']' => {
                self.pos += 1;
                Token::ArrayEnd
            }
            b'<' if self.d.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                Token::DictStart
            }
            b'>' if self.d.get(self.pos + 1) == Some(&b'>') => {
                self.pos += 2;
                Token::DictEnd
            }
            b'<' => Token::Obj(Obj::Str(self.hex_string())),
            b'(' => Token::Obj(Obj::Str(self.literal_string())),
            b'/' => {
                self.pos += 1;
                let start = self.pos;
                while self
                    .d
                    .get(self.pos)
                    .is_some_and(|&b| !is_white(b) && !is_delimiter(b))
                {
                    self.pos += 1;
                }
                Token::Obj(Obj::Name(name_text(&self.d[start..self.pos])))
            }
            _ => {
                let start = self.pos;
                while self
                    .d
                    .get(self.pos)
                    .is_some_and(|&b| !is_white(b) && !is_delimiter(b))
                {
                    self.pos += 1;
                }
                if self.pos == start {
                    // A stray delimiter (`)`, `>`, `{`): step over it.
                    self.pos += 1;
                    return Some(Token::Word(String::new()));
                }
                let word = &self.d[start..self.pos];
                let text = String::from_utf8_lossy(word);
                match &*text {
                    "true" => Token::Obj(Obj::Bool(true)),
                    "false" => Token::Obj(Obj::Bool(false)),
                    "null" => Token::Obj(Obj::Null),
                    _ => match text.parse::<f64>() {
                        Ok(n)
                            if word[0].is_ascii_digit()
                                || matches!(word[0], b'-' | b'+' | b'.') =>
                        {
                            Token::Obj(Obj::Num(n))
                        }
                        _ => Token::Word(text.into_owned()),
                    },
                }
            }
        })
    }

    fn hex_string(&mut self) -> Vec<u8> {
        self.pos += 1;
        let mut out = Vec::new();
        let mut high: Option<u8> = None;
        while let Some(&b) = self.d.get(self.pos) {
            self.pos += 1;
            if b == b'>' {
                break;
            }
            let Some(v) = (b as char).to_digit(16) else {
                continue;
            };
            match high.take() {
                Some(h) => out.push(h << 4 | v as u8),
                None => high = Some(v as u8),
            }
        }
        if let Some(h) = high {
            out.push(h << 4);
        }
        out
    }

    fn literal_string(&mut self) -> Vec<u8> {
        self.pos += 1;
        let mut out = Vec::new();
        let mut depth = 1;
        while let Some(&b) = self.d.get(self.pos) {
            self.pos += 1;
            match b {
                b'(' => {
                    depth += 1;
                    out.push(b);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(b);
                }
                b'\\' => {
                    let Some(&e) = self.d.get(self.pos) else {
                        break;
                    };
                    self.pos += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'0'..=b'7' => {
                            let mut v = u32::from(e - b'0');
                            for _ in 0..2 {
                                match self.d.get(self.pos) {
                                    Some(&o @ b'0'..=b'7') => {
                                        v = v * 8 + u32::from(o - b'0');
                                        self.pos += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        // A line break after a backslash continues the string.
                        b'\r' => {
                            if self.d.get(self.pos) == Some(&b'\n') {
                                self.pos += 1;
                            }
                        }
                        b'\n' => {}
                        other => out.push(other),
                    }
                }
                _ => out.push(b),
            }
        }
        out
    }

    /// The next object, with `n g R` read as a reference. None at the end or on a word
    /// that isn't part of an object (an operator).
    fn object(&mut self) -> Option<Obj> {
        let token = self.token()?;
        self.object_from(token, 0)
    }

    fn object_from(&mut self, token: Token, depth: usize) -> Option<Obj> {
        if depth > 64 {
            return None;
        }
        match token {
            Token::Obj(Obj::Num(n)) => {
                // `n g R`?
                let save = self.pos;
                if let Some(Token::Obj(Obj::Num(_))) = self.token()
                    && let Some(Token::Word(w)) = self.token()
                    && w == "R"
                {
                    return Some(Obj::Ref(n as u32));
                }
                self.pos = save;
                Some(Obj::Num(n))
            }
            Token::Obj(o) => Some(o),
            Token::ArrayStart => {
                let mut items = Vec::new();
                loop {
                    match self.token()? {
                        Token::ArrayEnd => return Some(Obj::Array(items)),
                        t => {
                            if let Some(o) = self.object_from(t, depth + 1) {
                                items.push(o);
                            }
                        }
                    }
                }
            }
            Token::DictStart => {
                let mut dict = Dict::new();
                loop {
                    match self.token()? {
                        Token::DictEnd => return Some(Obj::Dict(dict)),
                        Token::Obj(Obj::Name(key)) => {
                            let t = self.token()?;
                            if t == Token::DictEnd {
                                return Some(Obj::Dict(dict));
                            }
                            if let Some(v) = self.object_from(t, depth + 1) {
                                dict.insert(key, v);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Token::ArrayEnd | Token::DictEnd | Token::Word(_) => None,
        }
    }
}

/// A name's text, with its `#xx` escapes undone.
fn name_text(raw: &[u8]) -> String {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'#'
            && let Some(v) = raw
                .get(i + 1..i + 3)
                .and_then(|h| std::str::from_utf8(h).ok())
                .and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(raw[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---- The file's objects ------------------------------------------------------------------

pub struct Pdf {
    objects: HashMap<u32, Obj>,
    /// The trailer's dictionary (or the cross-reference stream's), last one wins.
    trailer: Dict,
}

/// Does `d[i..]` start with `word` as a whole word?
fn word_at(d: &[u8], i: usize, word: &[u8]) -> bool {
    d.get(i..i + word.len()) == Some(word)
        && d.get(i + word.len())
            .is_none_or(|&b| is_white(b) || is_delimiter(b))
}

fn find(d: &[u8], from: usize, what: &[u8]) -> Option<usize> {
    let first = *what.first()?;
    let mut i = from;
    while i + what.len() <= d.len() {
        let at = d[i..].iter().position(|&b| b == first)? + i;
        if d.get(at..at + what.len()) == Some(what) {
            return Some(at);
        }
        i = at + 1;
    }
    None
}

/// If `obj` at `at` is preceded by `n g` (the object's number and generation), the number.
fn object_number_before(d: &[u8], at: usize) -> Option<u32> {
    let mut i = at;
    let digits = |i: &mut usize| -> Option<(usize, usize)> {
        while *i > 0 && is_white(d[*i - 1]) {
            *i -= 1;
        }
        let end = *i;
        while *i > 0 && d[*i - 1].is_ascii_digit() {
            *i -= 1;
        }
        (*i < end).then_some((*i, end))
    };
    digits(&mut i)?; // the generation
    let (start, end) = digits(&mut i)?;
    if start > 0 && !is_white(d[start - 1]) && !is_delimiter(d[start - 1]) {
        return None;
    }
    std::str::from_utf8(&d[start..end]).ok()?.parse().ok()
}

impl Pdf {
    pub fn parse(d: &[u8]) -> Result<Self, String> {
        if find(&d[..d.len().min(1024)], 0, b"%PDF-").is_none() {
            return Err("This isn't a PDF file (it doesn't start with %PDF).".into());
        }
        let mut objects = HashMap::new();
        let mut trailer = Dict::new();
        let mut pos = 0;
        while let Some(at) = find(d, pos, b"obj") {
            pos = at + 3;
            if !word_at(d, at, b"obj") {
                continue;
            }
            let Some(number) = object_number_before(d, at) else {
                continue;
            };
            let mut p = Parser::new(d, at + 3);
            let Some(obj) = p.object() else {
                continue;
            };
            let save = p.pos;
            let obj = match (obj, p.token()) {
                (Obj::Dict(dict), Some(Token::Word(w))) if w == "stream" => {
                    let (data, end) = stream_data(d, p.pos, &dict);
                    pos = end;
                    Obj::Stream(dict, data)
                }
                (obj, _) => {
                    pos = save;
                    obj
                }
            };
            if let Obj::Stream(dict, _) = &obj
                && dict.get("Type").and_then(Obj::name) == Some("XRef")
            {
                trailer.extend(dict.clone());
            }
            objects.insert(number, obj);
        }
        // Trailers (one per update; later ones win).
        let mut pos = 0;
        while let Some(at) = find(d, pos, b"trailer") {
            pos = at + 7;
            if let Some(Obj::Dict(dict)) = Parser::new(d, pos).object() {
                trailer.extend(dict);
            }
        }
        let mut pdf = Self { objects, trailer };
        pdf.unpack_object_streams();
        if pdf.objects.is_empty() {
            return Err("This PDF has no objects that could be read: it may be damaged.".into());
        }
        Ok(pdf)
    }

    /// Objects packed in object streams, added where the file has none of that number.
    fn unpack_object_streams(&mut self) {
        let mut found = Vec::new();
        for obj in self.objects.values() {
            let Obj::Stream(dict, _) = obj else { continue };
            if dict.get("Type").and_then(Obj::name) != Some("ObjStm") {
                continue;
            }
            let Some(data) = self.decode(obj) else {
                continue;
            };
            let count = dict.get("N").and_then(Obj::num).unwrap_or(0.0) as usize;
            let first = dict.get("First").and_then(Obj::num).unwrap_or(0.0) as usize;
            let mut header = Parser::new(&data, 0);
            let mut pairs = Vec::new();
            for _ in 0..count {
                match (header.token(), header.token()) {
                    (Some(Token::Obj(Obj::Num(n))), Some(Token::Obj(Obj::Num(o)))) => {
                        pairs.push((n as u32, o as usize))
                    }
                    _ => break,
                }
            }
            for (number, offset) in pairs {
                if let Some(o) = Parser::new(&data, first + offset).object() {
                    found.push((number, o));
                }
            }
        }
        for (number, obj) in found {
            self.objects.entry(number).or_insert(obj);
        }
    }

    pub fn encrypted(&self) -> bool {
        self.trailer.contains_key("Encrypt")
    }

    /// `obj`, with a reference followed (a few times at most).
    fn resolve<'o>(&'o self, mut obj: &'o Obj) -> &'o Obj {
        for _ in 0..8 {
            match obj {
                Obj::Ref(n) => obj = self.objects.get(n).unwrap_or(&Obj::Null),
                _ => return obj,
            }
        }
        &Obj::Null
    }

    fn get<'o>(&'o self, dict: &'o Dict, key: &str) -> &'o Obj {
        dict.get(key).map_or(&Obj::Null, |o| self.resolve(o))
    }

    /// A stream's bytes with its filters undone; None when one can't be (an image's).
    fn decode(&self, obj: &Obj) -> Option<Vec<u8>> {
        let Obj::Stream(dict, raw) = self.resolve(obj) else {
            return None;
        };
        let filters = match self.get(dict, "Filter") {
            Obj::Name(n) => vec![n.clone()],
            Obj::Array(a) => a
                .iter()
                .filter_map(|f| self.resolve(f).name().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        let params = match self.get(dict, "DecodeParms") {
            Obj::Dict(p) => vec![Some(p)],
            Obj::Array(a) => a.iter().map(|p| self.resolve(p).dict()).collect(),
            _ => Vec::new(),
        };
        let mut data = raw.clone();
        for (i, filter) in filters.iter().enumerate() {
            data = match filter.as_str() {
                "FlateDecode" | "Fl" => {
                    let mut out = Vec::new();
                    // A damaged stream keeps what came before the damage.
                    let result = inflate::zlib_into(&data, MAX_STREAM, &mut out);
                    if result.is_err() && out.is_empty() {
                        return None;
                    }
                    match params.get(i).copied().flatten() {
                        Some(p) => predict(self, p, out)?,
                        None => out,
                    }
                }
                "ASCIIHexDecode" | "AHx" => {
                    let mut text = data.clone();
                    text.insert(0, b'<');
                    Parser::new(&text, 0).hex_string()
                }
                "ASCII85Decode" | "A85" => ascii85(&data),
                _ => return None,
            };
        }
        Some(data)
    }

    /// The pages, in order (through the page tree, else every page object found).
    fn pages(&self) -> Vec<(&Dict, Vec<&Dict>)> {
        let mut pages = Vec::new();
        let root = self.get(&self.trailer, "Root");
        let catalog = root.dict().or_else(|| {
            self.objects
                .values()
                .filter_map(Obj::dict)
                .find(|d| d.get("Type").and_then(Obj::name) == Some("Catalog"))
        });
        if let Some(catalog) = catalog
            && let Some(tree) = self.get(catalog, "Pages").dict()
        {
            self.walk(tree, &mut Vec::new(), &mut pages, 0);
        }
        if pages.is_empty() {
            let mut numbers: Vec<_> = self
                .objects
                .iter()
                .filter(|(_, o)| {
                    o.dict()
                        .is_some_and(|d| d.get("Type").and_then(Obj::name) == Some("Page"))
                })
                .map(|(n, _)| *n)
                .collect();
            numbers.sort_unstable();
            for n in numbers {
                if let Some(d) = self.objects[&n].dict() {
                    pages.push((d, Vec::new()));
                }
            }
        }
        pages
    }

    /// A page tree node's pages, each with the nodes above it (for inherited resources).
    fn walk<'o>(
        &'o self,
        node: &'o Dict,
        above: &mut Vec<&'o Dict>,
        pages: &mut Vec<(&'o Dict, Vec<&'o Dict>)>,
        depth: usize,
    ) {
        if depth > 64 || pages.len() > 100_000 {
            return;
        }
        match self.get(node, "Kids") {
            Obj::Array(kids) => {
                above.push(node);
                for kid in kids {
                    if let Some(kid) = self.resolve(kid).dict() {
                        self.walk(kid, above, pages, depth + 1);
                    }
                }
                above.pop();
            }
            _ => pages.push((node, above.clone())),
        }
    }

    pub fn text(&self) -> String {
        let pages = self.pages();
        let many = pages.len() > 1;
        let mut out = String::new();
        let mut fonts: HashMap<usize, Font> = HashMap::new();
        for (number, (page, above)) in pages.iter().enumerate() {
            let resources = std::iter::once(*page)
                .chain(above.iter().rev().copied())
                .find_map(|d| self.get(d, "Resources").dict());
            let content = match self.get(page, "Contents") {
                Obj::Array(parts) => {
                    let mut all = Vec::new();
                    for part in parts {
                        if let Some(bytes) = self.decode(part) {
                            all.extend(bytes);
                            all.push(b'\n');
                        }
                    }
                    all
                }
                c @ Obj::Stream(..) => self.decode(c).unwrap_or_default(),
                _ => Vec::new(),
            };
            let mut page_text = String::new();
            let mut state = TextState::new();
            self.show(
                &content,
                resources,
                &mut fonts,
                &mut state,
                &mut page_text,
                0,
            );
            let page_text = super::tidy(&page_text);
            if page_text.is_empty() {
                continue;
            }
            if many {
                out.push_str(&format!("[Page {}]\n", number + 1));
            }
            out.push_str(&page_text);
            out.push_str("\n\n");
            if out.len() > super::MAX_TEXT * 4 {
                break;
            }
        }
        super::tidy(&out)
    }

    /// The text a content stream shows, into `out`.
    fn show(
        &self,
        content: &[u8],
        resources: Option<&Dict>,
        fonts: &mut HashMap<usize, Font>,
        t: &mut TextState,
        out: &mut String,
        depth: usize,
    ) {
        if depth > MAX_DEPTH {
            return;
        }
        let font_dict = resources.and_then(|r| self.get(r, "Font").dict());
        let mut operands: Vec<Obj> = Vec::new();
        let mut p = Parser::new(content, 0);
        while let Some(token) = p.token() {
            let op = match token {
                Token::Word(w) => w,
                token => {
                    if let Some(o) = p.object_from(token, 0) {
                        operands.push(o);
                    }
                    if operands.len() > 4096 {
                        operands.clear();
                    }
                    continue;
                }
            };
            let num = |i: usize| operands.get(i).and_then(Obj::num).unwrap_or(0.0);
            let matrix = || -> Matrix { std::array::from_fn(|i| num(i)) };
            match op.as_str() {
                "q" => t.saved.push(t.graphics()),
                "Q" => {
                    if let Some(g) = t.saved.pop() {
                        t.restore(g);
                    }
                }
                "cm" => t.ctm = multiply(&matrix(), &t.ctm),
                "BT" => {
                    t.tm = IDENTITY;
                    t.tlm = IDENTITY;
                }
                "Tf" => {
                    t.size = num(1);
                    let dict = operands
                        .first()
                        .and_then(Obj::name)
                        .and_then(|n| font_dict.and_then(|f| self.get(f, n).dict()));
                    // Fonts are read once, then kept by their dictionary's address.
                    t.font = dict.map(|d| {
                        let key = d as *const Dict as usize;
                        fonts.entry(key).or_insert_with(|| Font::new(self, d));
                        key
                    });
                }
                "Tc" => t.char_spacing = num(0),
                "Tw" => t.word_spacing = num(0),
                "Tz" => t.scale = num(0) / 100.0,
                "TL" => t.leading = num(0),
                "Td" => t.move_line(num(0), num(1)),
                "TD" => {
                    t.leading = -num(1);
                    t.move_line(num(0), num(1));
                }
                "Tm" => {
                    t.tm = matrix();
                    t.tlm = t.tm;
                }
                "T*" => t.move_line(0.0, -t.leading),
                "Tj" | "'" | "\"" => {
                    if op == "\"" {
                        t.word_spacing = num(0);
                        t.char_spacing = num(1);
                    }
                    if op != "Tj" {
                        t.move_line(0.0, -t.leading);
                    }
                    if let Some(Obj::Str(s)) = operands.last() {
                        t.show(s, font_of(fonts, t.font), out);
                    }
                }
                "TJ" => {
                    if let Some(Obj::Array(items)) = operands.last() {
                        for item in items {
                            match item {
                                Obj::Str(s) => t.show(s, font_of(fonts, t.font), out),
                                Obj::Num(n) => {
                                    t.tm = multiply(
                                        &translate(-n / 1000.0 * t.size * t.scale, 0.0),
                                        &t.tm,
                                    )
                                }
                                _ => {}
                            }
                        }
                    }
                }
                // Marked content: a span's /ActualText stands for the glyphs inside it.
                "BMC" => t.marked += 1,
                "BDC" => {
                    t.marked += 1;
                    let actual = match operands.get(1) {
                        Some(Obj::Dict(d)) => d.get("ActualText"),
                        _ => None,
                    };
                    if t.actual.is_none()
                        && let Some(Obj::Str(text)) = actual
                    {
                        t.actual = Some((t.marked, text_string(text), false));
                    }
                }
                "EMC" => {
                    if let Some((level, text, shown)) = t.actual.take()
                        && level != t.marked
                    {
                        t.actual = Some((level, text, shown));
                    } else if let Some((_, text, false)) = t.actual.take() {
                        out.push_str(&text);
                    }
                    t.marked = t.marked.saturating_sub(1);
                }
                "Do" => {
                    if let Some(name) = operands.first().and_then(Obj::name)
                        && let Some(xobjects) =
                            resources.and_then(|r| self.get(r, "XObject").dict())
                        && let form @ Obj::Stream(dict, _) = self.get(xobjects, name)
                        && dict.get("Subtype").and_then(Obj::name) == Some("Form")
                        && let Some(bytes) = self.decode(form)
                    {
                        let own = self.get(dict, "Resources").dict().or(resources);
                        let saved = (t.graphics(), t.tm, t.tlm);
                        if let Obj::Array(m) = self.get(dict, "Matrix")
                            && m.len() == 6
                        {
                            let m: Matrix = std::array::from_fn(|i| m[i].num().unwrap_or(0.0));
                            t.ctm = multiply(&m, &t.ctm);
                        }
                        self.show(&bytes, own, fonts, t, out, depth + 1);
                        t.restore(saved.0);
                        (t.tm, t.tlm) = (saved.1, saved.2);
                    }
                }
                // An inline image: its bytes run from ID to EI.
                "ID" => {
                    let mut i = p.pos + 1;
                    while i + 2 <= content.len() {
                        if &content[i..i + 2] == b"EI"
                            && is_white(content[i - 1])
                            && content.get(i + 2).is_none_or(|&b| is_white(b))
                        {
                            break;
                        }
                        i += 1;
                    }
                    p.pos = i + 2;
                }
                _ => {}
            }
            operands.clear();
        }
    }
}

fn font_of(fonts: &HashMap<usize, Font>, key: Option<usize>) -> Option<&Font> {
    key.and_then(|k| fonts.get(&k))
}

/// A PDF matrix `[a b c d e f]`.
type Matrix = [f64; 6];
const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn multiply(m: &Matrix, n: &Matrix) -> Matrix {
    [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]
}

fn translate(x: f64, y: f64) -> Matrix {
    [1.0, 0.0, 0.0, 1.0, x, y]
}

/// The graphics state `q` saves: the matrix and the text state (font, size, spacing,
/// scale, leading).
type Graphics = (Matrix, Option<usize>, f64, f64, f64, f64, f64);

/// Where text goes on the page, followed so that a gap between two pieces of text can be
/// told apart: a new line, a space, or a word drawn in pieces.
struct TextState {
    ctm: Matrix,
    /// What `q` saves and `Q` brings back.
    saved: Vec<Graphics>,
    tm: Matrix,
    tlm: Matrix,
    font: Option<usize>,
    size: f64,
    char_spacing: f64,
    word_spacing: f64,
    scale: f64,
    leading: f64,
    /// Where the last text ended on the page, and its height there.
    last: Option<(f64, f64, f64)>,
    /// How deep in marked content, and the /ActualText being stood in for: its level, its
    /// text, and whether that has been written yet.
    marked: usize,
    actual: Option<(usize, String, bool)>,
}

impl TextState {
    fn new() -> Self {
        Self {
            ctm: IDENTITY,
            saved: Vec::new(),
            tm: IDENTITY,
            tlm: IDENTITY,
            font: None,
            size: 1.0,
            char_spacing: 0.0,
            word_spacing: 0.0,
            scale: 1.0,
            leading: 0.0,
            last: None,
            marked: 0,
            actual: None,
        }
    }

    fn graphics(&self) -> Graphics {
        (
            self.ctm,
            self.font,
            self.size,
            self.char_spacing,
            self.word_spacing,
            self.scale,
            self.leading,
        )
    }

    fn restore(&mut self, g: Graphics) {
        (
            self.ctm,
            self.font,
            self.size,
            self.char_spacing,
            self.word_spacing,
            self.scale,
            self.leading,
        ) = g;
    }

    fn move_line(&mut self, x: f64, y: f64) {
        self.tlm = multiply(&translate(x, y), &self.tlm);
        self.tm = self.tlm;
    }

    /// Shows a string: a line break or space first if it starts away from the last text,
    /// then its characters, moving along by their widths.
    fn show(&mut self, s: &[u8], font: Option<&Font>, out: &mut String) {
        let place = multiply(&self.tm, &self.ctm);
        let (x, y) = (place[4], place[5]);
        let height = (self.size * place[2].hypot(place[3])).abs().max(0.1);
        if let Some((lx, ly, lh)) = self.last {
            let h = height.max(lh);
            if (y - ly).abs() > h * 0.5 {
                new_line(out);
            } else if (x - lx) > h * 0.15 || (lx - x) > h * 2.0 {
                space(out);
            }
        }
        let mut text = String::new();
        let mut advance = 0.0;
        let mut i = 0;
        while i < s.len() {
            let (code, len) = font.map_or((u32::from(s[i]), 1), |f| f.next_code(&s[i..]));
            i += len;
            match font {
                Some(f) => {
                    f.push_char(code, &mut text);
                    advance += f.width(code) / 1000.0 * self.size;
                }
                None => {
                    text.push(WIN_ANSI[code as usize & 0xFF]);
                    advance += 0.5 * self.size;
                }
            }
            advance += self.char_spacing;
            if len == 1 && code == 32 {
                advance += self.word_spacing;
            }
        }
        match &mut self.actual {
            Some((_, actual, shown)) => {
                if !*shown {
                    out.push_str(actual);
                    *shown = true;
                }
            }
            None => out.push_str(&text),
        }
        self.tm = multiply(&translate(advance * self.scale, 0.0), &self.tm);
        let end = multiply(&self.tm, &self.ctm);
        self.last = Some((end[4], end[5], height));
    }
}

/// A text string outside a content stream (/ActualText): UTF-16 with its mark, or
/// PDFDocEncoding (close enough to Windows-1252).
fn text_string(s: &[u8]) -> String {
    match s {
        [0xFE, 0xFF, rest @ ..] => utf16(rest),
        [0xEF, 0xBB, 0xBF, rest @ ..] => String::from_utf8_lossy(rest).into_owned(),
        _ => s
            .iter()
            .map(|&b| WIN_ANSI[b as usize])
            .filter(|&c| c != '\0')
            .collect(),
    }
}

/// A stream's data: from after `stream` to `endstream`, by its length when that checks.
fn stream_data(d: &[u8], mut start: usize, dict: &Dict) -> (Vec<u8>, usize) {
    if d.get(start) == Some(&b'\r') {
        start += 1;
    }
    if d.get(start) == Some(&b'\n') {
        start += 1;
    }
    if let Some(Obj::Num(n)) = dict.get("Length") {
        let end = start + *n as usize;
        let mut after = end;
        while d.get(after).is_some_and(|&b| is_white(b)) {
            after += 1;
        }
        if word_at(d, after, b"endstream") {
            return (d[start..end].to_vec(), after + 9);
        }
    }
    // An indirect or wrong length: up to the keyword, less the line break before it.
    let at = find(d, start, b"endstream").unwrap_or(d.len());
    let mut end = at;
    if end > start && d[end - 1] == b'\n' {
        end -= 1;
    }
    if end > start && d[end - 1] == b'\r' {
        end -= 1;
    }
    (d[start..end].to_vec(), (at + 9).min(d.len()))
}

/// PNG predictors (`/Predictor` 10 and up), which some PDF writers put on their streams.
fn predict(pdf: &Pdf, params: &Dict, data: Vec<u8>) -> Option<Vec<u8>> {
    let predictor = pdf.get(params, "Predictor").num().unwrap_or(1.0) as u32;
    if predictor < 10 {
        return (predictor == 1).then_some(data);
    }
    let colors = pdf.get(params, "Colors").num().unwrap_or(1.0).max(1.0) as usize;
    let bits = pdf
        .get(params, "BitsPerComponent")
        .num()
        .unwrap_or(8.0)
        .max(1.0) as usize;
    let columns = pdf.get(params, "Columns").num().unwrap_or(1.0).max(1.0) as usize;
    let bpp = (colors * bits).div_ceil(8).max(1);
    let row = (colors * bits * columns).div_ceil(8);
    let mut out = Vec::with_capacity(data.len());
    let mut previous = vec![0u8; row];
    for chunk in data.chunks(row + 1) {
        let (&kind, bytes) = chunk.split_first()?;
        let mut current = bytes.to_vec();
        current.resize(row, 0);
        for i in 0..row {
            let left = if i >= bpp { current[i - bpp] } else { 0 };
            let up = previous[i];
            let up_left = if i >= bpp { previous[i - bpp] } else { 0 };
            let add = match kind {
                1 => left,
                2 => up,
                3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
                4 => {
                    let p = i16::from(left) + i16::from(up) - i16::from(up_left);
                    let (pa, pb, pc) = (
                        (p - i16::from(left)).abs(),
                        (p - i16::from(up)).abs(),
                        (p - i16::from(up_left)).abs(),
                    );
                    if pa <= pb && pa <= pc {
                        left
                    } else if pb <= pc {
                        up
                    } else {
                        up_left
                    }
                }
                _ => 0,
            };
            current[i] = current[i].wrapping_add(add);
        }
        out.extend_from_slice(&current);
        previous = current;
    }
    Some(out)
}

fn ascii85(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut n = 0;
    let start = if data.starts_with(b"<~") { 2 } else { 0 };
    for &b in &data[start..] {
        match b {
            b'~' => break,
            b'z' if n == 0 => out.extend_from_slice(&[0; 4]),
            b'!'..=b'u' => {
                group[n] = b - b'!';
                n += 1;
                if n == 5 {
                    let v = group.iter().fold(0u64, |v, &g| v * 85 + u64::from(g)) as u32;
                    out.extend_from_slice(&v.to_be_bytes());
                    n = 0;
                }
            }
            _ => {}
        }
    }
    if n > 1 {
        for g in group.iter_mut().skip(n) {
            *g = 84;
        }
        let v = group.iter().fold(0u64, |v, &g| v * 85 + u64::from(g)) as u32;
        out.extend_from_slice(&v.to_be_bytes()[..n - 1]);
    }
    out
}

fn space(out: &mut String) {
    if !out.is_empty() && !out.ends_with([' ', '\n']) {
        out.push(' ');
    }
}

fn new_line(out: &mut String) {
    while out.ends_with(' ') {
        out.pop();
    }
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

// ---- Fonts -------------------------------------------------------------------------------

/// How a font's codes become text, and how wide each is.
struct Font {
    /// The codespace ranges: each one's code length in bytes, its first and last code.
    code_spaces: Vec<(usize, u32, u32)>,
    /// How many bytes a code is outside those ranges: 2 for a Type0 font, else 1.
    code_bytes: usize,
    /// From the font's ToUnicode CMap.
    unicode: HashMap<u32, String>,
    /// A simple font's encoding, for a font without a CMap.
    encoding: Option<[char; 256]>,
    /// Glyph widths in thousandths of the font size, and the width of any other code.
    widths: HashMap<u32, f64>,
    default_width: f64,
}

impl Font {
    fn new(pdf: &Pdf, dict: &Dict) -> Self {
        let type0 = dict.get("Subtype").and_then(Obj::name) == Some("Type0");
        let mut font = Font {
            code_spaces: Vec::new(),
            code_bytes: if type0 { 2 } else { 1 },
            unicode: HashMap::new(),
            encoding: None,
            widths: HashMap::new(),
            default_width: if type0 { 1000.0 } else { 500.0 },
        };
        if let Some(cmap) = pdf.decode(pdf.get(dict, "ToUnicode")) {
            font.read_cmap(&cmap);
        }
        if type0 {
            // The widths are the descendant CID font's: /DW, and /W as `c [w...]` or `c1 c2 w`.
            if let Obj::Array(descendants) = pdf.get(dict, "DescendantFonts")
                && let Some(cid) = descendants.first().and_then(|d| pdf.resolve(d).dict())
            {
                if let Some(dw) = pdf.get(cid, "DW").num() {
                    font.default_width = dw;
                }
                if let Obj::Array(w) = pdf.get(cid, "W") {
                    let mut i = 0;
                    while i + 1 < w.len() {
                        let first = pdf.resolve(&w[i]).num().unwrap_or(0.0) as u32;
                        match pdf.resolve(&w[i + 1]) {
                            Obj::Array(each) => {
                                for (k, v) in each.iter().enumerate() {
                                    if let Some(v) = pdf.resolve(v).num() {
                                        font.widths.insert(first + k as u32, v);
                                    }
                                }
                                i += 2;
                            }
                            Obj::Num(last) if i + 2 < w.len() => {
                                let v = pdf.resolve(&w[i + 2]).num().unwrap_or(0.0);
                                for c in first..=(*last as u32).min(first + 65_535) {
                                    font.widths.insert(c, v);
                                }
                                i += 3;
                            }
                            _ => break,
                        }
                    }
                }
            }
        } else {
            font.code_spaces.clear();
            font.code_bytes = 1;
            if font.unicode.is_empty() {
                font.encoding = Some(simple_encoding(pdf, dict));
            }
            let first = pdf.get(dict, "FirstChar").num().unwrap_or(0.0) as u32;
            if let Obj::Array(w) = pdf.get(dict, "Widths") {
                for (k, v) in w.iter().enumerate() {
                    if let Some(v) = pdf.resolve(v).num() {
                        font.widths.insert(first + k as u32, v);
                    }
                }
                font.default_width = pdf
                    .get(dict, "FontDescriptor")
                    .dict()
                    .and_then(|d| pdf.get(d, "MissingWidth").num())
                    .unwrap_or(0.0);
            }
        }
        font
    }

    fn width(&self, code: u32) -> f64 {
        self.widths
            .get(&code)
            .copied()
            .unwrap_or(self.default_width)
    }

    /// The text for `code`: from the CMap, else the encoding (a code the CMap leaves out
    /// is a glyph with no text, like a ligature drawn for an /ActualText).
    fn push_char(&self, code: u32, out: &mut String) {
        if let Some(text) = self.unicode.get(&code) {
            out.extend(text.chars().filter(|&ch| ch != '\0'));
        } else if let Some(encoding) = &self.encoding {
            let ch = encoding[(code & 0xFF) as usize];
            if ch != '\0' {
                out.push(ch);
            }
        }
    }

    fn read_cmap(&mut self, cmap: &[u8]) {
        let mut p = Parser::new(cmap, 0);
        let mut strings: Vec<Vec<u8>> = Vec::new();
        let mut array: Option<Vec<Vec<u8>>> = None;
        let mut mode = "";
        while let Some(token) = p.token() {
            match token {
                Token::Obj(Obj::Str(s)) => match &mut array {
                    Some(a) => a.push(s),
                    None => strings.push(s),
                },
                Token::ArrayStart => array = Some(Vec::new()),
                Token::ArrayEnd => {
                    // bfrange's `<lo> <hi> [<dst> ...]`.
                    if let (Some(dsts), [.., lo, hi]) = (array.take(), strings.as_slice()) {
                        let (lo, hi) = (code(lo), code(hi));
                        for (k, dst) in dsts.iter().enumerate() {
                            if lo + (k as u32) <= hi {
                                self.unicode.insert(lo + k as u32, utf16(dst));
                            }
                        }
                        strings.clear();
                    }
                }
                Token::Word(w) => {
                    match w.as_str() {
                        "begincodespacerange" | "beginbfchar" | "beginbfrange" => {
                            mode = match w.as_str() {
                                "begincodespacerange" => "space",
                                "beginbfchar" => "char",
                                _ => "range",
                            };
                            strings.clear();
                        }
                        "endcodespacerange" | "endbfchar" | "endbfrange" => mode = "",
                        _ => {}
                    }
                    continue;
                }
                _ => continue,
            }
            match (mode, strings.len()) {
                ("space", 2) => {
                    let width = strings[0].len().max(1);
                    self.code_spaces
                        .push((width, code(&strings[0]), code(&strings[1])));
                    strings.clear();
                }
                ("char", 2) => {
                    self.unicode.insert(code(&strings[0]), utf16(&strings[1]));
                    strings.clear();
                }
                ("range", 3) => {
                    let (lo, hi) = (code(&strings[0]), code(&strings[1]));
                    let dst = &strings[2];
                    if hi >= lo && hi - lo <= 65_535 {
                        for k in 0..=(hi - lo) {
                            // The destination's last byte counts up.
                            let mut d = dst.clone();
                            if let Some(last) = d.len().checked_sub(1) {
                                let v = u32::from(d[last]) + k;
                                d[last] = v as u8;
                                if v > 255 && last > 0 {
                                    d[last - 1] = d[last - 1].wrapping_add((v >> 8) as u8);
                                }
                            }
                            self.unicode.insert(lo + k, utf16(&d));
                        }
                    }
                    strings.clear();
                }
                ("", _) => strings.clear(),
                _ => {}
            }
        }
        if let [(w, _, _)] = self.code_spaces.as_slice() {
            self.code_bytes = *w;
        }
    }

    /// The next code in `s`, and its length.
    fn next_code(&self, s: &[u8]) -> (u32, usize) {
        for &(w, lo, hi) in &self.code_spaces {
            if s.len() >= w {
                let c = code(&s[..w]);
                if (lo..=hi).contains(&c) {
                    return (c, w);
                }
            }
        }
        let w = self.code_bytes.min(s.len()).max(1);
        (code(&s[..w]), w)
    }
}

fn code(bytes: &[u8]) -> u32 {
    bytes.iter().take(4).fold(0, |v, &b| v << 8 | u32::from(b))
}

/// UTF-16BE text, as ToUnicode maps give it.
fn utf16(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .chunks(2)
        .map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]))
        .collect();
    char::decode_utf16(units)
        .map(|r| r.unwrap_or('\u{FFFD}'))
        .collect()
}

/// A simple font's 256 characters: its base encoding with its /Differences.
fn simple_encoding(pdf: &Pdf, dict: &Dict) -> [char; 256] {
    let base = |name: Option<&str>| match name {
        Some("MacRomanEncoding") => mac_roman(),
        _ => WIN_ANSI,
    };
    match pdf.get(dict, "Encoding") {
        Obj::Name(n) => base(Some(n)),
        Obj::Dict(e) => {
            let mut table = base(pdf.get(e, "BaseEncoding").name());
            if let Obj::Array(differences) = pdf.get(e, "Differences") {
                let mut at = 0usize;
                for d in differences {
                    match d {
                        Obj::Num(n) => at = *n as usize,
                        Obj::Name(glyph) => {
                            if at < 256 {
                                table[at] = glyph_char(glyph).unwrap_or(table[at]);
                            }
                            at += 1;
                        }
                        _ => {}
                    }
                }
            }
            table
        }
        _ => base(None),
    }
}

/// A glyph name's character: `uniXXXX`, `uXXXX`, one letter, or a common name.
fn glyph_char(name: &str) -> Option<char> {
    if let Some(hex) = name.strip_prefix("uni").or_else(|| name.strip_prefix('u'))
        && (4..=6).contains(&hex.len())
        && let Ok(v) = u32::from_str_radix(hex, 16)
    {
        return char::from_u32(v);
    }
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        return Some(c);
    }
    const NAMES: &[(&str, char)] = &[
        ("space", ' '),
        ("exclam", '!'),
        ("quotedbl", '"'),
        ("numbersign", '#'),
        ("dollar", '$'),
        ("percent", '%'),
        ("ampersand", '&'),
        ("quotesingle", '\''),
        ("parenleft", '('),
        ("parenright", ')'),
        ("asterisk", '*'),
        ("plus", '+'),
        ("comma", ','),
        ("hyphen", '-'),
        ("period", '.'),
        ("slash", '/'),
        ("zero", '0'),
        ("one", '1'),
        ("two", '2'),
        ("three", '3'),
        ("four", '4'),
        ("five", '5'),
        ("six", '6'),
        ("seven", '7'),
        ("eight", '8'),
        ("nine", '9'),
        ("colon", ':'),
        ("semicolon", ';'),
        ("less", '<'),
        ("equal", '='),
        ("greater", '>'),
        ("question", '?'),
        ("at", '@'),
        ("bracketleft", '['),
        ("backslash", '\\'),
        ("bracketright", ']'),
        ("asciicircum", '^'),
        ("underscore", '_'),
        ("grave", '`'),
        ("braceleft", '{'),
        ("bar", '|'),
        ("braceright", '}'),
        ("asciitilde", '~'),
        ("quoteleft", '\u{2018}'),
        ("quoteright", '\u{2019}'),
        ("quotedblleft", '\u{201C}'),
        ("quotedblright", '\u{201D}'),
        ("endash", '\u{2013}'),
        ("emdash", '\u{2014}'),
        ("bullet", '\u{2022}'),
        ("ellipsis", '\u{2026}'),
        ("fi", '\u{FB01}'),
        ("fl", '\u{FB02}'),
        ("Euro", '\u{20AC}'),
        ("copyright", '\u{A9}'),
        ("registered", '\u{AE}'),
        ("trademark", '\u{2122}'),
        ("degree", '\u{B0}'),
        ("minus", '\u{2212}'),
        ("nbspace", '\u{A0}'),
        ("sterling", '\u{A3}'),
        ("section", '\u{A7}'),
        ("dagger", '\u{2020}'),
        ("multiply", '\u{D7}'),
        ("eacute", 'é'),
        ("egrave", 'è'),
        ("ecircumflex", 'ê'),
        ("edieresis", 'ë'),
        ("aacute", 'á'),
        ("agrave", 'à'),
        ("acircumflex", 'â'),
        ("adieresis", 'ä'),
        ("atilde", 'ã'),
        ("aring", 'å'),
        ("ccedilla", 'ç'),
        ("iacute", 'í'),
        ("igrave", 'ì'),
        ("icircumflex", 'î'),
        ("idieresis", 'ï'),
        ("ntilde", 'ñ'),
        ("oacute", 'ó'),
        ("ograve", 'ò'),
        ("ocircumflex", 'ô'),
        ("odieresis", 'ö'),
        ("otilde", 'õ'),
        ("oslash", 'ø'),
        ("uacute", 'ú'),
        ("ugrave", 'ù'),
        ("ucircumflex", 'û'),
        ("udieresis", 'ü'),
        ("germandbls", 'ß'),
        ("Eacute", 'É'),
        ("Aacute", 'Á'),
        ("Adieresis", 'Ä'),
        ("Odieresis", 'Ö'),
        ("Udieresis", 'Ü'),
        ("Ccedilla", 'Ç'),
        ("Ntilde", 'Ñ'),
    ];
    NAMES.iter().find(|(n, _)| *n == name).map(|&(_, c)| c)
}

/// Windows-1252 (PDF's WinAnsiEncoding): Latin-1 with its 0x80-0x9F block.
pub const WIN_ANSI: [char; 256] = {
    let mut t = ['\0'; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = match char::from_u32(i as u32) {
            Some(c) => c,
            None => '\0',
        };
        i += 1;
    }
    let high = [
        '\u{20AC}', '\0', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}',
        '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\0', '\u{017D}', '\0', '\0',
        '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}',
        '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\0', '\u{017E}', '\u{0178}',
    ];
    let mut k = 0;
    while k < 32 {
        t[0x80 + k] = high[k];
        k += 1;
    }
    t
};

/// Mac OS Roman (PDF's MacRomanEncoding).
fn mac_roman() -> [char; 256] {
    const HIGH: &str = "ÄÅÇÉÑÖÜáàâäãåçéèêëíìîïñóòôöõúùûü†°¢£§•¶ß®©™´¨≠ÆØ∞±≤≥¥µ∂∑∏π∫ªºΩæø\
                        ¿¡¬√ƒ≈∆«»…\u{A0}ÀÃÕŒœ–—“”‘’÷◊ÿŸ⁄€‹›ﬁﬂ‡·‚„‰ÂÊÁËÈÍÎÏÌÓÔ\u{F8FF}ÒÚÛÙıˆ˜¯˘˙˚¸˝˛ˇ";
    let mut t = WIN_ANSI;
    for (i, c) in HIGH.chars().enumerate().take(128) {
        t[0x80 + i] = c;
    }
    t
}

// ---- Entry point ---------------------------------------------------------------------------

pub fn text(data: &[u8]) -> Result<String, String> {
    let pdf = Pdf::parse(data)?;
    if pdf.encrypted() {
        return Err(
            "This PDF is encrypted, so its text can't be read. Open it in Preview, \
                    export a copy without a password, and attach that."
                .into(),
        );
    }
    let text = pdf.text();
    if text.is_empty() {
        return Err(
            "This PDF has no text in it to read: it may be a scan or pictures of \
                    pages, and Vireo's engine can't read text in pictures yet."
                .into(),
        );
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PDF from parts: each object's body (numbered from 1), the root's number.
    pub fn build(objects: &[&[u8]], root: u32) -> Vec<u8> {
        let mut out = b"%PDF-1.7\n".to_vec();
        for (i, body) in objects.iter().enumerate() {
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(body);
            out.extend_from_slice(b"\nendobj\n");
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root {root} 0 R >>\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
        let mut s = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
        s.extend_from_slice(data);
        s.extend_from_slice(b"\nendstream");
        s
    }

    #[test]
    fn reads_a_pdf_from_macos() {
        // `cupsfilter` of a text file: Quartz's output, deflated, a MacRoman font and a
        // font with a ToUnicode map for the euro sign.
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/note.pdf"
        ))
        .unwrap();
        assert_eq!(
            text(&data).unwrap(),
            "Vireo attachment test.\nThe launch code is BLUE HERON 42.\nCafé prices: €3.50 for a flat white."
        );
    }

    #[test]
    fn reads_two_byte_fonts_object_streams_and_pages_in_order() {
        // Page 2 is listed first among the objects; the tree puts it second.
        let cmap = b"/CIDInit /ProcSet findresource begin begincmap\n\
            1 begincodespacerange <0000> <FFFF> endcodespacerange\n\
            2 beginbfchar <0001> <0048> <0002> <0069> endbfchar\n\
            1 beginbfrange <0010> <0012> <0041> endbfrange\n\
            1 beginbfrange <0020> <0021> [<FB01> <00E9>] endbfrange\n\
            endcmap end";
        let content1 =
            b"BT /F1 12 Tf 72 700 Td <00010002> Tj 0 -14 Td [<0010> -300 <00110012>] TJ ET";
        let content2 =
            b"BT /F1 12 Tf 1 0 0 1 72 700 Tm <0001> Tj ET q BI /W 1 /H 1 ID \xff\x00 EI Q \
            BT /F1 12 Tf 72 680 Td <0020 0021> Tj ET";
        // Objects 5 (the font) and 6 (its descendant) live in an object stream.
        let packed_font = b"<< /Type /Font /Subtype /Type0 /Encoding /Identity-H /ToUnicode 7 0 R /DescendantFonts [6 0 R] >>";
        let packed_cid = b"<< /Type /Font /Subtype /CIDFontType2 >>";
        let header = format!("5 0 6 {} ", packed_font.len() + 1);
        let mut objstm = header.clone().into_bytes();
        objstm.extend_from_slice(packed_font);
        objstm.push(b' ');
        objstm.extend_from_slice(packed_cid);
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [4 0 R 3 0 R] /Count 2 /Resources << /Font << /F1 5 0 R >> >> >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents 9 0 R >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents [8 0 R] >>".to_vec(),
            b"<< >>".to_vec(), // placeholder: the packed one is used
            b"<< >>".to_vec(),
            stream("", cmap),
            stream("", content1),
            stream("", content2),
            stream(&format!("/Type /ObjStm /N 2 /First {}", header.len()), &objstm),
        ];
        // Leave 5 and 6 out of the file's own objects so the packed ones are found.
        let objects: Vec<&[u8]> = objects
            .iter()
            .enumerate()
            .map(|(i, o)| {
                if i == 4 || i == 5 {
                    &b"null"[..]
                } else {
                    &o[..]
                }
            })
            .collect();
        let mut pdf = Pdf::parse(&build(&objects, 1)).unwrap();
        // `null` objects stand in the file; drop them as a file without them would.
        pdf.objects.remove(&5);
        pdf.objects.remove(&6);
        pdf.unpack_object_streams();
        assert_eq!(pdf.text(), "[Page 1]\nHi\nA BC\n\n[Page 2]\nH\n\u{FB01}é");
    }

    #[test]
    fn reads_simple_fonts_with_differences_and_literal_escapes() {
        let content = br"BT /F1 10 Tf 12 TL (Caf\351 \(ok\)\040\101) Tj T* (\001) Tj ET";
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_vec(),
            stream("", content),
            b"<< /Type /Font /Subtype /Type1 /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [1 /bullet] >> >>".to_vec(),
        ];
        let objects: Vec<&[u8]> = objects.iter().map(Vec::as_slice).collect();
        assert_eq!(text(&build(&objects, 1)).unwrap(), "Café (ok) A\n\u{2022}");
    }

    #[test]
    fn reports_encrypted_scanned_and_broken_pdfs() {
        let page: Vec<&[u8]> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Contents 4 0 R >>",
            b"<< /Length 8 >>\nstream\nq 1 0 0 \nendstream",
        ];
        let scanned = build(&page, 1);
        assert!(text(&scanned).unwrap_err().contains("no text"));
        let mut encrypted = scanned.clone();
        let at = find(&encrypted, 0, b"/Root").unwrap();
        encrypted.splice(at..at, b"/Encrypt 9 0 R ".iter().copied());
        assert!(text(&encrypted).unwrap_err().contains("encrypted"));
        assert!(text(b"hello").unwrap_err().contains("isn't a PDF"));
    }

    #[test]
    fn undoes_predictors_and_ascii85() {
        assert_eq!(ascii85(b"<~87cURD]i,\"Ebo80~>"), b"Hello World!");
        let pdf = Pdf {
            objects: HashMap::new(),
            trailer: Dict::new(),
        };
        let mut params = Dict::new();
        params.insert("Predictor".into(), Obj::Num(12.0));
        params.insert("Columns".into(), Obj::Num(2.0));
        // Two rows with the "up" filter: 1 2, then +1 +1.
        let out = predict(&pdf, &params, vec![2, 1, 2, 2, 1, 1]).unwrap();
        assert_eq!(out, [1, 2, 2, 3]);
    }
}
