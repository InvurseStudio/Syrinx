//! Grammars for constrained generation: what a tool call (or any structured output) may
//! say, enforced on the bytes the model writes.
//!
//! A [`Schema`] (a JSON value's shape: objects with fixed keys, strings, numbers, choices
//! among given strings, arrays) compiles into a [`Grammar`]. A [`Matcher`] follows one
//! generation through it a byte at a time. [`TokenTrie`] holds a vocabulary's tokens as
//! bytes, so the matcher can say which tokens may come next ([`Matcher::allowed`]) without
//! trying all of them: it walks the trie, and drops a whole branch at its first wrong
//! byte. Where the grammar leaves only one way on, [`Matcher::forced`] gives those bytes,
//! so they can be added without sampling ("write ahead").
//!
//! The JSON is written one way only: keys in the schema's order, `": "` and `", "` between
//! parts, no other spaces. That's the form models write unprompted (Python's
//! `json.dumps`), so it costs them nothing, and it leaves more text forced.

mod trie;

pub use trie::TokenTrie;

/// The shape of a value.
#[derive(Clone, Debug, PartialEq)]
pub enum Schema {
    /// An object with these keys, in this order; optional ones may be left out.
    Object(Vec<Property>),
    /// The same in other punctuation: Qwen's `<parameter=key>` lines, Gemma's `{key:value}`.
    Record {
        fields: Vec<Property>,
        style: RecordStyle,
    },
    /// Raw text, up to and including `until` (which must be ASCII and can't be in the
    /// text): a value in a format that isn't JSON. At most `max_bytes` bytes before `until`.
    Text {
        until: String,
        max_bytes: Option<usize>,
    },
    /// Any string (as JSON: quoted, escaped), of at most `max_bytes` bytes inside the quotes.
    String {
        max_bytes: Option<usize>,
    },
    /// One of these strings (as JSON strings): an enum, or values to copy (paths, links).
    OneOf(Vec<String>),
    Integer,
    Number,
    Boolean,
    Null,
    /// A list of `items`, at most `max_items` of them, as JSON writes it (`[a, b]`).
    Array {
        items: Box<Schema>,
        max_items: Option<usize>,
    },
    /// A list in other punctuation: Gemma 4's `[a,b]`, or an object with keys of its own
    /// (`{` and `}`, each item a key and its value). `open` and `close` are single ASCII
    /// bytes, and no item may start with `close`.
    List {
        items: Box<Schema>,
        max_items: Option<usize>,
        open: u8,
        separator: String,
        close: u8,
    },
    /// One of these values, told apart by their first byte (a string's quote, a number's
    /// digit, `t`/`f`/`n`): no two may start with the same byte.
    AnyOf(Vec<Schema>),
    /// These bytes exactly (not a JSON value: for the text around one).
    Literal(String),
    /// These one after another.
    Seq(Vec<Schema>),
    /// One of these texts, then what follows it: a tool's name, then that tool's
    /// arguments. No text may be the start of another. A branch is a decision worth
    /// making whole: [`Matcher::scored_options`] offers its texts to be scored.
    Branch(Vec<(String, Schema)>),
}

/// An object's punctuation: what opens and closes it, and what goes around each key.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordStyle {
    pub open: String,
    pub key_open: String,
    pub key_close: String,
    /// Between one value and the next key.
    pub separator: String,
    pub close: String,
    /// Keys written as JSON strings (in `key_open` and `key_close`'s place, quotes and all).
    pub json_keys: bool,
}

impl RecordStyle {
    /// `{"key": value, "key": value}`.
    pub fn json() -> Self {
        RecordStyle {
            open: "{".into(),
            key_open: String::new(),
            key_close: ": ".into(),
            separator: ", ".into(),
            close: "}".into(),
            json_keys: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub name: String,
    pub schema: Schema,
    pub required: bool,
}

impl Property {
    pub fn required(name: &str, schema: Schema) -> Self {
        Self {
            name: name.into(),
            schema,
            required: true,
        }
    }

    pub fn optional(name: &str, schema: Schema) -> Self {
        Self {
            name: name.into(),
            schema,
            required: false,
        }
    }
}

impl Schema {
    pub fn string() -> Self {
        Schema::String { max_bytes: None }
    }

    /// The call `{"name": "<tool>", "arguments": {...}}` for one of `tools` (a name and
    /// its arguments' schema each), the form most models write tool calls in.
    pub fn tool_call(tools: &[(String, Schema)]) -> Self {
        Schema::Seq(vec![
            Schema::Literal("{\"name\": ".into()),
            Schema::Branch(
                tools
                    .iter()
                    .map(|(name, args)| {
                        (
                            format!("{}, \"arguments\": ", json_string(name)),
                            args.clone(),
                        )
                    })
                    .collect(),
            ),
            Schema::Literal("}".into()),
        ])
    }
}

/// `text` as a JSON string, quotes included.
pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---- The compiled grammar -------------------------------------------------------------

/// What happens when a choice is made.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Then {
    Nothing,
    /// Go on with this node.
    Push(u32),
    /// In the object below: its property `j` comes next.
    Property(u16),
    /// The object below is closed.
    CloseObject,
    /// The array below gets another item.
    NextItem,
    /// The array below is closed.
    CloseArray,
}

/// A trie of byte strings, each ending in one option.
#[derive(Debug, Default)]
struct Options {
    /// Per node: its edges (byte, node), sorted.
    edges: Vec<Vec<(u8, u32)>>,
    /// Per node: the option that ends there.
    ends: Vec<Option<u16>>,
    then: Vec<Then>,
    /// A decision to score whole (a branch), not to take a token at a time.
    scored: bool,
}

impl Options {
    fn new(options: Vec<(Vec<u8>, Then)>) -> Self {
        let mut o = Options {
            edges: vec![Vec::new()],
            ends: vec![None],
            then: Vec::new(),
            scored: false,
        };
        for (k, (bytes, then)) in options.into_iter().enumerate() {
            let mut at = 0usize;
            for &b in &bytes {
                at = match o.edges[at].binary_search_by_key(&b, |e| e.0) {
                    Ok(i) => o.edges[at][i].1 as usize,
                    Err(i) => {
                        let n = o.edges.len() as u32;
                        o.edges[at].insert(i, (b, n));
                        o.edges.push(Vec::new());
                        o.ends.push(None);
                        n as usize
                    }
                };
            }
            assert!(
                o.ends[at].is_none() && o.edges[at].is_empty(),
                "two choices are the same, or one starts another: {:?}",
                String::from_utf8_lossy(&bytes)
            );
            o.ends[at] = Some(k as u16);
            o.then.push(then);
        }
        // No option may run on past another's end, either.
        for (node, end) in o.ends.iter().enumerate() {
            assert!(
                end.is_none() || o.edges[node].is_empty(),
                "one choice starts another"
            );
        }
        o
    }

    fn step(&self, at: u32, b: u8) -> Option<u32> {
        let edges = &self.edges[at as usize];
        edges
            .binary_search_by_key(&b, |e| e.0)
            .ok()
            .map(|i| edges[i].1)
    }
}

#[derive(Debug)]
enum Node {
    Choice(Options),
    Str {
        max: u32,
    },
    Text {
        until: Vec<u8>,
        /// KMP's failure function over `until`.
        fail: Vec<u8>,
        max: u32,
    },
    Num {
        integer: bool,
    },
    Obj {
        values: Vec<u32>,
        /// Per (next property, first?): the choice of what comes next.
        menus: Vec<u32>,
    },
    Arr {
        item: u32,
        max: u16,
        /// The choice after an item: another, or the end; and the end alone (at `max`).
        more: u32,
        last: u32,
        open: u8,
        close: u8,
    },
    Any(Vec<u32>),
    Seq(Vec<u32>),
}

/// A compiled [`Schema`]. Shared by every [`Matcher`] that follows it.
#[derive(Debug)]
pub struct Grammar {
    nodes: Vec<Node>,
    root: u32,
}

impl Grammar {
    pub fn new(schema: &Schema) -> Self {
        let mut g = Grammar {
            nodes: Vec::new(),
            root: 0,
        };
        g.root = g.add(schema);
        g
    }

    fn push(&mut self, node: Node) -> u32 {
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    fn choice(&mut self, options: Vec<(Vec<u8>, Then)>) -> u32 {
        self.push(Node::Choice(Options::new(options)))
    }

    fn add(&mut self, schema: &Schema) -> u32 {
        match schema {
            Schema::Literal(text) => self.choice(vec![(text.as_bytes().to_vec(), Then::Nothing)]),
            Schema::OneOf(values) => self.choice(
                values
                    .iter()
                    .map(|v| (json_string(v).into_bytes(), Then::Nothing))
                    .collect(),
            ),
            Schema::Boolean => self.choice(vec![
                (b"true".to_vec(), Then::Nothing),
                (b"false".to_vec(), Then::Nothing),
            ]),
            Schema::Null => self.choice(vec![(b"null".to_vec(), Then::Nothing)]),
            Schema::String { max_bytes } => self.push(Node::Str {
                max: max_bytes.map_or(u32::MAX, |m| m.min(u32::MAX as usize) as u32),
            }),
            Schema::Integer => self.push(Node::Num { integer: true }),
            Schema::Number => self.push(Node::Num { integer: false }),
            Schema::Seq(parts) => {
                let parts = parts.iter().map(|p| self.add(p)).collect();
                self.push(Node::Seq(parts))
            }
            Schema::Branch(options) => {
                let compiled: Vec<(Vec<u8>, Then)> = options
                    .iter()
                    .map(|(text, rest)| (text.as_bytes().to_vec(), Then::Push(self.add(rest))))
                    .collect();
                let mut options = Options::new(compiled);
                options.scored = true;
                self.push(Node::Choice(options))
            }
            Schema::Array { items, max_items } => self.list(items, *max_items, b'[', ", ", b']'),
            Schema::List {
                items,
                max_items,
                open,
                separator,
                close,
            } => self.list(items, *max_items, *open, separator, *close),
            Schema::AnyOf(options) => {
                let options = options.iter().map(|o| self.add(o)).collect();
                self.push(Node::Any(options))
            }
            Schema::Object(props) => self.record(props, &RecordStyle::json()),
            Schema::Record { fields, style } => self.record(fields, style),
            Schema::Text { until, max_bytes } => {
                let until = until.as_bytes().to_vec();
                assert!(
                    !until.is_empty() && until.is_ascii() && until.len() < 255,
                    "a text's end must be short ASCII"
                );
                let mut fail = vec![0u8; until.len()];
                let mut k = 0usize;
                for i in 1..until.len() {
                    while k > 0 && until[i] != until[k] {
                        k = fail[k - 1] as usize;
                    }
                    if until[i] == until[k] {
                        k += 1;
                    }
                    fail[i] = k as u8;
                }
                self.push(Node::Text {
                    until,
                    fail,
                    max: max_bytes.map_or(u32::MAX, |m| m.min(u32::MAX as usize) as u32),
                })
            }
        }
    }

    fn list(&mut self, items: &Schema, max_items: Option<usize>, open: u8, separator: &str, close: u8) -> u32 {
        assert!(open.is_ascii() && close.is_ascii(), "a list's brackets must be ASCII");
        let item = self.add(items);
        let more = self.choice(vec![
            (separator.as_bytes().to_vec(), Then::NextItem),
            (vec![close], Then::CloseArray),
        ]);
        let last = self.choice(vec![(vec![close], Then::CloseArray)]);
        self.push(Node::Arr {
            item,
            max: max_items.map_or(u16::MAX, |m| m.min(u16::MAX as usize) as u16),
            more,
            last,
            open,
            close,
        })
    }

    /// An object (any punctuation): its opening text, then its keys and values.
    fn record(&mut self, props: &[Property], style: &RecordStyle) -> u32 {
        let body = {
            {
                let values: Vec<u32> = props.iter().map(|p| self.add(&p.schema)).collect();
                let n = props.len();
                let mut menus = Vec::with_capacity((n + 1) * 2);
                for next in 0..=n {
                    for first in [false, true] {
                        let mut options = Vec::new();
                        // Properties from `next` up to the first required one.
                        for (j, p) in props.iter().enumerate().skip(next) {
                            let mut key = Vec::new();
                            if !first {
                                key.extend_from_slice(style.separator.as_bytes());
                            }
                            key.extend_from_slice(style.key_open.as_bytes());
                            if style.json_keys {
                                key.extend_from_slice(json_string(&p.name).as_bytes());
                            } else {
                                key.extend_from_slice(p.name.as_bytes());
                            }
                            key.extend_from_slice(style.key_close.as_bytes());
                            options.push((key, Then::Property(j as u16)));
                            if p.required {
                                break;
                            }
                        }
                        if props[next..].iter().all(|p| !p.required) {
                            options.push((style.close.as_bytes().to_vec(), Then::CloseObject));
                        }
                        menus.push(self.choice(options));
                    }
                }
                self.push(Node::Obj { values, menus })
            }
        };
        if style.open.is_empty() {
            body
        } else {
            let open = self.choice(vec![(style.open.as_bytes().to_vec(), Then::Nothing)]);
            self.push(Node::Seq(vec![open, body]))
        }
    }

    /// A matcher at the start.
    pub fn matcher(&self) -> Matcher<'_> {
        let mut m = Matcher {
            grammar: self,
            stack: Stack::default(),
        };
        m.stack.push(Frame::new(self, self.root));
        m
    }
}

// ---- Following a generation -------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Frame {
    /// In a choice's trie, at this node.
    Choice {
        node: u32,
        at: u32,
    },
    /// A string: `open` once past its opening quote; `escape` 1 after a backslash, then
    /// the hex digits of a \u escape still to come + 1; `utf8` the continuation bytes a
    /// character still needs.
    Str {
        max: u32,
        len: u32,
        open: bool,
        escape: u8,
        utf8: u8,
    },
    /// Raw text: how much of its end it has seen, its length, and the continuation
    /// bytes a character still needs.
    Text {
        node: u32,
        matched: u8,
        utf8: u8,
        len: u32,
    },
    /// A number, in the phase of JSON's number syntax it's at, and its length so far.
    Num {
        integer: bool,
        phase: u8,
        len: u8,
    },
    /// An object: `open` past its `{`; then waiting for the choice of what comes next.
    Obj {
        node: u32,
        next: u16,
        first: bool,
        open: bool,
    },
    /// An array: past its `[` (`open`), with `items` so far.
    Arr {
        node: u32,
        items: u16,
        open: bool,
    },
    /// One of several values, before its first byte picks which.
    Any {
        node: u32,
    },
    Seq {
        node: u32,
        next: u16,
    },
}

impl Frame {
    fn new(g: &Grammar, node: u32) -> Frame {
        match &g.nodes[node as usize] {
            Node::Choice(_) => Frame::Choice { node, at: 0 },
            Node::Str { max } => Frame::Str {
                max: *max,
                len: 0,
                open: false,
                escape: 0,
                utf8: 0,
            },
            Node::Text { .. } => Frame::Text {
                node,
                matched: 0,
                utf8: 0,
                len: 0,
            },
            Node::Num { integer } => Frame::Num {
                integer: *integer,
                phase: 0,
                len: 0,
            },
            Node::Obj { .. } => Frame::Obj {
                node,
                next: 0,
                first: true,
                open: false,
            },
            Node::Arr { .. } => Frame::Arr {
                node,
                items: 0,
                open: false,
            },
            Node::Any(_) => Frame::Any { node },
            Node::Seq(_) => Frame::Seq { node, next: 0 },
        }
    }
}

/// How deep values may nest (objects in arrays in objects...).
const DEPTH: usize = 32;

#[derive(Clone, Copy)]
struct Stack {
    frames: [Frame; DEPTH],
    len: usize,
}

impl Default for Stack {
    fn default() -> Self {
        Stack {
            frames: [Frame::Seq { node: 0, next: 0 }; DEPTH],
            len: 0,
        }
    }
}

impl Stack {
    fn push(&mut self, f: Frame) -> bool {
        if self.len == DEPTH {
            return false;
        }
        self.frames[self.len] = f;
        self.len += 1;
        true
    }

    fn pop(&mut self) {
        self.len -= 1;
    }

    fn top(&mut self) -> Option<&mut Frame> {
        self.len.checked_sub(1).map(|i| &mut self.frames[i])
    }
}

/// Where one generation is in a [`Grammar`]. Cheap to copy, so trying a token is a copy
/// and a few steps.
#[derive(Clone, Copy)]
pub struct Matcher<'g> {
    grammar: &'g Grammar,
    stack: Stack,
}

impl<'g> Matcher<'g> {
    /// Whether the whole value has been written.
    pub fn is_complete(&self) -> bool {
        let mut m = *self;
        m.settle_end();
        m.stack.len == 0
    }

    /// Takes the next byte, if the grammar allows it (else leaves the matcher as it was,
    /// and says no).
    pub fn advance(&mut self, b: u8) -> bool {
        let mut m = *self;
        if m.step(b) {
            *self = m;
            true
        } else {
            false
        }
    }

    /// Takes these bytes, if the grammar allows all of them (else changes nothing).
    pub fn advance_all(&mut self, bytes: &[u8]) -> bool {
        let mut m = *self;
        for &b in bytes {
            if !m.step(b) {
                return false;
            }
        }
        *self = m;
        true
    }

    /// Whether these bytes may come next.
    pub fn allows(&self, bytes: &[u8]) -> bool {
        let mut m = *self;
        bytes.iter().all(|&b| m.step(b))
    }

    /// The bytes that must come next, as long as there's only one way on: text that can be
    /// added without asking the model.
    pub fn forced(&self) -> Vec<u8> {
        let mut m = *self;
        let mut out = Vec::new();
        while !m.is_complete() {
            let mut only = None;
            for b in 0..=255u8 {
                let mut t = m;
                if t.step(b) {
                    if only.is_some() {
                        return out;
                    }
                    only = Some((b, t));
                }
            }
            match only {
                Some((b, t)) => {
                    out.push(b);
                    m = t;
                }
                None => break,
            }
        }
        out
    }

    /// When the next part is a branch (a tool's name), the rest of each of its texts:
    /// the options to score whole and choose between, rather than a token at a time.
    pub fn scored_options(&self) -> Option<Vec<Vec<u8>>> {
        // Parts not yet entered (a sequence's next) are entered here, as the next byte
        // would enter them.
        let mut m = *self;
        while let Some(Frame::Seq { node, next }) = m.stack.top().copied() {
            let Node::Seq(parts) = &self.grammar.nodes[node as usize] else {
                unreachable!()
            };
            if next as usize >= parts.len() {
                return None;
            }
            *m.stack.top().unwrap() = Frame::Seq {
                node,
                next: next + 1,
            };
            if !m.stack.push(Frame::new(self.grammar, parts[next as usize])) {
                return None;
            }
        }
        let top = m.stack.len.checked_sub(1).map(|i| m.stack.frames[i])?;
        let Frame::Choice { node, at } = top else {
            return None;
        };
        let Node::Choice(options) = &self.grammar.nodes[node as usize] else {
            unreachable!()
        };
        if !options.scored {
            return None;
        }
        let mut out = Vec::new();
        let mut stack = vec![(at, Vec::new())];
        while let Some((n, bytes)) = stack.pop() {
            if options.ends[n as usize].is_some() {
                out.push(bytes);
                continue;
            }
            for &(b, child) in options.edges[n as usize].iter().rev() {
                let mut next = bytes.clone();
                next.push(b);
                stack.push((child, next));
            }
        }
        Some(out)
    }

    /// Pops frames that are finished without more input (a number at the end, a sequence
    /// with nothing left).
    fn settle_end(&mut self) {
        loop {
            match self.stack.top() {
                Some(Frame::Num { integer, phase, .. }) if num_can_end(*integer, *phase) => {
                    self.stack.pop();
                }
                Some(Frame::Seq { node, next }) => {
                    let Node::Seq(parts) = &self.grammar.nodes[*node as usize] else {
                        unreachable!()
                    };
                    if (*next as usize) < parts.len() {
                        return;
                    }
                    self.stack.pop();
                }
                _ => return,
            }
        }
    }

    fn step(&mut self, b: u8) -> bool {
        let g = self.grammar;
        loop {
            let Some(top) = self.stack.top() else {
                return false; // complete: nothing more may come
            };
            match *top {
                Frame::Seq { node, next } => {
                    let Node::Seq(parts) = &g.nodes[node as usize] else {
                        unreachable!()
                    };
                    if (next as usize) < parts.len() {
                        *top = Frame::Seq {
                            node,
                            next: next + 1,
                        };
                        if !self.stack.push(Frame::new(g, parts[next as usize])) {
                            return false;
                        }
                    } else {
                        self.stack.pop();
                    }
                }
                Frame::Choice { node, at } => {
                    let Node::Choice(options) = &g.nodes[node as usize] else {
                        unreachable!()
                    };
                    let Some(to) = options.step(at, b) else {
                        return false;
                    };
                    match options.ends[to as usize] {
                        None => *top = Frame::Choice { node, at: to },
                        Some(k) => {
                            self.stack.pop();
                            return self.then(options.then[k as usize]);
                        }
                    }
                    return true;
                }
                Frame::Str {
                    max,
                    len,
                    open,
                    escape,
                    utf8,
                } => {
                    let mut f = (max, len, open, escape, utf8);
                    let ok = str_step(&mut f, b);
                    if ok == StrStep::Closed {
                        self.stack.pop();
                        return true;
                    }
                    if ok == StrStep::No {
                        return false;
                    }
                    *top = Frame::Str {
                        max: f.0,
                        len: f.1,
                        open: f.2,
                        escape: f.3,
                        utf8: f.4,
                    };
                    return true;
                }
                Frame::Text {
                    node,
                    matched,
                    utf8,
                    len,
                } => {
                    let Node::Text { until, fail, max } = &g.nodes[node as usize] else {
                        unreachable!()
                    };
                    let (mut matched, mut utf8) = (matched as usize, utf8);
                    if utf8 > 0 {
                        if b & 0xC0 != 0x80 {
                            return false;
                        }
                        utf8 -= 1;
                        matched = 0;
                    } else {
                        match b {
                            b'\n' | b'\t' | 0x20..=0x7E => {}
                            0xC2..=0xDF => utf8 = 1,
                            0xE0..=0xEF => utf8 = 2,
                            0xF0..=0xF4 => utf8 = 3,
                            _ => return false,
                        }
                        if utf8 > 0 {
                            matched = 0;
                        } else {
                            while matched > 0 && until[matched] != b {
                                matched = fail[matched - 1] as usize;
                            }
                            if until[matched] == b {
                                matched += 1;
                            }
                            if matched == until.len() {
                                self.stack.pop();
                                return true;
                            }
                        }
                    }
                    if len >= *max && matched == 0 {
                        return false;
                    }
                    *top = Frame::Text {
                        node,
                        matched: matched as u8,
                        utf8,
                        len: len + 1,
                    };
                    return true;
                }
                Frame::Num {
                    integer,
                    phase,
                    len,
                } => match num_step(integer, phase, b) {
                    Some(next) if len < 24 => {
                        *top = Frame::Num {
                            integer,
                            phase: next,
                            len: len + 1,
                        };
                        return true;
                    }
                    Some(_) => return false,
                    None if num_can_end(integer, phase) => {
                        // The number ended before this byte: it's the parent's.
                        self.stack.pop();
                    }
                    None => return false,
                },
                Frame::Obj {
                    node,
                    next,
                    first,
                    open,
                } => {
                    let Node::Obj { menus, .. } = &g.nodes[node as usize] else {
                        unreachable!()
                    };
                    if !open {
                        *top = Frame::Obj {
                            node,
                            next,
                            first,
                            open: true,
                        };
                        let menu = menus[next as usize * 2 + usize::from(first)];
                        if !self.stack.push(Frame::new(g, menu)) {
                            return false;
                        }
                        continue;
                    }
                    // Back from a value: what comes next.
                    let menu = menus[next as usize * 2 + usize::from(first)];
                    if !self.stack.push(Frame::new(g, menu)) {
                        return false;
                    }
                }
                Frame::Any { node } => {
                    let Node::Any(options) = &g.nodes[node as usize] else {
                        unreachable!()
                    };
                    // The first option this byte starts is the value (they start apart).
                    self.stack.pop();
                    for &option in options {
                        let mut t = *self;
                        if t.stack.push(Frame::new(g, option)) && t.step(b) {
                            *self = t;
                            return true;
                        }
                    }
                    return false;
                }
                Frame::Arr { node, items, open } => {
                    let Node::Arr {
                        item,
                        max,
                        more,
                        last,
                        open: open_byte,
                        close,
                    } = g.nodes[node as usize]
                    else {
                        unreachable!()
                    };
                    if !open {
                        if b != open_byte {
                            return false;
                        }
                        *top = Frame::Arr {
                            node,
                            items,
                            open: true,
                        };
                        return true;
                    }
                    if items == 0 {
                        // Just past the opening: the end, or the first item.
                        if b == close {
                            self.stack.pop();
                            return true;
                        }
                        if max == 0 {
                            return false;
                        }
                        *top = Frame::Arr {
                            node,
                            items: 1,
                            open,
                        };
                        if !self.stack.push(Frame::new(g, item)) {
                            return false;
                        }
                    } else {
                        let menu = if items >= max { last } else { more };
                        if !self.stack.push(Frame::new(g, menu)) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    /// After a choice: what it leads to, for the frame below it.
    fn then(&mut self, then: Then) -> bool {
        let g = self.grammar;
        match then {
            Then::Nothing => true,
            Then::Push(node) => self.stack.push(Frame::new(g, node)),
            Then::Property(j) => {
                let Some(Frame::Obj { node, .. }) = self.stack.top().copied() else {
                    unreachable!()
                };
                let Node::Obj { values, .. } = &g.nodes[node as usize] else {
                    unreachable!()
                };
                *self.stack.top().unwrap() = Frame::Obj {
                    node,
                    next: j + 1,
                    first: false,
                    open: true,
                };
                self.stack.push(Frame::new(g, values[j as usize]))
            }
            Then::CloseObject | Then::CloseArray => {
                self.stack.pop();
                true
            }
            Then::NextItem => {
                let Some(Frame::Arr { node, items, open }) = self.stack.top().copied() else {
                    unreachable!()
                };
                let Node::Arr { item, .. } = g.nodes[node as usize] else {
                    unreachable!()
                };
                *self.stack.top().unwrap() = Frame::Arr {
                    node,
                    items: items + 1,
                    open,
                };
                self.stack.push(Frame::new(g, item))
            }
        }
    }

    /// When the matcher is in raw text, past any partial character and with none of its
    /// end seen: the end's first byte, and the room left. Then any token of whole
    /// characters without that byte fits.
    fn in_plain_text(&self) -> Option<(u8, u32)> {
        match self.stack.len.checked_sub(1).map(|i| self.stack.frames[i]) {
            Some(Frame::Text {
                node,
                matched: 0,
                utf8: 0,
                len,
            }) => {
                let Node::Text { until, max, .. } = &self.grammar.nodes[node as usize] else {
                    unreachable!()
                };
                Some((until[0], max - len))
            }
            _ => None,
        }
    }

    /// Whether the matcher is inside a free string, past any escape or partial character,
    /// with room for at least `room` more bytes: then any token of plain text fits.
    fn in_plain_string(&self, room: u32) -> bool {
        matches!(
            self.stack.len.checked_sub(1).map(|i| self.stack.frames[i]),
            Some(Frame::Str { max, len, open: true, escape: 0, utf8: 0 }) if max - len >= room
        )
    }
}

#[derive(PartialEq)]
enum StrStep {
    No,
    Yes,
    Closed,
}

fn str_step(f: &mut (u32, u32, bool, u8, u8), b: u8) -> StrStep {
    let (max, len, open, escape, utf8) = f;
    if !*open {
        if b != b'"' {
            return StrStep::No;
        }
        *open = true;
        return StrStep::Yes;
    }
    if *escape == 1 {
        match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => *escape = 0,
            b'u' => *escape = 5,
            _ => return StrStep::No,
        }
    } else if *escape > 1 {
        if !b.is_ascii_hexdigit() {
            return StrStep::No;
        }
        *escape = if *escape == 2 { 0 } else { *escape - 1 };
    } else if *utf8 > 0 {
        if b & 0xC0 != 0x80 {
            return StrStep::No;
        }
        *utf8 -= 1;
    } else {
        match b {
            b'"' => return StrStep::Closed,
            b'\\' => *escape = 1,
            0x00..=0x1F | 0x7F => return StrStep::No,
            0x20..=0x7E => {}
            0xC2..=0xDF => *utf8 = 1,
            0xE0..=0xEF => *utf8 = 2,
            0xF0..=0xF4 => *utf8 = 3,
            _ => return StrStep::No,
        }
    }
    // Room for the closing quote is never needed: at `max`, only it is allowed.
    if *len >= *max {
        return StrStep::No;
    }
    *len += 1;
    StrStep::Yes
}

/// JSON's number syntax: the phase after `b`, if `b` may come in `phase`.
fn num_step(integer: bool, phase: u8, b: u8) -> Option<u8> {
    let digit = b.is_ascii_digit();
    match (phase, b) {
        (0, b'-') => Some(1),
        (0 | 1, b'0') => Some(3),
        (0 | 1, _) if digit => Some(2),
        (2, _) if digit => Some(2),
        (2 | 3, b'.') if !integer => Some(4),
        (2 | 3 | 5, b'e' | b'E') if !integer => Some(6),
        (4 | 5, _) if digit => Some(5),
        (6, b'+' | b'-') => Some(7),
        (6..=8, _) if digit => Some(8),
        _ => None,
    }
}

fn num_can_end(integer: bool, phase: u8) -> bool {
    matches!(phase, 2 | 3) || (!integer && matches!(phase, 5 | 8))
}

#[cfg(test)]
mod tests;
