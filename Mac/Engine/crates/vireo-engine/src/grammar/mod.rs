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
    /// A list of `items`, at most `max_items` of them.
    Array {
        items: Box<Schema>,
        max_items: Option<usize>,
    },
    /// These bytes exactly (not a JSON value: for the text around one).
    Literal(String),
    /// These one after another.
    Seq(Vec<Schema>),
    /// One of these texts, then what follows it: a tool's name, then that tool's
    /// arguments. No text may be the start of another.
    Branch(Vec<(String, Schema)>),
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
}

impl Options {
    fn new(options: Vec<(Vec<u8>, Then)>) -> Self {
        let mut o = Options {
            edges: vec![Vec::new()],
            ends: vec![None],
            then: Vec::new(),
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
    },
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
                self.choice(compiled)
            }
            Schema::Array { items, max_items } => {
                let item = self.add(items);
                let more = self.choice(vec![
                    (b", ".to_vec(), Then::NextItem),
                    (b"]".to_vec(), Then::CloseArray),
                ]);
                let last = self.choice(vec![(b"]".to_vec(), Then::CloseArray)]);
                self.push(Node::Arr {
                    item,
                    max: max_items.map_or(u16::MAX, |m| m.min(u16::MAX as usize) as u16),
                    more,
                    last,
                })
            }
            Schema::Object(props) => {
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
                                key.extend_from_slice(b", ");
                            }
                            key.extend_from_slice(json_string(&p.name).as_bytes());
                            key.extend_from_slice(b": ");
                            options.push((key, Then::Property(j as u16)));
                            if p.required {
                                break;
                            }
                        }
                        if props[next..].iter().all(|p| !p.required) {
                            options.push((b"}".to_vec(), Then::CloseObject));
                        }
                        menus.push(self.choice(options));
                    }
                }
                self.push(Node::Obj { values, menus })
            }
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
                        if b != b'{' {
                            return false;
                        }
                        *top = Frame::Obj {
                            node,
                            next,
                            first,
                            open: true,
                        };
                        let menu = menus[next as usize * 2 + usize::from(first)];
                        return self.stack.push(Frame::new(g, menu));
                    }
                    // Back from a value: what comes next.
                    let menu = menus[next as usize * 2 + usize::from(first)];
                    if !self.stack.push(Frame::new(g, menu)) {
                        return false;
                    }
                }
                Frame::Arr { node, items, open } => {
                    let Node::Arr {
                        item,
                        max,
                        more,
                        last,
                    } = g.nodes[node as usize]
                    else {
                        unreachable!()
                    };
                    if !open {
                        if b != b'[' {
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
                        // Just past "[": the end, or the first item.
                        if b == b']' {
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
