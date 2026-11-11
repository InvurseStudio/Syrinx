//! A vocabulary's tokens as bytes, in a trie, so a matcher can find every token that may
//! come next by walking it: a wrong byte drops every token that starts with it at once.

use super::Matcher;
use crate::Tokenizer;
use crate::tokenizer::TokenType;

pub struct TokenTrie {
    vocab: usize,
    /// Per node: its edges, `edges[edge_start[n]..edge_start[n + 1]]`, sorted by byte.
    edge_start: Vec<u32>,
    edges: Vec<(u8, u32)>,
    /// Per node: the tokens that end there, `ends[end_start[n]..end_start[n + 1]]`.
    end_start: Vec<u32>,
    ends: Vec<u32>,
    /// Tokens of plain text (printable, no quote or backslash, whole characters): any of
    /// them fits anywhere in a free string, so there they're taken without a walk.
    plain: Vec<u64>,
    /// The longest plain token, in bytes.
    longest_plain: u32,
    /// The other tokens, with their bytes, tried one by one inside a free string.
    others: Vec<(u32, Box<[u8]>)>,
    /// Every token with its bytes, for raw text's sets below.
    all: Vec<(u32, Box<[u8]>)>,
    /// Per first byte of a raw text's end: the tokens that fit anywhere in such text
    /// (whole characters, no control bytes but newline and tab, not that byte), the
    /// others' indexes in `all`, and the longest that fits. Made when first needed.
    text_sets: std::sync::Mutex<std::collections::HashMap<u8, std::sync::Arc<TextSet>>>,
}

struct TextSet {
    plain: Vec<u64>,
    others: Vec<u32>,
    longest: u32,
}

impl TokenTrie {
    /// Every token a constrained generation may use: text tokens, not control tokens
    /// (they end turns and calls, which the caller allows itself) or hand-added ones.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self::with_extra(tokenizer, &[])
    }

    /// The same, plus these special tokens as the text they're written as: ones a format
    /// uses inside a call, such as Gemma 4's `<|"|>` quote.
    pub fn with_extra(tokenizer: &Tokenizer, extra: &[u32]) -> Self {
        let vocab = tokenizer.vocab_size();
        let mut tokens: Vec<(Vec<u8>, u32)> = Vec::with_capacity(vocab);
        let mut bytes = Vec::new();
        for id in 0..vocab as u32 {
            if extra.contains(&id) {
                if let Some(text) = tokenizer.token_text(id) {
                    tokens.push((text.as_bytes().to_vec(), id));
                }
                continue;
            }
            if !matches!(
                tokenizer.token_type(id),
                Some(TokenType::Normal | TokenType::Byte)
            ) {
                continue;
            }
            bytes.clear();
            tokenizer.token_bytes(id, false, &mut bytes);
            if !bytes.is_empty() {
                tokens.push((bytes.clone(), id));
            }
        }
        Self::from_tokens(vocab, tokens)
    }

    /// From tokens' bytes directly (for tests, and vocabularies of our own).
    pub fn from_tokens(vocab: usize, mut tokens: Vec<(Vec<u8>, u32)>) -> Self {
        tokens.sort();
        let mut t = TokenTrie {
            vocab,
            edge_start: Vec::new(),
            edges: Vec::new(),
            end_start: Vec::new(),
            ends: Vec::new(),
            plain: vec![0; vocab.div_ceil(64)],
            longest_plain: 0,
            others: Vec::new(),
            all: tokens
                .iter()
                .map(|(b, id)| (*id, b.clone().into_boxed_slice()))
                .collect(),
            text_sets: Default::default(),
        };
        for (bytes, id) in &tokens {
            if is_plain(bytes) {
                t.plain[*id as usize / 64] |= 1 << (id % 64);
                t.longest_plain = t.longest_plain.max(bytes.len() as u32);
            } else {
                t.others.push((*id, bytes.clone().into_boxed_slice()));
            }
        }
        // Nodes breadth-first, so each node's edges are contiguous: a node is a range of
        // the sorted tokens sharing a prefix of length `depth`.
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((0usize, tokens.len(), 0usize));
        let mut node_edges: Vec<Vec<(u8, u32)>> = Vec::new();
        let mut node_ends: Vec<Vec<u32>> = Vec::new();
        let mut next_id = 1u32;
        while let Some((lo, hi, depth)) = queue.pop_front() {
            let mut edges = Vec::new();
            let mut ends = Vec::new();
            let mut i = lo;
            while i < hi && tokens[i].0.len() == depth {
                ends.push(tokens[i].1);
                i += 1;
            }
            while i < hi {
                let b = tokens[i].0[depth];
                let mut j = i;
                while j < hi && tokens[j].0[depth] == b {
                    j += 1;
                }
                edges.push((b, next_id));
                next_id += 1;
                queue.push_back((i, j, depth + 1));
                i = j;
            }
            node_edges.push(edges);
            node_ends.push(ends);
        }
        for (edges, ends) in node_edges.into_iter().zip(node_ends) {
            t.edge_start.push(t.edges.len() as u32);
            t.edges.extend(edges);
            t.end_start.push(t.ends.len() as u32);
            t.ends.extend(ends);
        }
        t.edge_start.push(t.edges.len() as u32);
        t.end_start.push(t.ends.len() as u32);
        t
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn text_set(&self, end: u8) -> std::sync::Arc<TextSet> {
        let mut sets = self.text_sets.lock().unwrap();
        sets.entry(end)
            .or_insert_with(|| {
                let mut set = TextSet {
                    plain: vec![0; self.vocab.div_ceil(64)],
                    others: Vec::new(),
                    longest: 0,
                };
                for (i, (id, bytes)) in self.all.iter().enumerate() {
                    let fits = !bytes.contains(&end)
                        && std::str::from_utf8(bytes).is_ok_and(|t| {
                            t.chars().all(|c| c == '\n' || c == '\t' || !c.is_control())
                        });
                    if fits {
                        set.plain[*id as usize / 64] |= 1 << (id % 64);
                        set.longest = set.longest.max(bytes.len() as u32);
                    } else {
                        set.others.push(i as u32);
                    }
                }
                std::sync::Arc::new(set)
            })
            .clone()
    }

    fn walk(&self, node: u32, m: &Matcher, out: &mut [u64]) {
        let n = node as usize;
        for &id in &self.ends[self.end_start[n] as usize..self.end_start[n + 1] as usize] {
            out[id as usize / 64] |= 1 << (id % 64);
        }
        for &(b, child) in &self.edges[self.edge_start[n] as usize..self.edge_start[n + 1] as usize]
        {
            let mut next = *m;
            if next.advance(b) {
                self.walk(child, &next, out);
            }
        }
    }
}

/// Printable ASCII but `"` and `\`, and whole UTF-8 characters.
fn is_plain(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    text.chars()
        .all(|c| !c.is_control() && c != '"' && c != '\\' && c != '\u{7f}')
}

impl Matcher<'_> {
    /// The tokens of `trie` that may come next, one bit per token id (64 a word).
    pub fn allowed(&self, trie: &TokenTrie) -> Vec<u64> {
        let mut out = vec![0u64; trie.vocab.div_ceil(64)];
        if let Some((end, room)) = self.in_plain_text() {
            let set = trie.text_set(end);
            if room >= set.longest {
                out.copy_from_slice(&set.plain);
                for &i in &set.others {
                    let (id, bytes) = &trie.all[i as usize];
                    if self.allows(bytes) {
                        out[*id as usize / 64] |= 1 << (id % 64);
                    }
                }
                return out;
            }
        }
        if self.in_plain_string(trie.longest_plain) {
            out.copy_from_slice(&trie.plain);
            for (id, bytes) in &trie.others {
                if self.allows(bytes) {
                    out[*id as usize / 64] |= 1 << (id % 64);
                }
            }
        } else {
            trie.walk(0, self, &mut out);
        }
        out
    }
}
