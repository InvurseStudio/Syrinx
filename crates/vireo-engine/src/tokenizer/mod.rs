//! Turns text into token ids and back, using the vocabulary inside a GGUF file.
//!
//! Supported so far: byte-level BPE (`tokenizer.ggml.model` = "gpt2"), which Llama 3 and
//! Qwen 2/3/3.5 use, with their pre-tokenizers; and Gemma 4's BPE ("gemma4"), which
//! works on the text's own UTF-8 characters with spaces written as "▁", splits only at
//! newlines, and falls back to `<0xXX>` byte tokens for characters not in the vocabulary.
//!
//! Encoding, as llama.cpp does it:
//! 1. With `parse_special`, special tokens written in the text (such as
//!    `<|start_header_id|>` from a chat template) are found first and become their own
//!    ids. User-defined tokens are always found.
//! 2. The rest is split into words by the pre-tokenizer.
//! 3. Each word's bytes are written in GPT-2's byte alphabet. If the whole word is a
//!    token (and the model's tokenizer does this, as Llama 3's does), that's its id;
//!    otherwise BPE merges it, starting from one token per byte.

mod bpe;
mod byte_map;
mod pretokenize;
pub mod unicode;
mod unicode_tables;

use std::fmt;

use crate::gguf::{Gguf, Value};
use crate::hash::FastHashMap;
use bpe::{Merge, Merges};
use byte_map::ByteMap;
pub use pretokenize::PreTokenizer;

/// GGUF's token types (`tokenizer.ggml.token_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    /// Special tokens that control the model, such as `<|begin_of_text|>`: only read
    /// from text when asked, and only shown when asked.
    Control,
    /// Tokens added to the vocabulary by hand: always read from text, shown as written.
    UserDefined,
    Unused,
    /// A token for one raw byte (SentencePiece vocabularies).
    Byte,
}

impl TokenType {
    fn from_gguf(id: i64) -> Self {
        match id {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }

    fn is_special(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined | Self::Unknown)
    }
}

#[derive(Debug)]
pub enum Error {
    /// The file has no tokenizer, or is missing part of one.
    Missing(&'static str),
    /// A tokenizer this engine can't run yet.
    Unsupported(String),
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Missing(key) => write!(f, "the model file has no {key}"),
            Error::Unsupported(what) => write!(f, "{what} isn't supported yet"),
            Error::Invalid(what) => write!(f, "the model's tokenizer is damaged: {what}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Text of tokens that end a model's turn, across the model families Vireo runs. As in
/// llama.cpp, a token found by its text here becomes a control token if the file didn't
/// already mark it so: it's then read from text only with `parse_special`.
const END_OF_GENERATION_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<end_of_turn>",
    "<|end|>",
    "<eos>",            // Gemma 4
    "<turn|>",          // Gemma 4
    "<|tool_response>", // Gemma 4
];

pub struct Tokenizer {
    tokens: Vec<String>,
    types: Vec<TokenType>,
    ids: FastHashMap<String, u32>,
    merges: Merges,
    bytes: ByteMap,
    /// The token for each single byte.
    byte_tokens: [Option<u32>; 256],
    pre: PreTokenizer,
    /// Use a word's own token when it has one, before merging (Llama 3).
    ignore_merges: bool,
    /// Gemma 4's style: UTF-8 characters, spaces as "▁", `<0xXX>` byte fallback.
    metaspace: bool,
    /// The `<0xXX>` token for each byte, in `metaspace` vocabularies.
    byte_fallback: [Option<u32>; 256],
    /// Special and user-defined tokens, longest first.
    specials: Vec<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    end_of_generation: Vec<u32>,
}

impl Tokenizer {
    /// The tokenizer described by a GGUF file's `tokenizer.ggml.*` metadata.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let model = gguf
            .get_str("tokenizer.ggml.model")
            .ok_or(Error::Missing("tokenizer (tokenizer.ggml.model)"))?;
        let metaspace = match model {
            "gpt2" => false,
            "gemma4" => true,
            other => return Err(Error::Unsupported(format!("The \"{other}\" tokenizer"))),
        };
        let pre = if metaspace {
            PreTokenizer::Gemma4
        } else {
            let pre_name = gguf.get_str("tokenizer.ggml.pre").unwrap_or("default");
            PreTokenizer::from_gguf_name(pre_name)
                .ok_or_else(|| Error::Unsupported(format!("The \"{pre_name}\" pre-tokenizer")))?
        };

        let tokens: Vec<String> = string_array(gguf, "tokenizer.ggml.tokens")?
            .ok_or(Error::Missing("vocabulary (tokenizer.ggml.tokens)"))?;
        let types = match gguf
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
        {
            None => vec![TokenType::Normal; tokens.len()],
            Some(items) if items.len() == tokens.len() => items
                .iter()
                .map(|v| TokenType::from_gguf(v.as_f64().unwrap_or(1.0) as i64))
                .collect(),
            Some(items) => {
                return Err(Error::Invalid(format!(
                    "{} token types for {} tokens",
                    items.len(),
                    tokens.len()
                )));
            }
        };
        let merges = string_array(gguf, "tokenizer.ggml.merges")?
            .ok_or(Error::Missing("BPE merges (tokenizer.ggml.merges)"))?;

        let id = |key: &str| -> Result<Option<u32>> {
            match gguf.get_u64(key) {
                None => Ok(None),
                Some(id) if (id as usize) < tokens.len() => Ok(Some(id as u32)),
                Some(id) => Err(Error::Invalid(format!(
                    "{key} is {id}, past the end of the vocabulary"
                ))),
            }
        };
        let bos = id("tokenizer.ggml.bos_token_id")?;
        let eos = id("tokenizer.ggml.eos_token_id")?;
        let mut end_of_generation: Vec<u32> = [
            eos,
            id("tokenizer.ggml.eot_token_id")?,
            id("tokenizer.ggml.eom_token_id")?,
        ]
        .into_iter()
        .flatten()
        .collect();

        // llama.cpp's defaults for each pre-tokenizer, then the file's own settings.
        let (default_add_bos, ignore_merges) = match pre {
            PreTokenizer::Llama3 => (true, true),
            PreTokenizer::Qwen2 | PreTokenizer::Qwen35 => (false, false),
            PreTokenizer::Gemma4 => (true, false),
        };
        let flag =
            |key: &str, default: bool| gguf.get(key).and_then(Value::as_bool).unwrap_or(default);
        let add_bos = flag("tokenizer.ggml.add_bos_token", default_add_bos);
        let add_eos = flag("tokenizer.ggml.add_eos_token", false);

        let mut tokenizer = Self::bpe(tokens, types, &merges, pre, ignore_merges)?;
        if metaspace {
            tokenizer.metaspace = true;
            for (b, slot) in tokenizer.byte_fallback.iter_mut().enumerate() {
                *slot = tokenizer.ids.get(&format!("<0x{b:02X}>")).copied();
            }
        }
        for text in END_OF_GENERATION_TEXTS {
            if let Some(&id) = tokenizer.ids.get(*text) {
                tokenizer.types[id as usize] = TokenType::Control;
                if !end_of_generation.contains(&id) {
                    end_of_generation.push(id);
                }
            }
        }
        // Gemma 4 (llama.cpp does the same): with a tool-response token, "</s>" is
        // ordinary text and doesn't end a turn.
        if tokenizer.ids.contains_key("<|tool_response>")
            && let Some(&id) = tokenizer.ids.get("</s>")
        {
            tokenizer.types[id as usize] = TokenType::Normal;
            end_of_generation.retain(|&e| e != id);
        }
        tokenizer.refresh_specials();
        tokenizer.bos = bos;
        tokenizer.eos = eos;
        tokenizer.add_bos = add_bos && bos.is_some();
        tokenizer.add_eos = add_eos && eos.is_some();
        tokenizer.end_of_generation = end_of_generation;
        Ok(tokenizer)
    }

    /// A byte-level BPE tokenizer from its parts. `merges` are "left right" pairs, best
    /// first.
    fn bpe(
        tokens: Vec<String>,
        types: Vec<TokenType>,
        merges: &[String],
        pre: PreTokenizer,
        ignore_merges: bool,
    ) -> Result<Self> {
        if tokens.len() > u32::MAX as usize {
            return Err(Error::Invalid("the vocabulary is too large".into()));
        }
        let mut ids = FastHashMap::with_capacity_and_hasher(tokens.len(), Default::default());
        for (i, text) in tokens.iter().enumerate() {
            // A few vocabularies list a text twice; the first id wins, as in llama.cpp.
            ids.entry(text.clone()).or_insert(i as u32);
        }

        let mut merge_table = Merges::with_capacity_and_hasher(merges.len(), Default::default());
        for (rank, line) in merges.iter().enumerate() {
            // Split at the first space after the first character, as llama.cpp does.
            let split = line
                .char_indices()
                .skip(1)
                .find(|&(_, c)| c == ' ')
                .map(|(i, _)| i)
                .ok_or_else(|| Error::Invalid(format!("the merge {line:?} has no space")))?;
            let (left, right) = (&line[..split], &line[split + 1..]);
            let (Some(&l), Some(&r)) = (ids.get(left), ids.get(right)) else {
                continue;
            };
            let Some(&token) = ids.get(&format!("{left}{right}")) else {
                continue;
            };
            merge_table.entry((l, r)).or_insert(Merge {
                rank: rank as u32,
                token,
            });
        }

        let bytes = ByteMap::new();
        let mut byte_tokens = [None; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            *slot = ids
                .get(bytes.char_of(b as u8).to_string().as_str())
                .copied();
        }

        let mut tokenizer = Self {
            tokens,
            types,
            ids,
            merges: merge_table,
            bytes,
            byte_tokens,
            pre,
            ignore_merges,
            metaspace: false,
            byte_fallback: [None; 256],
            specials: Vec::new(),
            bos: None,
            eos: None,
            add_bos: false,
            add_eos: false,
            end_of_generation: Vec::new(),
        };
        tokenizer.refresh_specials();
        Ok(tokenizer)
    }

    /// The special and user-defined tokens to look for in text, longest first.
    fn refresh_specials(&mut self) {
        let (tokens, types) = (&self.tokens, &self.types);
        let mut specials: Vec<u32> = (0..tokens.len() as u32)
            .filter(|&i| types[i as usize].is_special() && !tokens[i as usize].is_empty())
            .collect();
        specials.sort_by_key(|&i| (std::cmp::Reverse(tokens[i as usize].len()), i));
        self.specials = specials;
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn pre_tokenizer(&self) -> PreTokenizer {
        self.pre
    }

    /// A token's text as the vocabulary writes it (in GPT-2's byte alphabet: "Ġworld").
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_type(&self, id: u32) -> Option<TokenType> {
        self.types.get(id as usize).copied()
    }

    /// The id of the token whose vocabulary text is exactly `text`.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    /// Whether `encode` with `add_special` starts with the BOS token.
    pub fn adds_bos(&self) -> bool {
        self.add_bos
    }

    /// Tokens that end the model's turn: generation stops at any of them.
    pub fn is_end_of_generation(&self, id: u32) -> bool {
        self.end_of_generation.contains(&id)
    }

    /// Text to token ids. `add_special` adds the BOS (and EOS) tokens the model expects
    /// around a whole prompt; `parse_special` reads special tokens written in the text,
    /// such as a chat template's. These are llama.cpp's `llama_tokenize` arguments.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_special && self.add_bos {
            out.extend(self.bos);
        }
        for fragment in self.find_specials(text, parse_special) {
            match fragment {
                Fragment::Special(id) => out.push(id),
                Fragment::Text(t) => self.encode_text(t, &mut out),
            }
        }
        if add_special && self.add_eos {
            out.extend(self.eos);
        }
        out
    }

    /// Splits text around the special tokens written in it, longest tokens first.
    fn find_specials<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Fragment<'a>> {
        let mut fragments = vec![Fragment::Text(text)];
        for &id in &self.specials {
            let ty = self.types[id as usize];
            if !parse_special && ty != TokenType::UserDefined {
                continue;
            }
            let special = self.tokens[id as usize].as_str();
            if !text.contains(special) {
                continue;
            }
            let mut split = Vec::with_capacity(fragments.len() + 2);
            for fragment in fragments {
                let Fragment::Text(mut rest) = fragment else {
                    split.push(fragment);
                    continue;
                };
                while let Some(at) = rest.find(special) {
                    if at > 0 {
                        split.push(Fragment::Text(&rest[..at]));
                    }
                    split.push(Fragment::Special(id));
                    rest = &rest[at + special.len()..];
                }
                if !rest.is_empty() {
                    split.push(Fragment::Text(rest));
                }
            }
            fragments = split;
        }
        fragments
    }

    fn encode_text(&self, text: &str, out: &mut Vec<u32>) {
        if self.metaspace {
            return self.encode_metaspace(text, out);
        }
        let mut words = Vec::new();
        self.pre.split(text, &mut words);
        let mut symbols = Vec::new();
        for word in words {
            if self.ignore_merges
                && let Some(&id) = self.ids.get(&self.bytes.encode(word))
            {
                out.push(id);
                continue;
            }
            symbols.clear();
            // Every byte-level vocabulary has a token for each byte; a byte without one
            // is dropped rather than guessed at.
            symbols.extend(word.bytes().filter_map(|b| self.byte_tokens[b as usize]));
            bpe::merge(&mut symbols, &self.merges);
            out.extend_from_slice(&symbols);
        }
    }

    /// Gemma 4: spaces become "▁"; a run of newlines that is a token is that token; the
    /// rest is merged from one symbol per character. A character with no token of its own
    /// (it can't be part of any merge) becomes its UTF-8 bytes' `<0xXX>` tokens.
    fn encode_metaspace(&self, text: &str, out: &mut Vec<u32>) {
        let escaped = text.replace(' ', "\u{2581}");
        let mut words = Vec::new();
        self.pre.split(&escaped, &mut words);
        let vocab = self.tokens.len() as u32;
        let mut symbols = Vec::new();
        for word in words {
            if word.bytes().all(|b| b == b'\n')
                && let Some(&id) = self.ids.get(word)
            {
                out.push(id);
                continue;
            }
            symbols.clear();
            let mut buf = [0u8; 4];
            // A character outside the vocabulary gets an id past its end, which no merge
            // can match: vocab + its code point.
            symbols.extend(word.chars().map(|c| {
                self.ids
                    .get(&*c.encode_utf8(&mut buf))
                    .copied()
                    .unwrap_or(vocab + c as u32)
            }));
            bpe::merge(&mut symbols, &self.merges);
            for &id in &symbols {
                if id < vocab {
                    out.push(id);
                } else {
                    let c = char::from_u32(id - vocab).expect("made from a char");
                    for &b in c.encode_utf8(&mut buf).as_bytes() {
                        out.extend(self.byte_fallback[b as usize]);
                    }
                }
            }
        }
    }

    /// Appends the bytes a token stands for. A token can be part of a character, so
    /// stream generated text through bytes and turn them into text when they're whole.
    /// Control tokens are left out unless `include_special`.
    pub fn token_bytes(&self, id: u32, include_special: bool, out: &mut Vec<u8>) {
        let Some(text) = self.tokens.get(id as usize) else {
            return;
        };
        match self.types[id as usize] {
            TokenType::Control | TokenType::Unknown => {
                if include_special {
                    out.extend_from_slice(text.as_bytes());
                }
            }
            TokenType::UserDefined => out.extend_from_slice(text.as_bytes()),
            TokenType::Unused => {}
            TokenType::Normal | TokenType::Byte if self.metaspace => {
                // "<0xXX>" is one raw byte; anything else is text with "▁" for spaces.
                let hex = text
                    .strip_prefix("<0x")
                    .and_then(|t| t.strip_suffix('>'))
                    .filter(|h| h.len() == 2)
                    .and_then(|h| u8::from_str_radix(h, 16).ok());
                match hex {
                    Some(b) => out.push(b),
                    None => out.extend_from_slice(text.replace('\u{2581}', " ").as_bytes()),
                }
            }
            TokenType::Normal | TokenType::Byte => {
                for c in text.chars() {
                    match self.bytes.byte_of(c) {
                        Some(b) => out.push(b),
                        None => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
                    }
                }
            }
        }
    }

    /// Token ids to text. Bytes that don't make whole characters become U+FFFD.
    pub fn decode(&self, ids: &[u32], include_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.token_bytes(id, include_special, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

/// A metadata array of strings; `None` if the key isn't there.
fn string_array(gguf: &Gguf, key: &'static str) -> Result<Option<Vec<String>>> {
    let Some(value) = gguf.get(key) else {
        return Ok(None);
    };
    let items = value
        .as_array()
        .ok_or_else(|| Error::Invalid(format!("{key} isn't a list")))?;
    items
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::Invalid(format!("{key} has an item that isn't text")))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny byte-level vocabulary: every byte, a few merged words, and special tokens.
    fn tiny(ignore_merges: bool) -> Tokenizer {
        let map = ByteMap::new();
        let mut tokens: Vec<String> = (0..=255u8).map(|b| map.char_of(b).to_string()).collect();
        let mut types = vec![TokenType::Normal; 256];
        let words = [
            "he", "ll", "hell", "hello", "Ġw", "or", "Ġwor", "Ġworld", "ld", "Ġworl",
        ];
        for w in words {
            tokens.push(w.into());
            types.push(TokenType::Normal);
        }
        for (text, ty) in [
            ("<|bos|>", TokenType::Control),
            ("<|eot|>", TokenType::Control),
            ("<|eot|>x", TokenType::Control),
            ("[NAME]", TokenType::UserDefined),
        ] {
            tokens.push(text.into());
            types.push(ty);
        }
        // "Ġworld" is a token, but no merges reach it: only ignore_merges finds it.
        let merges: Vec<String> = ["h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut t =
            Tokenizer::bpe(tokens, types, &merges, PreTokenizer::Llama3, ignore_merges).unwrap();
        t.bos = t.token_id("<|bos|>");
        t.add_bos = true;
        t
    }

    fn texts(t: &Tokenizer, ids: &[u32]) -> Vec<String> {
        ids.iter()
            .map(|&id| t.token_text(id).unwrap().to_string())
            .collect()
    }

    #[test]
    fn merges_words() {
        let t = tiny(false);
        let ids = t.encode("hello world", false, false);
        assert_eq!(texts(&t, &ids), ["hello", "Ġwor", "l", "d"]);
        assert_eq!(t.decode(&ids, false), "hello world");
    }

    #[test]
    fn uses_whole_word_tokens_when_told_to() {
        let t = tiny(true);
        let ids = t.encode("hello world", false, false);
        assert_eq!(texts(&t, &ids), ["hello", "Ġworld"]);
    }

    #[test]
    fn adds_bos_and_reads_special_tokens() {
        let t = tiny(true);
        let ids = t.encode("hello<|eot|>", true, true);
        assert_eq!(texts(&t, &ids), ["<|bos|>", "hello", "<|eot|>"]);
        // The longer special token wins where both fit.
        let ids = t.encode("<|eot|>x", false, true);
        assert_eq!(texts(&t, &ids), ["<|eot|>x"]);
        // Without parse_special, control tokens are just text…
        let ids = t.encode("<|eot|>", false, false);
        assert!(ids.len() > 1);
        assert_eq!(t.decode(&ids, false), "<|eot|>");
        // …but user-defined tokens are always found.
        let ids = t.encode("hi [NAME]", false, false);
        assert_eq!(ids.last(), t.token_id("[NAME]").as_ref());
    }

    #[test]
    fn decodes_special_tokens_only_when_asked() {
        let t = tiny(true);
        let ids = t.encode("hello<|eot|>", true, true);
        assert_eq!(t.decode(&ids, false), "hello");
        assert_eq!(t.decode(&ids, true), "<|bos|>hello<|eot|>");
    }

    #[test]
    fn round_trips_any_text() {
        let t = tiny(false);
        for text in [
            "",
            " ",
            "tabs\tand\nnewlines\r\n",
            "héllo wörld",
            "中文 😀",
            "\u{0}\u{7f}",
            "a  b   c",
        ] {
            assert_eq!(
                t.decode(&t.encode(text, false, false), false),
                text,
                "{text:?}"
            );
        }
    }
}
