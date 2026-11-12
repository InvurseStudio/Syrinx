//! Splits text into words before BPE merges them, the way the model's own tokenizer did
//! when it was trained.
//!
//! Llama 3's split, as a regular expression:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)   contractions
//! |[^\r\n\p{L}\p{N}]?\p{L}+      letters, with one character before them (often a space)
//! |\p{N}{1,3}                    numbers, up to 3 digits at a time
//! | ?[^\s\p{L}\p{N}]+[\r\n]*     punctuation, with a space before and newlines after
//! |\s*[\r\n]+                    whitespace ending in newlines
//! |\s+(?!\S)                     whitespace, leaving the last space for the next word
//! |\s+                           any other whitespace
//! ```
//!
//! Qwen 2 and 3 use the same expression with numbers one digit at a time. Qwen 3.5
//! ("qwen35") also counts combining marks (\p{M}) as part of words:
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! |\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Llama 4 ("llama4", also Muse Glimmer) splits words at case changes, GPT-4o style:
//!
//! ```text
//! [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//! |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//! |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! This isn't a regular expression engine: it's a direct port of llama.cpp's hand-written
//! matcher for this expression (`unicode_regex_split_custom_llama3`), step for step, so
//! the words are exactly llama.cpp's.

use super::unicode::{is_letter, is_mark, is_number, is_whitespace};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    /// Llama 3 and 3.x ("llama-bpe" in GGUF files).
    Llama3,
    /// Qwen 2 and 3 ("qwen2").
    Qwen2,
    /// Qwen 3.5 to 3.8 ("qwen35"): Qwen 2's split, with combining marks part of words.
    Qwen35,
    /// Llama 4 and Muse Glimmer ("llama4"): words split at case changes, numbers up to
    /// 3 digits, contractions kept on their word ([`split_llama4`]).
    Llama4,
    /// Gemma 4: only newlines split the text (`[^\n]+|[\n]+`); BPE merges run over
    /// everything between them.
    Gemma4,
}

impl PreTokenizer {
    /// The pre-tokenizer for a GGUF file's `tokenizer.ggml.pre`, using llama.cpp's names.
    pub fn from_gguf_name(name: &str) -> Option<Self> {
        match name {
            "llama3" | "llama-v3" | "llama-bpe" => Some(Self::Llama3),
            "qwen2" | "deepseek-r1-qwen" => Some(Self::Qwen2),
            "qwen35" => Some(Self::Qwen35),
            "llama4" => Some(Self::Llama4),
            _ => None,
        }
    }

    fn max_digits(self) -> usize {
        match self {
            Self::Llama3 | Self::Llama4 => 3,
            Self::Qwen2 | Self::Qwen35 | Self::Gemma4 => 1,
        }
    }

    /// Whether combining marks count as part of words.
    fn marks_in_words(self) -> bool {
        self == Self::Qwen35
    }

    /// Appends `text`'s words to `out`, in order. Together they are exactly `text`.
    pub fn split<'a>(self, text: &'a str, out: &mut Vec<&'a str>) {
        if self == Self::Llama4 {
            return split_llama4(text, out);
        }
        if self == Self::Gemma4 {
            // Runs of newlines, and runs of everything else.
            let mut rest = text;
            while let Some(first) = rest.chars().next() {
                let newline = first == '\n';
                let end = rest
                    .find(|c: char| (c == '\n') != newline)
                    .unwrap_or(rest.len());
                out.push(&rest[..end]);
                rest = &rest[end..];
            }
            return;
        }
        let chars: Vec<char> = text.chars().collect();
        let mut byte_at = Vec::with_capacity(chars.len() + 1);
        let mut offset = 0;
        for c in &chars {
            byte_at.push(offset);
            offset += c.len_utf8();
        }
        byte_at.push(offset);

        let n = chars.len();
        let at = |i: usize| chars.get(i).copied();
        // "Letters" are \p{L}, or [\p{L}\p{M}] for Qwen 3.5.
        let marks = self.marks_in_words();
        let is_word = |c: char| is_letter(c) || (marks && is_mark(c));
        let letter = |i: usize| at(i).is_some_and(is_word);
        let number = |i: usize| at(i).is_some_and(is_number);
        let space = |i: usize| at(i).is_some_and(is_whitespace);
        let other =
            |i: usize| at(i).is_some_and(|c| !is_whitespace(c) && !is_word(c) && !is_number(c));
        let newline = |i: usize| matches!(at(i), Some('\r' | '\n'));
        let max_digits = self.max_digits();

        let mut start = 0;
        let mut emit = |start: &mut usize, end: usize| {
            if end > *start {
                out.push(&text[byte_at[*start]..byte_at[end]]);
            }
            *start = end;
        };

        let mut pos = 0;
        while pos < n {
            let c = chars[pos];

            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if c == '\'' && pos + 1 < n {
                let next = chars[pos + 1].to_ascii_lowercase();
                if matches!(next, 's' | 't' | 'm' | 'd') {
                    pos += 2;
                    emit(&mut start, pos);
                    continue;
                }
                if pos + 2 < n {
                    let after = chars[pos + 2].to_ascii_lowercase();
                    if matches!((next, after), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        pos += 3;
                        emit(&mut start, pos);
                        continue;
                    }
                }
            }

            // [^\r\n\p{L}\p{N}]?\p{L}+
            if !(c == '\r' || c == '\n' || is_number(c)) && (is_word(c) || letter(pos + 1)) {
                pos += 1;
                while letter(pos) {
                    pos += 1;
                }
                emit(&mut start, pos);
                continue;
            }

            // \p{N}{1,3}
            if is_number(c) {
                let mut run = pos;
                while number(pos) {
                    pos += 1;
                    if pos - run >= max_digits {
                        emit(&mut start, pos);
                        run = pos;
                    }
                }
                emit(&mut start, pos);
                continue;
            }

            // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
            let first = if c == ' ' { pos + 1 } else { pos };
            if other(first) {
                pos = first;
                while other(pos) {
                    pos += 1;
                }
                while newline(pos) {
                    pos += 1;
                }
                emit(&mut start, pos);
                continue;
            }

            let mut spaces = 0;
            let mut after_last_newline = None;
            while space(pos + spaces) {
                if newline(pos + spaces) {
                    after_last_newline = Some(pos + spaces + 1);
                }
                spaces += 1;
            }

            // \s*[\r\n]+
            if let Some(end) = after_last_newline {
                pos = end;
                emit(&mut start, pos);
                continue;
            }
            // \s+(?!\S): leave the last space to start the next word
            if spaces > 1 && pos + spaces < n {
                pos += spaces - 1;
                emit(&mut start, pos);
                continue;
            }
            // \s+
            if spaces > 0 {
                pos += spaces;
                emit(&mut start, pos);
                continue;
            }

            // Nothing matched (can't happen for valid text, but never loop forever).
            pos += 1;
            emit(&mut start, pos);
        }
    }
}

/// The Llama 4 expression (see the top of the file), alternative by alternative, each
/// matched as a backtracking regex engine would: the first alternative that matches at a
/// position wins, and within it the longest greedy match.
fn split_llama4<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    use super::unicode::{is_lower_class, is_upper_class};
    let chars: Vec<char> = text.chars().collect();
    let mut byte_at = Vec::with_capacity(chars.len() + 1);
    let mut offset = 0;
    for c in &chars {
        byte_at.push(offset);
        offset += c.len_utf8();
    }
    byte_at.push(offset);
    let n = chars.len();
    let at = |i: usize| chars.get(i).copied();
    let upper = |i: usize| at(i).is_some_and(is_upper_class);
    let lower = |i: usize| at(i).is_some_and(is_lower_class);
    let space = |i: usize| at(i).is_some_and(is_whitespace);
    // [^\r\n\p{L}\p{N}]
    let prefix =
        |i: usize| at(i).is_some_and(|c| c != '\r' && c != '\n' && !is_letter(c) && !is_number(c));
    // [^\s\p{L}\p{N}]
    let other =
        |i: usize| at(i).is_some_and(|c| !is_whitespace(c) && !is_letter(c) && !is_number(c));
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)? at i: its length.
    let contraction = |i: usize| -> usize {
        if at(i) != Some('\'') {
            return 0;
        }
        let lc = |j: usize| at(j).map(|c| c.to_ascii_lowercase());
        match (lc(i + 1), lc(i + 2)) {
            (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3,
            (Some('s' | 't' | 'm' | 'd'), _) => 2,
            _ => 0,
        }
    };
    // Alternative 1 from `from`: upper* lower+ contraction?
    let first_alt = |from: usize| -> Option<usize> {
        let mut u = from;
        while upper(u) {
            u += 1;
        }
        // Backtrack the upper run until a lowercase-class character follows it.
        let mut k = u;
        loop {
            if lower(k) {
                let mut end = k + 1;
                while lower(end) {
                    end += 1;
                }
                return Some(end + contraction(end));
            }
            if k == from {
                return None;
            }
            k -= 1;
        }
    };
    // Alternative 2 from `from`: upper+ lower* contraction?
    let second_alt = |from: usize| -> Option<usize> {
        if !upper(from) {
            return None;
        }
        let mut end = from + 1;
        while upper(end) {
            end += 1;
        }
        while lower(end) {
            end += 1;
        }
        Some(end + contraction(end))
    };
    let mut pos = 0;
    while pos < n {
        let start = pos;
        let with_prefix = |alt: &dyn Fn(usize) -> Option<usize>| {
            (if prefix(start) { alt(start + 1) } else { None }).or_else(|| alt(start))
        };
        let end = with_prefix(&first_alt)
            .or_else(|| with_prefix(&second_alt))
            .or_else(|| {
                // \p{N}{1,3}
                let mut e = start;
                while e < start + 3 && at(e).is_some_and(is_number) {
                    e += 1;
                }
                (e > start).then_some(e)
            })
            .or_else(|| {
                //  ?[^\s\p{L}\p{N}]+[\r\n/]*
                let first = if at(start) == Some(' ') {
                    start + 1
                } else {
                    start
                };
                if !other(first) {
                    return None;
                }
                let mut e = first;
                while other(e) {
                    e += 1;
                }
                while matches!(at(e), Some('\r' | '\n' | '/')) {
                    e += 1;
                }
                Some(e)
            })
            .unwrap_or_else(|| {
                let mut spaces = 0;
                let mut after_last_newline = None;
                while space(start + spaces) {
                    if matches!(at(start + spaces), Some('\r' | '\n')) {
                        after_last_newline = Some(start + spaces + 1);
                    }
                    spaces += 1;
                }
                match after_last_newline {
                    // \s*[\r\n]+
                    Some(e) => e,
                    // \s+(?!\S): the last space starts the next word
                    None if spaces > 1 && start + spaces < n => start + spaces - 1,
                    // \s+
                    None if spaces > 0 => start + spaces,
                    // Nothing matched (can't happen for valid text, but never loop forever).
                    None => start + 1,
                }
            });
        out.push(&text[byte_at[start]..byte_at[end]]);
        pos = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(pre: PreTokenizer, text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        pre.split(text, &mut out);
        assert_eq!(out.concat(), text, "the words must add up to the text");
        out
    }

    fn llama(text: &str) -> Vec<&str> {
        words(PreTokenizer::Llama3, text)
    }

    #[test]
    fn words_and_spaces() {
        assert_eq!(llama("Hello world"), ["Hello", " world"]);
        assert_eq!(llama("Hello, world!"), ["Hello", ",", " world", "!"]);
        assert_eq!(llama("a  b"), ["a", " ", " b"]);
        assert_eq!(llama("a\tb"), ["a", "\tb"]);
        assert_eq!(llama("x  "), ["x", "  "]);
        assert_eq!(llama(" "), [" "]);
        assert_eq!(llama(""), Vec::<&str>::new());
    }

    #[test]
    fn contractions() {
        assert_eq!(llama("I'm here"), ["I", "'m", " here"]);
        assert_eq!(llama("they'LL we've"), ["they", "'LL", " we", "'ve"]);
        assert_eq!(llama("it'S"), ["it", "'S"]);
        assert_eq!(llama("'x"), ["'x"]);
    }

    #[test]
    fn numbers() {
        assert_eq!(llama("12345"), ["123", "45"]);
        assert_eq!(llama("$100.50"), ["$", "100", ".", "50"]);
        assert_eq!(llama("abc123"), ["abc", "123"]);
        assert_eq!(words(PreTokenizer::Qwen2, "2024"), ["2", "0", "2", "4"]);
    }

    #[test]
    fn punctuation_and_newlines() {
        assert_eq!(llama("hi!!!\n"), ["hi", "!!!\n"]);
        assert_eq!(llama(" ?!x"), [" ?!", "x"]);
        assert_eq!(llama("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(llama("a \n b"), ["a", " \n", " b"]);
        assert_eq!(llama("line\r\nnext"), ["line", "\r\n", "next"]);
        assert_eq!(llama("fn main() {}"), ["fn", " main", "()", " {}"]);
    }

    #[test]
    fn gemma4_splits_only_at_newlines() {
        let gemma = |t| words(PreTokenizer::Gemma4, t);
        assert_eq!(
            gemma("Hello▁world!▁It's\n\nnext▁line\n"),
            ["Hello▁world!▁It's", "\n\n", "next▁line", "\n"]
        );
        assert_eq!(gemma("\n"), ["\n"]);
    }

    #[test]
    fn qwen35_keeps_marks_in_words() {
        let qwen35 = |t| words(PreTokenizer::Qwen35, t);
        assert_eq!(qwen35("हिन्दी पाठ"), ["हिन्दी", " पाठ"]);
        assert_eq!(
            qwen35("cafe\u{301} 2026"),
            ["cafe\u{301}", " ", "2", "0", "2", "6"]
        );
        assert_eq!(qwen35("I'M here"), ["I", "'M", " here"]);
    }

    #[test]
    fn llama4_splits_at_case_changes() {
        let w = |t| words(PreTokenizer::Llama4, t);
        assert_eq!(
            w("HelloWorld camelCase"),
            ["Hello", "World", " camel", "Case"]
        );
        assert_eq!(w(" XMLHttpRequest"), [" XMLHttp", "Request"]);
        // Contractions stay on their word, in either case.
        assert_eq!(w("I'M HERE don't"), ["I'M", " HERE", " don't"]);
        assert_eq!(w("abc123456 7"), ["abc", "123", "456", " ", "7"]);
        assert_eq!(
            w("path/to/file.rs\n\nend"),
            ["path", "/to", "/file", ".rs", "\n\n", "end"]
        );
        assert_eq!(w("ÉCOLE école"), ["ÉCOLE", " école"]);
    }

    #[test]
    fn other_scripts() {
        assert_eq!(llama("héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(llama("你好世界"), ["你好世界"]);
        // The vowel signs and virama aren't letters, so they break the run of letters,
        // and each one becomes the single character allowed before the next letters.
        assert_eq!(llama("हिन्दी"), ["ह", "िन", "्द", "ी"]);
        assert_eq!(llama("😀 ok"), ["😀", " ok"]);
    }
}
