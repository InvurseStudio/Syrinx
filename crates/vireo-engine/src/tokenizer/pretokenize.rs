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
            _ => None,
        }
    }

    fn max_digits(self) -> usize {
        match self {
            Self::Llama3 => 3,
            Self::Qwen2 | Self::Qwen35 | Self::Gemma4 => 1,
        }
    }

    /// Whether combining marks count as part of words.
    fn marks_in_words(self) -> bool {
        self == Self::Qwen35
    }

    /// Appends `text`'s words to `out`, in order. Together they are exactly `text`.
    pub fn split<'a>(self, text: &'a str, out: &mut Vec<&'a str>) {
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
    fn other_scripts() {
        assert_eq!(llama("héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(llama("你好世界"), ["你好世界"]);
        // The vowel signs and virama aren't letters, so they break the run of letters,
        // and each one becomes the single character allowed before the next letters.
        assert_eq!(llama("हिन्दी"), ["ह", "िन", "्द", "ी"]);
        assert_eq!(llama("😀 ok"), ["😀", " ok"]);
    }
}
