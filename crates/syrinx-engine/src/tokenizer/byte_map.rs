//! GPT-2's byte-to-character mapping, which byte-level BPE vocabularies are written in.
//!
//! Every byte gets a printable character: the printable Latin-1 bytes stand for
//! themselves, and the other 68 (control characters, space, and a few more) get the
//! characters from U+0100 on, in byte order. So a space is "Ġ" (U+0120) and a newline
//! "Ċ" (U+010A), which is why vocabularies are full of them.

/// Bytes that stand for themselves: '!' to '~', '¡' to '¬', '®' to 'ÿ'.
fn is_printable(b: u8) -> bool {
    matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// The 68 other bytes use U+0100 to U+0143, so every mapped character is below this.
const LAST_CHAR: usize = 0x144;

pub(super) struct ByteMap {
    to_char: [char; 256],
    /// Indexed by character; 0xFFFF for a character that doesn't stand for a byte.
    to_byte: [u16; LAST_CHAR],
}

impl ByteMap {
    pub(super) fn new() -> Self {
        let mut to_char = ['\0'; 256];
        let mut to_byte = [u16::MAX; LAST_CHAR];
        let mut next_extra = 0x100u32;
        for b in 0..=255u8 {
            let c = if is_printable(b) {
                char::from(b)
            } else {
                let c = char::from_u32(next_extra).expect("U+0100 to U+0143 are characters");
                next_extra += 1;
                c
            };
            to_char[b as usize] = c;
            to_byte[c as usize] = b.into();
        }
        Self { to_char, to_byte }
    }

    #[inline]
    pub(super) fn char_of(&self, b: u8) -> char {
        self.to_char[b as usize]
    }

    #[inline]
    pub(super) fn byte_of(&self, c: char) -> Option<u8> {
        let b = *self.to_byte.get(c as usize)?;
        u8::try_from(b).ok()
    }

    /// Text in the vocabulary's alphabet: each byte as its character.
    pub(super) fn encode(&self, text: &str) -> String {
        text.bytes().map(|b| self.char_of(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_gpt2() {
        let map = ByteMap::new();
        assert_eq!(map.char_of(b' '), 'Ġ');
        assert_eq!(map.char_of(b'\n'), 'Ċ');
        assert_eq!(map.char_of(b'\t'), 'ĉ');
        assert_eq!(map.char_of(0), 'Ā');
        assert_eq!(map.char_of(0xAD), 'Ń'); // the soft hyphen, the last extra one
        assert_eq!(map.char_of(b'A'), 'A');
        assert_eq!(map.encode("Hi there\n"), "HiĠthereĊ");
    }

    #[test]
    fn every_byte_round_trips() {
        let map = ByteMap::new();
        for b in 0..=255u8 {
            assert_eq!(map.byte_of(map.char_of(b)), Some(b), "byte {b}");
        }
        assert_eq!(map.byte_of('中'), None);
        assert_eq!(map.byte_of('\u{144}'), None);
    }
}
