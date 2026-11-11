//! DEFLATE (RFC 1951) and its zlib wrapper (RFC 1950), for the streams inside PDFs and
//! the files inside a .docx. Decoding only, a bit at a time through canonical Huffman
//! codes: simple, and fast enough for documents.

/// The longest code, in bits.
const MAX_BITS: usize = 15;

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buffer: u32,
    count: u32,
}

impl Bits<'_> {
    fn need(&mut self, n: u32) -> Result<u32, String> {
        while self.count < n {
            let Some(&byte) = self.data.get(self.pos) else {
                return Err("the compressed data ends early".into());
            };
            self.buffer |= u32::from(byte) << self.count;
            self.pos += 1;
            self.count += 8;
        }
        let value = self.buffer & ((1u32 << n) - 1);
        self.buffer >>= n;
        self.count -= n;
        Ok(value)
    }
}

/// A canonical Huffman code: how many codes of each length, and the symbols in code order.
struct Huffman {
    counts: [u16; MAX_BITS + 1],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        let mut counts = [0u16; MAX_BITS + 1];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut offsets = [0u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offsets[l as usize] as usize] = symbol as u16;
                offsets[l as usize] += 1;
            }
        }
        Self { counts, symbols }
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=MAX_BITS {
            code |= bits.need(1)? as i32;
            let count = i32::from(self.counts[len]);
            if code - count < first {
                return self
                    .symbols
                    .get((index + code - first) as usize)
                    .copied()
                    .ok_or_else(|| "a bad Huffman code".to_string());
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("a bad Huffman code".into())
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order the code-length code's lengths come in.
const LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Raw DEFLATE data, up to `limit` bytes out.
pub fn inflate(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    inflate_into(data, limit, &mut out)?;
    Ok(out)
}

/// zlib data (a two-byte header, DEFLATE, a checksum that isn't checked).
#[cfg(test)]
pub fn zlib(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    zlib_into(data, limit, &mut out)?;
    Ok(out)
}

/// As [`zlib`], keeping what was decoded before any error in `out`: a damaged PDF stream
/// often still has most of its text.
pub fn zlib_into(data: &[u8], limit: usize, out: &mut Vec<u8>) -> Result<(), String> {
    match data {
        [cmf, flg, rest @ ..]
            if cmf & 0x0F == 8 && (u16::from(*cmf) << 8 | u16::from(*flg)) % 31 == 0 =>
        {
            if flg & 0x20 != 0 {
                return Err("zlib data with a preset dictionary".into());
            }
            inflate_into(rest, limit, out)
        }
        // Some writers leave the header out.
        _ => inflate_into(data, limit, out),
    }
}

fn inflate_into(data: &[u8], limit: usize, out: &mut Vec<u8>) -> Result<(), String> {
    let mut bits = Bits {
        data,
        pos: 0,
        buffer: 0,
        count: 0,
    };
    loop {
        let last = bits.need(1)? == 1;
        match bits.need(2)? {
            0 => {
                // Stored: the rest of this byte is dropped (fewer than 8 bits are ever held).
                bits.buffer = 0;
                bits.count = 0;
                let header = data
                    .get(bits.pos..bits.pos + 4)
                    .ok_or("the compressed data ends early")?;
                let len = u16::from_le_bytes([header[0], header[1]]);
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if len != !nlen {
                    return Err("a stored block's length doesn't check".into());
                }
                let start = bits.pos + 4;
                let block = data
                    .get(start..start + len as usize)
                    .ok_or("the compressed data ends early")?;
                if out.len() + block.len() > limit {
                    return Err(too_large(limit));
                }
                out.extend_from_slice(block);
                bits.pos = start + len as usize;
            }
            1 => {
                let mut lengths = [0u8; 288];
                lengths[..144].fill(8);
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                lengths[280..].fill(8);
                let literals = Huffman::new(&lengths);
                let distances = Huffman::new(&[5u8; 30]);
                codes(&mut bits, &literals, &distances, limit, out)?;
            }
            2 => {
                let (literals, distances) = dynamic_codes(&mut bits)?;
                codes(&mut bits, &literals, &distances, limit, out)?;
            }
            _ => return Err("a bad block type".into()),
        }
        if last {
            return Ok(());
        }
    }
}

fn too_large(limit: usize) -> String {
    format!("it unpacks to more than {} MB", limit >> 20)
}

fn dynamic_codes(bits: &mut Bits) -> Result<(Huffman, Huffman), String> {
    let literal_count = bits.need(5)? as usize + 257;
    let distance_count = bits.need(5)? as usize + 1;
    let code_count = bits.need(4)? as usize + 4;
    if literal_count > 286 || distance_count > 30 {
        return Err("too many codes".into());
    }
    let mut code_lengths = [0u8; 19];
    for &i in &LENGTH_ORDER[..code_count] {
        code_lengths[i] = bits.need(3)? as u8;
    }
    let length_code = Huffman::new(&code_lengths);
    let mut lengths = vec![0u8; literal_count + distance_count];
    let mut i = 0;
    while i < lengths.len() {
        let symbol = length_code.decode(bits)?;
        let (value, repeat) = match symbol {
            0..=15 => (symbol as u8, 1),
            16 => {
                let previous = *lengths[..i]
                    .last()
                    .ok_or("a repeat with nothing before it")?;
                (previous, 3 + bits.need(2)? as usize)
            }
            17 => (0, 3 + bits.need(3)? as usize),
            _ => (0, 11 + bits.need(7)? as usize),
        };
        if i + repeat > lengths.len() {
            return Err("too many code lengths".into());
        }
        lengths[i..i + repeat].fill(value);
        i += repeat;
    }
    if lengths[256] == 0 {
        return Err("no end-of-block code".into());
    }
    Ok((
        Huffman::new(&lengths[..literal_count]),
        Huffman::new(&lengths[literal_count..]),
    ))
}

fn codes(
    bits: &mut Bits,
    literals: &Huffman,
    distances: &Huffman,
    limit: usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    loop {
        let symbol = literals.decode(bits)? as usize;
        match symbol {
            0..=255 => {
                if out.len() >= limit {
                    return Err(too_large(limit));
                }
                out.push(symbol as u8);
            }
            256 => return Ok(()),
            _ => {
                let i = symbol - 257;
                if i >= LENGTH_BASE.len() {
                    return Err("a bad length code".into());
                }
                let length =
                    LENGTH_BASE[i] as usize + bits.need(u32::from(LENGTH_EXTRA[i]))? as usize;
                let d = distances.decode(bits)? as usize;
                if d >= DISTANCE_BASE.len() {
                    return Err("a bad distance code".into());
                }
                let distance =
                    DISTANCE_BASE[d] as usize + bits.need(u32::from(DISTANCE_EXTRA[d]))? as usize;
                if distance > out.len() {
                    return Err("a distance back past the start".into());
                }
                if out.len() + length > limit {
                    return Err(too_large(limit));
                }
                let start = out.len() - distance;
                // Byte by byte: a copy can overlap what it writes.
                for k in 0..length {
                    let byte = out[start + k];
                    out.push(byte);
                }
            }
        }
    }
}

/// CRC-32 (the ZIP and PNG one), to check a .docx's files.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fixed_and_stored_blocks() {
        // zlib.compress(b"hello hello hello hello"): fixed codes with a back-reference.
        let fixed = [
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03,
            0x08, 0xb1,
        ];
        assert_eq!(zlib(&fixed, 1 << 20).unwrap(), b"hello hello hello hello");
        // A stored block: "abc".
        let stored = [0x01, 0x03, 0x00, 0xfc, 0xff, b'a', b'b', b'c'];
        assert_eq!(inflate(&stored, 1 << 20).unwrap(), b"abc");
    }

    #[test]
    fn stops_at_the_limit_and_on_damage() {
        let fixed = [
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03,
            0x08, 0xb1,
        ];
        assert!(zlib(&fixed, 10).unwrap_err().contains("more than"));
        assert!(zlib(&fixed[..6], 1 << 20).is_err());
        // What came before the damage is kept.
        let mut out = Vec::new();
        assert!(zlib_into(&fixed[..8], 1 << 20, &mut out).is_err());
        assert!(out.starts_with(b"hell"), "{out:?}");
    }

    #[test]
    fn checks_like_zip() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
