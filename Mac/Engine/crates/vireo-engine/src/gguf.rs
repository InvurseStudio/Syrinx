//! Reads GGUF files: the metadata (architecture, sizes, tokenizer, chat template) and
//! the table of tensors, with each tensor's data found in the memory-mapped file.
//!
//! A GGUF file, all little-endian:
//!
//! ```text
//! "GGUF"  version: u32  tensor count: u64  metadata count: u64
//! metadata: key (string), type (u32), value   … repeated
//! tensors:  name (string), dimension count (u32), dimensions (u64 each),
//!           type (u32), offset (u64, from the start of the data)   … repeated
//! padding to `general.alignment` (32 unless the metadata says otherwise)
//! tensor data
//! ```
//!
//! A string is a u64 length followed by that many bytes of UTF-8. Versions 2 and 3 are
//! read; version 1 (32-bit counts, from 2023) is refused.
//!
//! Everything is checked as it's read: lengths and counts against what's left of the
//! file (a damaged file can't make the reader allocate gigabytes), tensor sizes against
//! their type's blocks, and every tensor's data against the end of the file.

use std::collections::HashSet;
use std::fmt;
use std::ops::{Deref, Range};
use std::path::Path;

use crate::mmap::Mmap;
use crate::tensor_type::TensorType;

pub const MAGIC: &[u8; 4] = b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml's tensors have at most 4 dimensions.
pub const MAX_DIMS: usize = 4;
/// Arrays inside arrays, as deep as we follow them. Real files use one level at most.
const MAX_ARRAY_DEPTH: usize = 8;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The file doesn't start with "GGUF".
    NotGguf,
    /// A big-endian GGUF file, which Vireo doesn't read.
    BigEndian,
    UnsupportedVersion(u32),
    /// The file ended while reading `reading`, at byte `offset`.
    Truncated {
        reading: &'static str,
        offset: usize,
    },
    /// Something in the file is impossible or inconsistent.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "couldn't read the file: {e}"),
            Error::NotGguf => write!(f, "this isn't a GGUF file"),
            Error::BigEndian => write!(f, "this is a big-endian GGUF file, which isn't supported"),
            Error::UnsupportedVersion(v) => {
                write!(f, "GGUF version {v} isn't supported (versions 2 and 3 are)")
            }
            Error::Truncated { reading, offset } => {
                write!(
                    f,
                    "the file ends too early: it stops at byte {offset} while reading the {reading}"
                )
            }
            Error::Invalid(message) => write!(f, "the file is damaged: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// The type of a metadata value, with GGUF's ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub fn from_id(id: u32) -> Option<Self> {
        use ValueType::*;
        Some(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use ValueType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            Bool => "bool",
            String => "string",
            Array => "array",
            U64 => "u64",
            I64 => "i64",
            F64 => "f64",
        }
    }

    /// The fewest bytes one value of this type takes in the file.
    fn min_bytes(self) -> usize {
        use ValueType::*;
        match self {
            U8 | I8 | Bool => 1,
            U16 | I16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 | String => 8,
            Array => 12,
        }
    }
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    /// GGUF strings are UTF-8. Bytes that aren't are replaced with U+FFFD; the tokenizer
    /// will need the raw bytes if a model's vocabulary ever has them.
    String(String),
    /// The type of the items, and the items.
    Array(ValueType, Vec<Value>),
}

impl Value {
    /// Any integer that fits in a u64.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Any number, as an f64.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            Value::I8(v) => Some(v.into()),
            Value::I16(v) => Some(v.into()),
            Value::I32(v) => Some(v.into()),
            Value::I64(v) => Some(v as f64),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(_, items) => Some(items),
            _ => None,
        }
    }
}

/// Short enough for a terminal: strings longer than 60 characters and arrays longer than
/// 4 items are cut, with the full length given.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(s) => {
                let chars = s.chars().count();
                if chars > 60 {
                    let start: String = s.chars().take(60).collect();
                    write!(f, "\"{}…\" ({chars} characters)", start.escape_debug())
                } else {
                    write!(f, "\"{}\"", s.escape_debug())
                }
            }
            Value::Array(ty, items) => {
                const SHOWN: usize = 4;
                write!(f, "[")?;
                for (i, item) in items.iter().take(SHOWN).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                if items.len() > SHOWN {
                    write!(f, ", … {} {}s in all", items.len(), ty.name())?;
                }
                write!(f, "]")
            }
        }
    }
}

/// One tensor's entry in the file's table.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// ggml's order: the first dimension is the one that varies fastest in memory (a
    /// weight matrix's row length).
    pub dims: Vec<u64>,
    /// The type's GGUF id, kept even when the type isn't one we know.
    pub type_id: u32,
    /// `None` for a type this engine doesn't know: the tensor's data can't be found.
    pub kind: Option<TensorType>,
    /// Where its data starts, counted from the start of the file's data section.
    pub offset: u64,
    /// How many bytes its data takes, when its type is known.
    pub bytes: Option<u64>,
}

impl TensorInfo {
    /// How many numbers it holds.
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A GGUF file's header: everything but the tensor data itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Gguf {
    pub version: u32,
    /// In the order the file lists them.
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Where the data section starts, from the start of the file.
    pub data_offset: u64,
}

impl Gguf {
    /// Reads the header of a GGUF file whose bytes are `bytes` (the whole file, so
    /// tensor data can be checked against its end).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err(Error::NotGguf);
        }
        let mut r = Reader {
            bytes,
            pos: MAGIC.len(),
        };

        let version = r.u32("version")?;
        if matches!(version.swap_bytes(), 2 | 3) {
            return Err(Error::BigEndian);
        }
        if !matches!(version, 2 | 3) {
            return Err(Error::UnsupportedVersion(version));
        }
        // The fewest bytes a tensor entry (an empty name, no dimensions, type, offset)
        // and a metadata entry (an empty key, type, a one-byte value) can take.
        let tensor_count = r.count("tensor count", 8 + 4 + 4 + 8)?;
        let metadata_count = r.count("metadata count", 8 + 4 + 1)?;

        let mut metadata: Vec<(String, Value)> = Vec::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = r.string("metadata key")?;
            let ty = r.value_type("metadata value type")?;
            let value = r.value(ty, 0)?;
            if metadata.iter().any(|(k, _)| *k == key) {
                return Err(Error::Invalid(format!(
                    "the metadata key {key:?} appears twice"
                )));
            }
            metadata.push((key, value));
        }

        let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some((_, value)) => {
                value
                    .as_u64()
                    .filter(|a| a.is_power_of_two())
                    .ok_or_else(|| {
                        Error::Invalid(format!("general.alignment is {value}, not a power of two"))
                    })?
            }
        };

        let mut tensors = Vec::with_capacity(tensor_count);
        let mut names = HashSet::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let tensor = r.tensor_info(alignment)?;
            if !names.insert(tensor.name.clone()) {
                return Err(Error::Invalid(format!(
                    "the tensor {:?} appears twice",
                    tensor.name
                )));
            }
            tensors.push(tensor);
        }

        let data_offset = (r.pos as u64).next_multiple_of(alignment);
        for t in &tensors {
            // A tensor of a type we don't know has no size to check.
            let Some(size) = t.bytes else { continue };
            let fits = data_offset
                .checked_add(t.offset)
                .and_then(|start| start.checked_add(size))
                .is_some_and(|end| end <= bytes.len() as u64);
            if !fits {
                return Err(Error::Invalid(format!(
                    "the tensor {:?}'s data runs past the end of the file",
                    t.name
                )));
            }
        }

        Ok(Gguf {
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
        })
    }

    /// The value for a metadata key, such as "general.architecture".
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    /// "llama", "qwen3", "gemma3" and so on: the key prefix for the model's sizes
    /// (`llama.block_count`, `llama.embedding_length`, …).
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// A value under the architecture's prefix: `arch_u64("block_count")` reads
    /// `llama.block_count` in a Llama file.
    pub fn arch_u64(&self, key: &str) -> Option<u64> {
        self.get_u64(&format!("{}.{key}", self.architecture()?))
    }

    pub fn arch_f64(&self, key: &str) -> Option<f64> {
        self.get_f64(&format!("{}.{key}", self.architecture()?))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Where a tensor's data is, in the file's bytes. `None` for a type we don't know.
    pub fn tensor_range(&self, tensor: &TensorInfo) -> Option<Range<usize>> {
        let start = usize::try_from(self.data_offset + tensor.offset).ok()?;
        let len = usize::try_from(tensor.bytes?).ok()?;
        Some(start..start + len)
    }
}

/// A GGUF file opened from disk: its header, and its tensor data mapped into memory.
pub struct GgufFile {
    map: Mmap,
    header: Gguf,
}

impl GgufFile {
    /// Opens and checks the file at `path`. The file must not be changed while it's
    /// open (see `Mmap::open`): model files are only ever read.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let map = unsafe { Mmap::open(path.as_ref()) }?;
        let header = Gguf::parse(&map)?;
        Ok(Self { map, header })
    }

    /// The file's size in bytes.
    pub fn len(&self) -> u64 {
        self.map.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// A tensor's data, straight from the mapped file. `None` for a type we don't know.
    pub fn tensor_data(&self, tensor: &TensorInfo) -> Option<&[u8]> {
        self.header
            .tensor_range(tensor)
            .map(|range| &self.map[range])
    }
}

impl Deref for GgufFile {
    type Target = Gguf;

    fn deref(&self) -> &Gguf {
        &self.header
    }
}

/// Reads little-endian values from the front of a byte slice, checking every length.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: usize, reading: &'static str) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(Error::Truncated {
                reading,
                offset: self.bytes.len(),
            });
        }
        let taken = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(taken)
    }

    fn fixed<const N: usize>(&mut self, reading: &'static str) -> Result<[u8; N]> {
        Ok(self
            .take(N, reading)?
            .try_into()
            .expect("take returns N bytes"))
    }

    fn u8(&mut self, reading: &'static str) -> Result<u8> {
        Ok(self.fixed::<1>(reading)?[0])
    }

    fn u16(&mut self, reading: &'static str) -> Result<u16> {
        Ok(u16::from_le_bytes(self.fixed(reading)?))
    }

    fn u32(&mut self, reading: &'static str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed(reading)?))
    }

    fn u64(&mut self, reading: &'static str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed(reading)?))
    }

    /// A u64 count of things that each take at least `min_bytes`, checked against what's
    /// left of the file.
    fn count(&mut self, reading: &'static str, min_bytes: usize) -> Result<usize> {
        let count = self.u64(reading)?;
        match usize::try_from(count) {
            Ok(n) if n <= self.remaining() / min_bytes.max(1) => Ok(n),
            _ => Err(Error::Invalid(format!(
                "the {reading} is {count}, more than the rest of the file can hold"
            ))),
        }
    }

    fn string(&mut self, reading: &'static str) -> Result<String> {
        let len = self.u64(reading)?;
        let len = usize::try_from(len)
            .ok()
            .filter(|&n| n <= self.remaining())
            .ok_or(Error::Truncated {
                reading,
                offset: self.bytes.len(),
            })?;
        Ok(String::from_utf8_lossy(self.take(len, reading)?).into_owned())
    }

    fn value_type(&mut self, reading: &'static str) -> Result<ValueType> {
        let id = self.u32(reading)?;
        ValueType::from_id(id)
            .ok_or_else(|| Error::Invalid(format!("unknown metadata value type {id}")))
    }

    fn value(&mut self, ty: ValueType, depth: usize) -> Result<Value> {
        const READING: &str = "metadata value";
        Ok(match ty {
            ValueType::U8 => Value::U8(self.u8(READING)?),
            ValueType::I8 => Value::I8(self.u8(READING)? as i8),
            ValueType::U16 => Value::U16(self.u16(READING)?),
            ValueType::I16 => Value::I16(self.u16(READING)? as i16),
            ValueType::U32 => Value::U32(self.u32(READING)?),
            ValueType::I32 => Value::I32(self.u32(READING)? as i32),
            ValueType::U64 => Value::U64(self.u64(READING)?),
            ValueType::I64 => Value::I64(self.u64(READING)? as i64),
            ValueType::F32 => Value::F32(f32::from_bits(self.u32(READING)?)),
            ValueType::F64 => Value::F64(f64::from_bits(self.u64(READING)?)),
            ValueType::Bool => match self.u8(READING)? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => return Err(Error::Invalid(format!("a true/false value is {b}"))),
            },
            ValueType::String => Value::String(self.string(READING)?),
            ValueType::Array => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(Error::Invalid("arrays are nested too deeply".into()));
                }
                let item_type = self.value_type("array item type")?;
                let len = self.count("array length", item_type.min_bytes())?;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.value(item_type, depth + 1)?);
                }
                Value::Array(item_type, items)
            }
        })
    }

    fn tensor_info(&mut self, alignment: u64) -> Result<TensorInfo> {
        let name = self.string("tensor name")?;
        let dim_count = self.u32("tensor dimension count")? as usize;
        if dim_count > MAX_DIMS {
            return Err(Error::Invalid(format!(
                "the tensor {name:?} has {dim_count} dimensions; GGUF allows at most {MAX_DIMS}"
            )));
        }
        let mut dims = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            dims.push(self.u64("tensor dimensions")?);
        }
        let type_id = self.u32("tensor type")?;
        let offset = self.u64("tensor offset")?;

        let elements = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::Invalid(format!("the tensor {name:?} is impossibly large")))?;
        if offset % alignment != 0 {
            return Err(Error::Invalid(format!(
                "the tensor {name:?} starts at {offset}, which isn't a multiple of the alignment ({alignment})"
            )));
        }
        let kind = TensorType::from_id(type_id);
        let bytes = match kind {
            None => None,
            Some(kind) => {
                let row = dims.first().copied().unwrap_or(1);
                if row % kind.block_len() != 0 {
                    return Err(Error::Invalid(format!(
                        "the tensor {name:?}'s rows are {row} long, not a multiple of {kind}'s blocks of {}",
                        kind.block_len()
                    )));
                }
                Some(kind.bytes_for(elements).ok_or_else(|| {
                    Error::Invalid(format!("the tensor {name:?} is impossibly large"))
                })?)
            }
        };
        Ok(TensorInfo {
            name,
            dims,
            type_id,
            kind,
            offset,
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Builds GGUF files in memory, piece by piece, so each test can make exactly the
    // file it needs, including damaged ones.

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    fn header(version: u32, tensors: u64, metadata: u64) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend(version.to_le_bytes());
        out.extend(tensors.to_le_bytes());
        out.extend(metadata.to_le_bytes());
        out
    }

    fn kv(out: &mut Vec<u8>, key: &str, type_id: u32, value: &[u8]) {
        put_str(out, key);
        out.extend(type_id.to_le_bytes());
        out.extend(value);
    }

    fn kv_str(out: &mut Vec<u8>, key: &str, value: &str) {
        let mut v = Vec::new();
        put_str(&mut v, value);
        kv(out, key, 8, &v);
    }

    fn kv_str_array(out: &mut Vec<u8>, key: &str, items: &[&str]) {
        let mut v = 8u32.to_le_bytes().to_vec();
        v.extend((items.len() as u64).to_le_bytes());
        for item in items {
            put_str(&mut v, item);
        }
        kv(out, key, 9, &v);
    }

    fn tensor(out: &mut Vec<u8>, name: &str, dims: &[u64], type_id: u32, offset: u64) {
        put_str(out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(type_id.to_le_bytes());
        out.extend(offset.to_le_bytes());
    }

    fn pad(out: &mut Vec<u8>, alignment: usize) {
        out.resize(out.len().next_multiple_of(alignment), 0);
    }

    /// A small file: metadata of most types, an F32 tensor (4 × 2) and a Q8_0 tensor
    /// (32 × 2), aligned to `alignment`.
    fn sample(alignment: u32) -> Vec<u8> {
        let custom = alignment != 32;
        let mut f = header(3, 2, 6 + u64::from(custom));
        kv_str(&mut f, "general.architecture", "llama");
        kv(&mut f, "llama.block_count", 4, &16u32.to_le_bytes());
        kv(&mut f, "llama.rope.freq_base", 6, &500000f32.to_le_bytes());
        kv(&mut f, "tokenizer.ggml.add_bos_token", 7, &[1]);
        kv(&mut f, "llama.vocab_offset", 5, &(-3i32).to_le_bytes());
        kv_str_array(
            &mut f,
            "tokenizer.ggml.tokens",
            &["<s>", "a", "b", "c", "d", "é"],
        );
        if custom {
            kv(&mut f, "general.alignment", 4, &alignment.to_le_bytes());
        }
        let second = 32u64.next_multiple_of(alignment.into());
        tensor(&mut f, "token_embd.weight", &[4, 2], 0, 0);
        tensor(&mut f, "blk.0.attn_q.weight", &[32, 2], 8, second);
        pad(&mut f, alignment as usize);
        let data_start = f.len();
        f.extend((0..32u8).map(|i| i + 100)); // F32: 8 × 4 bytes
        f.resize(data_start + second as usize, 0);
        f.extend((0..68u8).map(|i| i + 1)); // Q8_0: 2 blocks × 34 bytes
        f
    }

    #[test]
    fn reads_metadata() {
        let g = Gguf::parse(&sample(32)).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.arch_u64("block_count"), Some(16));
        assert_eq!(g.arch_f64("rope.freq_base"), Some(500000.0));
        assert_eq!(
            g.get("tokenizer.ggml.add_bos_token")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(g.get("llama.vocab_offset"), Some(&Value::I32(-3)));
        assert_eq!(
            g.get_u64("llama.vocab_offset"),
            None,
            "negative numbers aren't a u64"
        );
        let tokens = g
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[5].as_str(), Some("é"));
        assert_eq!(g.alignment, 32);
    }

    #[test]
    fn finds_tensor_data() {
        let bytes = sample(32);
        let g = Gguf::parse(&bytes).unwrap();
        let embd = g.tensor("token_embd.weight").unwrap();
        assert_eq!(
            (embd.kind, embd.elements(), embd.bytes),
            (Some(TensorType::F32), 8, Some(32))
        );
        let q = g.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(
            (q.kind, q.elements(), q.bytes),
            (Some(TensorType::Q8_0), 64, Some(68))
        );

        assert_eq!(g.data_offset % 32, 0);
        let embd_data = &bytes[g.tensor_range(embd).unwrap()];
        assert_eq!(embd_data[0], 100);
        assert_eq!(embd_data[31], 131);
        let q_data = &bytes[g.tensor_range(q).unwrap()];
        assert_eq!((q_data[0], q_data[67], q_data.len()), (1, 68, 68));
    }

    #[test]
    fn follows_a_custom_alignment() {
        let bytes = sample(64);
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.alignment, 64);
        assert_eq!(g.data_offset % 64, 0);
        let q = g.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(bytes[g.tensor_range(q).unwrap()][0], 1);
    }

    #[test]
    fn opens_a_file_from_disk() {
        let path = std::env::temp_dir().join(format!("vireo-gguf-{}.gguf", std::process::id()));
        std::fs::write(&path, sample(32)).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let q = file.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(file.tensor_data(q).unwrap().len(), 68);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn keeps_tensors_of_unknown_types() {
        let mut f = header(3, 1, 0);
        tensor(&mut f, "future.weight", &[10], 200, 0);
        let g = Gguf::parse(&f).unwrap();
        let t = &g.tensors[0];
        assert_eq!((t.type_id, t.kind, t.bytes), (200, None, None));
        assert_eq!(g.tensor_range(t), None);
    }

    #[test]
    fn refuses_other_files_and_versions() {
        assert!(matches!(
            Gguf::parse(b"PK\x03\x04 a zip file"),
            Err(Error::NotGguf)
        ));
        assert!(matches!(Gguf::parse(b"GG"), Err(Error::NotGguf)));
        assert!(matches!(
            Gguf::parse(&header(1, 0, 0)),
            Err(Error::UnsupportedVersion(1))
        ));
        assert!(matches!(
            Gguf::parse(&header(3u32.swap_bytes(), 0, 0)),
            Err(Error::BigEndian)
        ));
        assert!(Gguf::parse(&header(2, 0, 0)).is_ok());
    }

    #[test]
    fn refuses_a_cut_off_file() {
        let bytes = sample(32);
        // Cut inside the header, inside the tensor table, and inside the tensor data.
        for len in [10, 30, 200, bytes.len() - 1] {
            assert!(Gguf::parse(&bytes[..len]).is_err(), "cut at {len}");
        }
    }

    #[test]
    fn refuses_impossible_lengths_without_allocating() {
        let mut f = header(3, 0, 1);
        put_str(&mut f, "key");
        f.extend(8u32.to_le_bytes());
        f.extend(u64::MAX.to_le_bytes()); // a string "as long as" u64::MAX
        assert!(matches!(Gguf::parse(&f), Err(Error::Truncated { .. })));

        let f = header(3, u64::MAX, 0); // more tensors than the file could hold
        assert!(matches!(Gguf::parse(&f), Err(Error::Invalid(_))));
    }

    #[test]
    fn refuses_inconsistent_tensors() {
        let mut misaligned = header(3, 1, 0);
        tensor(&mut misaligned, "w", &[4], 0, 4);
        pad(&mut misaligned, 32);
        misaligned.extend([0u8; 64]);
        assert!(
            matches!(Gguf::parse(&misaligned), Err(Error::Invalid(m)) if m.contains("alignment"))
        );

        let mut partial_block = header(3, 1, 0);
        tensor(&mut partial_block, "w", &[31], 8, 0);
        assert!(
            matches!(Gguf::parse(&partial_block), Err(Error::Invalid(m)) if m.contains("blocks"))
        );

        let mut past_end = header(3, 1, 0);
        tensor(&mut past_end, "w", &[1024], 0, 0);
        pad(&mut past_end, 32);
        assert!(
            matches!(Gguf::parse(&past_end), Err(Error::Invalid(m)) if m.contains("past the end"))
        );

        let mut twice = header(3, 2, 0);
        tensor(&mut twice, "w", &[1], 0, 0);
        tensor(&mut twice, "w", &[1], 0, 32);
        pad(&mut twice, 32);
        twice.extend([0u8; 64]);
        assert!(matches!(Gguf::parse(&twice), Err(Error::Invalid(m)) if m.contains("twice")));

        let mut too_many_dims = header(3, 1, 0);
        tensor(&mut too_many_dims, "w", &[1, 1, 1, 1, 1], 0, 0);
        assert!(
            matches!(Gguf::parse(&too_many_dims), Err(Error::Invalid(m)) if m.contains("dimensions"))
        );
    }

    #[test]
    fn refuses_bad_metadata() {
        let mut twice = header(3, 0, 2);
        kv_str(&mut twice, "general.name", "a");
        kv_str(&mut twice, "general.name", "b");
        assert!(matches!(Gguf::parse(&twice), Err(Error::Invalid(m)) if m.contains("twice")));

        let mut bad_bool = header(3, 0, 1);
        kv(&mut bad_bool, "flag", 7, &[2]);
        assert!(Gguf::parse(&bad_bool).is_err());

        let mut bad_alignment = header(3, 0, 1);
        kv(
            &mut bad_alignment,
            "general.alignment",
            4,
            &48u32.to_le_bytes(),
        );
        assert!(
            matches!(Gguf::parse(&bad_alignment), Err(Error::Invalid(m)) if m.contains("power of two"))
        );
    }

    #[test]
    fn shows_values_briefly() {
        let tokens = Value::Array(
            ValueType::String,
            ["a", "b", "c", "d", "e"]
                .iter()
                .map(|s| Value::String(s.to_string()))
                .collect(),
        );
        assert_eq!(
            tokens.to_string(),
            r#"["a", "b", "c", "d", … 5 strings in all]"#
        );
        assert_eq!(Value::String("line\n".into()).to_string(), r#""line\n""#);
        let long = Value::String("x".repeat(100));
        assert!(long.to_string().ends_with("…\" (100 characters)"));
    }
}
