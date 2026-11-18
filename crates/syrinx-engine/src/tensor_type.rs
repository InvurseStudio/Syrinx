//! How a GGUF file stores a tensor's numbers.
//!
//! Each type packs numbers in blocks: a run of `block_len` numbers stored in
//! `block_bytes` bytes, together with whatever scales the block needs. The ids and sizes
//! are GGUF's (ggml's `ggml_type`) and must match it exactly, since they decide where
//! every tensor's data starts and ends. Ids that ggml has retired (4, 5, 31–33, 36–38)
//! are left out.

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
    TQ1_0,
    TQ2_0,
    MXFP4,
}

struct Layout {
    kind: TensorType,
    id: u32,
    name: &'static str,
    block_len: u64,
    block_bytes: u64,
}

const fn layout(
    kind: TensorType,
    id: u32,
    name: &'static str,
    block_len: u64,
    block_bytes: u64,
) -> Layout {
    Layout {
        kind,
        id,
        name,
        block_len,
        block_bytes,
    }
}

use TensorType::*;

/// Every type, with its GGUF id, its name as llama.cpp prints it, and its block.
/// Block sizes in bytes, worked out from ggml's block structs:
/// - Q8_0: an f16 scale and 32 int8s (2 + 32)
/// - Q4_K: f16 scale and minimum, 12 bytes of packed sub-block scales, 128 bytes of
///   4-bit numbers (2 + 2 + 12 + 128)
/// - Q6_K: 128 bytes of low 4 bits, 64 of high 2 bits, 16 int8 scales, an f16 scale
///   (128 + 64 + 16 + 2)
const LAYOUTS: &[Layout] = &[
    layout(F32, 0, "F32", 1, 4),
    layout(F16, 1, "F16", 1, 2),
    layout(Q4_0, 2, "Q4_0", 32, 18),
    layout(Q4_1, 3, "Q4_1", 32, 20),
    layout(Q5_0, 6, "Q5_0", 32, 22),
    layout(Q5_1, 7, "Q5_1", 32, 24),
    layout(Q8_0, 8, "Q8_0", 32, 34),
    layout(Q8_1, 9, "Q8_1", 32, 36),
    layout(Q2_K, 10, "Q2_K", 256, 84),
    layout(Q3_K, 11, "Q3_K", 256, 110),
    layout(Q4_K, 12, "Q4_K", 256, 144),
    layout(Q5_K, 13, "Q5_K", 256, 176),
    layout(Q6_K, 14, "Q6_K", 256, 210),
    layout(Q8_K, 15, "Q8_K", 256, 292),
    layout(IQ2_XXS, 16, "IQ2_XXS", 256, 66),
    layout(IQ2_XS, 17, "IQ2_XS", 256, 74),
    layout(IQ3_XXS, 18, "IQ3_XXS", 256, 98),
    layout(IQ1_S, 19, "IQ1_S", 256, 50),
    layout(IQ4_NL, 20, "IQ4_NL", 32, 18),
    layout(IQ3_S, 21, "IQ3_S", 256, 110),
    layout(IQ2_S, 22, "IQ2_S", 256, 82),
    layout(IQ4_XS, 23, "IQ4_XS", 256, 136),
    layout(I8, 24, "I8", 1, 1),
    layout(I16, 25, "I16", 1, 2),
    layout(I32, 26, "I32", 1, 4),
    layout(I64, 27, "I64", 1, 8),
    layout(F64, 28, "F64", 1, 8),
    layout(IQ1_M, 29, "IQ1_M", 256, 56),
    layout(BF16, 30, "BF16", 1, 2),
    layout(TQ1_0, 34, "TQ1_0", 256, 54),
    layout(TQ2_0, 35, "TQ2_0", 256, 66),
    layout(MXFP4, 39, "MXFP4", 32, 17),
];

impl TensorType {
    /// The type with this GGUF id, or `None` for an id we don't know.
    pub fn from_id(id: u32) -> Option<Self> {
        LAYOUTS.iter().find(|l| l.id == id).map(|l| l.kind)
    }

    fn layout(self) -> &'static Layout {
        LAYOUTS
            .iter()
            .find(|l| l.kind == self)
            .expect("every TensorType has a layout")
    }

    /// Its GGUF id.
    pub fn id(self) -> u32 {
        self.layout().id
    }

    /// Its name as llama.cpp prints it, such as "Q4_K".
    pub fn name(self) -> &'static str {
        self.layout().name
    }

    /// How many numbers one block holds.
    pub fn block_len(self) -> u64 {
        self.layout().block_len
    }

    /// How many bytes one block takes.
    pub fn block_bytes(self) -> u64 {
        self.layout().block_bytes
    }

    /// The bytes `elements` numbers take, or `None` if they don't fill whole blocks
    /// or the size overflows.
    pub fn bytes_for(self, elements: u64) -> Option<u64> {
        if !elements.is_multiple_of(self.block_len()) {
            return None;
        }
        (elements / self.block_len()).checked_mul(self.block_bytes())
    }

    /// Bits each number takes, blocks' scales included: 4.5 for Q4_K.
    pub fn bits_per_number(self) -> f64 {
        self.block_bytes() as f64 * 8.0 / self.block_len() as f64
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_are_unique() {
        for l in LAYOUTS {
            assert_eq!(TensorType::from_id(l.id), Some(l.kind), "{}", l.name);
            assert_eq!(
                LAYOUTS.iter().filter(|o| o.id == l.id).count(),
                1,
                "{}",
                l.name
            );
            assert_eq!(
                LAYOUTS.iter().filter(|o| o.kind == l.kind).count(),
                1,
                "{}",
                l.name
            );
        }
        assert_eq!(TensorType::from_id(4), None);
        assert_eq!(TensorType::from_id(1000), None);
    }

    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!((Q8_0.block_len(), Q8_0.block_bytes()), (32, 34));
        assert_eq!((Q4_K.block_len(), Q4_K.block_bytes()), (256, 144));
        assert_eq!((Q6_K.block_len(), Q6_K.block_bytes()), (256, 210));
        assert_eq!(Q4_K.bits_per_number(), 4.5);
        assert_eq!(Q8_0.bits_per_number(), 8.5);
    }

    #[test]
    fn bytes_for_needs_whole_blocks() {
        assert_eq!(F32.bytes_for(3), Some(12));
        assert_eq!(Q8_0.bytes_for(64), Some(68));
        assert_eq!(Q4_K.bytes_for(256 * 3), Some(144 * 3));
        assert_eq!(Q4_K.bytes_for(255), None);
        assert_eq!(F32.bytes_for(u64::MAX), None);
    }
}
